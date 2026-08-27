/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/geometry_payload.hpp"

#include "chapter_binary_utils.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <format>
#include <limits>
#include <new>
#include <optional>
#include <ranges>
#include <set>
#include <string_view>
#include <utility>

namespace lfs::io::project {

    namespace {

        constexpr std::size_t PROPERTY_DESCRIPTOR_BYTES = 48;
        constexpr std::size_t PROPERTY_ALIGNMENT = 64;
        constexpr std::size_t PCLD_HEADER_BYTES = 24;
        constexpr std::size_t MESH_HEADER_BYTES = 32;

        using chapter_binary::align_up;
        using chapter_binary::all_zero;
        using chapter_binary::checked_add;
        using chapter_binary::checked_mul;
        using chapter_binary::read_f32;
        using chapter_binary::read_i32;
        using chapter_binary::read_u16;
        using chapter_binary::read_u32;
        using chapter_binary::read_u64;
        using chapter_binary::valid_utf8;
        using chapter_binary::write_f32;
        using chapter_binary::write_i32;
        using chapter_binary::write_u16;
        using chapter_binary::write_u32;
        using chapter_binary::write_u64;

        struct PropertyDescriptor {
            std::string name;
            std::uint16_t components = 0;
            GeometryDtype dtype = GeometryDtype::UInt8;
            std::uint32_t encoding = 0;
            std::uint64_t byte_offset = 0;
            std::uint64_t byte_length = 0;
        };

        struct PointProperty {
            std::string_view name;
            lfs::core::Tensor lfs::core::PointCloud::*tensor;
            std::uint16_t components;
        };

        constexpr std::array POINT_PROPERTIES{
            PointProperty{"means", &lfs::core::PointCloud::means, 3},
            PointProperty{"colors", &lfs::core::PointCloud::colors, 3},
            PointProperty{"normals", &lfs::core::PointCloud::normals, 3},
            PointProperty{"sh0", &lfs::core::PointCloud::sh0, 0},
            PointProperty{"shN", &lfs::core::PointCloud::shN, 0},
            PointProperty{"opacity", &lfs::core::PointCloud::opacity, 1},
            PointProperty{"scaling", &lfs::core::PointCloud::scaling, 3},
            PointProperty{"rotation", &lfs::core::PointCloud::rotation, 4},
            PointProperty{"attribute_names", nullptr, 0},
        };

        struct MeshProperty {
            std::string_view name;
            lfs::core::Tensor lfs::core::MeshData::*tensor;
            std::uint16_t components;
        };

        constexpr std::array MESH_PROPERTIES{
            MeshProperty{"vertices", &lfs::core::MeshData::vertices, 3},
            MeshProperty{"normals", &lfs::core::MeshData::normals, 3},
            MeshProperty{"tangents", &lfs::core::MeshData::tangents, 4},
            MeshProperty{"texcoords", &lfs::core::MeshData::texcoords, 2},
            MeshProperty{"colors", &lfs::core::MeshData::colors, 4},
            MeshProperty{"indices", nullptr, 0},
            MeshProperty{"submeshes", nullptr, 0},
            MeshProperty{"materials", nullptr, 0},
            MeshProperty{"textures", nullptr, 0},
        };

        lfs::Error geometry_error(const lfs::ErrorCode code,
                                  std::string user_message,
                                  std::string detail,
                                  const std::string_view field = {}) {
            lfs::SmallFields fields;
            if (!field.empty()) {
                fields.add("field", field);
            }
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(user_message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            });
        }

        template <typename T>
        lfs::Result<T> data_loss(std::string detail, const std::string_view field = {}) {
            auto error = geometry_error(
                lfs::ErrorCode::DataLoss,
                "The project contains invalid geometry data.",
                std::move(detail),
                field);
            if constexpr (std::is_void_v<T>) {
                return lfs::Result<void>::failure(std::move(error));
            } else {
                return error;
            }
        }

        lfs::Result<void> invalid_geometry(std::string detail,
                                           const std::string_view field = {}) {
            return lfs::Result<void>::failure(geometry_error(
                lfs::ErrorCode::InvalidArgument,
                "The geometry cannot be saved.",
                std::move(detail),
                field));
        }

        std::optional<std::size_t> geometry_dtype_size(
            const GeometryDtype dtype) {
            switch (dtype) {
            case GeometryDtype::Float32:
            case GeometryDtype::UInt32:
            case GeometryDtype::Int32:
                return 4;
            case GeometryDtype::Float16:
            case GeometryDtype::UInt16:
                return 2;
            case GeometryDtype::UInt8:
                return 1;
            case GeometryDtype::Float64:
                return 8;
            }
            return std::nullopt;
        }

        std::optional<lfs::core::DataType> tensor_dtype(
            const GeometryDtype dtype) {
            switch (dtype) {
            case GeometryDtype::Float32:
                return lfs::core::DataType::Float32;
            case GeometryDtype::UInt8:
                return lfs::core::DataType::UInt8;
            default:
                return std::nullopt;
            }
        }

        bool is_pcld_known_property(const std::string_view name) {
            return std::ranges::find(
                       POINT_PROPERTIES,
                       name,
                       &PointProperty::name) != POINT_PROPERTIES.end();
        }

        bool is_mesh_known_property(const std::string_view name) {
            return std::ranges::find(
                       MESH_PROPERTIES,
                       name,
                       &MeshProperty::name) != MESH_PROPERTIES.end();
        }

        template <typename IsKnown>
        lfs::Result<void> append_opaque_property(
            std::vector<GeometryPropertyPlane>& retained,
            GeometryPropertyPlane property,
            const std::string_view fourcc,
            IsKnown is_known) {
            if (is_known(property.name)) {
                return invalid_geometry(
                    std::format(
                        "'{}' is a {} v1 well-known property and cannot be "
                        "registered as opaque",
                        property.name,
                        fourcc),
                    property.name);
            }
            if (std::ranges::any_of(
                    retained,
                    [&](const GeometryPropertyPlane& existing) {
                        return existing.name == property.name;
                    })) {
                return invalid_geometry(
                    std::format(
                        "{} opaque property '{}' already exists",
                        fourcc,
                        property.name),
                    property.name);
            }
            retained.push_back(std::move(property));
            return {};
        }

        lfs::Result<std::vector<PropertyDescriptor>> parse_descriptors(
            const std::span<const std::byte> payload,
            const std::size_t table_offset,
            const std::uint16_t property_count,
            const std::uint64_t minimum_plane_offset) {
            std::uint64_t table_bytes = 0;
            if (!checked_mul<std::uint64_t>(
                    property_count, PROPERTY_DESCRIPTOR_BYTES, table_bytes)) {
                return data_loss<std::vector<PropertyDescriptor>>(
                    "Geometry property descriptor table size overflows u64",
                    "property_count");
            }
            std::uint64_t table_end = 0;
            if (!checked_add<std::uint64_t>(
                    table_offset, table_bytes, table_end) ||
                table_end > payload.size()) {
                return data_loss<std::vector<PropertyDescriptor>>(
                    std::format(
                        "Geometry descriptor table [0x{:x}, 0x{:x}) exceeds "
                        "{}-byte payload",
                        table_offset,
                        table_end,
                        payload.size()),
                    "property_count");
            }

            std::vector<PropertyDescriptor> descriptors;
            descriptors.reserve(property_count);
            std::set<std::string> names;
            for (std::uint16_t index = 0; index < property_count; ++index) {
                const std::size_t offset =
                    table_offset +
                    static_cast<std::size_t>(index) *
                        PROPERTY_DESCRIPTOR_BYTES;
                const auto name_bytes = payload.subspan(offset, 16);
                const auto zero = std::ranges::find(name_bytes, std::byte{0});
                const std::size_t name_length =
                    zero == name_bytes.end()
                        ? name_bytes.size()
                        : static_cast<std::size_t>(
                              std::distance(name_bytes.begin(), zero));
                if (name_length == 0) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property descriptor {} has an empty name",
                            index),
                        "properties.name");
                }
                if (zero != name_bytes.end() &&
                    !all_zero(name_bytes.subspan(name_length))) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property descriptor {} name padding is "
                            "not zero",
                            index),
                        "properties.name");
                }
                std::string name(
                    reinterpret_cast<const char*>(name_bytes.data()),
                    name_length);
                if (!valid_utf8(name)) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property descriptor {} name is not valid "
                            "UTF-8",
                            index),
                        "properties.name");
                }
                if (!names.emplace(name).second) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property '{}' appears more than once",
                            name),
                        "properties.name");
                }

                const auto components = read_u16(payload, offset + 16);
                const auto raw_dtype = read_u16(payload, offset + 18);
                const auto dtype = static_cast<GeometryDtype>(raw_dtype);
                const auto dtype_bytes = geometry_dtype_size(dtype);
                const auto encoding = read_u32(payload, offset + 20);
                const auto byte_offset = read_u64(payload, offset + 24);
                const auto byte_length = read_u64(payload, offset + 32);
                if (components == 0) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property '{}' has zero components", name),
                        "properties.components");
                }
                if (!dtype_bytes) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property '{}' uses unknown dtype {}", name,
                            raw_dtype),
                        "properties.dtype");
                }
                if (encoding == 0) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property '{}' uses zero encoding", name),
                        "properties.encoding");
                }
                if (byte_offset % PROPERTY_ALIGNMENT != 0) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property '{}' plane offset 0x{:x} is not "
                            "{}-byte aligned",
                            name,
                            byte_offset,
                            PROPERTY_ALIGNMENT),
                        "properties.byte_offset");
                }
                if (byte_offset < minimum_plane_offset) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property '{}' plane starts at 0x{:x}, "
                            "before minimum data offset 0x{:x}",
                            name,
                            byte_offset,
                            minimum_plane_offset),
                        "properties.byte_offset");
                }
                std::uint64_t byte_end = 0;
                if (!checked_add(byte_offset, byte_length, byte_end) ||
                    byte_end > payload.size()) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property '{}' plane [0x{:x}, 0x{:x}) "
                            "exceeds {}-byte payload",
                            name,
                            byte_offset,
                            byte_end,
                            payload.size()),
                        "properties.byte_length");
                }
                if (!all_zero(payload.subspan(offset + 40, 8))) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property '{}' descriptor reserved bytes "
                            "are nonzero",
                            name),
                        "properties.reserved");
                }

                descriptors.push_back(PropertyDescriptor{
                    .name = std::move(name),
                    .components = components,
                    .dtype = dtype,
                    .encoding = encoding,
                    .byte_offset = byte_offset,
                    .byte_length = byte_length,
                });
            }

            std::vector<const PropertyDescriptor*> ordered;
            ordered.reserve(descriptors.size());
            for (const auto& descriptor : descriptors) {
                if (descriptor.byte_length != 0) {
                    ordered.push_back(&descriptor);
                }
            }
            std::ranges::sort(
                ordered,
                {},
                [](const PropertyDescriptor* descriptor) {
                    return descriptor->byte_offset;
                });
            for (std::size_t index = 1; index < ordered.size(); ++index) {
                std::uint64_t prior_end = 0;
                [[maybe_unused]] const bool valid_end = checked_add(
                    ordered[index - 1]->byte_offset,
                    ordered[index - 1]->byte_length,
                    prior_end);
                assert(valid_end);
                if (ordered[index]->byte_offset < prior_end) {
                    return data_loss<std::vector<PropertyDescriptor>>(
                        std::format(
                            "Geometry property planes '{}' and '{}' overlap",
                            ordered[index - 1]->name,
                            ordered[index]->name),
                        "properties.byte_offset");
                }
            }
            return descriptors;
        }

        lfs::Result<void> validate_raw_length(
            const PropertyDescriptor& descriptor,
            const std::uint64_t element_count) {
            const auto dtype_bytes = geometry_dtype_size(descriptor.dtype);
            assert(dtype_bytes.has_value());
            std::uint64_t expected = 0;
            if (!checked_mul<std::uint64_t>(
                    element_count, descriptor.components, expected) ||
                !checked_mul<std::uint64_t>(
                    expected, *dtype_bytes, expected)) {
                return data_loss<void>(
                    std::format(
                        "Geometry property '{}' declared size arithmetic "
                        "overflows u64",
                        descriptor.name),
                    "properties.byte_length");
            }
            if (descriptor.byte_length != expected) {
                return data_loss<void>(
                    std::format(
                        "Geometry property '{}' has {} bytes; expected {} = {} "
                        "elements x {} components x {} bytes",
                        descriptor.name,
                        descriptor.byte_length,
                        expected,
                        element_count,
                        descriptor.components,
                        *dtype_bytes),
                    "properties.byte_length");
            }
            return {};
        }

        const PropertyDescriptor* find_descriptor(
            const std::span<const PropertyDescriptor> descriptors,
            const std::string_view name) {
            const auto found = std::ranges::find(
                descriptors, name, &PropertyDescriptor::name);
            return found == descriptors.end() ? nullptr : &*found;
        }

        lfs::Result<void> require_descriptor_shape(
            const PropertyDescriptor& descriptor,
            const GeometryDtype dtype,
            const std::uint16_t components,
            const std::uint32_t encoding = GEOMETRY_ENCODING_RAW) {
            if (descriptor.dtype != dtype ||
                descriptor.components != components ||
                descriptor.encoding != encoding) {
                return data_loss<void>(
                    std::format(
                        "Geometry property '{}' descriptor is "
                        "components={}, dtype={}, encoding={}; expected "
                        "components={}, dtype={}, encoding={}",
                        descriptor.name,
                        descriptor.components,
                        std::to_underlying(descriptor.dtype),
                        descriptor.encoding,
                        components,
                        std::to_underlying(dtype),
                        encoding),
                    "properties");
            }
            return {};
        }

        lfs::Result<void> validate_pcld_known_descriptor(
            const PropertyDescriptor& descriptor,
            const std::uint64_t point_count,
            const std::span<const std::byte> plane) {
            const auto spec = std::ranges::find(
                POINT_PROPERTIES,
                descriptor.name,
                &PointProperty::name);
            assert(spec != POINT_PROPERTIES.end());
            if (descriptor.name == "colors") {
                if (descriptor.components != 3 ||
                    (descriptor.dtype != GeometryDtype::UInt8 &&
                     descriptor.dtype != GeometryDtype::Float32) ||
                    descriptor.encoding != GEOMETRY_ENCODING_RAW) {
                    return data_loss<void>(
                        "PCLD colors must be raw u8x3 or f32x3",
                        "properties.colors");
                }
            } else if (descriptor.name == "sh0" ||
                       descriptor.name == "shN") {
                if (descriptor.dtype != GeometryDtype::Float32 ||
                    descriptor.encoding != GEOMETRY_ENCODING_RAW ||
                    descriptor.components % 3 != 0) {
                    return data_loss<void>(
                        std::format(
                            "PCLD {} must use raw f32 components in RGB "
                            "triples",
                            descriptor.name),
                        "properties");
                }
            } else if (descriptor.name == "attribute_names") {
                if (auto result = require_descriptor_shape(
                        descriptor,
                        GeometryDtype::UInt8,
                        1,
                        PCLD_ENCODING_ATTRIBUTE_NAMES);
                    !result) {
                    return result;
                }
                if (plane.size() < 4) {
                    return data_loss<void>(
                        "PCLD attribute_names string table is truncated",
                        "properties.attribute_names");
                }
                const std::uint32_t count = read_u32(plane, 0);
                std::size_t cursor = 4;
                for (std::uint32_t index = 0; index < count; ++index) {
                    if (cursor > plane.size() ||
                        plane.size() - cursor < sizeof(std::uint16_t)) {
                        return data_loss<void>(
                            std::format(
                                "PCLD attribute_names entry {} length is "
                                "truncated",
                                index),
                            "properties.attribute_names");
                    }
                    const auto length = read_u16(plane, cursor);
                    cursor += sizeof(std::uint16_t);
                    if (length > plane.size() - cursor) {
                        return data_loss<void>(
                            std::format(
                                "PCLD attribute_names entry {} exceeds the "
                                "string table",
                                index),
                            "properties.attribute_names");
                    }
                    const std::string_view name(
                        reinterpret_cast<const char*>(plane.data() + cursor),
                        length);
                    if (!valid_utf8(name)) {
                        return data_loss<void>(
                            std::format(
                                "PCLD attribute_names entry {} is not valid "
                                "UTF-8",
                                index),
                            "properties.attribute_names");
                    }
                    cursor += length;
                }
                if (cursor != plane.size()) {
                    return data_loss<void>(
                        "PCLD attribute_names string table has trailing bytes",
                        "properties.attribute_names");
                }
                return {};
            } else {
                assert(spec->components != 0);
                if (auto result = require_descriptor_shape(
                        descriptor,
                        GeometryDtype::Float32,
                        spec->components);
                    !result) {
                    return result;
                }
            }

            if (descriptor.encoding == GEOMETRY_ENCODING_RAW) {
                return validate_raw_length(descriptor, point_count);
            }
            return {};
        }

        lfs::Result<lfs::core::Tensor> allocate_tensor(
            const GeometryDecodeOptions& options,
            const lfs::core::TensorShape& shape,
            const lfs::core::DataType dtype,
            const std::span<const std::byte> bytes,
            const std::string_view property_name) {
            try {
                auto tensor =
                    options.tensor_allocator
                        ? options.tensor_allocator(shape, dtype)
                        : lfs::core::Tensor::empty(
                              shape, lfs::core::Device::CPU, dtype);
                if (!tensor.is_valid() ||
                    tensor.device() != lfs::core::Device::CPU ||
                    tensor.dtype() != dtype || tensor.shape() != shape ||
                    !tensor.is_contiguous() || tensor.bytes() != bytes.size()) {
                    return geometry_error(
                        lfs::ErrorCode::ContractViolation,
                        "The geometry tensor allocator returned an invalid tensor.",
                        std::format(
                            "Allocator result for '{}' must be a contiguous CPU "
                            "tensor with shape {}, dtype {}, and {} bytes",
                            property_name,
                            shape.str(),
                            lfs::core::dtype_name(dtype),
                            bytes.size()),
                        property_name);
                }
                if (!bytes.empty()) {
                    std::memcpy(
                        tensor.data_ptr(), bytes.data(), bytes.size());
                }
                return tensor;
            } catch (const std::bad_alloc&) {
                return geometry_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "Not enough memory to load the geometry.",
                    std::format(
                        "Allocation failed for geometry property '{}' ({} "
                        "bytes)",
                        property_name,
                        bytes.size()),
                    property_name);
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): translate allocator exceptions to the typed IO boundary.
                return geometry_error(
                    lfs::ErrorCode::Internal,
                    "The geometry tensor could not be created.",
                    std::format(
                        "Tensor allocation for geometry property '{}' failed: "
                        "{}",
                        property_name,
                        error.what()),
                    property_name);
            }
        }

        lfs::Result<GeometryPropertyPlane> tensor_plane(
            const std::string_view name,
            const lfs::core::Tensor& source,
            const std::uint64_t element_count,
            const std::uint16_t required_components,
            const GeometryDtype dtype) {
            if (!source.is_valid()) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The geometry cannot be saved.",
                    std::format(
                        "Geometry property '{}' does not contain a tensor",
                        name),
                    name);
            }
            if (source.ndim() < 1 ||
                source.shape()[0] != element_count) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The geometry cannot be saved.",
                    std::format(
                        "Geometry property '{}' has shape {}; first dimension "
                        "must equal {}",
                        name,
                        source.shape().str(),
                        element_count),
                    name);
            }
            std::uint64_t components = 1;
            for (std::size_t dimension = 1; dimension < source.ndim();
                 ++dimension) {
                if (!checked_mul<std::uint64_t>(
                        components, source.shape()[dimension], components)) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The geometry cannot be saved.",
                        std::format(
                            "Geometry property '{}' component count overflows "
                            "u64",
                            name),
                        name);
                }
            }
            if (components == 0 ||
                components > std::numeric_limits<std::uint16_t>::max()) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The geometry cannot be saved.",
                    std::format(
                        "Geometry property '{}' has unsupported component count "
                        "{}",
                        name,
                        components),
                    name);
            }
            if (required_components != 0 &&
                components != required_components) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The geometry cannot be saved.",
                    std::format(
                        "Geometry property '{}' has {} components; expected {}",
                        name,
                        components,
                        required_components),
                    name);
            }
            const auto expected_dtype = tensor_dtype(dtype);
            assert(expected_dtype.has_value());
            if (source.dtype() != *expected_dtype) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The geometry cannot be saved.",
                    std::format(
                        "Geometry property '{}' uses dtype {}; expected {}",
                        name,
                        lfs::core::dtype_name(source.dtype()),
                        lfs::core::dtype_name(*expected_dtype)),
                    name);
            }

            const auto cpu = source.cpu().contiguous();
            GeometryPropertyPlane plane{
                .name = std::string(name),
                .components = static_cast<std::uint16_t>(components),
                .dtype = dtype,
                .encoding = GEOMETRY_ENCODING_RAW,
                .bytes = std::vector<std::byte>(cpu.bytes()),
            };
            if (!plane.bytes.empty()) {
                std::memcpy(
                    plane.bytes.data(), cpu.data_ptr(), plane.bytes.size());
            }
            return plane;
        }

        lfs::Result<std::vector<std::byte>> encode_attribute_names(
            const std::span<const std::string> names) {
            if (names.size() > std::numeric_limits<std::uint32_t>::max()) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The point cloud cannot be saved.",
                    "Point-cloud attribute name count exceeds u32",
                    "attribute_names");
            }
            std::uint64_t total = 4;
            for (const auto& name : names) {
                if (!valid_utf8(name) ||
                    name.size() > std::numeric_limits<std::uint16_t>::max()) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The point cloud cannot be saved.",
                        std::format(
                            "Point-cloud attribute name is invalid UTF-8 or "
                            "longer than 65535 bytes: '{}'",
                            name),
                        "attribute_names");
                }
                if (!checked_add<std::uint64_t>(
                        total, sizeof(std::uint16_t) + name.size(), total) ||
                    total > std::numeric_limits<std::size_t>::max()) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The point cloud cannot be saved.",
                        "Point-cloud attribute-name table size overflows",
                        "attribute_names");
                }
            }
            std::vector<std::byte> result(
                static_cast<std::size_t>(total));
            write_u32(
                result,
                0,
                static_cast<std::uint32_t>(names.size()));
            std::size_t cursor = 4;
            for (const auto& name : names) {
                write_u16(
                    result,
                    cursor,
                    static_cast<std::uint16_t>(name.size()));
                cursor += sizeof(std::uint16_t);
                if (!name.empty()) {
                    std::memcpy(
                        result.data() + cursor, name.data(), name.size());
                    cursor += name.size();
                }
            }
            assert(cursor == result.size());
            return result;
        }

        std::vector<std::string> decode_attribute_names(
            const std::span<const std::byte> bytes) {
            const auto count = read_u32(bytes, 0);
            std::vector<std::string> result;
            result.reserve(count);
            std::size_t cursor = 4;
            for (std::uint32_t index = 0; index < count; ++index) {
                const auto length = read_u16(bytes, cursor);
                cursor += sizeof(std::uint16_t);
                result.emplace_back(
                    reinterpret_cast<const char*>(bytes.data() + cursor),
                    length);
                cursor += length;
            }
            assert(cursor == bytes.size());
            return result;
        }

        lfs::Result<std::vector<std::byte>> encode_planes(
            const std::array<char, 4> magic,
            const std::uint16_t payload_version,
            const std::uint16_t header_field_at_6,
            const std::uint64_t count_at_8,
            const std::optional<std::uint64_t> count_at_16,
            const std::size_t header_bytes,
            const std::size_t property_count_offset,
            const std::span<const GeometryPropertyPlane> properties) {
            if (properties.size() >
                std::numeric_limits<std::uint16_t>::max()) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The geometry cannot be saved.",
                    "Geometry property count exceeds u16",
                    "property_count");
            }

            std::set<std::string> names;
            std::uint64_t descriptor_bytes = 0;
            if (!checked_mul<std::uint64_t>(
                    properties.size(),
                    PROPERTY_DESCRIPTOR_BYTES,
                    descriptor_bytes)) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The geometry cannot be saved.",
                    "Geometry descriptor table size overflows",
                    "property_count");
            }
            std::uint64_t cursor = 0;
            if (!checked_add<std::uint64_t>(
                    header_bytes, descriptor_bytes, cursor) ||
                !align_up(cursor, PROPERTY_ALIGNMENT, cursor)) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The geometry cannot be saved.",
                    "Geometry descriptor/data offset overflows",
                    "property_count");
            }

            std::vector<std::uint64_t> offsets;
            offsets.reserve(properties.size());
            for (const auto& property : properties) {
                if (property.name.empty() || property.name.size() > 16 ||
                    !valid_utf8(property.name)) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The geometry cannot be saved.",
                        std::format(
                            "Geometry property name '{}' must contain 1..16 "
                            "UTF-8 bytes",
                            property.name),
                        "properties.name");
                }
                if (!names.emplace(property.name).second) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The geometry cannot be saved.",
                        std::format(
                            "Geometry property '{}' is duplicated",
                            property.name),
                        "properties.name");
                }
                if (property.components == 0 ||
                    !geometry_dtype_size(property.dtype) ||
                    property.encoding == 0) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The geometry cannot be saved.",
                        std::format(
                            "Geometry property '{}' has an invalid descriptor",
                            property.name),
                        "properties");
                }
                offsets.push_back(cursor);
                if (!checked_add<std::uint64_t>(
                        cursor, property.bytes.size(), cursor) ||
                    !align_up(cursor, PROPERTY_ALIGNMENT, cursor)) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The geometry cannot be saved.",
                        std::format(
                            "Geometry property '{}' payload size overflows",
                            property.name),
                        "properties.byte_length");
                }
            }
            if (cursor > std::numeric_limits<std::size_t>::max()) {
                return geometry_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "The geometry payload is too large for this process.",
                    std::format(
                        "Geometry payload requires {} bytes", cursor),
                    "payload_size");
            }

            std::vector<std::byte> result(
                static_cast<std::size_t>(cursor), std::byte{0});
            std::memcpy(result.data(), magic.data(), magic.size());
            write_u16(result, 4, payload_version);
            write_u16(result, 6, header_field_at_6);
            write_u64(result, 8, count_at_8);
            if (count_at_16) {
                write_u64(result, 16, *count_at_16);
            }
            write_u16(
                result,
                property_count_offset,
                static_cast<std::uint16_t>(properties.size()));

            for (std::size_t index = 0; index < properties.size(); ++index) {
                const auto& property = properties[index];
                const std::size_t descriptor_offset =
                    header_bytes + index * PROPERTY_DESCRIPTOR_BYTES;
                std::memcpy(
                    result.data() + descriptor_offset,
                    property.name.data(),
                    property.name.size());
                write_u16(
                    result, descriptor_offset + 16, property.components);
                write_u16(
                    result,
                    descriptor_offset + 18,
                    std::to_underlying(property.dtype));
                write_u32(
                    result, descriptor_offset + 20, property.encoding);
                write_u64(
                    result, descriptor_offset + 24, offsets[index]);
                write_u64(
                    result,
                    descriptor_offset + 32,
                    property.bytes.size());
                if (!property.bytes.empty()) {
                    std::memcpy(
                        result.data() + offsets[index],
                        property.bytes.data(),
                        property.bytes.size());
                }
            }
            return result;
        }

        lfs::Result<std::vector<GeometryPropertyPlane>>
        build_point_cloud_properties(const PointCloudPayload& payload) {
            const auto& point_cloud = payload.point_cloud();
            if (!point_cloud || !point_cloud->means.is_valid() ||
                point_cloud->means.ndim() != 2 ||
                point_cloud->means.shape()[1] != 3 ||
                point_cloud->means.dtype() !=
                    lfs::core::DataType::Float32) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The point cloud cannot be saved.",
                    "PCLD means must be a valid f32[N,3] tensor",
                    "means");
            }
            const auto point_count =
                static_cast<std::uint64_t>(point_cloud->means.shape()[0]);

            const auto make_known =
                [&](const std::string_view name)
                -> lfs::Result<std::optional<GeometryPropertyPlane>> {
                if (name == "attribute_names") {
                    if (point_cloud->attribute_names.empty()) {
                        return std::optional<GeometryPropertyPlane>{};
                    }
                    auto bytes =
                        encode_attribute_names(point_cloud->attribute_names);
                    if (!bytes) {
                        return std::move(bytes).error();
                    }
                    return GeometryPropertyPlane{
                        .name = "attribute_names",
                        .components = 1,
                        .dtype = GeometryDtype::UInt8,
                        .encoding = PCLD_ENCODING_ATTRIBUTE_NAMES,
                        .bytes = std::move(*bytes),
                    };
                }

                const auto spec = std::ranges::find(
                    POINT_PROPERTIES, name, &PointProperty::name);
                assert(spec != POINT_PROPERTIES.end() && spec->tensor);
                const auto& tensor = point_cloud.get()->*(spec->tensor);
                if (!tensor.is_valid() ||
                    (tensor.numel() == 0 && name != "means")) {
                    return std::optional<GeometryPropertyPlane>{};
                }
                auto dtype = GeometryDtype::Float32;
                if (name == "colors") {
                    if (tensor.dtype() == lfs::core::DataType::UInt8) {
                        dtype = GeometryDtype::UInt8;
                    } else if (
                        tensor.dtype() != lfs::core::DataType::Float32) {
                        return geometry_error(
                            lfs::ErrorCode::InvalidArgument,
                            "The point cloud cannot be saved.",
                            "PCLD colors must be u8x3 or f32x3",
                            "colors");
                    }
                }
                auto plane = tensor_plane(
                    name,
                    tensor,
                    point_count,
                    spec->components,
                    dtype);
                if (!plane) {
                    return std::move(plane).error();
                }
                return std::optional<GeometryPropertyPlane>(
                    std::move(*plane));
            };

            std::set<std::string_view> emitted;
            std::vector<GeometryPropertyPlane> properties;
            properties.reserve(
                POINT_PROPERTIES.size() +
                payload.retained_properties().size());
            for (const auto& retained : payload.retained_properties()) {
                if (!is_pcld_known_property(retained.name)) {
                    properties.push_back(retained);
                    continue;
                }
                if (!emitted.emplace(retained.name).second) {
                    continue;
                }
                auto current = make_known(retained.name);
                if (!current) {
                    return std::move(current).error();
                }
                if (*current) {
                    properties.push_back(std::move(**current));
                }
            }
            for (const auto& spec : POINT_PROPERTIES) {
                if (!emitted.emplace(spec.name).second) {
                    continue;
                }
                auto current = make_known(spec.name);
                if (!current) {
                    return std::move(current).error();
                }
                if (*current) {
                    properties.push_back(std::move(**current));
                }
            }

            for (const auto& property : properties) {
                if (property.encoding == GEOMETRY_ENCODING_RAW) {
                    PropertyDescriptor descriptor{
                        .name = property.name,
                        .components = property.components,
                        .dtype = property.dtype,
                        .encoding = property.encoding,
                        .byte_length = property.bytes.size(),
                    };
                    if (auto result =
                            validate_raw_length(descriptor, point_count);
                        !result) {
                        return std::move(result).error();
                    }
                }
            }
            return properties;
        }

        lfs::Result<std::vector<std::byte>> encode_submeshes(
            const std::span<const lfs::core::Submesh> submeshes,
            const std::uint64_t index_count,
            const std::size_t material_count) {
            std::uint64_t byte_count = 0;
            if (!checked_mul<std::uint64_t>(
                    submeshes.size(), 12, byte_count) ||
                byte_count > std::numeric_limits<std::size_t>::max()) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The mesh cannot be saved.",
                    "MESH submesh table size overflows",
                    "submeshes");
            }
            for (std::size_t index = 0; index < submeshes.size(); ++index) {
                const auto& submesh = submeshes[index];
                if (submesh.start_index >
                        std::numeric_limits<std::uint32_t>::max() ||
                    submesh.index_count >
                        std::numeric_limits<std::uint32_t>::max() ||
                    submesh.material_index >
                        std::numeric_limits<std::uint32_t>::max()) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        std::format(
                            "MESH submesh {} cannot be represented by u32 "
                            "fields",
                            index),
                        "submeshes");
                }
                std::uint64_t range_end = 0;
                if (!checked_add<std::uint64_t>(
                        submesh.start_index,
                        submesh.index_count,
                        range_end) ||
                    range_end > index_count) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        std::format(
                            "MESH submesh {} range [{}, {}) exceeds {} "
                            "indices",
                            index,
                            submesh.start_index,
                            range_end,
                            index_count),
                        "submeshes");
                }
                if (material_count == 0 ||
                    submesh.material_index >= material_count) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        std::format(
                            "MESH submesh {} material index {} is outside "
                            "[0, {})",
                            index,
                            submesh.material_index,
                            material_count),
                        "submeshes");
                }
            }

            std::vector<std::byte> result(
                static_cast<std::size_t>(byte_count));
            for (std::size_t index = 0; index < submeshes.size(); ++index) {
                const auto& submesh = submeshes[index];
                const auto offset = index * 12;
                write_u32(
                    result,
                    offset,
                    static_cast<std::uint32_t>(submesh.start_index));
                write_u32(
                    result,
                    offset + 4,
                    static_cast<std::uint32_t>(submesh.index_count));
                write_u32(
                    result,
                    offset + 8,
                    static_cast<std::uint32_t>(
                        submesh.material_index));
            }
            return result;
        }

        lfs::Result<std::vector<std::byte>> encode_materials(
            const std::span<const lfs::core::Material> materials,
            const std::size_t texture_count) {
            const auto validate_texture =
                [&](const std::uint32_t runtime_index)
                -> lfs::Result<void> {
                if (runtime_index > texture_count) {
                    return invalid_geometry(
                        std::format(
                            "Material texture handle {} is outside the "
                            "0..{} runtime range",
                            runtime_index,
                            texture_count),
                        "materials.texture_index");
                }
                if (runtime_index != 0 &&
                    static_cast<std::uint64_t>(runtime_index - 1) >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<std::int32_t>::max())) {
                    return invalid_geometry(
                        "Material texture index exceeds i32",
                        "materials.texture_index");
                }
                return {};
            };

            std::uint64_t byte_count = 0;
            for (const auto& material : materials) {
                if (!valid_utf8(material.name) ||
                    material.name.size() >
                        std::numeric_limits<std::uint16_t>::max()) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        std::format(
                            "Material name is invalid UTF-8 or too long: '{}'",
                            material.name),
                        "materials.name");
                }
                std::uint64_t record_bytes = 70;
                if (!checked_add<std::uint64_t>(
                        record_bytes, material.name.size(), record_bytes) ||
                    !align_up(record_bytes, 8, record_bytes) ||
                    !checked_add(byte_count, record_bytes, byte_count) ||
                    byte_count >
                        std::numeric_limits<std::size_t>::max()) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        "MESH material table size overflows",
                        "materials");
                }
                const std::array runtime_indices = {
                    material.albedo_tex,
                    material.normal_tex,
                    material.metallic_roughness_tex,
                    material.emissive_tex,
                    material.ao_tex,
                };
                for (const auto runtime_index : runtime_indices) {
                    if (auto result = validate_texture(runtime_index);
                        !result) {
                        return std::move(result).error();
                    }
                }
            }
            std::vector<std::byte> result(
                static_cast<std::size_t>(byte_count), std::byte{0});
            std::size_t cursor = 0;
            const auto encode_texture =
                [&](const std::uint32_t runtime_index)
                -> lfs::Result<std::int32_t> {
                if (runtime_index == 0) {
                    return std::int32_t{-1};
                }
                if (runtime_index > texture_count) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        std::format(
                            "Material texture handle {} is outside the "
                            "1..{} runtime range",
                            runtime_index,
                            texture_count),
                        "materials.texture_index");
                }
                const auto zero_based =
                    static_cast<std::uint64_t>(runtime_index - 1);
                if (zero_based >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int32_t>::max())) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        "Material texture index exceeds i32",
                        "materials.texture_index");
                }
                return static_cast<std::int32_t>(zero_based);
            };

            for (const auto& material : materials) {
                write_u16(
                    result,
                    cursor,
                    static_cast<std::uint16_t>(material.name.size()));
                cursor += 2;
                if (!material.name.empty()) {
                    std::memcpy(
                        result.data() + cursor,
                        material.name.data(),
                        material.name.size());
                    cursor += material.name.size();
                }
                for (std::size_t component = 0; component < 4; ++component) {
                    write_f32(
                        result,
                        cursor,
                        material.base_color[component]);
                    cursor += 4;
                }
                for (std::size_t component = 0; component < 3; ++component) {
                    write_f32(
                        result,
                        cursor,
                        material.emissive[component]);
                    cursor += 4;
                }
                write_f32(result, cursor, material.metallic);
                cursor += 4;
                write_f32(result, cursor, material.roughness);
                cursor += 4;
                write_f32(result, cursor, material.ao);
                cursor += 4;

                constexpr std::size_t TEXTURE_FIELDS = 5;
                const std::array runtime_indices = {
                    material.albedo_tex,
                    material.normal_tex,
                    material.metallic_roughness_tex,
                    material.emissive_tex,
                    material.ao_tex,
                };
                static_assert(
                    runtime_indices.size() == TEXTURE_FIELDS);
                for (const auto runtime_index : runtime_indices) {
                    auto encoded = encode_texture(runtime_index);
                    if (!encoded) {
                        return std::move(encoded).error();
                    }
                    write_i32(result, cursor, *encoded);
                    cursor += 4;
                }
                result[cursor++] = static_cast<std::byte>(
                    material.double_sided ? 1 : 0);
                cursor += 7;
                const auto aligned =
                    (cursor + 7u) & ~std::size_t{7};
                assert(aligned <= result.size());
                cursor = aligned;
            }
            assert(cursor == result.size());
            return result;
        }

        lfs::Result<std::vector<std::byte>> encode_textures(
            const std::span<const lfs::core::TextureImage> textures) {
            std::uint64_t total = 0;
            for (std::size_t index = 0; index < textures.size(); ++index) {
                const auto& texture = textures[index];
                if (texture.width < 0 || texture.height < 0 ||
                    texture.channels < 0) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        std::format(
                            "Texture {} has negative dimensions {}x{}x{}",
                            index,
                            texture.width,
                            texture.height,
                            texture.channels),
                        "textures");
                }
                std::uint64_t expected = 0;
                if (!checked_mul<std::uint64_t>(
                        static_cast<std::uint64_t>(texture.width),
                        static_cast<std::uint64_t>(texture.height),
                        expected) ||
                    !checked_mul<std::uint64_t>(
                        expected,
                        static_cast<std::uint64_t>(texture.channels),
                        expected) ||
                    expected != texture.pixels.size()) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        std::format(
                            "Texture {} byte count {} does not equal {}x{}x{}",
                            index,
                            texture.pixels.size(),
                            texture.width,
                            texture.height,
                            texture.channels),
                        "textures");
                }
                if (!checked_add<std::uint64_t>(
                        total, 20, total) ||
                    !checked_add<std::uint64_t>(
                        total, expected, total) ||
                    total > std::numeric_limits<std::size_t>::max()) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        "MESH texture table size overflows",
                        "textures");
                }
            }
            std::vector<std::byte> result(
                static_cast<std::size_t>(total));
            std::size_t cursor = 0;
            for (const auto& texture : textures) {
                write_u32(
                    result,
                    cursor,
                    static_cast<std::uint32_t>(texture.width));
                write_u32(
                    result,
                    cursor + 4,
                    static_cast<std::uint32_t>(texture.height));
                write_u32(
                    result,
                    cursor + 8,
                    static_cast<std::uint32_t>(texture.channels));
                write_u64(
                    result,
                    cursor + 12,
                    texture.pixels.size());
                cursor += 20;
                if (!texture.pixels.empty()) {
                    std::memcpy(
                        result.data() + cursor,
                        texture.pixels.data(),
                        texture.pixels.size());
                    cursor += texture.pixels.size();
                }
            }
            assert(cursor == result.size());
            return result;
        }

        lfs::Result<GeometryPropertyPlane> encode_mesh_indices(
            const lfs::core::Tensor& indices,
            const std::uint64_t index_count,
            const std::uint64_t vertex_count) {
            if (!indices.is_valid() || indices.ndim() != 2 ||
                indices.shape()[1] != 3 ||
                indices.dtype() != lfs::core::DataType::Int32 ||
                indices.numel() != index_count) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The mesh cannot be saved.",
                    "MESH indices must be a valid i32[F,3] tensor",
                    "indices");
            }
            const auto cpu = indices.cpu().contiguous();
            const auto* source = cpu.ptr<std::int32_t>();
            for (std::size_t index = 0; index < cpu.numel(); ++index) {
                if (source[index] < 0 ||
                    static_cast<std::uint64_t>(source[index]) >=
                        vertex_count) {
                    return geometry_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The mesh cannot be saved.",
                        std::format(
                            "MESH index {} value {} is outside [0, {})",
                            index,
                            source[index],
                            vertex_count),
                        "indices");
                }
            }
            GeometryPropertyPlane result{
                .name = "indices",
                .components = 1,
                .dtype = GeometryDtype::UInt32,
                .encoding = GEOMETRY_ENCODING_RAW,
                .bytes =
                    std::vector<std::byte>(cpu.numel() * sizeof(std::uint32_t)),
            };
            for (std::size_t index = 0; index < cpu.numel(); ++index) {
                write_u32(
                    result.bytes,
                    index * sizeof(std::uint32_t),
                    static_cast<std::uint32_t>(source[index]));
            }
            return result;
        }

        lfs::Result<std::vector<GeometryPropertyPlane>>
        build_mesh_properties(const MeshPayload& payload) {
            const auto& mesh = payload.mesh();
            if (!mesh || !mesh->vertices.is_valid() ||
                mesh->vertices.ndim() != 2 ||
                mesh->vertices.shape()[1] != 3 ||
                mesh->vertices.dtype() !=
                    lfs::core::DataType::Float32 ||
                !mesh->indices.is_valid() ||
                mesh->indices.ndim() != 2 ||
                mesh->indices.shape()[1] != 3 ||
                mesh->indices.dtype() !=
                    lfs::core::DataType::Int32) {
                return geometry_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The mesh cannot be saved.",
                    "MESH requires f32[V,3] vertices and i32[F,3] indices",
                    "mesh");
            }
            const std::uint64_t vertex_count =
                mesh->vertices.shape()[0];
            const std::uint64_t index_count = mesh->indices.numel();

            const auto make_known =
                [&](const std::string_view name)
                -> lfs::Result<std::optional<GeometryPropertyPlane>> {
                if (name == "indices") {
                    auto plane = encode_mesh_indices(
                        mesh->indices, index_count, vertex_count);
                    if (!plane) {
                        return std::move(plane).error();
                    }
                    return std::optional<GeometryPropertyPlane>(
                        std::move(*plane));
                }

                const auto special_plane =
                    [&](const std::string_view property,
                        const std::uint16_t components,
                        const GeometryDtype dtype,
                        const std::uint32_t encoding,
                        auto encode)
                    -> lfs::Result<std::optional<GeometryPropertyPlane>> {
                    auto bytes = encode();
                    if (!bytes) {
                        return std::move(bytes).error();
                    }
                    if (bytes->empty()) {
                        return std::optional<GeometryPropertyPlane>{};
                    }
                    return GeometryPropertyPlane{
                        .name = std::string(property),
                        .components = components,
                        .dtype = dtype,
                        .encoding = encoding,
                        .bytes = std::move(*bytes),
                    };
                };
                if (name == "submeshes") {
                    return special_plane(
                        name, 3, GeometryDtype::UInt32,
                        MESH_ENCODING_SUBMESHES, [&] {
                            return encode_submeshes(
                                mesh->submeshes,
                                index_count,
                                mesh->materials.size());
                        });
                }
                if (name == "materials") {
                    return special_plane(
                        name, 1, GeometryDtype::UInt8,
                        MESH_ENCODING_MATERIALS, [&] {
                            return encode_materials(
                                mesh->materials,
                                mesh->texture_images.size());
                        });
                }
                if (name == "textures") {
                    return special_plane(
                        name, 1, GeometryDtype::UInt8,
                        MESH_ENCODING_TEXTURES, [&] {
                            return encode_textures(mesh->texture_images);
                        });
                }

                const auto spec = std::ranges::find(
                    MESH_PROPERTIES, name, &MeshProperty::name);
                assert(spec != MESH_PROPERTIES.end() && spec->tensor);
                const auto& tensor = mesh.get()->*(spec->tensor);
                if (!tensor.is_valid() ||
                    (tensor.numel() == 0 && name != "vertices")) {
                    return std::optional<GeometryPropertyPlane>{};
                }
                auto plane = tensor_plane(
                    name,
                    tensor,
                    vertex_count,
                    spec->components,
                    GeometryDtype::Float32);
                if (!plane) {
                    return std::move(plane).error();
                }
                return std::optional<GeometryPropertyPlane>(
                    std::move(*plane));
            };

            std::set<std::string_view> emitted;
            std::vector<GeometryPropertyPlane> properties;
            properties.reserve(
                MESH_PROPERTIES.size() +
                payload.retained_properties().size());
            for (const auto& retained : payload.retained_properties()) {
                if (!is_mesh_known_property(retained.name)) {
                    properties.push_back(retained);
                    continue;
                }
                if (!emitted.emplace(retained.name).second) {
                    continue;
                }
                auto current = make_known(retained.name);
                if (!current) {
                    return std::move(current).error();
                }
                if (*current) {
                    properties.push_back(std::move(**current));
                }
            }
            for (const auto& spec : MESH_PROPERTIES) {
                if (!emitted.emplace(spec.name).second) {
                    continue;
                }
                auto current = make_known(spec.name);
                if (!current) {
                    return std::move(current).error();
                }
                if (*current) {
                    properties.push_back(std::move(**current));
                }
            }
            return properties;
        }

        lfs::Result<void> validate_mesh_special_planes(
            const std::span<const PropertyDescriptor> descriptors,
            const std::span<const std::byte> payload,
            const std::uint64_t vertex_count,
            const std::uint64_t index_count) {
            for (const auto& descriptor : descriptors) {
                const auto plane = payload.subspan(
                    static_cast<std::size_t>(descriptor.byte_offset),
                    static_cast<std::size_t>(descriptor.byte_length));
                if (descriptor.encoding == GEOMETRY_ENCODING_RAW) {
                    const auto elements =
                        descriptor.name == "indices" ? index_count
                                                     : vertex_count;
                    if (auto result =
                            validate_raw_length(descriptor, elements);
                        !result) {
                        return result;
                    }
                }

                const auto spec = std::ranges::find(
                    MESH_PROPERTIES,
                    descriptor.name,
                    &MeshProperty::name);
                if (spec != MESH_PROPERTIES.end() && spec->tensor) {
                    if (auto result = require_descriptor_shape(
                            descriptor,
                            GeometryDtype::Float32,
                            spec->components);
                        !result) {
                        return result;
                    }
                } else if (descriptor.name == "indices") {
                    if (auto result = require_descriptor_shape(
                            descriptor,
                            GeometryDtype::UInt32,
                            1);
                        !result) {
                        return result;
                    }
                    for (std::uint64_t index = 0; index < index_count;
                         ++index) {
                        const auto stored = read_u32(
                            plane,
                            static_cast<std::size_t>(
                                index * sizeof(std::uint32_t)));
                        if (stored >
                                static_cast<std::uint32_t>(
                                    std::numeric_limits<std::int32_t>::max()) ||
                            stored >= vertex_count) {
                            return data_loss<void>(
                                std::format(
                                    "MESH index {} value {} is outside "
                                    "the runtime range [0, {})",
                                    index,
                                    stored,
                                    vertex_count),
                                "indices");
                        }
                    }
                } else if (descriptor.name == "submeshes") {
                    if (auto result = require_descriptor_shape(
                            descriptor,
                            GeometryDtype::UInt32,
                            3,
                            MESH_ENCODING_SUBMESHES);
                        !result) {
                        return result;
                    }
                    if (descriptor.byte_length % 12 != 0) {
                        return data_loss<void>(
                            "MESH submeshes plane length is not a multiple of "
                            "12",
                            "submeshes");
                    }
                } else if (descriptor.name == "materials") {
                    if (auto result = require_descriptor_shape(
                            descriptor,
                            GeometryDtype::UInt8,
                            1,
                            MESH_ENCODING_MATERIALS);
                        !result) {
                        return result;
                    }
                    std::size_t cursor = 0;
                    while (cursor < plane.size()) {
                        if (plane.size() - cursor < 2) {
                            return data_loss<void>(
                                "MESH material name length is truncated",
                                "materials");
                        }
                        const auto name_bytes =
                            read_u16(plane, cursor);
                        std::uint64_t record_end = cursor;
                        if (!checked_add<std::uint64_t>(
                                record_end,
                                70u + name_bytes,
                                record_end) ||
                            record_end > plane.size()) {
                            return data_loss<void>(
                                "MESH material record exceeds its plane",
                                "materials");
                        }
                        const std::string_view name(
                            reinterpret_cast<const char*>(
                                plane.data() + cursor + 2),
                            name_bytes);
                        if (!valid_utf8(name)) {
                            return data_loss<void>(
                                "MESH material name is not valid UTF-8",
                                "materials");
                        }
                        const std::size_t double_sided_offset =
                            cursor + 2 + name_bytes + 40 + 20;
                        if (std::to_integer<std::uint8_t>(
                                plane[double_sided_offset]) > 1 ||
                            !all_zero(plane.subspan(
                                double_sided_offset + 1, 7))) {
                            return data_loss<void>(
                                "MESH material boolean/reserved bytes are "
                                "invalid",
                                "materials");
                        }
                        std::uint64_t aligned = 0;
                        if (!align_up(record_end, 8, aligned) ||
                            aligned > plane.size() ||
                            !all_zero(plane.subspan(
                                static_cast<std::size_t>(record_end),
                                static_cast<std::size_t>(
                                    aligned - record_end)))) {
                            return data_loss<void>(
                                "MESH material record padding is invalid",
                                "materials");
                        }
                        cursor = static_cast<std::size_t>(aligned);
                    }
                } else if (descriptor.name == "textures") {
                    if (auto result = require_descriptor_shape(
                            descriptor,
                            GeometryDtype::UInt8,
                            1,
                            MESH_ENCODING_TEXTURES);
                        !result) {
                        return result;
                    }
                    std::size_t cursor = 0;
                    while (cursor < plane.size()) {
                        if (plane.size() - cursor < 20) {
                            return data_loss<void>(
                                "MESH texture record header is truncated",
                                "textures");
                        }
                        const auto width = read_u32(plane, cursor);
                        const auto height = read_u32(plane, cursor + 4);
                        const auto channels =
                            read_u32(plane, cursor + 8);
                        const auto pixel_bytes =
                            read_u64(plane, cursor + 12);
                        std::uint64_t expected = 0;
                        if (!checked_mul<std::uint64_t>(
                                width, height, expected) ||
                            !checked_mul<std::uint64_t>(
                                expected, channels, expected) ||
                            expected != pixel_bytes) {
                            return data_loss<void>(
                                "MESH texture pixel count does not match its "
                                "dimensions",
                                "textures");
                        }
                        std::uint64_t record_end = 0;
                        if (!checked_add<std::uint64_t>(
                                cursor + 20u,
                                pixel_bytes,
                                record_end) ||
                            record_end > plane.size()) {
                            return data_loss<void>(
                                "MESH texture record exceeds its plane",
                                "textures");
                        }
                        cursor = static_cast<std::size_t>(record_end);
                    }
                }
            }

            std::size_t material_count = 0;
            if (const auto* materials =
                    find_descriptor(descriptors, "materials")) {
                const auto plane = payload.subspan(
                    static_cast<std::size_t>(materials->byte_offset),
                    static_cast<std::size_t>(materials->byte_length));
                std::size_t cursor = 0;
                while (cursor < plane.size()) {
                    const auto name_bytes = read_u16(plane, cursor);
                    cursor =
                        (cursor + 70u + name_bytes + 7u) &
                        ~std::size_t{7};
                    ++material_count;
                }
                assert(cursor == plane.size());
            }
            if (const auto* submeshes =
                    find_descriptor(descriptors, "submeshes")) {
                const auto plane = payload.subspan(
                    static_cast<std::size_t>(submeshes->byte_offset),
                    static_cast<std::size_t>(submeshes->byte_length));
                for (std::size_t offset = 0; offset < plane.size();
                     offset += 12) {
                    const auto start_index = read_u32(plane, offset);
                    const auto count = read_u32(plane, offset + 4);
                    const auto material_index =
                        read_u32(plane, offset + 8);
                    std::uint64_t range_end = 0;
                    if (!checked_add<std::uint64_t>(
                            start_index, count, range_end) ||
                        range_end > index_count ||
                        material_count == 0 ||
                        material_index >= material_count) {
                        return data_loss<void>(
                            std::format(
                                "MESH submesh at byte {} has range "
                                "[{}, {}) and material {}; limits are "
                                "{} indices and {} materials",
                                offset,
                                start_index,
                                range_end,
                                material_index,
                                index_count,
                                material_count),
                            "submeshes");
                    }
                }
            }
            return {};
        }

        std::vector<lfs::core::Submesh> decode_submeshes(
            const std::span<const std::byte> plane) {
            std::vector<lfs::core::Submesh> result;
            result.reserve(plane.size() / 12);
            for (std::size_t offset = 0; offset < plane.size();
                 offset += 12) {
                result.push_back(lfs::core::Submesh{
                    .start_index = read_u32(plane, offset),
                    .index_count = read_u32(plane, offset + 4),
                    .material_index = read_u32(plane, offset + 8),
                });
            }
            return result;
        }

        lfs::Result<std::vector<lfs::core::TextureImage>>
        decode_textures(const std::span<const std::byte> plane) {
            std::vector<lfs::core::TextureImage> result;
            std::size_t cursor = 0;
            while (cursor < plane.size()) {
                const auto width = read_u32(plane, cursor);
                const auto height = read_u32(plane, cursor + 4);
                const auto channels = read_u32(plane, cursor + 8);
                const auto pixel_bytes = read_u64(plane, cursor + 12);
                if (width >
                        static_cast<std::uint32_t>(
                            std::numeric_limits<int>::max()) ||
                    height >
                        static_cast<std::uint32_t>(
                            std::numeric_limits<int>::max()) ||
                    channels >
                        static_cast<std::uint32_t>(
                            std::numeric_limits<int>::max()) ||
                    pixel_bytes >
                        std::numeric_limits<std::size_t>::max()) {
                    return geometry_error(
                        lfs::ErrorCode::ResourceExhausted,
                        "The mesh texture is too large for this process.",
                        "MESH texture dimensions cannot be represented by the "
                        "runtime",
                        "textures");
                }
                cursor += 20;
                lfs::core::TextureImage texture{
                    .pixels = std::vector<std::uint8_t>(
                        static_cast<std::size_t>(pixel_bytes)),
                    .width = static_cast<int>(width),
                    .height = static_cast<int>(height),
                    .channels = static_cast<int>(channels),
                };
                if (pixel_bytes != 0) {
                    std::memcpy(
                        texture.pixels.data(),
                        plane.data() + cursor,
                        texture.pixels.size());
                }
                cursor += texture.pixels.size();
                result.push_back(std::move(texture));
            }
            return result;
        }

        lfs::Result<std::vector<lfs::core::Material>>
        decode_materials(const std::span<const std::byte> plane,
                         const std::size_t texture_count) {
            std::vector<lfs::core::Material> result;
            std::size_t cursor = 0;
            while (cursor < plane.size()) {
                const auto name_bytes = read_u16(plane, cursor);
                cursor += 2;
                lfs::core::Material material;
                material.name.assign(
                    reinterpret_cast<const char*>(plane.data() + cursor),
                    name_bytes);
                cursor += name_bytes;
                for (std::size_t component = 0; component < 4; ++component) {
                    material.base_color[component] =
                        read_f32(plane, cursor);
                    cursor += 4;
                }
                for (std::size_t component = 0; component < 3; ++component) {
                    material.emissive[component] =
                        read_f32(plane, cursor);
                    cursor += 4;
                }
                material.metallic = read_f32(plane, cursor);
                cursor += 4;
                material.roughness = read_f32(plane, cursor);
                cursor += 4;
                material.ao = read_f32(plane, cursor);
                cursor += 4;

                std::array<std::uint32_t*, 5> destinations = {
                    &material.albedo_tex,
                    &material.normal_tex,
                    &material.metallic_roughness_tex,
                    &material.emissive_tex,
                    &material.ao_tex,
                };
                for (auto* destination : destinations) {
                    const auto stored = read_i32(plane, cursor);
                    cursor += 4;
                    if (stored < -1 ||
                        (stored >= 0 &&
                         static_cast<std::uint64_t>(stored) >=
                             texture_count)) {
                        return geometry_error(
                            lfs::ErrorCode::DataLoss,
                            "The project contains invalid geometry data.",
                            std::format(
                                "MESH material texture index {} is outside "
                                "[-1, {})",
                                stored,
                                texture_count),
                            "materials.texture_index");
                    }
                    *destination =
                        stored < 0
                            ? 0u
                            : static_cast<std::uint32_t>(stored) + 1u;
                }
                material.double_sided =
                    std::to_integer<std::uint8_t>(plane[cursor]) != 0;
                cursor += 8;
                cursor = (cursor + 7u) & ~std::size_t{7};
                result.push_back(std::move(material));
            }
            assert(cursor == plane.size());
            return result;
        }

    } // namespace

    PointCloudPayload::PointCloudPayload(
        std::shared_ptr<lfs::core::PointCloud> point_cloud)
        : point_cloud_(std::move(point_cloud)) {}

    lfs::Result<void> PointCloudPayload::add_opaque_property(
        GeometryPropertyPlane property) {
        return append_opaque_property(
            retained_properties_,
            std::move(property),
            "PCLD",
            is_pcld_known_property);
    }

    MeshPayload::MeshPayload(std::shared_ptr<lfs::core::MeshData> mesh)
        : mesh_(std::move(mesh)) {}

    lfs::Result<void> MeshPayload::add_opaque_property(
        GeometryPropertyPlane property) {
        return append_opaque_property(
            retained_properties_,
            std::move(property),
            "MESH",
            is_mesh_known_property);
    }

    lfs::Result<std::vector<std::byte>>
    encode_point_cloud_payload(const PointCloudPayload& payload) {
        auto properties = build_point_cloud_properties(payload);
        if (!properties) {
            return std::move(properties).error();
        }
        const auto point_count = static_cast<std::uint64_t>(
            payload.point_cloud()->means.shape()[0]);
        return encode_planes(
            {'L', 'P', 'C', 'D'},
            PCLD_PAYLOAD_VERSION,
            GEOMETRY_COORD_F32_ABSOLUTE_LOCAL,
            point_count,
            std::nullopt,
            PCLD_HEADER_BYTES,
            16,
            *properties);
    }

    lfs::Result<PointCloudPayload> decode_point_cloud_payload(
        const std::span<const std::byte> payload,
        const GeometryDecodeOptions& options) {
        if (payload.size() < PCLD_HEADER_BYTES ||
            std::memcmp(payload.data(), "LPCD", 4) != 0) {
            return data_loss<PointCloudPayload>(
                "PCLD payload is truncated or has the wrong magic",
                "magic");
        }
        if (read_u16(payload, 4) != PCLD_PAYLOAD_VERSION) {
            return geometry_error(
                lfs::ErrorCode::Unsupported,
                "This point-cloud payload requires a newer LichtFeld version.",
                std::format(
                    "Unsupported PCLD payload version {}",
                    read_u16(payload, 4)),
                "payload_version");
        }
        if (read_u16(payload, 6) !=
            GEOMETRY_COORD_F32_ABSOLUTE_LOCAL) {
            return geometry_error(
                lfs::ErrorCode::Unsupported,
                "This point-cloud coordinate encoding is not supported.",
                std::format(
                    "Unsupported PCLD coord_encoding {}",
                    read_u16(payload, 6)),
                "coord_encoding");
        }
        if (!all_zero(payload.subspan(18, 6))) {
            return data_loss<PointCloudPayload>(
                "PCLD header reserved bytes are nonzero",
                "reserved");
        }
        const auto point_count = read_u64(payload, 8);
        if (point_count >
            std::numeric_limits<std::size_t>::max()) {
            return geometry_error(
                lfs::ErrorCode::ResourceExhausted,
                "The point cloud is too large for this process.",
                std::format(
                    "PCLD point_count {} exceeds size_t", point_count),
                "point_count");
        }
        const auto property_count = read_u16(payload, 16);
        std::uint64_t table_end = 0;
        std::uint64_t minimum_plane_offset = 0;
        if (!checked_mul<std::uint64_t>(
                property_count,
                PROPERTY_DESCRIPTOR_BYTES,
                table_end) ||
            !checked_add<std::uint64_t>(
                PCLD_HEADER_BYTES, table_end, table_end) ||
            !align_up(
                table_end,
                PROPERTY_ALIGNMENT,
                minimum_plane_offset)) {
            return data_loss<PointCloudPayload>(
                "PCLD descriptor table offset overflows",
                "property_count");
        }
        auto descriptors = parse_descriptors(
            payload,
            PCLD_HEADER_BYTES,
            property_count,
            minimum_plane_offset);
        if (!descriptors) {
            return std::move(descriptors).error();
        }
        if (find_descriptor(*descriptors, "means") == nullptr) {
            return data_loss<PointCloudPayload>(
                "PCLD payload does not contain required 'means' property",
                "properties.means");
        }

        for (const auto& descriptor : *descriptors) {
            const auto plane = payload.subspan(
                static_cast<std::size_t>(descriptor.byte_offset),
                static_cast<std::size_t>(descriptor.byte_length));
            if (is_pcld_known_property(descriptor.name)) {
                if (auto result = validate_pcld_known_descriptor(
                        descriptor, point_count, plane);
                    !result) {
                    return std::move(result).error();
                }
            } else if (descriptor.encoding ==
                       GEOMETRY_ENCODING_RAW) {
                if (auto result =
                        validate_raw_length(descriptor, point_count);
                    !result) {
                    return std::move(result).error();
                }
            }
        }

        auto point_cloud =
            std::make_shared<lfs::core::PointCloud>();
        PointCloudPayload result(point_cloud);
        result.retained_properties_.reserve(descriptors->size());
        for (const auto& descriptor : *descriptors) {
            const auto plane = payload.subspan(
                static_cast<std::size_t>(descriptor.byte_offset),
                static_cast<std::size_t>(descriptor.byte_length));
            result.retained_properties_.push_back(
                GeometryPropertyPlane{
                    .name = descriptor.name,
                    .components = descriptor.components,
                    .dtype = descriptor.dtype,
                    .encoding = descriptor.encoding,
                    .bytes =
                        std::vector<std::byte>(
                            plane.begin(), plane.end()),
                });

            if (descriptor.name == "attribute_names") {
                point_cloud->attribute_names =
                    decode_attribute_names(plane);
                continue;
            }
            const auto spec = std::ranges::find(
                POINT_PROPERTIES,
                descriptor.name,
                &PointProperty::name);
            if (spec == POINT_PROPERTIES.end()) {
                continue;
            }
            assert(spec->tensor);
            const auto dtype = tensor_dtype(descriptor.dtype);
            assert(dtype.has_value());
            lfs::core::TensorShape shape{
                static_cast<std::size_t>(point_count),
                descriptor.components};
            if (descriptor.name == "sh0" ||
                descriptor.name == "shN") {
                if (descriptor.components % 3 != 0) {
                    return data_loss<PointCloudPayload>(
                        std::format(
                            "PCLD {} component count {} is not divisible by "
                            "three",
                            descriptor.name,
                            descriptor.components),
                        descriptor.name);
                }
                shape = lfs::core::TensorShape{
                    static_cast<std::size_t>(point_count),
                    3,
                    static_cast<std::size_t>(
                        descriptor.components / 3)};
            }
            auto tensor = allocate_tensor(
                options,
                shape,
                *dtype,
                plane,
                descriptor.name);
            if (!tensor) {
                return std::move(tensor).error();
            }
            point_cloud.get()->*(spec->tensor) =
                std::move(*tensor);
        }
        assert(point_cloud->means.is_valid());
        assert(
            point_cloud->means.ndim() == 2 &&
            point_cloud->means.shape()[1] == 3);
        return result;
    }

    lfs::Result<std::vector<std::byte>>
    encode_mesh_payload(const MeshPayload& payload) {
        auto properties = build_mesh_properties(payload);
        if (!properties) {
            return std::move(properties).error();
        }
        const auto vertex_count = static_cast<std::uint64_t>(
            payload.mesh()->vertices.shape()[0]);
        const auto index_count =
            static_cast<std::uint64_t>(
                payload.mesh()->indices.numel());
        return encode_planes(
            {'L', 'M', 'S', 'H'},
            MESH_PAYLOAD_VERSION,
            0,
            vertex_count,
            index_count,
            MESH_HEADER_BYTES,
            24,
            *properties);
    }

    lfs::Result<MeshPayload> decode_mesh_payload(
        const std::span<const std::byte> payload,
        const GeometryDecodeOptions& options) {
        if (payload.size() < MESH_HEADER_BYTES ||
            std::memcmp(payload.data(), "LMSH", 4) != 0) {
            return data_loss<MeshPayload>(
                "MESH payload is truncated or has the wrong magic",
                "magic");
        }
        if (read_u16(payload, 4) != MESH_PAYLOAD_VERSION) {
            return geometry_error(
                lfs::ErrorCode::Unsupported,
                "This mesh payload requires a newer LichtFeld version.",
                std::format(
                    "Unsupported MESH payload version {}",
                    read_u16(payload, 4)),
                "payload_version");
        }
        if (read_u16(payload, 6) != 0 ||
            !all_zero(payload.subspan(26, 6))) {
            return data_loss<MeshPayload>(
                "MESH header reserved bytes are nonzero",
                "reserved");
        }
        const auto vertex_count = read_u64(payload, 8);
        const auto index_count = read_u64(payload, 16);
        if (vertex_count >
                std::numeric_limits<std::size_t>::max() ||
            index_count >
                std::numeric_limits<std::size_t>::max()) {
            return geometry_error(
                lfs::ErrorCode::ResourceExhausted,
                "The mesh is too large for this process.",
                std::format(
                    "MESH counts vertex={} index={} exceed size_t",
                    vertex_count,
                    index_count),
                "counts");
        }
        if (index_count % 3 != 0) {
            return data_loss<MeshPayload>(
                std::format(
                    "MESH index_count {} is not divisible by three",
                    index_count),
                "index_count");
        }
        const auto property_count = read_u16(payload, 24);
        std::uint64_t table_end = 0;
        std::uint64_t minimum_plane_offset = 0;
        if (!checked_mul<std::uint64_t>(
                property_count,
                PROPERTY_DESCRIPTOR_BYTES,
                table_end) ||
            !checked_add<std::uint64_t>(
                MESH_HEADER_BYTES, table_end, table_end) ||
            !align_up(
                table_end,
                PROPERTY_ALIGNMENT,
                minimum_plane_offset)) {
            return data_loss<MeshPayload>(
                "MESH descriptor table offset overflows",
                "property_count");
        }
        auto descriptors = parse_descriptors(
            payload,
            MESH_HEADER_BYTES,
            property_count,
            minimum_plane_offset);
        if (!descriptors) {
            return std::move(descriptors).error();
        }
        if (find_descriptor(*descriptors, "vertices") == nullptr ||
            find_descriptor(*descriptors, "indices") == nullptr) {
            return data_loss<MeshPayload>(
                "MESH payload must contain vertices and indices properties",
                "properties");
        }
        if (auto result = validate_mesh_special_planes(
                *descriptors,
                payload,
                vertex_count,
                index_count);
            !result) {
            return std::move(result).error();
        }

        std::size_t texture_count = 0;
        if (const auto* textures =
                find_descriptor(*descriptors, "textures")) {
            const auto plane = payload.subspan(
                static_cast<std::size_t>(textures->byte_offset),
                static_cast<std::size_t>(textures->byte_length));
            std::size_t cursor = 0;
            while (cursor < plane.size()) {
                const auto pixel_bytes = read_u64(plane, cursor + 12);
                cursor += 20 + static_cast<std::size_t>(pixel_bytes);
                ++texture_count;
            }
        }
        if (const auto* materials =
                find_descriptor(*descriptors, "materials")) {
            const auto plane = payload.subspan(
                static_cast<std::size_t>(materials->byte_offset),
                static_cast<std::size_t>(materials->byte_length));
            std::size_t cursor = 0;
            while (cursor < plane.size()) {
                const auto name_bytes = read_u16(plane, cursor);
                const std::size_t texture_offset =
                    cursor + 2 + name_bytes + 40;
                for (std::size_t texture = 0; texture < 5; ++texture) {
                    const auto index =
                        read_i32(plane, texture_offset + texture * 4);
                    if (index < -1 ||
                        (index >= 0 &&
                         static_cast<std::uint64_t>(index) >=
                             texture_count)) {
                        return data_loss<MeshPayload>(
                            std::format(
                                "MESH material texture index {} is outside "
                                "[-1, {})",
                                index,
                                texture_count),
                            "materials.texture_index");
                    }
                }
                cursor =
                    (cursor + 70 + name_bytes + 7u) &
                    ~std::size_t{7};
            }
        }

        auto mesh = std::make_shared<lfs::core::MeshData>();
        MeshPayload result(mesh);
        result.retained_properties_.reserve(descriptors->size());
        for (const auto& descriptor : *descriptors) {
            const auto plane = payload.subspan(
                static_cast<std::size_t>(descriptor.byte_offset),
                static_cast<std::size_t>(descriptor.byte_length));
            result.retained_properties_.push_back(
                GeometryPropertyPlane{
                    .name = descriptor.name,
                    .components = descriptor.components,
                    .dtype = descriptor.dtype,
                    .encoding = descriptor.encoding,
                    .bytes =
                        std::vector<std::byte>(
                            plane.begin(), plane.end()),
                });

            if (descriptor.name == "indices") {
                auto tensor = allocate_tensor(
                    options,
                    lfs::core::TensorShape{
                        static_cast<std::size_t>(index_count / 3),
                        3},
                    lfs::core::DataType::Int32,
                    plane,
                    descriptor.name);
                if (!tensor) {
                    return std::move(tensor).error();
                }
                auto* destination = tensor->ptr<std::int32_t>();
                for (std::size_t index = 0; index < index_count;
                     ++index) {
                    const auto stored =
                        read_u32(plane, index * sizeof(std::uint32_t));
                    if (stored >
                        static_cast<std::uint32_t>(
                            std::numeric_limits<std::int32_t>::max())) {
                        return data_loss<MeshPayload>(
                            std::format(
                                "MESH index {} value {} cannot be represented "
                                "by runtime i32",
                                index,
                                stored),
                            "indices");
                    }
                    destination[index] =
                        static_cast<std::int32_t>(stored);
                }
                mesh->indices = std::move(*tensor);
            } else if (descriptor.name == "submeshes") {
                mesh->submeshes = decode_submeshes(plane);
            } else if (descriptor.name == "textures") {
                auto textures = decode_textures(plane);
                if (!textures) {
                    return std::move(textures).error();
                }
                mesh->texture_images = std::move(*textures);
            } else if (descriptor.name == "materials") {
                auto materials =
                    decode_materials(plane, texture_count);
                if (!materials) {
                    return std::move(materials).error();
                }
                mesh->materials = std::move(*materials);
            } else {
                const auto spec = std::ranges::find(
                    MESH_PROPERTIES,
                    descriptor.name,
                    &MeshProperty::name);
                if (spec == MESH_PROPERTIES.end() || !spec->tensor) {
                    continue;
                }
                auto tensor = allocate_tensor(
                    options,
                    lfs::core::TensorShape{
                        static_cast<std::size_t>(vertex_count),
                        descriptor.components},
                    lfs::core::DataType::Float32,
                    plane,
                    descriptor.name);
                if (!tensor) {
                    return std::move(tensor).error();
                }
                mesh.get()->*(spec->tensor) = std::move(*tensor);
            }
        }
        assert(
            mesh->vertices.is_valid() &&
            mesh->vertices.ndim() == 2 &&
            mesh->vertices.shape()[1] == 3);
        assert(
            mesh->indices.is_valid() && mesh->indices.ndim() == 2 &&
            mesh->indices.shape()[1] == 3);
        return result;
    }

} // namespace lfs::io::project
