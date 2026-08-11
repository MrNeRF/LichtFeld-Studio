/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "model.hpp"
#include "wire_reader.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <unordered_set>

namespace fs = std::filesystem;

namespace lfs::onnx_vulkan::detail {
    namespace {
        constexpr std::int64_t kSupportedIrVersion = 7;
        constexpr std::size_t kMaximumRank = 16;

        [[nodiscard]] Error malformed(std::string message) {
            return {ErrorCode::MalformedModel, std::move(message), {}, {}};
        }

        [[nodiscard]] Error unsupported(std::string message, std::string capability = {}) {
            return {ErrorCode::UnsupportedModel, std::move(message), {}, std::move(capability)};
        }

        [[nodiscard]] std::expected<void, Error>
        require_wire(const FieldKey key, const WireType expected, const std::string_view context) {
            if (key.type != expected) {
                return std::unexpected(malformed(std::string(context) + " field " +
                                                 std::to_string(key.number) + " has the wrong wire type"));
            }
            return {};
        }

        [[nodiscard]] std::string as_string(const std::span<const std::byte> bytes) {
            return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
        }

        [[nodiscard]] std::expected<std::string, Error>
        read_string(WireReader& reader, const std::string_view context) {
            auto bytes = reader.bytes();
            if (!bytes)
                return std::unexpected(malformed(std::string(context) + ": " + bytes.error()));
            if (std::ranges::find(*bytes, std::byte{0}) != bytes->end())
                return std::unexpected(malformed(std::string(context) + " contains an embedded NUL"));
            return as_string(*bytes);
        }

        [[nodiscard]] std::expected<std::uint64_t, Error>
        read_varint(WireReader& reader, const std::string_view context) {
            auto value = reader.varint();
            if (!value)
                return std::unexpected(malformed(std::string(context) + ": " + value.error()));
            return *value;
        }

        [[nodiscard]] std::expected<std::span<const std::byte>, Error>
        read_message(WireReader& reader, const std::string_view context) {
            auto value = reader.bytes();
            if (!value)
                return std::unexpected(malformed(std::string(context) + ": " + value.error()));
            return *value;
        }

        [[nodiscard]] std::expected<void, Error>
        skip(WireReader& reader, const FieldKey key, const std::string_view context) {
            if (auto result = reader.skip(key.type); !result)
                return std::unexpected(malformed(std::string(context) + ": " + result.error()));
            return {};
        }

        [[nodiscard]] std::expected<ElementType, Error> parse_element_type(const std::uint64_t value) {
            switch (value) {
            case 1: return ElementType::Float32;
            case 6: return ElementType::Int32;
            case 7: return ElementType::Int64;
            case 9: return ElementType::Bool;
            default:
                return std::unexpected(unsupported("unsupported ONNX tensor element type " +
                                                   std::to_string(value),
                                                   "tensor type " + std::to_string(value)));
            }
        }

        [[nodiscard]] std::expected<std::size_t, Error>
        checked_element_count(const std::span<const std::int64_t> shape) {
            std::size_t count = 1;
            for (const auto extent : shape) {
                if (extent < 0)
                    return std::unexpected(malformed("initializer has a negative dimension"));
                if (extent == 0)
                    return std::size_t{0};
                const auto size = static_cast<std::size_t>(extent);
                if (count > std::numeric_limits<std::size_t>::max() / size)
                    return std::unexpected(malformed("tensor element count overflows size_t"));
                count *= size;
            }
            return count;
        }

        [[nodiscard]] std::expected<std::size_t, Error>
        checked_byte_count(const ElementType type, const std::span<const std::int64_t> shape) {
            auto count = checked_element_count(shape);
            if (!count)
                return std::unexpected(count.error());
            const auto width = element_size(type);
            if (*count > std::numeric_limits<std::size_t>::max() / width)
                return std::unexpected(malformed("tensor byte count overflows size_t"));
            return *count * width;
        }

        [[nodiscard]] std::expected<std::uint64_t, Error>
        parse_decimal_u64(const std::string_view text, const std::string_view label) {
            if (text.empty())
                return std::unexpected(malformed(std::string(label) + " is empty"));
            std::uint64_t value = 0;
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc{} || end != text.data() + text.size())
                return std::unexpected(malformed(std::string(label) + " is not an unsigned decimal integer"));
            return value;
        }

        struct ParseContext {
            fs::path model_path;
            std::uint64_t max_external_bytes = 0;
            std::uint64_t external_bytes_read = 0;
        };

        struct ExternalEntry {
            std::string key;
            std::string value;
        };

        [[nodiscard]] std::expected<ExternalEntry, Error>
        parse_external_entry(const std::span<const std::byte> bytes) {
            WireReader reader(bytes);
            ExternalEntry result;
            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("external_data entry: " + key.error()));
                if (key->number == 1 || key->number == 2) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "external_data"); !valid)
                        return std::unexpected(valid.error());
                    auto text = read_string(reader, "external_data");
                    if (!text)
                        return std::unexpected(text.error());
                    (key->number == 1 ? result.key : result.value) = std::move(*text);
                } else if (auto ignored = skip(reader, *key, "external_data"); !ignored) {
                    return std::unexpected(ignored.error());
                }
            }
            if (result.key.empty())
                return std::unexpected(malformed("external_data entry has no key"));
            return result;
        }

        [[nodiscard]] bool has_parent_component(const fs::path& path) {
            return std::ranges::any_of(path, [](const fs::path& part) { return part == ".."; });
        }

        [[nodiscard]] bool is_path_within(const fs::path& child, const fs::path& parent) {
            auto child_it = child.begin();
            for (auto parent_it = parent.begin(); parent_it != parent.end(); ++parent_it, ++child_it) {
                if (child_it == child.end() || *child_it != *parent_it)
                    return false;
            }
            return true;
        }

        [[nodiscard]] std::expected<std::shared_ptr<std::vector<std::byte>>, Error>
        load_external_data(ParseContext& context,
                           const std::vector<ExternalEntry>& entries,
                           const std::size_t required_bytes) {
            std::optional<std::string> location;
            std::uint64_t offset = 0;
            std::optional<std::uint64_t> length;
            std::unordered_set<std::string> seen;
            for (const auto& entry : entries) {
                if (!seen.emplace(entry.key).second)
                    return std::unexpected(malformed("duplicate external_data key '" + entry.key + "'"));
                if (entry.key == "location") {
                    location = entry.value;
                } else if (entry.key == "offset") {
                    auto parsed = parse_decimal_u64(entry.value, "external_data offset");
                    if (!parsed)
                        return std::unexpected(parsed.error());
                    offset = *parsed;
                } else if (entry.key == "length") {
                    auto parsed = parse_decimal_u64(entry.value, "external_data length");
                    if (!parsed)
                        return std::unexpected(parsed.error());
                    length = *parsed;
                } else {
                    return std::unexpected(unsupported("unsupported external_data key '" + entry.key + "'",
                                                       "external_data " + entry.key));
                }
            }
            if (!location || location->empty())
                return std::unexpected(malformed("external tensor data has no location"));

            std::string portable_location = *location;
            std::ranges::replace(portable_location, '\\', '/');
            const fs::path relative = fs::path(portable_location).lexically_normal();
            if (relative.empty() || relative.is_absolute() || relative.has_root_name() || has_parent_component(relative))
                return std::unexpected(malformed("external tensor location escapes the model directory: " + *location));

            std::error_code ec;
            const fs::path model_dir = fs::weakly_canonical(context.model_path.parent_path(), ec);
            if (ec)
                return std::unexpected(Error{ErrorCode::Io, "cannot resolve model directory: " + ec.message()});
            const fs::path sidecar = fs::weakly_canonical(model_dir / relative, ec);
            if (ec || !is_path_within(sidecar, model_dir))
                return std::unexpected(malformed("external tensor location resolves outside the model directory: " + *location));
            const auto file_size = fs::file_size(sidecar, ec);
            if (ec)
                return std::unexpected(Error{ErrorCode::Io, "cannot stat external tensor data '" +
                                                               sidecar.string() + "': " + ec.message()});
            const std::uint64_t selected_length = length.value_or(file_size >= offset ? file_size - offset : 0);
            if (offset > file_size || selected_length > file_size - offset)
                return std::unexpected(malformed("external tensor offset/length exceeds sidecar size"));
            if (selected_length != required_bytes)
                return std::unexpected(malformed("external tensor byte length does not match its shape and type"));
            if (selected_length > context.max_external_bytes -
                                      std::min(context.external_bytes_read, context.max_external_bytes))
                return std::unexpected(malformed("external tensor data exceeds the configured byte limit"));

            auto storage = std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(selected_length));
            std::ifstream stream(sidecar, std::ios::binary);
            if (!stream)
                return std::unexpected(Error{ErrorCode::Io, "cannot open external tensor data '" + sidecar.string() + "'"});
            stream.seekg(static_cast<std::streamoff>(offset));
            if (selected_length != 0)
                stream.read(reinterpret_cast<char*>(storage->data()), static_cast<std::streamsize>(selected_length));
            if (!stream)
                return std::unexpected(Error{ErrorCode::Io, "cannot read external tensor data '" + sidecar.string() + "'"});
            context.external_bytes_read += selected_length;
            return storage;
        }

        template <typename T>
        void append_scalar_bytes(std::vector<std::byte>& destination, const T value) {
            const auto old_size = destination.size();
            destination.resize(old_size + sizeof(T));
            std::memcpy(destination.data() + old_size, &value, sizeof(T));
        }

        [[nodiscard]] std::expected<void, Error>
        append_packed_varints(std::vector<std::uint64_t>& destination,
                              const std::span<const std::byte> bytes,
                              const std::string_view context) {
            WireReader packed(bytes);
            while (!packed.empty()) {
                auto value = packed.varint();
                if (!value)
                    return std::unexpected(malformed(std::string(context) + ": " + value.error()));
                destination.push_back(*value);
            }
            return {};
        }

        [[nodiscard]] std::expected<void, Error>
        append_packed_fixed32(std::vector<std::uint32_t>& destination,
                              const std::span<const std::byte> bytes,
                              const std::string_view context) {
            if (bytes.size() % sizeof(std::uint32_t) != 0)
                return std::unexpected(malformed(std::string(context) + " is not fixed32-aligned"));
            WireReader packed(bytes);
            while (!packed.empty()) {
                auto value = packed.fixed32();
                if (!value)
                    return std::unexpected(malformed(std::string(context) + ": " + value.error()));
                destination.push_back(*value);
            }
            return {};
        }

        [[nodiscard]] std::expected<TensorData, Error>
        parse_tensor(const std::span<const std::byte> bytes, ParseContext& context) {
            WireReader reader(bytes);
            TensorData tensor;
            std::optional<ElementType> type;
            std::span<const std::byte> raw_data;
            std::vector<std::uint32_t> float_data;
            std::vector<std::uint64_t> int32_data;
            std::vector<std::uint64_t> int64_data;
            std::vector<ExternalEntry> external_entries;
            std::uint64_t data_location = 0;
            bool has_segment = false;

            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("TensorProto: " + key.error()));
                switch (key->number) {
                case 1: {
                    if (key->type == WireType::Varint) {
                        auto value = read_varint(reader, "TensorProto.dims");
                        if (!value)
                            return std::unexpected(value.error());
                        tensor.shape.push_back(WireReader::as_int64(*value));
                    } else if (key->type == WireType::LengthDelimited) {
                        auto packed = read_message(reader, "TensorProto.dims");
                        if (!packed)
                            return std::unexpected(packed.error());
                        std::vector<std::uint64_t> values;
                        if (auto result = append_packed_varints(values, *packed, "TensorProto.dims"); !result)
                            return std::unexpected(result.error());
                        for (const auto value : values)
                            tensor.shape.push_back(WireReader::as_int64(value));
                    } else {
                        return std::unexpected(malformed("TensorProto.dims has the wrong wire type"));
                    }
                    if (tensor.shape.size() > kMaximumRank)
                        return std::unexpected(unsupported("tensor rank exceeds the v1 limit", "tensor rank"));
                    break;
                }
                case 2: {
                    if (auto valid = require_wire(*key, WireType::Varint, "TensorProto.data_type"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_varint(reader, "TensorProto.data_type");
                    if (!value)
                        return std::unexpected(value.error());
                    auto parsed = parse_element_type(*value);
                    if (!parsed)
                        return std::unexpected(parsed.error());
                    type = *parsed;
                    break;
                }
                case 3:
                    has_segment = true;
                    if (auto ignored = skip(reader, *key, "TensorProto.segment"); !ignored)
                        return std::unexpected(ignored.error());
                    break;
                case 4: {
                    if (key->type == WireType::Fixed32) {
                        auto value = reader.fixed32();
                        if (!value)
                            return std::unexpected(malformed("TensorProto.float_data: " + value.error()));
                        float_data.push_back(*value);
                    } else if (key->type == WireType::LengthDelimited) {
                        auto packed = read_message(reader, "TensorProto.float_data");
                        if (!packed)
                            return std::unexpected(packed.error());
                        if (auto result = append_packed_fixed32(float_data, *packed, "TensorProto.float_data"); !result)
                            return std::unexpected(result.error());
                    } else {
                        return std::unexpected(malformed("TensorProto.float_data has the wrong wire type"));
                    }
                    break;
                }
                case 5:
                case 7: {
                    auto& destination = key->number == 5 ? int32_data : int64_data;
                    if (key->type == WireType::Varint) {
                        auto value = read_varint(reader, "TensorProto integer data");
                        if (!value)
                            return std::unexpected(value.error());
                        destination.push_back(*value);
                    } else if (key->type == WireType::LengthDelimited) {
                        auto packed = read_message(reader, "TensorProto integer data");
                        if (!packed)
                            return std::unexpected(packed.error());
                        if (auto result = append_packed_varints(destination, *packed, "TensorProto integer data"); !result)
                            return std::unexpected(result.error());
                    } else {
                        return std::unexpected(malformed("TensorProto integer data has the wrong wire type"));
                    }
                    break;
                }
                case 6:
                case 10:
                case 11:
                    return std::unexpected(unsupported("unsupported typed tensor payload", "tensor payload field " +
                                                                                           std::to_string(key->number)));
                case 8: {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "TensorProto.name"); !valid)
                        return std::unexpected(valid.error());
                    auto name = read_string(reader, "TensorProto.name");
                    if (!name)
                        return std::unexpected(name.error());
                    tensor.name = std::move(*name);
                    break;
                }
                case 9: {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "TensorProto.raw_data"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_message(reader, "TensorProto.raw_data");
                    if (!value)
                        return std::unexpected(value.error());
                    if (!raw_data.empty())
                        return std::unexpected(malformed("TensorProto has duplicate raw_data"));
                    raw_data = *value;
                    break;
                }
                case 13: {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "TensorProto.external_data"); !valid)
                        return std::unexpected(valid.error());
                    auto entry_bytes = read_message(reader, "TensorProto.external_data");
                    if (!entry_bytes)
                        return std::unexpected(entry_bytes.error());
                    auto entry = parse_external_entry(*entry_bytes);
                    if (!entry)
                        return std::unexpected(entry.error());
                    external_entries.push_back(std::move(*entry));
                    break;
                }
                case 14: {
                    if (auto valid = require_wire(*key, WireType::Varint, "TensorProto.data_location"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_varint(reader, "TensorProto.data_location");
                    if (!value)
                        return std::unexpected(value.error());
                    data_location = *value;
                    break;
                }
                default:
                    if (auto ignored = skip(reader, *key, "TensorProto"); !ignored)
                        return std::unexpected(ignored.error());
                }
            }

            if (has_segment)
                return std::unexpected(unsupported("segmented tensors are not supported", "TensorProto.segment"));
            if (!type)
                return std::unexpected(malformed("TensorProto has no data_type"));
            tensor.type = *type;
            auto required_bytes = checked_byte_count(*type, tensor.shape);
            if (!required_bytes)
                return std::unexpected(required_bytes.error());

            const unsigned payload_count = static_cast<unsigned>(!raw_data.empty()) +
                                           static_cast<unsigned>(!float_data.empty()) +
                                           static_cast<unsigned>(!int32_data.empty()) +
                                           static_cast<unsigned>(!int64_data.empty()) +
                                           static_cast<unsigned>(!external_entries.empty() || data_location == 1);
            if (payload_count > 1)
                return std::unexpected(malformed("TensorProto has multiple data payload encodings"));
            if (data_location > 1)
                return std::unexpected(unsupported("unsupported TensorProto data_location " +
                                                   std::to_string(data_location),
                                                   "TensorProto.data_location"));

            if (data_location == 1 || !external_entries.empty()) {
                if (data_location != 1)
                    return std::unexpected(malformed("external_data requires EXTERNAL data_location"));
                auto storage = load_external_data(context, external_entries, *required_bytes);
                if (!storage)
                    return std::unexpected(storage.error());
                tensor.external_owner = std::move(*storage);
                tensor.bytes = *tensor.external_owner;
                return tensor;
            }
            if (!raw_data.empty() || *required_bytes == 0) {
                if (raw_data.size() != *required_bytes)
                    return std::unexpected(malformed("raw tensor byte length does not match its shape and type"));
                tensor.bytes = raw_data;
                return tensor;
            }

            auto storage = std::make_shared<std::vector<std::byte>>();
            storage->reserve(*required_bytes);
            if (*type == ElementType::Float32) {
                if (float_data.size() * sizeof(float) != *required_bytes)
                    return std::unexpected(malformed("float_data length does not match tensor shape"));
                for (const auto bits : float_data)
                    append_scalar_bytes(*storage, std::bit_cast<float>(bits));
            } else if (*type == ElementType::Int64) {
                if (int64_data.size() * sizeof(std::int64_t) != *required_bytes)
                    return std::unexpected(malformed("int64_data length does not match tensor shape"));
                for (const auto value : int64_data)
                    append_scalar_bytes(*storage, WireReader::as_int64(value));
            } else if (*type == ElementType::Int32) {
                if (int32_data.size() * sizeof(std::int32_t) != *required_bytes)
                    return std::unexpected(malformed("int32_data length does not match tensor shape"));
                for (const auto value : int32_data)
                    append_scalar_bytes(*storage, static_cast<std::int32_t>(value));
            } else if (*type == ElementType::Bool) {
                if (int32_data.size() != *required_bytes)
                    return std::unexpected(malformed("bool int32_data length does not match tensor shape"));
                for (const auto value : int32_data)
                    append_scalar_bytes(*storage, static_cast<std::uint8_t>(value != 0));
            }
            if (storage->size() != *required_bytes)
                return std::unexpected(malformed("TensorProto has no compatible data payload"));
            tensor.external_owner = std::move(storage);
            tensor.bytes = *tensor.external_owner;
            return tensor;
        }

        [[nodiscard]] std::expected<std::vector<std::int64_t>, Error>
        parse_shape(const std::span<const std::byte> bytes) {
            WireReader reader(bytes);
            std::vector<std::int64_t> result;
            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("TensorShapeProto: " + key.error()));
                if (key->number != 1) {
                    if (auto ignored = skip(reader, *key, "TensorShapeProto"); !ignored)
                        return std::unexpected(ignored.error());
                    continue;
                }
                if (auto valid = require_wire(*key, WireType::LengthDelimited, "TensorShapeProto.dim"); !valid)
                    return std::unexpected(valid.error());
                auto dim_bytes = read_message(reader, "TensorShapeProto.dim");
                if (!dim_bytes)
                    return std::unexpected(dim_bytes.error());
                WireReader dim_reader(*dim_bytes);
                std::optional<std::int64_t> value;
                bool symbolic = false;
                while (!dim_reader.empty()) {
                    auto dim_key = dim_reader.key();
                    if (!dim_key)
                        return std::unexpected(malformed("TensorShapeProto.Dimension: " + dim_key.error()));
                    if (dim_key->number == 1) {
                        if (auto valid = require_wire(*dim_key, WireType::Varint, "Dimension.dim_value"); !valid)
                            return std::unexpected(valid.error());
                        auto raw = read_varint(dim_reader, "Dimension.dim_value");
                        if (!raw)
                            return std::unexpected(raw.error());
                        value = WireReader::as_int64(*raw);
                        if (*value < 0)
                            return std::unexpected(malformed("ValueInfo dimension is negative"));
                    } else if (dim_key->number == 2) {
                        if (auto valid = require_wire(*dim_key, WireType::LengthDelimited, "Dimension.dim_param"); !valid)
                            return std::unexpected(valid.error());
                        auto param = read_string(dim_reader, "Dimension.dim_param");
                        if (!param)
                            return std::unexpected(param.error());
                        symbolic = !param->empty();
                    } else if (auto ignored = skip(dim_reader, *dim_key, "Dimension"); !ignored) {
                        return std::unexpected(ignored.error());
                    }
                }
                if (value && symbolic)
                    return std::unexpected(malformed("Dimension has both dim_value and dim_param"));
                result.push_back(value.value_or(-1));
                if (result.size() > kMaximumRank)
                    return std::unexpected(unsupported("tensor rank exceeds the v1 limit", "tensor rank"));
            }
            return result;
        }

        [[nodiscard]] std::expected<std::pair<ElementType, std::vector<std::int64_t>>, Error>
        parse_tensor_type(const std::span<const std::byte> bytes) {
            WireReader reader(bytes);
            std::optional<ElementType> element_type;
            std::vector<std::int64_t> shape;
            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("TypeProto.Tensor: " + key.error()));
                if (key->number == 1) {
                    if (auto valid = require_wire(*key, WireType::Varint, "TypeProto.Tensor.elem_type"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_varint(reader, "TypeProto.Tensor.elem_type");
                    if (!value)
                        return std::unexpected(value.error());
                    auto parsed = parse_element_type(*value);
                    if (!parsed)
                        return std::unexpected(parsed.error());
                    element_type = *parsed;
                } else if (key->number == 2) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "TypeProto.Tensor.shape"); !valid)
                        return std::unexpected(valid.error());
                    auto shape_bytes = read_message(reader, "TypeProto.Tensor.shape");
                    if (!shape_bytes)
                        return std::unexpected(shape_bytes.error());
                    auto parsed = parse_shape(*shape_bytes);
                    if (!parsed)
                        return std::unexpected(parsed.error());
                    shape = std::move(*parsed);
                } else if (auto ignored = skip(reader, *key, "TypeProto.Tensor"); !ignored) {
                    return std::unexpected(ignored.error());
                }
            }
            if (!element_type)
                return std::unexpected(malformed("tensor TypeProto has no elem_type"));
            return std::pair{*element_type, std::move(shape)};
        }

        [[nodiscard]] std::expected<ValueInfo, Error>
        parse_value_info(const std::span<const std::byte> bytes) {
            WireReader reader(bytes);
            ValueInfo result;
            bool has_type = false;
            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("ValueInfoProto: " + key.error()));
                if (key->number == 1) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "ValueInfoProto.name"); !valid)
                        return std::unexpected(valid.error());
                    auto name = read_string(reader, "ValueInfoProto.name");
                    if (!name)
                        return std::unexpected(name.error());
                    result.name = std::move(*name);
                } else if (key->number == 2) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "ValueInfoProto.type"); !valid)
                        return std::unexpected(valid.error());
                    auto type_bytes = read_message(reader, "ValueInfoProto.type");
                    if (!type_bytes)
                        return std::unexpected(type_bytes.error());
                    WireReader type_reader(*type_bytes);
                    bool tensor_seen = false;
                    while (!type_reader.empty()) {
                        auto type_key = type_reader.key();
                        if (!type_key)
                            return std::unexpected(malformed("TypeProto: " + type_key.error()));
                        if (type_key->number == 1) {
                            if (tensor_seen)
                                return std::unexpected(malformed("TypeProto has duplicate tensor_type"));
                            if (auto valid = require_wire(*type_key, WireType::LengthDelimited, "TypeProto.tensor_type"); !valid)
                                return std::unexpected(valid.error());
                            auto tensor_bytes = read_message(type_reader, "TypeProto.tensor_type");
                            if (!tensor_bytes)
                                return std::unexpected(tensor_bytes.error());
                            auto parsed = parse_tensor_type(*tensor_bytes);
                            if (!parsed)
                                return std::unexpected(parsed.error());
                            result.type = parsed->first;
                            result.shape = std::move(parsed->second);
                            tensor_seen = true;
                            has_type = true;
                        } else if (type_key->number >= 4 && type_key->number <= 9) {
                            return std::unexpected(unsupported("container and non-tensor TypeProto values are not supported",
                                                               "TypeProto field " + std::to_string(type_key->number)));
                        } else if (auto ignored = skip(type_reader, *type_key, "TypeProto"); !ignored) {
                            return std::unexpected(ignored.error());
                        }
                    }
                } else if (auto ignored = skip(reader, *key, "ValueInfoProto"); !ignored) {
                    return std::unexpected(ignored.error());
                }
            }
            if (result.name.empty())
                return std::unexpected(malformed("ValueInfoProto has an empty name"));
            if (!has_type)
                return std::unexpected(malformed("ValueInfoProto '" + result.name + "' has no tensor type"));
            return result;
        }

        [[nodiscard]] std::expected<Graph, Error>
        parse_graph(const std::span<const std::byte> bytes, ParseContext& context);

        [[nodiscard]] std::expected<Attribute, Error>
        parse_attribute(const std::span<const std::byte> bytes, ParseContext& context) {
            WireReader reader(bytes);
            Attribute result;
            std::optional<std::uint64_t> declared_type;
            std::optional<float> float_value;
            std::optional<std::int64_t> int_value;
            std::optional<std::string> string_value;
            std::optional<TensorData> tensor_value;
            std::shared_ptr<Graph> graph_value;
            std::vector<float> floats;
            std::vector<std::int64_t> ints;
            std::vector<std::string> strings;
            std::vector<TensorData> tensors;
            std::vector<std::shared_ptr<Graph>> graphs;
            bool unsupported_payload = false;

            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("AttributeProto: " + key.error()));
                switch (key->number) {
                case 1: {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "AttributeProto.name"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_string(reader, "AttributeProto.name");
                    if (!value)
                        return std::unexpected(value.error());
                    result.name = std::move(*value);
                    break;
                }
                case 2: {
                    if (auto valid = require_wire(*key, WireType::Fixed32, "AttributeProto.f"); !valid)
                        return std::unexpected(valid.error());
                    auto bits = reader.fixed32();
                    if (!bits)
                        return std::unexpected(malformed("AttributeProto.f: " + bits.error()));
                    float_value = std::bit_cast<float>(*bits);
                    break;
                }
                case 3: {
                    if (auto valid = require_wire(*key, WireType::Varint, "AttributeProto.i"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_varint(reader, "AttributeProto.i");
                    if (!value)
                        return std::unexpected(value.error());
                    int_value = WireReader::as_int64(*value);
                    break;
                }
                case 4: {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "AttributeProto.s"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_string(reader, "AttributeProto.s");
                    if (!value)
                        return std::unexpected(value.error());
                    string_value = std::move(*value);
                    break;
                }
                case 5:
                case 10: {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "AttributeProto tensor"); !valid)
                        return std::unexpected(valid.error());
                    auto tensor_bytes = read_message(reader, "AttributeProto tensor");
                    if (!tensor_bytes)
                        return std::unexpected(tensor_bytes.error());
                    auto tensor = parse_tensor(*tensor_bytes, context);
                    if (!tensor)
                        return std::unexpected(tensor.error());
                    if (key->number == 5)
                        tensor_value = std::move(*tensor);
                    else
                        tensors.push_back(std::move(*tensor));
                    break;
                }
                case 6:
                case 11: {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "AttributeProto graph"); !valid)
                        return std::unexpected(valid.error());
                    auto graph_bytes = read_message(reader, "AttributeProto graph");
                    if (!graph_bytes)
                        return std::unexpected(graph_bytes.error());
                    auto graph = parse_graph(*graph_bytes, context);
                    if (!graph)
                        return std::unexpected(graph.error());
                    auto shared = std::make_shared<Graph>(std::move(*graph));
                    if (key->number == 6)
                        graph_value = std::move(shared);
                    else
                        graphs.push_back(std::move(shared));
                    break;
                }
                case 7: {
                    if (key->type == WireType::Fixed32) {
                        auto bits = reader.fixed32();
                        if (!bits)
                            return std::unexpected(malformed("AttributeProto.floats: " + bits.error()));
                        floats.push_back(std::bit_cast<float>(*bits));
                    } else if (key->type == WireType::LengthDelimited) {
                        auto packed = read_message(reader, "AttributeProto.floats");
                        if (!packed)
                            return std::unexpected(packed.error());
                        std::vector<std::uint32_t> bits;
                        if (auto parsed = append_packed_fixed32(bits, *packed, "AttributeProto.floats"); !parsed)
                            return std::unexpected(parsed.error());
                        for (const auto value : bits)
                            floats.push_back(std::bit_cast<float>(value));
                    } else {
                        return std::unexpected(malformed("AttributeProto.floats has the wrong wire type"));
                    }
                    break;
                }
                case 8: {
                    if (key->type == WireType::Varint) {
                        auto value = read_varint(reader, "AttributeProto.ints");
                        if (!value)
                            return std::unexpected(value.error());
                        ints.push_back(WireReader::as_int64(*value));
                    } else if (key->type == WireType::LengthDelimited) {
                        auto packed = read_message(reader, "AttributeProto.ints");
                        if (!packed)
                            return std::unexpected(packed.error());
                        std::vector<std::uint64_t> values;
                        if (auto parsed = append_packed_varints(values, *packed, "AttributeProto.ints"); !parsed)
                            return std::unexpected(parsed.error());
                        for (const auto value : values)
                            ints.push_back(WireReader::as_int64(value));
                    } else {
                        return std::unexpected(malformed("AttributeProto.ints has the wrong wire type"));
                    }
                    break;
                }
                case 9: {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "AttributeProto.strings"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_string(reader, "AttributeProto.strings");
                    if (!value)
                        return std::unexpected(value.error());
                    strings.push_back(std::move(*value));
                    break;
                }
                case 20: {
                    if (auto valid = require_wire(*key, WireType::Varint, "AttributeProto.type"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_varint(reader, "AttributeProto.type");
                    if (!value)
                        return std::unexpected(value.error());
                    declared_type = *value;
                    break;
                }
                case 14:
                case 21:
                case 22:
                case 23:
                    unsupported_payload = true;
                    if (auto ignored = skip(reader, *key, "AttributeProto"); !ignored)
                        return std::unexpected(ignored.error());
                    break;
                default:
                    if (auto ignored = skip(reader, *key, "AttributeProto"); !ignored)
                        return std::unexpected(ignored.error());
                }
            }
            if (result.name.empty())
                return std::unexpected(malformed("AttributeProto has an empty name"));
            if (!declared_type || *declared_type == 0 || *declared_type > 10)
                return std::unexpected(malformed("attribute '" + result.name + "' has an invalid or missing type"));
            if (unsupported_payload)
                return std::unexpected(unsupported("attribute '" + result.name + "' uses an unsupported payload",
                                                   "AttributeProto payload"));
            result.type = static_cast<AttributeType>(*declared_type);
            switch (result.type) {
            case AttributeType::Float:
                if (!float_value) return std::unexpected(malformed("FLOAT attribute '" + result.name + "' has no value"));
                result.value = *float_value;
                break;
            case AttributeType::Int:
                if (!int_value) return std::unexpected(malformed("INT attribute '" + result.name + "' has no value"));
                result.value = *int_value;
                break;
            case AttributeType::String:
                if (!string_value) return std::unexpected(malformed("STRING attribute '" + result.name + "' has no value"));
                result.value = std::move(*string_value);
                break;
            case AttributeType::Tensor:
                if (!tensor_value) return std::unexpected(malformed("TENSOR attribute '" + result.name + "' has no value"));
                result.value = std::move(*tensor_value);
                break;
            case AttributeType::Graph:
                if (!graph_value) return std::unexpected(malformed("GRAPH attribute '" + result.name + "' has no value"));
                result.value = std::move(graph_value);
                break;
            case AttributeType::Floats: result.value = std::move(floats); break;
            case AttributeType::Ints: result.value = std::move(ints); break;
            case AttributeType::Strings: result.value = std::move(strings); break;
            case AttributeType::Tensors: result.value = std::move(tensors); break;
            case AttributeType::Graphs: result.value = std::move(graphs); break;
            case AttributeType::Undefined:
                return std::unexpected(malformed("attribute '" + result.name + "' has undefined type"));
            }
            return result;
        }

        [[nodiscard]] std::expected<Node, Error>
        parse_node(const std::span<const std::byte> bytes, ParseContext& context) {
            WireReader reader(bytes);
            Node result;
            std::unordered_set<std::string> attribute_names;
            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("NodeProto: " + key.error()));
                if (key->number == 1 || key->number == 2 || key->number == 3 ||
                    key->number == 4 || key->number == 7) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "NodeProto string"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_string(reader, "NodeProto string");
                    if (!value)
                        return std::unexpected(value.error());
                    if (key->number == 1) result.inputs.push_back(std::move(*value));
                    if (key->number == 2) result.outputs.push_back(std::move(*value));
                    if (key->number == 3) result.name = std::move(*value);
                    if (key->number == 4) result.op_type = std::move(*value);
                    if (key->number == 7) result.domain = std::move(*value);
                } else if (key->number == 5) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "NodeProto.attribute"); !valid)
                        return std::unexpected(valid.error());
                    auto attr_bytes = read_message(reader, "NodeProto.attribute");
                    if (!attr_bytes)
                        return std::unexpected(attr_bytes.error());
                    auto attribute = parse_attribute(*attr_bytes, context);
                    if (!attribute)
                        return std::unexpected(attribute.error());
                    if (!attribute_names.emplace(attribute->name).second)
                        return std::unexpected(malformed("node '" + result.name + "' has duplicate attribute '" +
                                                         attribute->name + "'"));
                    result.attributes.push_back(std::move(*attribute));
                } else if (auto ignored = skip(reader, *key, "NodeProto"); !ignored) {
                    return std::unexpected(ignored.error());
                }
            }
            if (result.op_type.empty())
                return std::unexpected(malformed("NodeProto has an empty op_type"));
            if (result.outputs.empty())
                return std::unexpected(malformed("node '" + result.name + "' has no outputs"));
            return result;
        }

        [[nodiscard]] std::expected<Graph, Error>
        parse_graph(const std::span<const std::byte> bytes, ParseContext& context) {
            WireReader reader(bytes);
            Graph result;
            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("GraphProto: " + key.error()));
                if (key->number == 1) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "GraphProto.node"); !valid)
                        return std::unexpected(valid.error());
                    auto node_bytes = read_message(reader, "GraphProto.node");
                    if (!node_bytes)
                        return std::unexpected(node_bytes.error());
                    auto node = parse_node(*node_bytes, context);
                    if (!node)
                        return std::unexpected(node.error());
                    result.nodes.push_back(std::move(*node));
                } else if (key->number == 2) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "GraphProto.name"); !valid)
                        return std::unexpected(valid.error());
                    auto name = read_string(reader, "GraphProto.name");
                    if (!name)
                        return std::unexpected(name.error());
                    result.name = std::move(*name);
                } else if (key->number == 5) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "GraphProto.initializer"); !valid)
                        return std::unexpected(valid.error());
                    auto tensor_bytes = read_message(reader, "GraphProto.initializer");
                    if (!tensor_bytes)
                        return std::unexpected(tensor_bytes.error());
                    auto tensor = parse_tensor(*tensor_bytes, context);
                    if (!tensor)
                        return std::unexpected(tensor.error());
                    result.initializers.push_back(std::move(*tensor));
                } else if (key->number == 11 || key->number == 12 || key->number == 13) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "GraphProto ValueInfo"); !valid)
                        return std::unexpected(valid.error());
                    auto info_bytes = read_message(reader, "GraphProto ValueInfo");
                    if (!info_bytes)
                        return std::unexpected(info_bytes.error());
                    auto info = parse_value_info(*info_bytes);
                    if (!info)
                        return std::unexpected(info.error());
                    if (key->number == 11) result.inputs.push_back(std::move(*info));
                    if (key->number == 12) result.outputs.push_back(std::move(*info));
                    if (key->number == 13) result.value_info.push_back(std::move(*info));
                } else if (key->number == 15) {
                    return std::unexpected(unsupported("sparse initializers are not supported", "GraphProto.sparse_initializer"));
                } else if (auto ignored = skip(reader, *key, "GraphProto"); !ignored) {
                    return std::unexpected(ignored.error());
                }
            }
            if (result.outputs.empty())
                return std::unexpected(malformed("GraphProto has no outputs"));
            return result;
        }

        [[nodiscard]] std::expected<std::pair<std::string, std::int64_t>, Error>
        parse_opset(const std::span<const std::byte> bytes) {
            WireReader reader(bytes);
            std::string domain;
            std::optional<std::int64_t> version;
            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("OperatorSetIdProto: " + key.error()));
                if (key->number == 1) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "OperatorSetIdProto.domain"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_string(reader, "OperatorSetIdProto.domain");
                    if (!value)
                        return std::unexpected(value.error());
                    domain = std::move(*value);
                } else if (key->number == 2) {
                    if (auto valid = require_wire(*key, WireType::Varint, "OperatorSetIdProto.version"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_varint(reader, "OperatorSetIdProto.version");
                    if (!value)
                        return std::unexpected(value.error());
                    version = WireReader::as_int64(*value);
                } else if (auto ignored = skip(reader, *key, "OperatorSetIdProto"); !ignored) {
                    return std::unexpected(ignored.error());
                }
            }
            if (!version || *version <= 0)
                return std::unexpected(malformed("OperatorSetIdProto has an invalid version"));
            if (domain == "ai.onnx")
                domain.clear();
            return std::pair{std::move(domain), *version};
        }

        [[nodiscard]] std::expected<Model, Error>
        parse_model_bytes(const std::span<const std::byte> bytes, ParseContext& context) {
            WireReader reader(bytes);
            Model model;
            bool graph_seen = false;
            while (!reader.empty()) {
                auto key = reader.key();
                if (!key)
                    return std::unexpected(malformed("ModelProto: " + key.error()));
                if (key->number == 1) {
                    if (auto valid = require_wire(*key, WireType::Varint, "ModelProto.ir_version"); !valid)
                        return std::unexpected(valid.error());
                    auto value = read_varint(reader, "ModelProto.ir_version");
                    if (!value)
                        return std::unexpected(value.error());
                    model.ir_version = WireReader::as_int64(*value);
                } else if (key->number == 7) {
                    if (graph_seen)
                        return std::unexpected(malformed("ModelProto has duplicate graph fields"));
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "ModelProto.graph"); !valid)
                        return std::unexpected(valid.error());
                    auto graph_bytes = read_message(reader, "ModelProto.graph");
                    if (!graph_bytes)
                        return std::unexpected(graph_bytes.error());
                    auto graph = parse_graph(*graph_bytes, context);
                    if (!graph)
                        return std::unexpected(graph.error());
                    model.graph = std::move(*graph);
                    graph_seen = true;
                } else if (key->number == 8) {
                    if (auto valid = require_wire(*key, WireType::LengthDelimited, "ModelProto.opset_import"); !valid)
                        return std::unexpected(valid.error());
                    auto opset_bytes = read_message(reader, "ModelProto.opset_import");
                    if (!opset_bytes)
                        return std::unexpected(opset_bytes.error());
                    auto opset = parse_opset(*opset_bytes);
                    if (!opset)
                        return std::unexpected(opset.error());
                    if (!model.opsets.emplace(opset->first, opset->second).second)
                        return std::unexpected(malformed("ModelProto has duplicate opset import for domain '" +
                                                         opset->first + "'"));
                } else if (key->number == 20) {
                    return std::unexpected(unsupported("training graphs are not supported", "ModelProto.training_info"));
                } else if (key->number == 25) {
                    return std::unexpected(unsupported("model-local functions are not supported", "ModelProto.functions"));
                } else if (auto ignored = skip(reader, *key, "ModelProto"); !ignored) {
                    return std::unexpected(ignored.error());
                }
            }
            if (model.ir_version != kSupportedIrVersion)
                return std::unexpected(unsupported("unsupported ONNX IR version " +
                                                   std::to_string(model.ir_version) + "; expected IR 7",
                                                   "IR version " + std::to_string(model.ir_version)));
            if (!graph_seen)
                return std::unexpected(malformed("ModelProto has no graph"));
            if (!model.opsets.contains(""))
                return std::unexpected(malformed("ModelProto has no ai.onnx opset import"));
            for (const auto& [domain, version] : model.opsets) {
                if (!domain.empty())
                    return std::unexpected(unsupported("unsupported ONNX operator domain '" + domain + "'",
                                                       "domain " + domain));
                if (version != 14)
                    return std::unexpected(unsupported("unsupported ai.onnx opset " + std::to_string(version) +
                                                       "; expected opset 14",
                                                       "ai.onnx opset " + std::to_string(version)));
            }
            return model;
        }
    } // namespace

    std::expected<Model, Error> parse_model(const fs::path& path, const SessionOptions& options) {
        std::error_code ec;
        const auto size = fs::file_size(path, ec);
        if (ec)
            return std::unexpected(Error{ErrorCode::Io, "cannot stat ONNX model '" + path.string() + "': " + ec.message()});
        if (size == 0)
            return std::unexpected(malformed("ONNX model is empty"));
        if (size > options.max_model_bytes || size > std::numeric_limits<std::size_t>::max())
            return std::unexpected(malformed("ONNX model exceeds the configured byte limit"));

        auto storage = std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(size));
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
            return std::unexpected(Error{ErrorCode::Io, "cannot open ONNX model '" + path.string() + "'"});
        stream.read(reinterpret_cast<char*>(storage->data()), static_cast<std::streamsize>(size));
        if (!stream)
            return std::unexpected(Error{ErrorCode::Io, "cannot read ONNX model '" + path.string() + "'"});

        ParseContext context{.model_path = path,
                             .max_external_bytes = options.max_external_data_bytes};
        auto parsed = parse_model_bytes(*storage, context);
        if (!parsed)
            return std::unexpected(parsed.error());
        parsed->model_bytes = std::move(storage);
        parsed->path = path;
        return parsed;
    }

    const Attribute* find_attribute(const Node& node, const std::string_view name) noexcept {
        const auto it = std::ranges::find(node.attributes, name, &Attribute::name);
        return it == node.attributes.end() ? nullptr : &*it;
    }

} // namespace lfs::onnx_vulkan::detail
