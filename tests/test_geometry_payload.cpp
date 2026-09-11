/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/geometry_payload.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::io::project::GeometryDecodeOptions;
    using lfs::io::project::GeometryDtype;
    using lfs::io::project::GeometryPropertyPlane;
    using lfs::io::project::MeshPayload;
    using lfs::io::project::PointCloudPayload;

    struct PlaneView {
        std::uint16_t components = 0;
        std::uint16_t dtype = 0;
        std::uint32_t encoding = 0;
        std::uint64_t offset = 0;
        std::span<const std::byte> bytes;
    };

    std::uint16_t read_u16(const std::span<const std::byte> bytes,
                           const std::size_t offset) {
        return static_cast<std::uint16_t>(
                   std::to_integer<std::uint8_t>(bytes[offset])) |
               static_cast<std::uint16_t>(
                   std::to_integer<std::uint8_t>(bytes[offset + 1]))
                   << 8u;
    }

    std::uint32_t read_u32(const std::span<const std::byte> bytes,
                           const std::size_t offset) {
        std::uint32_t result = 0;
        for (std::size_t byte = 0; byte < 4; ++byte) {
            result |=
                static_cast<std::uint32_t>(
                    std::to_integer<std::uint8_t>(bytes[offset + byte]))
                << (byte * 8u);
        }
        return result;
    }

    std::uint64_t read_u64(const std::span<const std::byte> bytes,
                           const std::size_t offset) {
        std::uint64_t result = 0;
        for (std::size_t byte = 0; byte < 8; ++byte) {
            result |=
                static_cast<std::uint64_t>(
                    std::to_integer<std::uint8_t>(bytes[offset + byte]))
                << (byte * 8u);
        }
        return result;
    }

    void write_u16(const std::span<std::byte> bytes,
                   const std::size_t offset,
                   const std::uint16_t value) {
        for (std::size_t byte = 0; byte < 2; ++byte) {
            bytes[offset + byte] =
                static_cast<std::byte>(value >> (byte * 8u));
        }
    }

    void write_u32(const std::span<std::byte> bytes,
                   const std::size_t offset,
                   const std::uint32_t value) {
        for (std::size_t byte = 0; byte < 4; ++byte) {
            bytes[offset + byte] =
                static_cast<std::byte>(value >> (byte * 8u));
        }
    }

    void write_u64(const std::span<std::byte> bytes,
                   const std::size_t offset,
                   const std::uint64_t value) {
        for (std::size_t byte = 0; byte < 8; ++byte) {
            bytes[offset + byte] =
                static_cast<std::byte>(value >> (byte * 8u));
        }
    }

    std::size_t descriptor_offset(
        const std::span<const std::byte> payload,
        const std::string_view name) {
        const bool pcld =
            payload.size() >= 4 &&
            std::memcmp(payload.data(), "LPCD", 4) == 0;
        const std::size_t header_bytes = pcld ? 24 : 32;
        const std::uint16_t count =
            read_u16(payload, pcld ? 16 : 24);
        for (std::uint16_t index = 0; index < count; ++index) {
            const std::size_t offset = header_bytes + index * 48;
            const auto* raw = reinterpret_cast<const char*>(
                payload.data() + offset);
            std::size_t length = 0;
            while (length < 16 && raw[length] != '\0') {
                ++length;
            }
            if (std::string_view(raw, length) == name) {
                return offset;
            }
        }
        return payload.size();
    }

    PlaneView plane_view(const std::span<const std::byte> payload,
                         const std::string_view name) {
        const auto descriptor = descriptor_offset(payload, name);
        EXPECT_LT(descriptor, payload.size());
        const auto offset = read_u64(payload, descriptor + 24);
        const auto length = read_u64(payload, descriptor + 32);
        EXPECT_LE(offset + length, payload.size());
        return PlaneView{
            .components = read_u16(payload, descriptor + 16),
            .dtype = read_u16(payload, descriptor + 18),
            .encoding = read_u32(payload, descriptor + 20),
            .offset = offset,
            .bytes = payload.subspan(
                static_cast<std::size_t>(offset),
                static_cast<std::size_t>(length)),
        };
    }

    std::vector<std::byte> byte_values(
        const std::initializer_list<std::uint8_t> values) {
        std::vector<std::byte> result;
        result.reserve(values.size());
        for (const auto value : values) {
            result.push_back(static_cast<std::byte>(value));
        }
        return result;
    }

    std::shared_ptr<lfs::core::PointCloud> sample_point_cloud() {
        auto point_cloud = std::make_shared<lfs::core::PointCloud>();
        point_cloud->means = Tensor::from_vector(
            {0.0f, 1.0f, 2.0f, -3.0f, 4.5f, 6.0f},
            {2, 3},
            Device::CPU);
        point_cloud->colors = Tensor::from_vector(
                                  {1.0f, 0.5f, 0.25f, 0.0f, 0.75f, 1.0f},
                                  {2, 3},
                                  Device::CPU)
                                  .to(DataType::Float32);
        point_cloud->normals = Tensor::from_vector(
            {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f},
            {2, 3},
            Device::CPU);
        point_cloud->sh0 = Tensor::from_vector(
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            {2, 3, 1},
            Device::CPU);
        point_cloud->shN = Tensor::from_vector(
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
             7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
            {2, 3, 2},
            Device::CPU);
        point_cloud->opacity =
            Tensor::from_vector({0.1f, 0.9f}, {2, 1}, Device::CPU);
        point_cloud->scaling = Tensor::from_vector(
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            {2, 3},
            Device::CPU);
        point_cloud->rotation = Tensor::from_vector(
            {1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f},
            {2, 4},
            Device::CPU);
        point_cloud->attribute_names = {
            "x", "y", "z", "nx", "ny", "nz", "opacity"};
        return point_cloud;
    }

    std::shared_ptr<lfs::core::MeshData> sample_mesh() {
        auto mesh = std::make_shared<lfs::core::MeshData>();
        mesh->vertices = Tensor::from_vector(
            {-1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f, 1.0f, 0.0f,
             -1.0f, 1.0f, 0.0f},
            {4, 3},
            Device::CPU);
        mesh->normals = Tensor::from_vector(
            {0.0f, 0.0f, 1.0f,
             0.0f, 0.0f, 1.0f,
             0.0f, 0.0f, 1.0f,
             0.0f, 0.0f, 1.0f},
            {4, 3},
            Device::CPU);
        mesh->texcoords = Tensor::from_vector(
            {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f},
            {4, 2},
            Device::CPU);
        mesh->indices = Tensor::from_vector(
            {0, 1, 2, 0, 2, 3}, {2, 3}, Device::CPU);
        mesh->texture_images.push_back(lfs::core::TextureImage{
            .pixels = {1, 2, 3, 4, 5, 6},
            .width = 2,
            .height = 1,
            .channels = 3,
        });
        lfs::core::Material material;
        material.name = "plane";
        material.base_color = {0.25f, 0.5f, 0.75f, 1.0f};
        material.emissive = {0.1f, 0.2f, 0.3f};
        material.metallic = 0.4f;
        material.roughness = 0.6f;
        material.ao = 0.8f;
        material.albedo_tex = 1;
        material.double_sided = true;
        mesh->materials.push_back(material);
        mesh->submeshes.push_back(lfs::core::Submesh{
            .start_index = 0,
            .index_count = 6,
            .material_index = 0,
        });
        return mesh;
    }

    TEST(GeometryPayloadTest,
         PointCloudRoundTripsAllKnownFieldsAndOpaquePlaneByteForByte) {
        PointCloudPayload source(sample_point_cloud());
        ASSERT_TRUE(source.add_opaque_property(GeometryPropertyPlane{
            .name = "vendor_score",
            .components = 1,
            .dtype = GeometryDtype::UInt16,
            .encoding = 77,
            .bytes = byte_values({0x10, 0x20, 0x30, 0x40}),
        }));

        auto encoded = lfs::io::project::encode_point_cloud_payload(source);
        ASSERT_TRUE(encoded) << lfs::format_for_developer(encoded.error());
        auto decoded =
            lfs::io::project::decode_point_cloud_payload(*encoded);
        ASSERT_TRUE(decoded) << lfs::format_for_developer(decoded.error());

        EXPECT_EQ(decoded->point_cloud()->means.shape().str(), "[2, 3]");
        EXPECT_EQ(decoded->point_cloud()->sh0.shape().str(), "[2, 3, 1]");
        EXPECT_EQ(decoded->point_cloud()->shN.shape().str(), "[2, 3, 2]");
        EXPECT_EQ(decoded->point_cloud()->attribute_names,
                  source.point_cloud()->attribute_names);

        decoded->point_cloud()->opacity.ptr<float>()[0] = 0.25f;
        auto reencoded =
            lfs::io::project::encode_point_cloud_payload(*decoded);
        ASSERT_TRUE(reencoded)
            << lfs::format_for_developer(reencoded.error());
        const auto before = plane_view(*encoded, "vendor_score");
        const auto after = plane_view(*reencoded, "vendor_score");
        EXPECT_EQ(after.components, before.components);
        EXPECT_EQ(after.dtype, before.dtype);
        EXPECT_EQ(after.encoding, before.encoding);
        EXPECT_TRUE(std::ranges::equal(after.bytes, before.bytes));
    }

    TEST(GeometryPayloadTest,
         MeshPlaneRoundTripsMaterialsTexturesAndOpaqueProperty) {
        MeshPayload source(sample_mesh());
        ASSERT_TRUE(source.add_opaque_property(GeometryPropertyPlane{
            .name = "vendor_ids",
            .components = 1,
            .dtype = GeometryDtype::UInt32,
            .encoding = 91,
            .bytes = byte_values(
                {0x01, 0x00, 0x00, 0x00,
                 0x02, 0x00, 0x00, 0x00,
                 0x03, 0x00, 0x00, 0x00,
                 0x04, 0x00, 0x00, 0x00}),
        }));
        auto encoded = lfs::io::project::encode_mesh_payload(source);
        ASSERT_TRUE(encoded) << lfs::format_for_developer(encoded.error());
        auto decoded = lfs::io::project::decode_mesh_payload(*encoded);
        ASSERT_TRUE(decoded) << lfs::format_for_developer(decoded.error());

        ASSERT_EQ(decoded->mesh()->materials.size(), 1);
        EXPECT_EQ(decoded->mesh()->materials[0].name, "plane");
        EXPECT_EQ(decoded->mesh()->materials[0].albedo_tex, 1u);
        EXPECT_TRUE(decoded->mesh()->materials[0].double_sided);
        ASSERT_EQ(decoded->mesh()->texture_images.size(), 1);
        EXPECT_EQ(decoded->mesh()->texture_images[0].pixels,
                  (std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6}));
        ASSERT_EQ(decoded->mesh()->submeshes.size(), 1);
        EXPECT_EQ(decoded->mesh()->submeshes[0].index_count, 6u);

        decoded->mesh()->vertices.ptr<float>()[0] = -2.0f;
        auto reencoded =
            lfs::io::project::encode_mesh_payload(*decoded);
        ASSERT_TRUE(reencoded)
            << lfs::format_for_developer(reencoded.error());
        const auto before = plane_view(*encoded, "vendor_ids");
        const auto after = plane_view(*reencoded, "vendor_ids");
        EXPECT_EQ(after.components, before.components);
        EXPECT_EQ(after.dtype, before.dtype);
        EXPECT_EQ(after.encoding, before.encoding);
        EXPECT_TRUE(std::ranges::equal(after.bytes, before.bytes));
    }

    TEST(GeometryPayloadTest,
         HostilePointCloudDescriptorsFailBeforeTensorAllocation) {
        auto valid = lfs::io::project::encode_point_cloud_payload(
            PointCloudPayload(sample_point_cloud()));
        ASSERT_TRUE(valid) << lfs::format_for_developer(valid.error());

        std::size_t allocations = 0;
        const GeometryDecodeOptions options{
            .tensor_allocator =
                [&](const lfs::core::TensorShape shape,
                    const DataType dtype) {
                    ++allocations;
                    return Tensor::empty(shape, Device::CPU, dtype);
                },
        };

        auto overlap = *valid;
        const auto means = descriptor_offset(overlap, "means");
        const auto colors = descriptor_offset(overlap, "colors");
        ASSERT_LT(means, overlap.size());
        ASSERT_LT(colors, overlap.size());
        write_u64(
            overlap,
            colors + 24,
            read_u64(overlap, means + 24));
        auto overlap_result =
            lfs::io::project::decode_point_cloud_payload(
                overlap, options);
        EXPECT_FALSE(overlap_result);
        EXPECT_EQ(allocations, 0u);

        auto absurd_count = *valid;
        write_u64(
            absurd_count,
            8,
            std::numeric_limits<std::uint64_t>::max());
        auto count_result =
            lfs::io::project::decode_point_cloud_payload(
                absurd_count, options);
        EXPECT_FALSE(count_result);
        EXPECT_EQ(allocations, 0u);

        auto wrong_components = *valid;
        write_u16(wrong_components, means + 16, 4);
        auto shape_result =
            lfs::io::project::decode_point_cloud_payload(
                wrong_components, options);
        EXPECT_FALSE(shape_result);
        EXPECT_EQ(allocations, 0u);
    }

    TEST(GeometryPayloadTest,
         HostileMeshIndicesAndOutOfBoundsPlanesFailBeforeAllocation) {
        auto valid = lfs::io::project::encode_mesh_payload(
            MeshPayload(sample_mesh()));
        ASSERT_TRUE(valid) << lfs::format_for_developer(valid.error());

        std::size_t allocations = 0;
        const GeometryDecodeOptions options{
            .tensor_allocator =
                [&](const lfs::core::TensorShape shape,
                    const DataType dtype) {
                    ++allocations;
                    return Tensor::empty(shape, Device::CPU, dtype);
                },
        };

        auto bad_index = *valid;
        const auto indices = descriptor_offset(bad_index, "indices");
        ASSERT_LT(indices, bad_index.size());
        const auto index_offset =
            read_u64(bad_index, indices + 24);
        write_u32(bad_index, index_offset, 99);
        auto index_result =
            lfs::io::project::decode_mesh_payload(
                bad_index, options);
        EXPECT_FALSE(index_result);
        EXPECT_EQ(allocations, 0u);

        auto out_of_bounds = *valid;
        const auto vertices =
            descriptor_offset(out_of_bounds, "vertices");
        ASSERT_LT(vertices, out_of_bounds.size());
        write_u64(
            out_of_bounds,
            vertices + 24,
            static_cast<std::uint64_t>(out_of_bounds.size()) + 64);
        auto bounds_result =
            lfs::io::project::decode_mesh_payload(
                out_of_bounds, options);
        EXPECT_FALSE(bounds_result);
        EXPECT_EQ(allocations, 0u);
    }

    TEST(GeometryPayloadTest, EmptyRequiredPlanesRemainRepresentable) {
        auto point_cloud = std::make_shared<lfs::core::PointCloud>();
        point_cloud->means =
            Tensor::empty({0, 3}, Device::CPU, DataType::Float32);
        auto pcld = lfs::io::project::encode_point_cloud_payload(
            PointCloudPayload(point_cloud));
        ASSERT_TRUE(pcld) << lfs::format_for_developer(pcld.error());
        auto decoded_pcld =
            lfs::io::project::decode_point_cloud_payload(*pcld);
        ASSERT_TRUE(decoded_pcld)
            << lfs::format_for_developer(decoded_pcld.error());
        EXPECT_EQ(decoded_pcld->point_cloud()->means.shape().str(),
                  "[0, 3]");

        auto mesh = std::make_shared<lfs::core::MeshData>();
        mesh->vertices =
            Tensor::empty({0, 3}, Device::CPU, DataType::Float32);
        mesh->indices =
            Tensor::empty({0, 3}, Device::CPU, DataType::Int32);
        auto mesh_bytes = lfs::io::project::encode_mesh_payload(
            MeshPayload(mesh));
        ASSERT_TRUE(mesh_bytes)
            << lfs::format_for_developer(mesh_bytes.error());
        auto decoded_mesh =
            lfs::io::project::decode_mesh_payload(*mesh_bytes);
        ASSERT_TRUE(decoded_mesh)
            << lfs::format_for_developer(decoded_mesh.error());
        EXPECT_EQ(decoded_mesh->mesh()->vertices.shape().str(),
                  "[0, 3]");
        EXPECT_EQ(decoded_mesh->mesh()->indices.shape().str(),
                  "[0, 3]");
    }

} // namespace
