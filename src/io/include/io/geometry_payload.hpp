/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace lfs::io::project {

    inline constexpr std::uint16_t PCLD_PAYLOAD_VERSION = 1;
    inline constexpr std::uint16_t MESH_PAYLOAD_VERSION = 1;
    inline constexpr std::uint16_t GEOMETRY_COORD_F32_ABSOLUTE_LOCAL = 1;
    inline constexpr std::uint32_t GEOMETRY_ENCODING_RAW = 1;
    inline constexpr std::uint32_t PCLD_ENCODING_ATTRIBUTE_NAMES = 2;
    inline constexpr std::uint32_t MESH_ENCODING_SUBMESHES = 2;
    inline constexpr std::uint32_t MESH_ENCODING_MATERIALS = 3;
    inline constexpr std::uint32_t MESH_ENCODING_TEXTURES = 4;

    enum class GeometryDtype : std::uint16_t {
        Float32 = 1,
        Float16 = 2,
        UInt8 = 3,
        UInt16 = 4,
        UInt32 = 5,
        Int32 = 6,
        Float64 = 7,
    };

    struct LFS_IO_API GeometryPropertyPlane {
        std::string name;
        std::uint16_t components = 0;
        GeometryDtype dtype = GeometryDtype::UInt8;
        std::uint32_t encoding = GEOMETRY_ENCODING_RAW;
        std::vector<std::byte> bytes;
    };

    using GeometryTensorAllocator =
        std::function<lfs::core::Tensor(lfs::core::TensorShape,
                                        lfs::core::DataType)>;

    struct GeometryDecodeOptions {
        GeometryTensorAllocator tensor_allocator;
    };

    class LFS_IO_API PointCloudPayload {
    public:
        explicit PointCloudPayload(std::shared_ptr<lfs::core::PointCloud> point_cloud);

        [[nodiscard]] std::shared_ptr<const lfs::core::PointCloud> point_cloud() const noexcept {
            return point_cloud_;
        }
        [[nodiscard]] std::shared_ptr<lfs::core::PointCloud>& point_cloud() noexcept {
            return point_cloud_;
        }
        [[nodiscard]] const std::vector<GeometryPropertyPlane>& retained_properties() const noexcept {
            return retained_properties_;
        }

        [[nodiscard]] lfs::Result<void> add_opaque_property(GeometryPropertyPlane property);

    private:
        friend LFS_IO_API lfs::Result<PointCloudPayload>
        decode_point_cloud_payload(std::span<const std::byte>, const GeometryDecodeOptions&);
        friend LFS_IO_API lfs::Result<std::vector<std::byte>>
        encode_point_cloud_payload(const PointCloudPayload&);

        std::shared_ptr<lfs::core::PointCloud> point_cloud_;
        std::vector<GeometryPropertyPlane> retained_properties_;
    };

    class LFS_IO_API MeshPayload {
    public:
        explicit MeshPayload(std::shared_ptr<lfs::core::MeshData> mesh);

        [[nodiscard]] std::shared_ptr<const lfs::core::MeshData> mesh() const noexcept {
            return mesh_;
        }
        [[nodiscard]] std::shared_ptr<lfs::core::MeshData>& mesh() noexcept {
            return mesh_;
        }
        [[nodiscard]] const std::vector<GeometryPropertyPlane>& retained_properties() const noexcept {
            return retained_properties_;
        }

        [[nodiscard]] lfs::Result<void> add_opaque_property(GeometryPropertyPlane property);

    private:
        friend LFS_IO_API lfs::Result<MeshPayload>
        decode_mesh_payload(std::span<const std::byte>, const GeometryDecodeOptions&);
        friend LFS_IO_API lfs::Result<std::vector<std::byte>>
        encode_mesh_payload(const MeshPayload&);

        std::shared_ptr<lfs::core::MeshData> mesh_;
        std::vector<GeometryPropertyPlane> retained_properties_;
    };

    [[nodiscard]] LFS_IO_API lfs::Result<PointCloudPayload>
    decode_point_cloud_payload(
        std::span<const std::byte> payload,
        const GeometryDecodeOptions& options = {});

    [[nodiscard]] LFS_IO_API lfs::Result<std::vector<std::byte>>
    encode_point_cloud_payload(const PointCloudPayload& payload);

    [[nodiscard]] LFS_IO_API lfs::Result<MeshPayload>
    decode_mesh_payload(
        std::span<const std::byte> payload,
        const GeometryDecodeOptions& options = {});

    [[nodiscard]] LFS_IO_API lfs::Result<std::vector<std::byte>>
    encode_mesh_payload(const MeshPayload& payload);

} // namespace lfs::io::project
