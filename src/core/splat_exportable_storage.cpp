/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_exportable_storage.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_impl.hpp"

#include <cstring>
#include <format>

namespace lfs::core {

    namespace {

        constexpr std::size_t kFloatBytes = sizeof(float);
        constexpr std::size_t kRegionAlignment = 256;

        std::size_t align_up(std::size_t v, std::size_t a) {
            return ((v + a - 1) / a) * a;
        }

        std::size_t region_bytes_for(std::size_t capacity,
                                     std::size_t per_primitive_floats) {
            return capacity * per_primitive_floats * kFloatBytes;
        }

        SplatExportableStorage::Region region_from_name(std::string_view name) {
            if (name == "SplatData.means")
                return SplatExportableStorage::Means;
            if (name == "SplatData.scaling")
                return SplatExportableStorage::Scaling;
            if (name == "SplatData.rotation")
                return SplatExportableStorage::Rotation;
            if (name == "SplatData.opacity")
                return SplatExportableStorage::Opacity;
            if (name == "SplatData.sh0")
                return SplatExportableStorage::Sh0;
            if (name == "SplatData.shN")
                return SplatExportableStorage::ShN;
            throw std::runtime_error(
                std::format("SplatExportableStorage: unknown allocator name '{}'", name));
        }

    } // namespace

    std::expected<SplatExportableStorage, std::string>
    SplatExportableStorage::create(std::size_t capacity, int sh_degree, int device) {
        if (capacity == 0) {
            return std::unexpected("SplatExportableStorage::create: capacity must be > 0");
        }

        SplatExportableStorage out{};

        // Per-region byte sizes derived from the same shapes splat_data.cpp uses
        // (means {N,3}, scaling {N,3}, rotation {N,4}, opacity {N,1}, sh0 {N,1,3},
        //  shN swizzled {capacity_floats}).
        const auto rest_coeffs =
            static_cast<std::uint32_t>(sh_rest_coefficients_for_degree(sh_degree));
        const std::size_t shN_capacity_floats = sh_swizzled_float_count(capacity, rest_coeffs);

        const std::array<std::size_t, Count> raw_bytes{
            region_bytes_for(capacity, 3),     // Means {N,3}
            region_bytes_for(capacity, 3),     // Scaling {N,3}
            region_bytes_for(capacity, 4),     // Rotation {N,4}
            region_bytes_for(capacity, 1),     // Opacity {N,1}
            region_bytes_for(capacity, 3),     // Sh0 {N,1,3} ⇒ 3 floats/primitive
            shN_capacity_floats * kFloatBytes, // ShN {capacity_floats}
        };

        std::size_t cursor = 0;
        for (std::size_t i = 0; i < Count; ++i) {
            cursor = align_up(cursor, kRegionAlignment);
            out.region_offsets[i] = cursor;
            out.region_bytes[i] = raw_bytes[i];
            cursor += raw_bytes[i];
        }
        const std::size_t total_bytes = cursor;

        auto block_result = allocateExportableDeviceBlock(total_bytes, device);
        if (!block_result) {
            return std::unexpected(std::format(
                "SplatExportableStorage::create: backing-block allocation failed: {}",
                block_result.error()));
        }
        out.block = std::move(*block_result);

        LOG_INFO("SplatExportableStorage: total={} MiB capacity={} sh_degree={} (means={}, "
                 "scaling={}, rotation={}, opacity={}, sh0={}, shN={} MiB)",
                 total_bytes >> 20,
                 capacity,
                 sh_degree,
                 raw_bytes[Means] >> 20,
                 raw_bytes[Scaling] >> 20,
                 raw_bytes[Rotation] >> 20,
                 raw_bytes[Opacity] >> 20,
                 raw_bytes[Sh0] >> 20,
                 raw_bytes[ShN] >> 20);

        return out;
    }

    SplatTensorAllocator SplatExportableStorage::make_allocator() const {
        auto block_copy = block;
        auto offsets = region_offsets;
        return [block = std::move(block_copy), offsets](TensorShape shape,
                                                        std::size_t capacity,
                                                        DataType dtype,
                                                        std::string_view name) -> Tensor {
            const Region region = region_from_name(name);
            void* const data = static_cast<char*>(block->device_ptr) + offsets[region];
            // Share ownership with the backing block. Use the implicit upcast
            // shared_ptr<void>(shared_ptr<ExportableBlock>) so owner.get() is non-null
            // (from_external_owner checks `if (!owner)` which is `get() != nullptr`).
            std::shared_ptr<void> owner = block;
            return Tensor::from_external_owner(data,
                                               std::move(shape),
                                               Device::CUDA,
                                               dtype,
                                               std::move(owner),
                                               capacity,
                                               /*stream=*/getCurrentCUDAStream(),
                                               "splat.exportable");
        };
    }

} // namespace lfs::core
