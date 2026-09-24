/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/rad_ops.hpp"
#include "../../internal/tensor_impl.hpp"
#include "core/tensor_backend.hpp"
#include "kernels/rad_page.cuh"
#include <stdexcept>

namespace lfs::core::internal {
    namespace {
        LodPoolDeviceView cuda_pool(const RadPagePool& pool) {
            std::array<void*, 9> ptr{};
            for (size_t i = 0; i < 9; ++i)
                if (pool.regions[i].is_valid())
                    ptr[i] = const_cast<void*>(pool.regions[i].data_ptr());
            return {static_cast<float*>(ptr[0]), static_cast<uint2*>(ptr[1]), static_cast<uint32_t*>(ptr[2]),
                    static_cast<uint2*>(ptr[3]), static_cast<uint2*>(ptr[4]), static_cast<uint16_t*>(ptr[5]),
                    static_cast<float4*>(ptr[6]), static_cast<uint2*>(ptr[7]), static_cast<uint32_t*>(ptr[8]),
                    pool.sh_slots};
        }
        void check(cudaError_t result) {
            if (result != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(result));
        }
    } // namespace

    void cuda_rad_page_dequant(const Tensor& packed, const RadPagePackedDesc& desc, const RadPagePool& pool, uint32_t page) {
        const auto stream = prepare_inputs_for_stream({&packed}, getCurrentCUDAStream());
        pin_operands({&packed});
        check(launchLodPageDequant(static_cast<const uint8_t*>(packed.data_ptr()) + sizeof(desc),
                                   desc, cuda_pool(pool), page, pool.page_splats, stream));
    }

    void cuda_rad_page_quantize(const RadPageSources& src, const RadPagePool& pool, uint32_t page) {
        const auto stream = getCurrentCUDAStream();
        for (const auto* t : {&src.means, &src.sh0, &src.shN, &src.rotation, &src.scaling, &src.opacity, &src.shN_bounds})
            if (t->is_valid()) {
                prepare_inputs_for_stream({t}, stream);
                pin_operands({t});
            }
        const LodPageTensorSources sources{
            .means = src.means.ptr<float>() + size_t(src.offset) * 3,
            .sh0 = src.sh0.ptr<float>() + size_t(src.offset) * 3,
            .shN = src.sh_rest && src.shN.is_valid() ? src.shN.data_ptr() : nullptr,
            .shN_bounds = src.sh_q16 ? static_cast<const float2*>(src.shN_bounds.data_ptr()) : nullptr,
            .rotation = src.rotation.ptr<float>() + size_t(src.offset) * 4,
            .scaling = src.scaling.ptr<float>() + size_t(src.offset) * 3,
            .opacity = src.opacity.ptr<float>() + src.offset,
            .src_rest = src.sh_rest,
            .src_splat_offset = src.offset,
            .shN_f16 = src.shN.is_valid() && src.shN.dtype() == DataType::Float16,
            .shN_q16 = src.sh_q16,
            .count = src.count};
        check(launchLodPageQuantizeFromTensors(sources, cuda_pool(pool), page, pool.page_splats, stream));
    }
} // namespace lfs::core::internal
