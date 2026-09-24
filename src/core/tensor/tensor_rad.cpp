/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_rad.hpp"
#include "internal/rad_ops.hpp"
#include <stdexcept>

namespace lfs::core {
    namespace {
        GpuBackend validate_pool(const RadPagePool& pool, uint32_t page) {
            const auto backend = gpu_backend_of(pool.regions[0]);
            if (!backend || pool.page_splats == 0 || pool.page_splats % 32 || pool.sh_slots > 12)
                throw std::invalid_argument("RAD pool requires GPU byte tensors and whole SH tiles");
            const size_t n = (size_t(page) + 1) * pool.page_splats;
            const std::array<size_t, 9> sizes{n * 12, n * 8, n * pool.sh_slots * 4, n * 8, n * 8, n * 2,
                                              (size_t(page) + 1) * radq::kPageFrameBytes, n * 8, n * 12};
            for (size_t i = 0; i < 9; ++i) {
                const auto& tensor = pool.regions[i];
                if ((i >= 7 || (i == 2 && !pool.sh_slots)) && !tensor.is_valid())
                    continue;
                if (gpu_backend_of(tensor) != backend || !tensor.is_contiguous() ||
                    tensor.dtype() != DataType::UInt8 || tensor.bytes() < sizes[i])
                    throw std::invalid_argument("RAD page exceeds its pool region or has incompatible storage");
            }
            if (pool.regions[7].is_valid() != pool.regions[8].is_valid())
                throw std::invalid_argument("RAD metadata requires both bounds and links");
            return *backend;
        }
    } // namespace
    void rad_page_validate(const RadPagePool& pool, const RadPagePackedDesc& desc, size_t bytes, uint32_t page) {
        (void)validate_pool(pool, page);
        if (!desc.count || desc.count > pool.page_splats || desc.meta_node_count > pool.page_splats ||
            desc.property_count > kRadPackedMaxProps || desc.sh_coeffs_rest > 15 || !desc.used_bytes ||
            bytes < sizeof(desc) || desc.used_bytes > bytes - sizeof(desc))
            throw std::invalid_argument("RAD packed page exceeds descriptor or staging bounds");
        const auto fits = [&](size_t offset, size_t n) { return offset <= desc.used_bytes && n <= desc.used_bytes - offset; };
        constexpr std::array<size_t, 8> dims{3, 1, 3, 3, 3, 9, 15, 21};
        std::array<bool, 8> seen{};
        for (uint32_t i = 0; i < desc.property_count; ++i) {
            const auto& p = desc.props[i];
            if (p.kind >= 8 || seen[p.kind] || p.encoding > 8)
                throw std::invalid_argument("Invalid RAD property");
            seen[p.kind] = true;
            const size_t width = p.encoding <= 1 ? 4 : ((p.encoding <= 3 || p.encoding == 7) ? 2 : 1);
            if (p.plane_offset % 4 || p.plane_bytes < dims[p.kind] * desc.count * width ||
                !fits(p.plane_offset, p.plane_bytes) || (p.encoding == 8 && p.kind != 4))
                throw std::invalid_argument("RAD property exceeds staging bounds");
        }
        if (pool.regions[7].is_valid() && (desc.meta_bounds_offset % 4 || desc.meta_links_offset % 4 ||
                                           !fits(desc.meta_bounds_offset, size_t(desc.meta_node_count) * 8) ||
                                           !fits(desc.meta_links_offset, size_t(desc.meta_node_count) * 12)))
            throw std::invalid_argument("RAD metadata exceeds staging bounds");
    }
    void rad_page_dequant(const Tensor& packed, const RadPagePackedDesc& desc, const RadPagePool& pool, uint32_t page) {
        rad_page_validate(pool, desc, packed.bytes(), page);
        if (gpu_backend_of(packed) != gpu_backend_of(pool.regions[0]) || !packed.is_contiguous() || packed.dtype() != DataType::UInt8)
            throw std::invalid_argument("RAD packed tensor must match the pool backend");
        if (gpu_backend_of(packed) == GpuBackend::Vulkan) {
#if LFS_TENSOR_VULKAN
            internal::vulkan_rad_page_dequant(packed, pool, page);
#else
            throw std::runtime_error("Vulkan tensor backend not built");
#endif
            return;
        }
#if LFS_HAS_CUDA
        internal::cuda_rad_page_dequant(packed, desc, pool, page);
#else
        throw std::runtime_error("CUDA tensor backend is unavailable");
#endif
    }
    void rad_page_quantize(const RadPageSources& src, const RadPagePool& pool, uint32_t page) {
        const auto backend = validate_pool(pool, page);
        if (!src.count || src.count > pool.page_splats || src.sh_rest > 15)
            throw std::invalid_argument("Invalid RAD resident page");
        const size_t n = size_t(src.offset) + src.count;
        for (const auto* t : {&src.means, &src.sh0, &src.rotation, &src.scaling, &src.opacity}) {
            const size_t dims = t == &src.rotation ? 4 : (t == &src.opacity ? 1 : 3);
            if (gpu_backend_of(*t) != backend || t->dtype() != DataType::Float32 || !t->is_contiguous() || t->numel() < n * dims)
                throw std::invalid_argument("RAD resident attributes require matching contiguous float GPU tensors");
        }
        if (src.shN.is_valid() && src.sh_rest && (gpu_backend_of(src.shN) != backend || !src.shN.is_contiguous() || (src.shN.dtype() != DataType::Float32 && src.shN.dtype() != DataType::Float16)))
            throw std::invalid_argument("Invalid RAD resident SH storage");
        if (src.shN.is_valid() && src.sh_rest) {
            const size_t tiled_splats = ((n + 31) / 32) * 32;
            const size_t values_per_splat = src.sh_q16 ? src.sh_rest * 3 : ((src.sh_rest * 3 + 3) / 4) * 4;
            const size_t element_bytes = src.sh_q16 ? 2 : dtype_size(src.shN.dtype());
            if (src.shN.bytes() < tiled_splats * values_per_splat * element_bytes)
                throw std::invalid_argument("RAD resident SH storage is too small");
        }
        if (src.sh_q16 && (!src.shN.is_valid() || src.shN.dtype() != DataType::Float16 ||
                           !src.shN_bounds.is_valid() || gpu_backend_of(src.shN_bounds) != backend ||
                           src.shN_bounds.dtype() != DataType::Float32 || !src.shN_bounds.is_contiguous() ||
                           src.shN_bounds.numel() < ((n + 255) / 256) * 2))
            throw std::invalid_argument("Invalid RAD resident SH bounds");
        if (backend == GpuBackend::Vulkan) {
#if LFS_TENSOR_VULKAN
            internal::vulkan_rad_page_quantize(src, pool, page);
#else
            throw std::runtime_error("Vulkan tensor backend not built");
#endif
            return;
        }
#if LFS_HAS_CUDA
        internal::cuda_rad_page_quantize(src, pool, page);
#else
        throw std::runtime_error("CUDA tensor backend is unavailable");
#endif
    }
} // namespace lfs::core
