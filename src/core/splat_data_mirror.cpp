/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data_mirror.hpp"
#include "core/crash_handler.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/gpu_kernel_module.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "mirror_centroid_program.hpp"
#include "splat_data_mirror_centroid.hpp"
#include <limits>
#include <map>
#include <mutex>

namespace lfs::core {

    namespace {

        // Sign multipliers for position reflection per axis
        constexpr float POS_MULT[3][3] = {
            {-1.0f, 1.0f, 1.0f}, // X
            {1.0f, -1.0f, 1.0f}, // Y
            {1.0f, 1.0f, -1.0f}  // Z
        };

        // Quaternion sign multipliers per axis (w,x,y,z)
        constexpr float QUAT_MULT[3][4] = {
            {1.0f, 1.0f, -1.0f, -1.0f}, // X
            {1.0f, -1.0f, 1.0f, -1.0f}, // Y
            {1.0f, -1.0f, -1.0f, 1.0f}  // Z
        };

        // SH coefficient multipliers per axis (15 coeffs for degrees 1-3, excluding DC)
        constexpr float SH_MULT[3][15] = {
            {1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1}, // X
            {-1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1}, // Y
            {1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1}  // Z
        };

        // Cumulative SH coefficient counts per degree (excluding DC)
        constexpr int SH_COEFF_COUNT[4] = {0, 3, 8, 15};

        struct MirrorCache {
            Tensor pos_mult[3];
            Tensor quat_mult[3];
            Tensor sh_mult[3][3]; // [axis][degree 1-3] - no degree 0 (empty)
            Device device = Device::CPU;
            GpuBackend backend = GpuBackend::CUDA;
            bool valid = false;
        };

        std::mutex g_cache_mutex;
        MirrorCache g_cache;
        std::map<GpuBackend, std::unique_ptr<GpuKernelModule>> g_centroid_programs;

        void clear_mirror_cache() noexcept {
            // g_cache is a non-local static constructed before the
            // CudaMemoryPool singleton. Free CUDA mult tensors while the pool
            // is still alive so static destruction finds empty holders.
            std::lock_guard lock(g_cache_mutex);
            for (int a = 0; a < 3; ++a) {
                g_cache.pos_mult[a] = {};
                g_cache.quat_mult[a] = {};
                for (int d = 0; d < 3; ++d) {
                    g_cache.sh_mult[a][d] = {};
                }
            }
            g_centroid_programs.clear();
            g_cache.valid = false;
            g_cache.device = Device::CPU;
        }

        void ensure_cache(const Device device, const GpuBackend backend) {
            std::lock_guard lock(g_cache_mutex);

            if (g_cache.valid && g_cache.device == device &&
                (device == Device::CPU || g_cache.backend == backend))
                return;

            for (int a = 0; a < 3; ++a) {
                g_cache.pos_mult[a] = Tensor::from_vector(
                    {POS_MULT[a][0], POS_MULT[a][1], POS_MULT[a][2]}, {1, 3}, device);
                g_cache.quat_mult[a] = Tensor::from_vector(
                    {QUAT_MULT[a][0], QUAT_MULT[a][1], QUAT_MULT[a][2], QUAT_MULT[a][3]}, {1, 4}, device);

                // Only degrees 1-3 (degree 0 has no coeffs in shN)
                for (int d = 1; d <= 3; ++d) {
                    const int n = SH_COEFF_COUNT[d];
                    std::vector<float> v(n);
                    for (int i = 0; i < n; ++i)
                        v[i] = SH_MULT[a][i];
                    g_cache.sh_mult[a][d - 1] = Tensor::from_vector(v, {1, static_cast<size_t>(n), 1}, device);
                }
            }
            g_cache.device = device;
            g_cache.backend = backend;
            g_cache.valid = true;
        }

        const bool g_mirror_cache_release_hook_registered = [] {
            register_gpu_pre_shutdown_hook([]() noexcept { clear_mirror_cache(); });
            return true;
        }();

    } // namespace

    glm::vec3 compute_selection_center(const SplatData& splat_data, const Tensor& selection_mask) {
        const auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0)
            return glm::vec3(0.0f);

        const GpuBackendScope backend_scope(gpu_backend_of(means).value_or(default_gpu_backend()));
#if LFS_HAS_CUDA
        if (gpu_backend_of(means) == GpuBackend::CUDA) {
            const auto selected = selection_mask.ne(0)
                                      .reshape(TensorShape{means.size(0)})
                                      .to(means.device())
                                      .contiguous();
            return detail::selected_centroid_cuda(means.contiguous(), selected);
        }
#endif

        if (const auto backend = gpu_backend_of(means);
            backend && means.size(0) <= std::numeric_limits<uint32_t>::max()) {
            std::lock_guard lock(g_cache_mutex);
            auto& program = g_centroid_programs[*backend];
            if (!program) {
                auto loaded = GpuKernelModule::load(mirror_centroid_program_entries(), *backend);
                if (!loaded)
                    throw lfs::Exception(std::move(loaded).error());
                program = std::move(*loaded);
            }
            const auto positions = means.contiguous();
            const auto selected = selection_mask.ne(0).reshape(TensorShape{means.size(0)}).to(means.device()).to(DataType::Int32).contiguous();
            const auto blocks = static_cast<uint32_t>(std::min(size_t{256}, (means.size(0) + 255) / 256));
            auto partials = Tensor::empty({blocks, size_t{7}}, means.device());
            struct Params {
                uint64_t means = 0, selected = 0, partials = 0;
                uint32_t count, blocks;
            } params{.count = static_cast<uint32_t>(means.size(0)), .blocks = blocks};
            const std::array bindings{GpuKernelModule::Binding{0, &positions},
                                      GpuKernelModule::Binding{8, &selected},
                                      GpuKernelModule::Binding{16, &partials, GpuKernelModule::Access::ReadWrite}};
            auto dispatched = program->dispatch({.function = "mirrorCentroid",
                                                 .arguments = {std::as_bytes(std::span(&params, 1)), bindings},
                                                 .groups = {blocks, 1, 1},
                                                 .group = {256, 1, 1}});
            if (!dispatched)
                throw lfs::Exception(std::move(dispatched).error());
            const auto cpu = partials.cpu();
            const auto* values = cpu.ptr<float>();
            double sum[3]{}, count = 0.0;
            for (uint32_t block = 0; block < blocks; ++block) {
                for (int axis = 0; axis < 3; ++axis)
                    sum[axis] += static_cast<double>(values[block * 7 + axis * 2]) + values[block * 7 + axis * 2 + 1];
                count += values[block * 7 + 6];
            }
            if (count == 0)
                return glm::vec3(0.0f);
            return {static_cast<float>(sum[0] / count), static_cast<float>(sum[1] / count),
                    static_cast<float>(sum[2] / count)};
        }

        // Keep the centroid accurate enough that rounding a reflection does not
        // repeatedly move its pivot. Divide in double before rounding once.
        const auto positions = means.cpu().contiguous();
        const auto selected_cpu = selection_mask.cpu().ne(0).reshape(TensorShape{means.size(0)}).contiguous();
        const auto* p = positions.ptr<float>();
        const auto* mask = selected_cpu.ptr<bool>();
        double sum[3]{};
        size_t count = 0;
        for (size_t i = 0; i < means.size(0); ++i) {
            if (!mask[i])
                continue;
            ++count;
            for (int axis = 0; axis < 3; ++axis)
                sum[axis] += p[i * 3 + axis];
        }
        if (count == 0)
            return glm::vec3(0.0f);
        return {static_cast<float>(sum[0] / count), static_cast<float>(sum[1] / count),
                static_cast<float>(sum[2] / count)};
    }

    void mirror_gaussians(SplatData& splat_data,
                          const Tensor& selection_mask,
                          const MirrorAxis axis,
                          const glm::vec3& center) {
        LOG_TIMER("mirror_gaussians");

        auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0)
            return;

        const auto backend = gpu_backend_of(means).value_or(default_gpu_backend());
        const GpuBackendScope backend_scope(backend);
        const int a = static_cast<int>(axis);
        const auto device = means.device();
        ensure_cache(device, backend);

        const auto selected = selection_mask.ne(0);
        if (selected.sum_scalar() == 0)
            return;

        auto indices = selected.nonzero();
        if (indices.ndim() == 2)
            indices = indices.squeeze(1);
        if (indices.dtype() != DataType::Int32)
            indices = indices.to(DataType::Int32);

        // Position: p' = p * mult + offset
        {
            const auto sel = means.index_select(0, indices);
            const float off = 2.0f * center[a];
            const auto offset = Tensor::from_vector(
                {a == 0 ? off : 0.0f, a == 1 ? off : 0.0f, a == 2 ? off : 0.0f}, {1, 3}, device);
            means.index_copy_(0, indices, sel * g_cache.pos_mult[a] + offset);
        }

        // Quaternion
        if (auto& rot = splat_data.rotation_raw(); rot.is_valid() && rot.size(0) > 0) {
            rot.index_copy_(0, indices, rot.index_select(0, indices) * g_cache.quat_mult[a]);
        }

        // SH coefficients (degrees 1-3 only, shN excludes DC). Use
        // shN_canonical() so q16-resident models dequant correctly (raw
        // shN().ptr<float>() aborts on Float16 codes).
        const size_t layout_rest = splat_data.max_sh_coeffs_rest();
        if (splat_data.shN().is_valid() && splat_data.shN().numel() > 0 && layout_rest > 0) {
            const int degree = static_cast<int>(std::sqrt(layout_rest + 1)) - 1;
            if (degree >= 1 && degree <= 3) {
                Tensor shN_canon = splat_data.shN_canonical();
                if (shN_canon.device() != device)
                    shN_canon = shN_canon.to(device);
                const auto selected =
                    shN_canon.index_select(0, indices) * g_cache.sh_mult[a][degree - 1];
                shN_canon.index_copy_(0, indices, selected);
                splat_data.shN_set_from_canonical(shN_canon, splat_data.means().capacity());
                if (sh_value_quant::enabled()) {
                    (void)splat_data.apply_shN_value_quant();
                } else if (splat_data.has_tensor_allocator()) {
                    auto& shN = splat_data.shN();
                    if (shN.is_valid() && shN.dtype() == DataType::Float32) {
                        const size_t n = splat_data.size();
                        const size_t cap = std::max(n, splat_data.means().capacity());
                        const auto rest = static_cast<uint32_t>(splat_data.max_sh_coeffs_rest());
                        Tensor dest = splat_data.allocate_named_param(
                            shN.shape(),
                            sh_swizzled_float_count(cap, rest),
                            DataType::Float32,
                            "SplatData.shN");
                        dest.copy_from(shN);
                        shN = std::move(dest);
                    }
                }
            }
        }
    }

} // namespace lfs::core
