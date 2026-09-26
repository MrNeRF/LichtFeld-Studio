/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../export_pipeline.hpp"
#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"
#include "vk_context.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <algorithm>
#include <array>

namespace lfs::core::internal {
    namespace {
        class VulkanExportKernels final : public ExportKernels {
        public:
            GpuBackend backend() const override { return GpuBackend::Vulkan; }

            bool float_atomics() const override { return context_->caps().shader_atomic_float; }

            bool screened_assignment() const override {
                const auto& caps = context_->caps();
                return caps.cooperative_matrix && caps.shader_float16 && caps.vulkan_memory_model &&
                       caps.vulkan_memory_model_device_scope;
            }

            uint64_t address(const Tensor& tensor) const override { return vk::address(storage_ref(tensor)); }

            void launch(const char* const module, const uint32_t phase, const std::span<const std::byte> params,
                        const std::span<const StorageRef> reads, const std::span<const StorageRef> writes,
                        const size_t work) override {
                const std::array<uint32_t, 1> constants{phase};
                const auto bytes = static_cast<uint32_t>(params.size());
                const auto& pipeline = context_->pipelines().specialized(module, bytes, constants);
                context_->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, params.data());
                    const size_t groups = (work + pipeline.local_size_x - 1) / pipeline.local_size_x;
                    vkCmdDispatch(command, static_cast<uint32_t>(std::min<size_t>(groups, context_->caps().max_workgroup_count[0])), 1, 1);
                });
            }

        private:
            std::shared_ptr<VulkanContext> context_ = acquire_vulkan_context();
        };
    } // namespace

    Tensor VulkanBackendOps::morton_sort(const Tensor& positions, Tensor* sorted_keys, ExecContext) {
        LFS_FACADE_TRACE(morton_sort);
        VulkanExportKernels kernels;
        return export_morton_sort(kernels, positions, sorted_keys);
    }

    std::tuple<Tensor, Tensor> VulkanBackendOps::kmeans_sh(const Tensor& sh, int n_points, int sh_coeffs, int k, int iterations,
                                                           bool, ExecContext) {
        LFS_FACADE_TRACE(kmeans_sh);
        VulkanExportKernels kernels;
        return export_kmeans_sh(kernels, sh, n_points, sh_coeffs, k, iterations);
    }

    void VulkanBackendOps::assign_sh3(const Tensor& sh, const Tensor& centroids, const Tensor& norms, Tensor& labels, bool fast,
                                      bool have_labels, ExecContext) {
        LFS_FACADE_TRACE(assign_sh3);
        VulkanExportKernels kernels;
        export_assign_sh3(kernels, sh, centroids, norms, labels, fast, have_labels);
    }

    void VulkanBackendOps::decimate_candidates(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                               const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest,
                                               std::vector<uint32_t>& idx, std::vector<float>& cost, ExecContext) {
        LFS_FACADE_TRACE(decimate_candidates);
        VulkanExportKernels kernels;
        export_decimate_candidates(kernels, position, rotation, scale, opacity, dc, sh, rest, idx, cost);
    }

    DecimateMerge VulkanBackendOps::decimate_merge(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                                   const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest,
                                                   const std::vector<int>& member_group, const std::vector<uint32_t>& minimum,
                                                   const std::vector<uint32_t>& members, const std::vector<uint32_t>& offsets,
                                                   size_t removed, ExecContext) {
        LFS_FACADE_TRACE(decimate_merge);
        VulkanExportKernels kernels;
        return export_decimate_merge(kernels, position, rotation, scale, opacity, dc, sh, rest, member_group, minimum,
                                     members, offsets, removed);
    }
} // namespace lfs::core::internal
