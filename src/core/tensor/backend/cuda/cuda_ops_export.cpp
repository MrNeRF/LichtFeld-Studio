/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "kernels/export_entry.hpp"

#include "core/tensor_export.hpp"

#include <initializer_list>

namespace lfs::core::internal {
    namespace {
        // The export kernels allocate scratch on CUDA and run on the legacy default stream,
        // so each input's pending producer work is bridged onto it first.
        void order_inputs(std::initializer_list<const Tensor*> inputs) {
            for (const Tensor* input : inputs) {
                if (!input->is_valid() || input->device() != Device::GPU || input->numel() == 0)
                    continue;
                (void)input->data_ptr();
                input->sync_to_stream(nullptr);
            }
        }
    } // namespace

    Tensor CudaBackendOps::morton_sort(const Tensor& positions, Tensor* sorted_keys, ExecContext) {
        LFS_FACADE_TRACE(morton_sort);
        const GpuBackendScope scope(GpuBackend::CUDA);
        order_inputs({&positions});
        return export_cuda::morton_sort_indices_for_positions(positions, sorted_keys);
    }

    std::tuple<Tensor, Tensor> CudaBackendOps::kmeans_sh(const Tensor& shN, int n_points, int sh_coeffs, int k,
                                                         int iterations, bool fast, ExecContext) {
        LFS_FACADE_TRACE(kmeans_sh);
        const GpuBackendScope scope(GpuBackend::CUDA);
        order_inputs({&shN});
        return export_cuda::kmeans_sh_swizzled(shN, n_points, sh_coeffs, k, iterations, fast);
    }

    void CudaBackendOps::assign_sh3(const Tensor& shN, const Tensor& centroids, const Tensor& norms, Tensor& labels,
                                    bool fast, bool have_labels, ExecContext) {
        LFS_FACADE_TRACE(assign_sh3);
        const GpuBackendScope scope(GpuBackend::CUDA);
        order_inputs({&shN, &centroids, &norms, &labels});
        export_cuda::assign_sh3_labels(shN, centroids, norms, labels, fast, have_labels);
    }

    void CudaBackendOps::decimate_candidates(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                             const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest,
                                             std::vector<uint32_t>& idx, std::vector<float>& cost, ExecContext) {
        LFS_FACADE_TRACE(decimate_candidates);
        const GpuBackendScope scope(GpuBackend::CUDA);
        order_inputs({&position, &rotation, &scale, &opacity, &dc, &sh});
        const decimate::Data data{position, rotation, scale, opacity, dc, sh, position.size(0), rest};
        auto found = export_cuda::gpu_candidates(data);
        idx = std::move(found.idx);
        cost = std::move(found.cost);
    }

    DecimateMerge CudaBackendOps::decimate_merge(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                                 const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest,
                                                 const std::vector<int>& member_group, const std::vector<uint32_t>& minimum,
                                                 const std::vector<uint32_t>& members, const std::vector<uint32_t>& offsets,
                                                 size_t removed, ExecContext) {
        LFS_FACADE_TRACE(decimate_merge);
        const GpuBackendScope scope(GpuBackend::CUDA);
        order_inputs({&position, &rotation, &scale, &opacity, &dc, &sh});
        const decimate::Data data{position, rotation, scale, opacity, dc, sh, position.size(0), rest};
        decimate::Selection selection;
        selection.member_group = member_group;
        selection.minimum = minimum;
        selection.members = members;
        selection.offsets = offsets;
        selection.removed = removed;
        auto out = export_cuda::gpu_merge(data, selection);
        return {std::move(out.pos), std::move(out.rot), std::move(out.scale), std::move(out.opacity), std::move(out.dc), std::move(out.sh)};
    }
} // namespace lfs::core::internal
