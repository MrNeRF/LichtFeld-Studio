/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_export.hpp"
#include "gpu_backend_ops.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

namespace lfs::core::internal {

    // The export kernels of a GPU backend. Every backend ports the same
    // modules with the same parameter layouts and phase numbers, so one
    // driver sequences them for all.
    class ExportKernels {
    public:
        virtual ~ExportKernels() = default;
        [[nodiscard]] virtual GpuBackend backend() const = 0;
        // Without float atomics, k-means sums label-sorted runs instead.
        [[nodiscard]] virtual bool float_atomics() const = 0;
        // Half matrix products screen the SH3 assignment when available.
        [[nodiscard]] virtual bool screened_assignment() const = 0;
        [[nodiscard]] virtual uint64_t address(const Tensor& tensor) const = 0;
        // Runs `work` invocations of a module phase.
        virtual void launch(const char* module, uint32_t phase, std::span<const std::byte> params,
                            std::span<const StorageRef> reads, std::span<const StorageRef> writes, size_t work) = 0;
    };

    Tensor export_morton_sort(ExportKernels& kernels, const Tensor& positions, Tensor* sorted_keys);
    std::tuple<Tensor, Tensor> export_kmeans_sh(ExportKernels& kernels, const Tensor& sh, int n_points, int sh_coeffs,
                                                int k, int iterations);
    void export_assign_sh3(ExportKernels& kernels, const Tensor& sh, const Tensor& centroids, const Tensor& norms,
                           Tensor& labels, bool fast, bool have_labels);
    void export_decimate_candidates(ExportKernels& kernels, const Tensor& position, const Tensor& rotation,
                                    const Tensor& scale, const Tensor& opacity, const Tensor& dc, const Tensor& sh,
                                    int rest, std::vector<uint32_t>& idx, std::vector<float>& cost);
    DecimateMerge export_decimate_merge(ExportKernels& kernels, const Tensor& position, const Tensor& rotation,
                                        const Tensor& scale, const Tensor& opacity, const Tensor& dc, const Tensor& sh,
                                        int rest, const std::vector<int>& member_group,
                                        const std::vector<uint32_t>& minimum, const std::vector<uint32_t>& members,
                                        const std::vector<uint32_t>& offsets, size_t removed);

} // namespace lfs::core::internal
