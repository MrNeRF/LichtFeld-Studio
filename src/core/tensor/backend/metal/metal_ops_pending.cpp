/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Operations not yet ported to Metal. They fail like the Vulkan backend's
// unsupported cases, so callers and the backend corpus report them uniformly.

#include "metal_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/tensor_export.hpp"

#include <format>
#include <string_view>

namespace lfs::core::internal {

    namespace {
        [[noreturn]] void not_ported(const std::string_view operation) {
            throw TensorError(std::format("Metal backend: {} is not implemented yet", operation));
        }
    } // namespace

    Tensor MetalBackendOps::morton_sort(const Tensor&, Tensor*, ExecContext) {
        not_ported("morton_sort");
    }

    std::tuple<Tensor, Tensor> MetalBackendOps::kmeans_sh(const Tensor&, int, int, int, int, bool, ExecContext) {
        not_ported("kmeans_sh");
    }

    void MetalBackendOps::assign_sh3(const Tensor&, const Tensor&, const Tensor&, Tensor&, bool, bool, ExecContext) {
        not_ported("assign_sh3");
    }

    void MetalBackendOps::decimate_candidates(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, int, std::vector<uint32_t>&, std::vector<float>&, ExecContext) {
        not_ported("decimate_candidates");
    }

    DecimateMerge MetalBackendOps::decimate_merge(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, int, const std::vector<int>&, const std::vector<uint32_t>&, const std::vector<uint32_t>&, const std::vector<uint32_t>&, size_t, ExecContext) {
        not_ported("decimate_merge");
    }

} // namespace lfs::core::internal
