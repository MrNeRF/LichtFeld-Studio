/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal ExtraLossOps; see ops/extra_loss_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include "core/assert.hpp"

#include <algorithm>
#include <format>

namespace lfs::training {
    namespace {
        using namespace lfs::gpu_ops;
        namespace mk = metal;

        constexpr uint32_t kMaxGroups = 1024;

        uint32_t element_count(In tensor, const char* what) {
            LFS_ASSERT_MSG(tensor.numel() <= UINT32_MAX,
                           std::format("{} supports at most 2^32-1 elements (count={})", what, tensor.numel()));
            return mk::count32(tensor.numel(), "extra loss");
        }

        void regularize(In raw, Out gradient, Out loss, Out reduction_temp, Regularizer kind, float weight) {
            const uint32_t n = element_count(raw, "regularization");
            if (n == 0 || weight == 0.0f)
                return;
            const uint32_t groups = std::min<uint32_t>((n + 255) / 256, kMaxGroups);
            LFS_ASSERT_MSG(reduction_temp.numel() >= groups,
                           std::format("regularization reduction scratch is too small (have={}, need={})",
                                       reduction_temp.numel(), groups));
            struct {
                uint64_t raw, gradient, partials;
                uint32_t count, has_gradient, opacity;
                float grad_scale;
            } const params{mk::address(raw), mk::address(gradient), mk::address(reduction_temp), n,
                           gradient.is_valid() ? 1u : 0u, kind == Regularizer::Opacity ? 1u : 0u,
                           weight / static_cast<float>(n)};
            mk::launch("extra_regularize", params, {&raw, &gradient, &reduction_temp}, groups);
            struct {
                uint64_t partials, result, mask_sum;
                uint32_t count, masked;
                float channels, scale, divisor, offset;
            } const reduce{mk::address(reduction_temp), mk::address(loss), 0, groups, 0, 0.f, weight,
                           static_cast<float>(n), 0.f};
            mk::launch("loss_reduce_final", reduce, {&reduction_temp, &loss}, 1, kMaxGroups);
        }

        void admm(In sigmoid, In z, In u, Out gradient, float rho, float grad_loss, bool accumulate) {
            const uint32_t n = element_count(sigmoid, "ADMM backward");
            if (n == 0)
                return;
            struct {
                uint64_t gradient, sigmoid, z, u;
                uint32_t count, accumulate;
                float rho, grad_loss;
            } const params{mk::address(gradient), mk::address(sigmoid), mk::address(z), mk::address(u), n,
                           accumulate ? 1u : 0u, rho, grad_loss};
            mk::launch_items("extra_admm", params, {&gradient, &sigmoid, &z, &u}, n);
        }
    } // namespace

    const lfs::gpu_ops::ExtraLossOps& metal_extra_loss_ops() {
        static const lfs::gpu_ops::ExtraLossOps ops{
            .regularize = regularize,
            .admm = admm,
        };
        return ops;
    }
} // namespace lfs::training
