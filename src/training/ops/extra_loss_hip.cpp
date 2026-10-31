/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/extra_loss_hip.hpp"

#include "core/tensor_backend.hpp"

#include <cstdint>
#include <format>
#include <stdexcept>

namespace lfs::training {
    namespace {
        using namespace lfs::gpu_ops;

        void require_tensor(In tensor, const char* name) {
            if (!tensor.is_valid() || tensor.dtype() != core::DataType::Float32 ||
                core::gpu_backend_of(tensor) != core::GpuBackend::HIP || !tensor.is_contiguous()) {
                throw std::invalid_argument(std::format("HIP regularization requires a contiguous Float32 HIP {}", name));
            }
        }

        bool overlaps(In a, In b) {
            if (a.numel() == 0 || b.numel() == 0)
                return false;
            const auto first = reinterpret_cast<std::uintptr_t>(a.data_ptr());
            const auto second = reinterpret_cast<std::uintptr_t>(b.data_ptr());
            return first <= second ? second - first < a.bytes() : first - second < b.bytes();
        }

        void regularize(In raw, Out gradient, Out loss, Out, Regularizer kind, float weight) {
            require_tensor(raw, "parameter");
            require_tensor(loss, "loss");
            if (loss.numel() != 1 || (kind != Regularizer::Scale && kind != Regularizer::Opacity))
                throw std::invalid_argument("HIP regularization requires one loss element and a supported regularizer");
            if (gradient.is_valid()) {
                require_tensor(gradient, "gradient");
                if (gradient.shape() != raw.shape())
                    throw std::invalid_argument("HIP regularization gradient must match the parameter shape");
            }
            if (overlaps(raw, loss) || (gradient.is_valid() && (overlaps(gradient, raw) || overlaps(gradient, loss))))
                throw std::invalid_argument("HIP regularization parameter, gradient and loss must not overlap");
            if (raw.numel() == 0 || weight == 0.f)
                return;

            // All allocation, expression lowering, reduction and ordering stay
            // inside Tensor. No host scalar readback or foreign queue is needed.
            const core::GpuBackendScope scope(core::GpuBackend::HIP);
            const Tensor activated = kind == Regularizer::Scale ? raw.exp() : raw.sigmoid();
            const Tensor reduced = (activated.mean() * weight).reshape({1});
            Tensor updated;
            if (gradient.is_valid()) {
                const float scale = weight / static_cast<float>(raw.numel());
                if (kind == Regularizer::Scale) {
                    updated = gradient + activated * scale;
                } else {
                    // Match the native stable derivative, including saturated
                    // positive logits where sigmoid(x) * (1-sigmoid(x)) is zero.
                    const Tensor e = (-raw.abs()).exp();
                    const Tensor denominator = e + 1.f;
                    updated = gradient + e / (denominator * denominator) * scale;
                }
            }
            loss.copy_from(reduced.reshape(loss.shape()));
            if (gradient.is_valid())
                gradient.copy_from(updated);
        }
    } // namespace

    const gpu_ops::ExtraLossOps& hip_extra_loss_ops() {
        static const gpu_ops::ExtraLossOps ops{.regularize = regularize};
        return ops;
    }
} // namespace lfs::training
