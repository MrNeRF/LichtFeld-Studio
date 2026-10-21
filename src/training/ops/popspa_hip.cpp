/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/popspa_hip.hpp"
#include "training/hip/application_popspa_api.h"
#include "training/hip/application_tensor.hpp"
#include <core/tensor_execution.hpp>
#include <climits>
#include <cmath>
#include <limits>

namespace lfs::training {
    namespace {
        using namespace gpu_ops;
        using core::DataType;
        using core::Device;

        void require(bool condition, const char* message) {
            if (!condition) throw std::invalid_argument(message);
        }
        void shape(In tensor, size_t elements, DataType dtype) {
            require(tensor.is_valid() && tensor.numel() == elements && tensor.dtype() == dtype,
                    "HIP POPSpa tensor shape or dtype mismatch");
        }
        size_t rows(In raw) {
            require(raw.is_valid() && raw.dtype() == DataType::Float32 && raw.numel() <= INT_MAX,
                    "HIP POPSpa requires bounded Float32 parameters");
            return raw.numel();
        }
        LfsHipPopspaCall descriptor(hip::ApplicationCall& owner, uint32_t operation, size_t count) {
            LfsHipPopspaCall call{};
            call.struct_bytes = sizeof(call);
            call.operation = operation;
            call.device = owner.queue.device;
            call.queue = owner.queue.native_queue;
            call.count = count;
            return call;
        }
        void submit(hip::ApplicationCall& owner, LfsHipPopspaCall& call) {
            if (call.operation <= LFS_HIP_POPSPA_PROJECT && call.keep_count && call.keep_count < call.count) {
                const auto query = owner.module->symbol<decltype(&lfs_hip_application_popspa_workspace)>(
                    "lfs_hip_application_popspa_workspace");
                LfsHipError error{};
                if (int code = query(&call, &call.workspace_bytes, &error)) throw hip::TrainingError(code, error);
                require(call.workspace_bytes <= std::numeric_limits<size_t>::max(), "HIP POPSpa workspace exceeds address space");
                // Core owns, budgets and caches this storage. Native selection
                // cannot mutate outputs until its complete workspace is ready.
                auto scratch = Tensor::empty_exact({size_t(call.workspace_bytes)}, DataType::UInt8);
                call.workspace = owner.bind(scratch, DataType::UInt8);
            }
            const auto run = owner.module->symbol<decltype(&lfs_hip_application_popspa)>("lfs_hip_application_popspa");
            LfsHipError error{};
            if (int code = run(&call, &error)) throw hip::TrainingError(code, error);
        }

        void keep_indices(In scores, In active, In frozen, size_t k, Out keep) {
            require(scores.is_valid() && scores.ndim() == 2 && scores.shape()[1] == 8 &&
                        scores.dtype() == DataType::UInt8 && scores.shape()[0] <= INT_MAX && k <= scores.shape()[0],
                    "HIP POPSpa scores require UInt8 [N,8] Float64 rows and k <= N");
            const size_t n = scores.shape()[0];
            shape(active, n, DataType::Bool); shape(frozen, n, DataType::Bool);
            const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
            keep = Tensor::empty_exact({k}, DataType::Int64);
            if (!k) return;
            hip::ApplicationCall owner;
            auto call = descriptor(owner, LFS_HIP_POPSPA_KEEP, n);
            call.keep_count = k;
            call.scores = owner.bind(scores, DataType::UInt8);
            call.active = owner.bind(active, DataType::Bool); call.frozen = owner.bind(frozen, DataType::Bool);
            call.keep = owner.bind(keep, DataType::Int64);
            submit(owner, call);
        }

        void project(In raw, Out z, Out u, In active, In frozen, size_t k, bool update_dual) {
            const size_t n = rows(raw);
            require(k <= n, "HIP POPSpa projection k exceeds row count");
            shape(z, n, DataType::Float32); shape(u, n, DataType::Float32);
            shape(active, n, DataType::Bool); shape(frozen, n, DataType::Bool);
            if (!n) return;
            const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
            hip::ApplicationCall owner;
            auto call = descriptor(owner, LFS_HIP_POPSPA_PROJECT, n);
            call.keep_count = k; call.update_dual = update_dual;
            call.raw = owner.bind(raw, DataType::Float32); call.z = owner.bind(z, DataType::Float32);
            call.u = owner.bind(u, DataType::Float32);
            call.active = owner.bind(active, DataType::Bool); call.frozen = owner.bind(frozen, DataType::Bool);
            submit(owner, call);
        }

        void penalty(In raw, In z, In u, In active, In frozen, Out gradient, Out loss, float rho) {
            const size_t n = rows(raw);
            require(std::isfinite(rho) && rho >= 0.f, "Invalid HIP POPSpa penalty weight");
            shape(z, n, DataType::Float32); shape(u, n, DataType::Float32);
            shape(active, n, DataType::Bool); shape(frozen, n, DataType::Bool);
            shape(gradient, n, DataType::Float32); shape(loss, 1, DataType::Float32);
            if (!n) return;
            const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
            hip::ApplicationCall owner;
            auto call = descriptor(owner, LFS_HIP_POPSPA_PENALTY, n);
            call.rho = rho;
            call.raw = owner.bind(raw, DataType::Float32); call.z = owner.bind(z, DataType::Float32);
            call.u = owner.bind(u, DataType::Float32); call.gradient = owner.bind(gradient, DataType::Float32);
            call.active = owner.bind(active, DataType::Bool); call.frozen = owner.bind(frozen, DataType::Bool);
            call.loss = owner.bind(loss, DataType::Float32);
            submit(owner, call);
        }

        void effective_rank(In scales, Out gradient, In active, In frozen, Out loss, size_t active_count,
                            float rank_weight, float thin_weight, float epsilon) {
            const size_t elements = rows(scales), n = elements / 3;
            require(elements % 3 == 0 && active_count <= n && std::isfinite(rank_weight) && rank_weight >= 0.f &&
                        std::isfinite(thin_weight) && thin_weight >= 0.f && std::isfinite(epsilon) && epsilon > 0.f,
                    "Invalid HIP effective-rank dimensions or weights");
            if (gradient.is_valid() && gradient.numel()) shape(gradient, elements, DataType::Float32);
            if (active.is_valid() && active.numel()) shape(active, n, DataType::Bool);
            if (frozen.is_valid() && frozen.numel()) shape(frozen, n, DataType::Bool);
            shape(loss, 1, DataType::Float32);
            if (!n || !active_count) return;
            const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
            hip::ApplicationCall owner;
            auto call = descriptor(owner, LFS_HIP_POPSPA_EFFECTIVE_RANK, n);
            call.active_count = active_count;
            call.rank_weight = rank_weight; call.thin_weight = thin_weight; call.epsilon = epsilon;
            call.raw = owner.bind(scales, DataType::Float32); call.gradient = owner.bind(gradient, DataType::Float32);
            call.active = owner.bind(active, DataType::Bool); call.frozen = owner.bind(frozen, DataType::Bool);
            call.loss = owner.bind(loss, DataType::Float32);
            submit(owner, call);
        }
    }

    const gpu_ops::PopspaOps& hip_popspa_ops() {
        static const gpu_ops::PopspaOps ops{
            .score_encoding = gpu_ops::PopScoreEncoding::Float64,
            .keep_indices = keep_indices,
            .project = project,
            .penalty = penalty,
            .effective_rank = effective_rank,
        };
        return ops;
    }
}
