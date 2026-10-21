/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "session.hpp"
#include "splat_storage_layout.hpp"
#include <cmath>
#include <core/tensor_backend.hpp>
#include <core/tensor_execution.hpp>
#include <optional>
#include <utility>

namespace lfs::training::hip {
    // Host-side connection to the actual Tensor allocation. The shallow snapshot
    // pins the original allocation even if the source Tensor is reassigned.
    // Every operation validates its source/control generation before GPU work.
    // Callers must still serialize model mutations; Control is not thread-safe.
    class TensorModel {
        using Control = core::SplatExportableStorage::Control;
        struct Lease {
            std::shared_ptr<core::Tensor> source;
            core::Tensor snapshot;
            std::shared_ptr<Control> control;
            std::shared_ptr<core::ExportableBlock> block;
            std::array<size_t, core::SplatExportableStorage::Count> offsets, bytes;
            size_t capacity, live_count;
            uint64_t generation;
            LfsHipTrainDeviceModel device;
            LfsHipTrainModelLayout layout;
            std::shared_ptr<core::Tensor> sh_source;
            core::Tensor sh_snapshot;
            std::optional<LfsHipTrainResidentSh> sh;
            void validate() const {
                if (!source || !control || control->poisoned || control->generation != generation ||
                    control->block != block || control->capacity != capacity || control->region_offsets != offsets || control->region_bytes != bytes ||
                    !source->is_valid() || core::gpu_backend_of(*source) != core::GpuBackend::HIP ||
                    source->dtype() != core::DataType::UInt8 || !source->is_contiguous() || source->bytes() != device.allocation_bytes ||
                    reinterpret_cast<uintptr_t>(std::as_const(*source).data_ptr()) != device.address ||
                    reinterpret_cast<uintptr_t>(block->device_ptr) != device.address || block->reserved_bytes != layout.required_bytes)
                    throw std::invalid_argument("HIP training model storage changed; close and attach a new session");
                if (sh && (!sh_source || !sh_source->is_valid() ||
                           core::gpu_backend_of(*sh_source) != core::GpuBackend::HIP ||
                           sh_source->dtype() != core::DataType::Float32 || !sh_source->is_contiguous() ||
                           sh_source->bytes() != sh_snapshot.bytes() ||
                           reinterpret_cast<uintptr_t>(std::as_const(*sh_source).data_ptr()) != device.address + sh->offset_bytes))
                    throw std::invalid_argument("HIP training SH view changed; close and attach a new session");
            }
            void prepare() const {
                validate();
                // Tensor operations and the isolated session own distinct native
                // streams. Complete producers before borrowing their allocation;
                // session calls complete their own stream before returning.
                try {
                    core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP).wait();
                } catch (...) {
                    control->poisoned = true;
                    throw;
                }
            }
        };
        std::shared_ptr<Lease> lease_;

    public:
        TensorModel(std::shared_ptr<core::Tensor> source, std::shared_ptr<Control> control, size_t live_count, int ordinal,
                    std::shared_ptr<core::Tensor> fp32_sh = {}, uint32_t max_degree = 0,
                    uint32_t active_degree = 0, float rest_learning_rate = 0.002f) {
            if (!source || !source->is_valid() || core::gpu_backend_of(*source) != core::GpuBackend::HIP ||
                source->dtype() != core::DataType::UInt8 || !source->is_contiguous() || !control)
                throw std::invalid_argument("HIP training requires a contiguous native HIP byte Tensor and its storage control");
            const auto layout = splat_storage_layout(*control, live_count, bool(fp32_sh));
            if (layout.required_bytes > source->bytes())
                throw std::invalid_argument("HIP model control exceeds Tensor allocation");
            const LfsHipTrainDeviceModel device{sizeof(device), ordinal, reinterpret_cast<uintptr_t>(std::as_const(*source).data_ptr()), source->bytes()};
            lease_ = std::make_shared<Lease>(Lease{source, *source, control, control->block, control->region_offsets, control->region_bytes,
                                                   control->capacity, live_count, control->generation, device, layout});
            if (fp32_sh) {
                if (max_degree < 1 || max_degree > 3 || active_degree > max_degree ||
                    !std::isfinite(rest_learning_rate) || rest_learning_rate < 0)
                    throw std::invalid_argument("Invalid HIP FP32 SH configuration");
                const size_t coefficients = ((max_degree + 1) * (max_degree + 1) - 1) * 3;
                const size_t required = ((live_count + 31) / 32 * 32) * ((coefficients + 3) / 4) * 16;
                if (fp32_sh->bytes() != required || required > control->region_bytes[core::SplatExportableStorage::ShN])
                    throw std::invalid_argument("HIP FP32 SH view has an incompatible swizzled size");
                lease_->sh_source = fp32_sh;
                lease_->sh_snapshot = *fp32_sh;
                lease_->sh = LfsHipTrainResidentSh{sizeof(LfsHipTrainResidentSh), max_degree, active_degree,
                                                   rest_learning_rate, control->region_offsets[core::SplatExportableStorage::ShN]};
            } else if (max_degree || active_degree) {
                throw std::invalid_argument("HIP higher SH requires an explicit FP32 Tensor view");
            }
            lease_->validate();
        }
        std::unique_ptr<Session> open(const std::filesystem::path& module, const LfsHipTrainConfig& config) const {
            lease_->validate();
            if (config.gaussian_count != lease_->live_count || config.device_ordinal != lease_->device.device_ordinal)
                throw std::invalid_argument("HIP training configuration differs from its Tensor model");
            auto session = std::make_unique<Session>(module, config, lease_->device, lease_->layout, lease_, [lease = lease_] { lease->prepare(); }, [lease = lease_](int status) {
                    if (status == LFS_HIP_TRAIN_GPU_ERROR || status == LFS_HIP_TRAIN_POISONED)
                        lease->control->poisoned = true; });
            if (lease_->sh)
                session->attach_resident_sh(*lease_->sh);
            return session;
        }
    };
} // namespace lfs::training::hip
