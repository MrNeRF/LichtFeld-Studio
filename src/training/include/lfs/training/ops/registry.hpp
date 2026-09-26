/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/gpu_backend_fwd.hpp"
#include "lfs/training/ops/adam.hpp"
#include "lfs/training/ops/geometry.hpp"
#include "lfs/training/ops/loss.hpp"
#include "lfs/training/ops/mcmc.hpp"
#include "lfs/training/ops/mrnf.hpp"

#include <bitset>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::core::param {
    struct TrainingParameters;
}

namespace lfs::training {

    namespace ops = lfs::gpu_ops;

    struct TrainingOps {
        core::GpuBackend backend = core::GpuBackend::CUDA;
        const ops::PhotometricOps* photometric = nullptr;
        const ops::AdamOps* adam = nullptr;
        const ops::McmcOps* mcmc = nullptr;
        const ops::MrnfOps* mrnf = nullptr;
        const ops::GeometryLossOps* geometry = nullptr;
    };

    enum class Family {
        Session,
        Fast,
        Gsplat,
        Photometric,
        Geometry,
        Masks,
        ExtraLoss,
        Adam,
        Mcmc,
        Mrnf,
        Refine,
        Bilateral,
        PPISP,
        Controller,
        Morton,
        Sh,
        TrainingImage,
        SharedImage,
        Lpips,
        Count
    };

    using FamilySet = std::bitset<static_cast<size_t>(Family::Count)>;

    const TrainingOps& training_ops(core::GpuBackend backend);

    // Called after configuration defaults and input-dependent options are resolved.
    // input_dependencies covers the selected loader and preprocessing path.
    FamilySet required_training_families(
        const core::param::TrainingParameters& resolved,
        FamilySet input_dependencies);

    // Image conversion used by every training loader.
    FamilySet training_loader_dependencies(const core::param::TrainingParameters& resolved);

    std::vector<std::string_view> missing_training_families(
        const TrainingOps& ops, FamilySet required);

    std::string_view training_family_name(Family family);

    std::string training_unavailable_message(
        core::GpuBackend backend, const std::vector<std::string_view>& missing);

    // Empty when the cached table can run this configuration.
    std::optional<std::string> unavailable_training_reason(
        const core::param::TrainingParameters& resolved,
        core::GpuBackend backend,
        FamilySet input_dependencies);

    // Empty when this one family is present on the backend's table.
    std::optional<std::string> unavailable_training_family(
        core::GpuBackend backend, Family family);

} // namespace lfs::training
