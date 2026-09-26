/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

#include "core/parameters.hpp"

namespace lfs::training {
    namespace {

        using core::param::BackgroundMode;
        using core::param::MaskMode;
        using core::param::RasterBackendId;

    } // namespace

    FamilySet training_loader_dependencies(const core::param::TrainingParameters&) {
        FamilySet dependencies;
        // Training images go through the shared image conversion kernels.
        dependencies.set(static_cast<size_t>(Family::SharedImage));
        return dependencies;
    }

    FamilySet required_training_families(
        const core::param::TrainingParameters& resolved,
        const FamilySet input_dependencies) {
        FamilySet required = input_dependencies;
        const auto& opt = resolved.optimization;

        required.set(static_cast<size_t>(Family::Photometric));
        required.set(static_cast<size_t>(Family::Adam));
        required.set(static_cast<size_t>(Family::Sh));
        // The trainer updates the camera-loss heatmap on every step.
        required.set(static_cast<size_t>(Family::TrainingImage));

        if (opt.raster_backend() == RasterBackendId::ThreeDGUT) {
            required.set(static_cast<size_t>(Family::Gsplat));
        } else {
            required.set(static_cast<size_t>(Family::Fast));
        }

        const auto strategy = core::param::canonical_strategy_name(opt.strategy);
        const bool mcmc = strategy == core::param::kStrategyMCMC;
        const bool mrnf = strategy == core::param::kStrategyMRNF;
        const bool igs = strategy == core::param::kStrategyIGSPlus;
        if (mcmc || igs) {
            required.set(static_cast<size_t>(Family::Mcmc));
        }
        if (mrnf || igs) {
            required.set(static_cast<size_t>(Family::Mrnf));
        }
        if (mcmc || mrnf || igs || opt.use_edge_map) {
            required.set(static_cast<size_t>(Family::Refine));
        }
        if (opt.morton_reorder_interval > 0) {
            required.set(static_cast<size_t>(Family::Morton));
        }
        if (opt.mask_mode != MaskMode::None) {
            required.set(static_cast<size_t>(Family::Masks));
            required.set(static_cast<size_t>(Family::SharedImage));
        }
        if (opt.use_depth_loss || opt.use_normal_loss) {
            required.set(static_cast<size_t>(Family::Geometry));
            required.set(static_cast<size_t>(Family::SharedImage));
        }
        if (opt.scale_reg > 0.f || opt.opacity_reg > 0.f || opt.enable_sparsity) {
            required.set(static_cast<size_t>(Family::ExtraLoss));
        }
        if (opt.bilateral_grid_active()) {
            required.set(static_cast<size_t>(Family::Bilateral));
        }
        if (opt.ppisp_active()) {
            required.set(static_cast<size_t>(Family::PPISP));
        }
        if (opt.ppisp_active() && opt.ppisp_use_controller) {
            required.set(static_cast<size_t>(Family::Controller));
        }
        if (opt.use_error_map || opt.use_edge_map || opt.bg_modulation ||
            opt.bg_mode == BackgroundMode::Image || opt.bg_mode == BackgroundMode::Random) {
            required.set(static_cast<size_t>(Family::TrainingImage));
        }
        if (opt.undistort || resolved.dataset.resize_factor > 0) {
            required.set(static_cast<size_t>(Family::SharedImage));
        }
        if (opt.enable_eval) {
            required.set(static_cast<size_t>(Family::Lpips));
        }
        if (opt.perf_bench || opt.profile_start_iter >= 0) {
            required.set(static_cast<size_t>(Family::Session));
        }
        return required;
    }

    std::optional<std::string> unavailable_training_reason(
        const core::param::TrainingParameters& resolved,
        const core::GpuBackend backend,
        const FamilySet input_dependencies) {
        const TrainingOps& ops = training_ops(backend);
        const FamilySet required = required_training_families(resolved, input_dependencies);
        const auto missing = missing_training_families(ops, required);
        if (missing.empty()) {
            return std::nullopt;
        }
        return training_unavailable_message(backend, missing);
    }

    std::optional<std::string> unavailable_training_family(
        const core::GpuBackend backend, const Family family) {
        FamilySet required;
        required.set(static_cast<size_t>(family));
        const auto missing = missing_training_families(training_ops(backend), required);
        if (missing.empty()) {
            return std::nullopt;
        }
        return training_unavailable_message(backend, missing);
    }

} // namespace lfs::training
