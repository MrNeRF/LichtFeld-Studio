/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

#include "lfs/training/ops/adam_cuda.hpp"
#include "lfs/training/ops/mcmc_cuda.hpp"
#include "lfs/training/ops/mrnf_cuda.hpp"
#include "lfs/training/ops/photometric_cuda.hpp"

#include <format>

namespace lfs::training {
    namespace {

        // Families without a table slot still run in the CUDA trainer.
        [[nodiscard]] bool family_available(const TrainingOps& ops, const Family family) {
            switch (family) {
            case Family::Photometric:
                return ops.photometric != nullptr;
            case Family::Adam:
                return ops.adam != nullptr;
            case Family::Mcmc:
                return ops.mcmc != nullptr;
            case Family::Mrnf:
                return ops.mrnf != nullptr;
            case Family::Count:
                return false;
            default:
                return ops.backend == core::GpuBackend::CUDA;
            }
        }

    } // namespace

    const TrainingOps& training_ops(const core::GpuBackend backend) {
        static const TrainingOps kCuda{
            .backend = core::GpuBackend::CUDA,
            .photometric = &cuda_photometric_ops(),
            .adam = &cuda_adam_ops(),
            .mcmc = &cuda_mcmc_ops(),
            .mrnf = &cuda_mrnf_ops(),
        };
        static const TrainingOps kVulkan{
            .backend = core::GpuBackend::Vulkan,
            .photometric = nullptr,
            .adam = nullptr,
            .mrnf = nullptr,
        };
        static const TrainingOps kMetal{
            .backend = core::GpuBackend::Metal,
            .photometric = nullptr,
            .adam = nullptr,
            .mrnf = nullptr,
        };
        switch (backend) {
        case core::GpuBackend::CUDA:
            return kCuda;
        case core::GpuBackend::Vulkan:
            return kVulkan;
        case core::GpuBackend::Metal:
            return kMetal;
        }
        return kVulkan;
    }

    std::string_view training_family_name(const Family family) {
        switch (family) {
        case Family::Session: return "Session";
        case Family::Fast: return "Fast";
        case Family::Gsplat: return "Gsplat";
        case Family::Photometric: return "Photometric";
        case Family::Geometry: return "Geometry";
        case Family::Masks: return "Masks";
        case Family::ExtraLoss: return "ExtraLoss";
        case Family::Adam: return "Adam";
        case Family::Mcmc: return "Mcmc";
        case Family::Mrnf: return "Mrnf";
        case Family::Refine: return "Refine";
        case Family::Bilateral: return "Bilateral";
        case Family::PPISP: return "PPISP";
        case Family::Controller: return "Controller";
        case Family::Morton: return "Morton";
        case Family::Sh: return "Sh";
        case Family::TrainingImage: return "TrainingImage";
        case Family::SharedImage: return "SharedImage";
        case Family::Lpips: return "Lpips";
        case Family::Count: break;
        }
        return {};
    }

    std::vector<std::string_view> missing_training_families(
        const TrainingOps& ops, const FamilySet required) {
        std::vector<std::string_view> missing;
        for (size_t index = 0; index < static_cast<size_t>(Family::Count); ++index) {
            if (!required.test(index)) {
                continue;
            }
            const auto family = static_cast<Family>(index);
            if (!family_available(ops, family)) {
                missing.push_back(training_family_name(family));
            }
        }
        return missing;
    }

    std::string training_unavailable_message(
        const core::GpuBackend backend, const std::vector<std::string_view>& missing) {
        std::string text = std::format(
            "{} training is unavailable for this configuration.\nMissing families: ",
            core::gpu_backend_name(backend));
        for (size_t index = 0; index < missing.size(); ++index) {
            if (index != 0) {
                text += ", ";
            }
            text += missing[index];
        }
        text += '.';
        return text;
    }

} // namespace lfs::training
