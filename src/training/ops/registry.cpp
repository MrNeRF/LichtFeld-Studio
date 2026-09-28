/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

#include <array>
#include <format>
#include <stdexcept>

namespace lfs::training {
    namespace {

        // Families without a table slot still run in the CUDA trainer.
        [[nodiscard]] bool family_available(const TrainingOps& ops, const Family family) {
            switch (family) {
            case Family::Session:
                return ops.session != nullptr;
            case Family::Photometric:
                return ops.photometric != nullptr;
            case Family::Adam:
                return ops.adam != nullptr;
            case Family::Mcmc:
                return ops.mcmc != nullptr;
            case Family::Mrnf:
                return ops.mrnf != nullptr;
            case Family::Geometry:
                return ops.geometry != nullptr;
            case Family::Fast:
                return ops.fast != nullptr;
            case Family::Morton:
                return ops.morton != nullptr;
            case Family::Masks:
                return ops.masks != nullptr;
            case Family::ExtraLoss:
                return ops.extra_loss != nullptr;
            case Family::Bilateral:
                return ops.bilateral != nullptr;
            case Family::TrainingImage:
                return ops.training_image != nullptr;
            case Family::Sh:
                return ops.sh != nullptr;
            case Family::PPISP:
                return ops.ppisp != nullptr;
            case Family::Controller:
                return ops.controller != nullptr;
            case Family::Gsplat:
                return ops.gsplat != nullptr;
            case Family::Refine:
                return ops.refine != nullptr;
            case Family::SharedImage:
                return ops.shared_image != nullptr;
            case Family::Lpips:
                return ops.lpips != nullptr;
            case Family::Count:
                return false;
            }
            return false;
        }

    } // namespace

    const TrainingOps& cuda_training_ops_table();
    const TrainingOps& metal_training_ops_table();

    const TrainingOps& training_ops(const core::GpuBackend backend) {
        static const TrainingOps& kCuda = cuda_training_ops_table();
        static const TrainingOps kVulkan{
            .backend = core::GpuBackend::Vulkan,
            .photometric = nullptr,
            .adam = nullptr,
            .mrnf = nullptr,
            .fast = nullptr,
            .sh = nullptr,
        };
        static const TrainingOps& kMetal = metal_training_ops_table();
        static const std::array tables{&kCuda, &kVulkan, &kMetal};
        const auto index = static_cast<size_t>(backend);
        return index < tables.size() ? *tables[index] : kVulkan;
    }

    const ops::MortonOps& training_morton_ops() {
        const auto backend = core::default_gpu_backend();
        const auto* morton = training_ops(backend).morton;
        if (!morton)
            throw std::runtime_error(unavailable_training_family(backend, Family::Morton)
                                         .value_or("Morton training ops are unavailable"));
        return *morton;
    }

    const ops::SessionOps& training_session_ops() {
        const auto backend = core::default_gpu_backend();
        const auto* session = training_ops(backend).session;
        if (!session)
            throw std::runtime_error(unavailable_training_family(backend, Family::Session)
                                         .value_or("Session training ops are unavailable"));
        return *session;
    }

    const ops::ShOps& training_sh_ops() {
        const auto* sh = training_ops(core::default_gpu_backend()).sh;
        if (sh == nullptr) {
            throw std::runtime_error(
                unavailable_training_family(core::default_gpu_backend(), Family::Sh)
                    .value_or("Sh training ops are unavailable"));
        }
        return *sh;
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
