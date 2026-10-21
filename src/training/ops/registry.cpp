/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"
#ifdef LFS_HIP_APPLICATION_POPSPA
#include "lfs/training/ops/popspa_hip.hpp"
#endif
#ifdef LFS_TENSOR_HIP
#include "core/tensor_execution.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_FAST
#include "lfs/training/ops/fast_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_MCMC
#include "lfs/training/ops/mcmc_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_REFINE
#include "lfs/training/ops/refine_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_PHOTOMETRIC
#include "lfs/training/ops/photometric_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_ADAM
#include "lfs/training/ops/adam_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_SH
#include "lfs/training/ops/sh_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_SESSION
#include "lfs/training/ops/session_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_TRAINING_IMAGE
#include "lfs/training/ops/training_image_hip.hpp"
#endif

#ifdef LFS_HIP_APPLICATION_EXTRA_LOSS
#include "lfs/training/ops/extra_loss_hip.hpp"
#endif

#ifdef LFS_HIP_APPLICATION_MORTON
#include "lfs/training/ops/morton_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_LPIPS
#include "lfs/training/ops/lpips_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_STRUCTURE
#include "lfs/training/ops/structure_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_BLOB
#include "lfs/training/ops/blob_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_CONTROLLER
#include "lfs/training/ops/controller_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_MRNF
#include "lfs/training/ops/mrnf_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_BILATERAL
#include "lfs/training/ops/bilateral_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_PPISP
#include "lfs/training/ops/ppisp_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_GEOMETRY
#include "lfs/training/ops/geometry_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_MASKS
#include "lfs/training/ops/masks_hip.hpp"
#endif
#ifdef LFS_HIP_APPLICATION_GSPLAT
#include "lfs/training/ops/gsplat_hip.hpp"
#endif

#include <array>
#include <format>
#include <stdexcept>

namespace lfs::training {
    namespace {
        core::GpuBackend active_training_backend() {
#ifdef LFS_TENSOR_HIP
            // Model mutation in an explicit HIP execution scope must use its
            // native kernels even when the process default selects another backend.
            if (core::TensorExecutionTarget::current().backend() == core::GpuBackend::HIP)
                return core::GpuBackend::HIP;
#endif
            return core::default_gpu_backend();
        }
    }
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
            case Family::Blob:
                return ops.blob != nullptr;
            case Family::Structure:
                return ops.structure != nullptr;
            case Family::Popspa:
                return ops.popspa != nullptr && (ops.fast == nullptr || ops.fast->pop_scores != nullptr) &&
                       (ops.gsplat == nullptr || ops.gsplat->pop_scores != nullptr);
            case Family::Count:
                return false;
            }
            return false;
        }

    } // namespace

    const TrainingOps& cuda_training_ops_table();
    const TrainingOps& metal_training_ops_table();
    const TrainingOps& vulkan_training_ops_table();

    const TrainingOps& training_ops(const core::GpuBackend backend) {
#ifdef LFS_TENSOR_HIP
        if (backend == core::GpuBackend::HIP) {
            static const TrainingOps pending = [] {
                TrainingOps result{.backend = core::GpuBackend::HIP};
#ifdef LFS_HIP_APPLICATION_POPSPA
                result.popspa = &hip_popspa_ops();
#endif
#ifdef LFS_HIP_APPLICATION_BILATERAL
                result.bilateral = &hip_bilateral_ops();
#endif
#ifdef LFS_HIP_APPLICATION_PPISP
                result.ppisp = &hip_ppisp_ops();
#endif
#ifdef LFS_HIP_APPLICATION_GEOMETRY
                result.geometry = &hip_geometry_ops();
#endif
#ifdef LFS_HIP_APPLICATION_MASKS
                result.masks = &hip_masks_ops();
#endif
#ifdef LFS_HIP_APPLICATION_GSPLAT
                result.gsplat = &hip_gsplat_ops();
#endif
#ifdef LFS_HIP_APPLICATION_MRNF
                result.mrnf = &hip_mrnf_ops();
#endif
#ifdef LFS_HIP_APPLICATION_CONTROLLER
                result.controller = &hip_controller_ops();
#endif
#ifdef LFS_HIP_APPLICATION_BLOB
                result.blob = &hip_blob_ops();
#endif
#ifdef LFS_HIP_APPLICATION_STRUCTURE
                result.structure = &hip_structure_ops();
#endif
#ifdef LFS_HIP_APPLICATION_MORTON
                result.morton = &hip_morton_ops();
#endif
#ifdef LFS_HIP_APPLICATION_LPIPS
                result.lpips = &hip_lpips_ops();
#endif
#ifdef LFS_HIP_APPLICATION_EXTRA_LOSS
                result.extra_loss = &hip_extra_loss_ops();
#endif
#ifdef LFS_HIP_APPLICATION_FAST
                result.fast = &hip_fast_ops();
#endif
#ifdef LFS_HIP_APPLICATION_MCMC
                result.mcmc = &hip_mcmc_ops();
#endif
#ifdef LFS_HIP_APPLICATION_REFINE
                result.refine = &hip_refine_ops();
#endif
#ifdef LFS_HIP_APPLICATION_PHOTOMETRIC
                result.photometric = &hip_photometric_ops();
#endif
#ifdef LFS_HIP_APPLICATION_ADAM
                result.adam = &hip_adam_ops();
#endif
#ifdef LFS_HIP_APPLICATION_SH
                result.sh = &hip_sh_ops();
#endif
#ifdef LFS_HIP_APPLICATION_SESSION
                result.session = &hip_session_ops();
#endif
#ifdef LFS_HIP_APPLICATION_TRAINING_IMAGE
                result.training_image = &hip_training_image_ops();
#endif
#ifdef LFS_HIP_APPLICATION_SHARED_IMAGE
                result.shared_image = core::shared_image_ops(core::GpuBackend::HIP);
#endif
                return result;
            }();
            return pending;
        }
#endif
        static const TrainingOps& kCuda = cuda_training_ops_table();
        static const TrainingOps& kVulkan = vulkan_training_ops_table();
        static const TrainingOps& kMetal = metal_training_ops_table();
        static const std::array tables{&kCuda, &kVulkan, &kMetal};
        const auto index = static_cast<size_t>(backend);
        return index < tables.size() ? *tables[index] : kVulkan;
    }

    const ops::MortonOps& training_morton_ops() {
        const auto backend = active_training_backend();
        const auto* morton = training_ops(backend).morton;
        if (!morton)
            throw std::runtime_error(unavailable_training_family(backend, Family::Morton)
                                         .value_or("Morton training ops are unavailable"));
        return *morton;
    }

    const ops::SessionOps& training_session_ops() {
        const auto backend = active_training_backend();
        const auto* session = training_ops(backend).session;
        if (!session)
            throw std::runtime_error(unavailable_training_family(backend, Family::Session)
                                         .value_or("Session training ops are unavailable"));
        return *session;
    }

    const ops::ShOps& training_sh_ops() {
        const auto backend = active_training_backend();
        const auto* sh = training_ops(backend).sh;
        if (sh == nullptr) {
            throw std::runtime_error(
                unavailable_training_family(backend, Family::Sh)
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
        case Family::Blob: return "Blob";
        case Family::Structure: return "Structure";
        case Family::Popspa: return "Popspa";
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
