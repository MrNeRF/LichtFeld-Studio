/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_appearance_correction.hpp"
#include "core/logger.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/export_post_process.hpp"
#include "scene/scene_manager.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller.hpp"
#include "training/components/ppisp_controller_pool.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <exception>
#include <format>
#include <mutex>
#include <vector>

namespace lfs::vis {

    namespace {
        [[nodiscard]] lfs::training::PPISPRenderOverrides toRenderOverrides(const PPISPOverrides& ov) {
            lfs::training::PPISPRenderOverrides r;
            r.exposure_offset = ov.exposure_offset;
            r.vignette_enabled = ov.vignette_enabled;
            r.vignette_strength = ov.vignette_strength;
            r.wb_temperature = ov.wb_temperature;
            r.wb_tint = ov.wb_tint;
            r.color_red_x = ov.color_red_x;
            r.color_red_y = ov.color_red_y;
            r.color_green_x = ov.color_green_x;
            r.color_green_y = ov.color_green_y;
            r.color_blue_x = ov.color_blue_x;
            r.color_blue_y = ov.color_blue_y;
            r.gamma_multiplier = ov.gamma_multiplier;
            r.gamma_red = ov.gamma_red;
            r.gamma_green = ov.gamma_green;
            r.gamma_blue = ov.gamma_blue;
            r.crf_toe = ov.crf_toe;
            r.crf_shoulder = ov.crf_shoulder;
            return r;
        }

        [[nodiscard]] bool isImageWithAlpha(const lfs::core::Tensor& image) {
            return image.ndim() == 3 &&
                   ((image.shape()[0] == 4) || (image.shape()[2] == 4));
        }

        [[nodiscard]] lfs::core::Tensor preparePpispInput(lfs::core::Tensor input) {
            if (!input.is_valid() || input.device() == lfs::core::Device::CUDA) {
                return input;
            }
            return input.cuda();
        }

        // Applies PPISP to an image regardless of who owns the component (standalone
        // appearance model or an active trainer). A non-null pool selects the
        // controller path; controller_params (from one thumbnail prediction) keeps
        // banded application seam-free.
        [[nodiscard]] lfs::core::Tensor applyPpispAppearance(
            lfs::training::PPISP& ppisp,
            lfs::training::PPISPControllerPool* const pool,
            const lfs::core::Tensor& rgb,
            const int camera_uid,
            const PPISPOverrides& overrides,
            const lfs::training::PPISPRegion& region = {},
            const lfs::core::Tensor& controller_params = {}) {
            const bool was_hwc = (rgb.ndim() == 3 && rgb.shape()[2] == 3);
            const auto input = was_hwc ? rgb.permute({2, 0, 1}).contiguous() : rgb;
            const bool is_training_camera = ppisp.is_known_frame(camera_uid);
            const int camera_idx = is_training_camera ? ppisp.camera_index(ppisp.camera_for_frame(camera_uid)) : 0;

            lfs::core::Tensor result;

            if (pool != nullptr) {
                const int controller_idx =
                    (camera_idx >= 0 && camera_idx < pool->num_cameras()) ? camera_idx : 0;
                const auto params = controller_params.is_valid()
                                        ? controller_params
                                        : pool->predict(controller_idx, input.unsqueeze(0), 1.0f);
                result = overrides.isIdentity()
                             ? ppisp.apply_with_controller_params(input, params, controller_idx, region)
                             : ppisp.apply_with_controller_params_and_overrides(
                                   input, params, controller_idx, toRenderOverrides(overrides), region);
            } else if (is_training_camera) {
                const int camera_id = ppisp.camera_for_frame(camera_uid);
                result = overrides.isIdentity()
                             ? ppisp.apply(input, camera_id, camera_uid, region)
                             : ppisp.apply_with_overrides(
                                   input, camera_id, camera_uid, toRenderOverrides(overrides), region);
            } else {
                // Keep manual overrides active on novel views by anchoring them to any valid learned PPISP
                // frame/camera pair when no controller is available.
                const int fallback_camera = ppisp.any_camera_id();
                const int fallback_frame = ppisp.any_frame_uid();
                result = overrides.isIdentity()
                             ? ppisp.apply(input, fallback_camera, fallback_frame, region)
                             : ppisp.apply_with_overrides(
                                   input, fallback_camera, fallback_frame, toRenderOverrides(overrides), region);
            }

            return (was_hwc && result.is_valid()) ? result.permute({1, 2, 0}).contiguous() : result;
        }

        [[nodiscard]] lfs::core::Tensor applyStandaloneAppearance(
            const lfs::core::Tensor& rgb,
            SceneManager& scene_mgr,
            const int camera_uid,
            const PPISPOverrides& overrides,
            const bool use_controller) {
            auto* ppisp = scene_mgr.getAppearancePPISP();
            if (!ppisp) {
                return rgb;
            }
            auto* const pool = use_controller && scene_mgr.hasAppearanceController()
                                   ? scene_mgr.getAppearanceControllerPool()
                                   : nullptr;
            return applyPpispAppearance(*ppisp, pool, rgb, camera_uid, overrides);
        }
    } // namespace

    std::shared_ptr<lfs::core::Tensor> applyViewportAppearanceCorrection(
        std::shared_ptr<lfs::core::Tensor> image,
        SceneManager* const scene_manager,
        const RenderSettings& settings,
        const int camera_uid) {
        if (!image || !scene_manager || !settings.apply_appearance_correction) {
            return image;
        }

        lfs::core::Tensor rgb_input = *image;
        lfs::core::Tensor alpha;
        const bool has_alpha = isImageWithAlpha(*image);
        int alpha_concat_dim = 0;
        if (has_alpha) {
            if (image->shape()[0] == 4) {
                alpha = image->slice(0, 3, 4).contiguous();
                rgb_input = image->slice(0, 0, 3).contiguous();
                alpha_concat_dim = 0;
            } else {
                alpha = image->slice(2, 3, 4).contiguous();
                rgb_input = image->slice(2, 0, 3).contiguous();
                alpha_concat_dim = 2;
            }
        }

        const auto restore_alpha = [&](lfs::core::Tensor corrected_rgb) {
            if (!has_alpha || !corrected_rgb.is_valid()) {
                return corrected_rgb;
            }
            auto alpha_for_copy = alpha;
            if (alpha_for_copy.device() != corrected_rgb.device()) {
                alpha_for_copy = alpha_for_copy.to(corrected_rgb.device());
            }

            std::vector<size_t> restored_shape = corrected_rgb.shape().dims();
            restored_shape[static_cast<size_t>(alpha_concat_dim)] = 4;
            auto restored = lfs::core::Tensor::empty(
                lfs::core::TensorShape(restored_shape),
                corrected_rgb.device(),
                corrected_rgb.dtype());

            restored.slice(static_cast<size_t>(alpha_concat_dim), 0, 3).copy_from(corrected_rgb);
            restored.slice(static_cast<size_t>(alpha_concat_dim), 3, 4).copy_from(alpha_for_copy);
            return restored.contiguous();
        };

        if (const auto* tm = scene_manager->getTrainerManager()) {
            if (const auto* trainer = tm->getTrainer(); trainer && trainer->hasPPISP()) {
                lfs::training::PPISPViewportOverrides trainer_overrides{};
                if (settings.ppisp_mode == RenderSettings::PPISPMode::MANUAL) {
                    trainer_overrides.exposure_offset = settings.ppisp_overrides.exposure_offset;
                    trainer_overrides.vignette_enabled = settings.ppisp_overrides.vignette_enabled;
                    trainer_overrides.vignette_strength = settings.ppisp_overrides.vignette_strength;
                    trainer_overrides.wb_temperature = settings.ppisp_overrides.wb_temperature;
                    trainer_overrides.wb_tint = settings.ppisp_overrides.wb_tint;
                    trainer_overrides.gamma_multiplier = settings.ppisp_overrides.gamma_multiplier;
                }
                const bool use_controller = (settings.ppisp_mode == RenderSettings::PPISPMode::AUTO);
                try {
                    auto ppisp_input = preparePpispInput(rgb_input);
                    auto corrected = trainer->applyPPISPForViewport(
                        ppisp_input, camera_uid, trainer_overrides, use_controller);
                    corrected = restore_alpha(std::move(corrected));
                    return std::make_shared<lfs::core::Tensor>(std::move(corrected));
                } catch (const std::exception& e) {
                    LOG_WARN("Viewport trainer PPISP correction failed: {}", e.what());
                    return image;
                } catch (...) {
                    LOG_WARN("Viewport trainer PPISP correction failed");
                    return image;
                }
            }
        }

        if (!scene_manager->hasAppearanceModel()) {
            return image;
        }

        const auto& overrides = (settings.ppisp_mode == RenderSettings::PPISPMode::MANUAL)
                                    ? settings.ppisp_overrides
                                    : PPISPOverrides{};
        const bool use_controller = (settings.ppisp_mode == RenderSettings::PPISPMode::AUTO);
        lfs::core::Tensor corrected;
        try {
            auto ppisp_input = preparePpispInput(rgb_input);
            corrected = applyStandaloneAppearance(ppisp_input, *scene_manager, camera_uid, overrides, use_controller);
        } catch (const std::exception& e) {
            LOG_WARN("Standalone viewport PPISP correction failed: {}", e.what());
            return image;
        } catch (...) {
            LOG_WARN("Standalone viewport PPISP correction failed");
            return image;
        }
        if (!corrected.is_valid()) {
            return has_alpha ? std::make_shared<lfs::core::Tensor>(restore_alpha(rgb_input)) : image;
        }

        corrected = restore_alpha(std::move(corrected));
        return std::make_shared<lfs::core::Tensor>(std::move(corrected));
    }

    namespace {

        // Bounds the GPU working set of one export band (~1.1 GB at 32M pixels across
        // the unpack/PPISP/composite intermediates) independent of export size.
        constexpr int kMaxExportBandPixels = 32 * 1024 * 1024;
        constexpr int kThumbnailFallbackWidth = 1920;
        constexpr int kThumbnailFallbackHeight = 1080;

        // Stride-sampled CUDA float CHW thumbnail for the one-shot controller
        // prediction, targeting the live viewport size so every export resolution
        // predicts the same params the on-screen view uses.
        [[nodiscard]] lfs::core::Tensor makeExportThumbnailChw(const lfs::core::Tensor& image_u8_hwc_cpu,
                                                               glm::ivec2 target_size) {
            if (target_size.x <= 0 || target_size.y <= 0) {
                target_size = {kThumbnailFallbackWidth, kThumbnailFallbackHeight};
            }
            const int height = static_cast<int>(image_u8_hwc_cpu.size(0));
            const int width = static_cast<int>(image_u8_hwc_cpu.size(1));
            const int channels = static_cast<int>(image_u8_hwc_cpu.size(2));
            const int stride = std::max({1,
                                         (width + target_size.x - 1) / target_size.x,
                                         (height + target_size.y - 1) / target_size.y});
            const int thumb_height = (height + stride - 1) / stride;
            const int thumb_width = (width + stride - 1) / stride;

            auto thumbnail = lfs::core::Tensor::empty(
                {size_t{3}, static_cast<size_t>(thumb_height), static_cast<size_t>(thumb_width)},
                lfs::core::Device::CPU,
                lfs::core::DataType::Float32);
            const auto* const src = image_u8_hwc_cpu.ptr<uint8_t>();
            auto* const dst = thumbnail.ptr<float>();
            const auto plane = static_cast<size_t>(thumb_height) * static_cast<size_t>(thumb_width);
            constexpr float kInv255 = 1.0f / 255.0f;
            for (int ty = 0; ty < thumb_height; ++ty) {
                const auto src_row = static_cast<size_t>(ty) * static_cast<size_t>(stride) *
                                     static_cast<size_t>(width);
                for (int tx = 0; tx < thumb_width; ++tx) {
                    const auto s = (src_row + static_cast<size_t>(tx) * static_cast<size_t>(stride)) *
                                   static_cast<size_t>(channels);
                    const auto d = static_cast<size_t>(ty) * static_cast<size_t>(thumb_width) +
                                   static_cast<size_t>(tx);
                    dst[d] = static_cast<float>(src[s]) * kInv255;
                    dst[plane + d] = static_cast<float>(src[s + 1]) * kInv255;
                    dst[2 * plane + d] = static_cast<float>(src[s + 2]) * kInv255;
                }
            }
            return thumbnail.cuda();
        }

    } // namespace

    std::expected<lfs::core::Tensor, std::string> applyExportPostProcess(
        lfs::core::Tensor image_u8_hwc_cpu,
        SceneManager* const scene_manager,
        const RenderSettings& settings,
        const int camera_uid,
        const ExportPostProcessMode mode,
        const ExportPostProcessView& view) {
        const bool needs_env = mode == ExportPostProcessMode::EnvironmentComposite;
        const int expected_channels = mode == ExportPostProcessMode::Opaque ? 3 : 4;
        if (!image_u8_hwc_cpu.is_valid() || image_u8_hwc_cpu.device() != lfs::core::Device::CPU ||
            image_u8_hwc_cpu.dtype() != lfs::core::DataType::UInt8 || image_u8_hwc_cpu.ndim() != 3 ||
            static_cast<int>(image_u8_hwc_cpu.size(2)) != expected_channels ||
            !image_u8_hwc_cpu.is_contiguous()) {
            return std::unexpected(std::format(
                "export post-process expects a contiguous CPU u8 HWC image with {} channels", expected_channels));
        }
        const int height = static_cast<int>(image_u8_hwc_cpu.size(0));
        const int width = static_cast<int>(image_u8_hwc_cpu.size(1));

        // Resolve the PPISP component regardless of owner: an active trainer keeps it
        // internal; the viewer keeps a standalone appearance model on the scene.
        const bool use_controller = settings.ppisp_mode == RenderSettings::PPISPMode::AUTO;
        lfs::training::PPISP* ppisp = nullptr;
        lfs::training::PPISPControllerPool* controller_pool = nullptr;
        bool trainer_owned = false;
        if (scene_manager) {
            if (const auto* tm = scene_manager->getTrainerManager()) {
                if (const auto* trainer = tm->getTrainer(); trainer && trainer->hasPPISP()) {
                    ppisp = trainer->getPPISP();
                    controller_pool = use_controller && trainer->hasPPISPController()
                                          ? trainer->getPPISPControllerPool()
                                          : nullptr;
                    trainer_owned = true;
                }
            }
            if (ppisp == nullptr && scene_manager->hasAppearanceModel()) {
                ppisp = scene_manager->getAppearancePPISP();
                controller_pool = use_controller && scene_manager->hasAppearanceController()
                                      ? scene_manager->getAppearanceControllerPool()
                                      : nullptr;
            }
        }
        const bool apply_ppisp = settings.apply_appearance_correction && ppisp != nullptr;

        if (!apply_ppisp && !needs_env) {
            return image_u8_hwc_cpu;
        }

        PPISPOverrides overrides{};
        if (settings.ppisp_mode == RenderSettings::PPISPMode::MANUAL) {
            if (trainer_owned) {
                // The live viewport forwards only this subset to trainer-owned PPISP;
                // keep exports identical to what the user sees on screen.
                overrides.exposure_offset = settings.ppisp_overrides.exposure_offset;
                overrides.vignette_enabled = settings.ppisp_overrides.vignette_enabled;
                overrides.vignette_strength = settings.ppisp_overrides.vignette_strength;
                overrides.wb_temperature = settings.ppisp_overrides.wb_temperature;
                overrides.wb_tint = settings.ppisp_overrides.wb_tint;
                overrides.gamma_multiplier = settings.ppisp_overrides.gamma_multiplier;
            } else {
                overrides = settings.ppisp_overrides;
            }
        }

        std::shared_ptr<const lfs::rendering::CudaEnvironmentMap> env_map;
        lfs::rendering::EnvironmentCompositeBandParams env_params;
        if (needs_env) {
            auto loaded = lfs::rendering::getOrLoadCudaEnvironmentMap(settings.environment_map_path);
            if (!loaded) {
                return std::unexpected(std::format("export environment map failed: {}", loaded.error()));
            }
            env_map = std::move(*loaded);
            const auto [focal_x, focal_y] =
                lfs::rendering::computePixelFocalLengths({width, height}, view.focal_length_mm);
            env_params.camera_rotation = view.rotation;
            env_params.full_size = {width, height};
            env_params.focal_x = focal_x;
            env_params.focal_y = focal_y;
            env_params.center_x = static_cast<float>(width) * 0.5f;
            env_params.center_y = static_cast<float>(height) * 0.5f;
            env_params.equirectangular_view = view.equirectangular_view;
            env_params.exposure = settings.environment_exposure;
            env_params.rotation_degrees = settings.environment_rotation_degrees;
        }

        try {
            // Controller params come from one thumbnail prediction so every band shares
            // identical exposure/color (per-band prediction would cause visible seams and
            // allocate CNN intermediates at export resolution).
            lfs::core::Tensor controller_params;
            if (apply_ppisp && controller_pool != nullptr) {
                const bool is_training_camera = ppisp->is_known_frame(camera_uid);
                const int camera_idx =
                    is_training_camera ? ppisp->camera_index(ppisp->camera_for_frame(camera_uid)) : 0;
                const int controller_idx =
                    (camera_idx >= 0 && camera_idx < controller_pool->num_cameras()) ? camera_idx : 0;
                const auto thumbnail = makeExportThumbnailChw(image_u8_hwc_cpu, view.controller_predict_size);

                // The pool's forward buffers are shared with a live controller-training
                // step; the transaction lock plus the device syncs keep this prediction
                // (which resizes those buffers and clones from one) out of an in-flight
                // trainer predict/backward pair on another stream.
                std::lock_guard<std::mutex> controller_lock(controller_pool->predict_mutex());
                cudaDeviceSynchronize();
                controller_pool->allocate_buffers(thumbnail.size(1), thumbnail.size(2));
                controller_params = controller_pool->predict(controller_idx, thumbnail.unsqueeze(0), 1.0f).clone();
                cudaDeviceSynchronize();
            }

            lfs::core::Tensor output = image_u8_hwc_cpu;
            if (needs_env) {
                output = lfs::core::Tensor::empty(
                    {static_cast<size_t>(height), static_cast<size_t>(width), size_t{3}},
                    lfs::core::Device::CPU,
                    lfs::core::DataType::UInt8);
                if (!output.is_valid()) {
                    return std::unexpected("export post-process failed to allocate the output image");
                }
            }

            const int band_rows = std::clamp(kMaxExportBandPixels / std::max(width, 1), 1, height);
            for (int y0 = 0; y0 < height; y0 += band_rows) {
                const int band_height = std::min(band_rows, height - y0);
                const auto band_cuda = image_u8_hwc_cpu
                                           .slice(0, static_cast<size_t>(y0), static_cast<size_t>(y0 + band_height))
                                           .cuda();

                lfs::core::Tensor rgb_chw;
                lfs::core::Tensor alpha;
                if (auto unpacked = lfs::rendering::unpackU8HwcBandToChwFloat(
                        band_cuda, rgb_chw, expected_channels == 4 ? &alpha : nullptr);
                    !unpacked) {
                    return std::unexpected(std::format("export band unpack failed: {}", unpacked.error()));
                }

                lfs::core::Tensor corrected = rgb_chw;
                if (apply_ppisp) {
                    const lfs::training::PPISPRegion region{.y_offset = y0, .full_height = height};
                    corrected = applyPpispAppearance(*ppisp, controller_pool, rgb_chw, camera_uid, overrides,
                                                     region, controller_params);
                    if (!corrected.is_valid()) {
                        return std::unexpected("export PPISP correction produced no image");
                    }
                }

                lfs::core::Tensor band_u8;
                if (needs_env) {
                    env_params.y_offset = y0;
                    if (auto composited = lfs::rendering::compositeEnvironmentBand(
                            *env_map, env_params, corrected, alpha, band_u8);
                        !composited) {
                        return std::unexpected(
                            std::format("export environment composite failed: {}", composited.error()));
                    }
                } else {
                    if (auto packed = lfs::rendering::packChwFloatBandToU8Hwc(
                            corrected, mode == ExportPostProcessMode::Transparent ? &alpha : nullptr, band_u8);
                        !packed) {
                        return std::unexpected(std::format("export band pack failed: {}", packed.error()));
                    }
                }

                output.slice(0, static_cast<size_t>(y0), static_cast<size_t>(y0 + band_height)).copy_from(band_u8);
            }

            lfs::core::Tensor::trim_memory_pool();
            return output;
        } catch (const std::exception& e) {
            lfs::core::Tensor::trim_memory_pool();
            return std::unexpected(std::format("export post-process failed: {}", e.what()));
        }
    }

} // namespace lfs::vis
