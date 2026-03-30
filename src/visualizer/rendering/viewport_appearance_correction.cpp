/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_appearance_correction.hpp"
#include "scene/scene_manager.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"

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

            const bool was_hwc = (rgb.ndim() == 3 && rgb.shape()[2] == 3);
            const auto input = was_hwc ? rgb.permute({2, 0, 1}).contiguous() : rgb;
            const bool is_training_camera = ppisp->is_known_frame(camera_uid);
            const bool has_controller = use_controller && scene_mgr.hasAppearanceController();
            const int camera_idx = is_training_camera ? ppisp->camera_index(ppisp->camera_for_frame(camera_uid)) : 0;

            lfs::core::Tensor result;

            if (has_controller) {
                auto* pool = scene_mgr.getAppearanceControllerPool();
                const int controller_idx =
                    (camera_idx >= 0 && camera_idx < pool->num_cameras()) ? camera_idx : 0;
                const auto params = pool->predict(controller_idx, input.unsqueeze(0), 1.0f);
                result = overrides.isIdentity()
                             ? ppisp->apply_with_controller_params(input, params, controller_idx)
                             : ppisp->apply_with_controller_params_and_overrides(
                                   input, params, controller_idx, toRenderOverrides(overrides));
            } else if (is_training_camera) {
                const int camera_id = ppisp->camera_for_frame(camera_uid);
                result = overrides.isIdentity()
                             ? ppisp->apply(input, camera_id, camera_uid)
                             : ppisp->apply_with_overrides(
                                   input, camera_id, camera_uid, toRenderOverrides(overrides));
            } else {
                // Keep manual overrides active on novel views by anchoring them to any valid learned PPISP
                // frame/camera pair when no controller is available.
                const int fallback_camera = ppisp->any_camera_id();
                const int fallback_frame = ppisp->any_frame_uid();
                result = overrides.isIdentity()
                             ? ppisp->apply(input, fallback_camera, fallback_frame)
                             : ppisp->apply_with_overrides(
                                   input, fallback_camera, fallback_frame, toRenderOverrides(overrides));
            }

            return (was_hwc && result.is_valid()) ? result.permute({1, 2, 0}).contiguous() : result;
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
                auto corrected = trainer->applyPPISPForViewport(
                    *image, camera_uid, trainer_overrides, use_controller);
                return std::make_shared<lfs::core::Tensor>(std::move(corrected));
            }
        }

        if (!scene_manager->hasAppearanceModel()) {
            return image;
        }

        const auto& overrides = (settings.ppisp_mode == RenderSettings::PPISPMode::MANUAL)
                                    ? settings.ppisp_overrides
                                    : PPISPOverrides{};
        const bool use_controller = (settings.ppisp_mode == RenderSettings::PPISPMode::AUTO);
        auto corrected = applyStandaloneAppearance(*image, *scene_manager, camera_uid, overrides, use_controller);
        if (!corrected.is_valid()) {
            return image;
        }

        return std::make_shared<lfs::core::Tensor>(std::move(corrected));
    }

} // namespace lfs::vis
