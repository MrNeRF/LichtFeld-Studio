#pragma once
#include <optional>
namespace lfs::training {
    struct PPISPViewportOverrides {
        float exposure_offset = 0.0f;

        bool vignette_enabled = true;
        float vignette_strength = 1.0f;

        float wb_temperature = 0.0f;
        float wb_tint = 0.0f;
        float color_red_x = 0.0f;
        float color_red_y = 0.0f;
        float color_green_x = 0.0f;
        float color_green_y = 0.0f;
        float color_blue_x = 0.0f;
        float color_blue_y = 0.0f;

        float gamma_multiplier = 1.0f;
        float gamma_red = 0.0f;
        float gamma_green = 0.0f;
        float gamma_blue = 0.0f;
        float crf_toe = 0.0f;
        float crf_shoulder = 0.0f;

        [[nodiscard]] bool operator==(const PPISPViewportOverrides&) const = default;

        [[nodiscard]] bool isIdentity() const {
            return exposure_offset == 0.0f && vignette_enabled && vignette_strength == 1.0f &&
                   wb_temperature == 0.0f && wb_tint == 0.0f && color_red_x == 0.0f && color_red_y == 0.0f &&
                   color_green_x == 0.0f && color_green_y == 0.0f && color_blue_x == 0.0f && color_blue_y == 0.0f &&
                   gamma_multiplier == 1.0f && gamma_red == 0.0f && gamma_green == 0.0f && gamma_blue == 0.0f &&
                   crf_toe == 0.0f && crf_shoulder == 0.0f;
        }
    };

    struct CameraMetricsAppearanceConfig {
        bool enabled = false;
        PPISPViewportOverrides overrides{};
        bool use_controller = true;
    };

    struct CameraMetricsSnapshot {
        float psnr = 0.0f;
        std::optional<float> ssim;
        bool used_mask = false;
    };

} // namespace lfs::training
