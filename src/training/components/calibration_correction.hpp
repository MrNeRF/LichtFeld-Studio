// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "core/camera_types.h"
#include "core/tensor.hpp"

#include <functional>

namespace lfs::training {
    namespace kernels {
        struct DecoupledRawGradient;
    }

    class CalibrationCorrection {
    public:
        static constexpr int grid = 9;
        explicit CalibrationCorrection(int slots);
        core::Tensor& field() { return field_; }
        const core::Tensor& field() const { return field_; }

        // Pixels per radian of the photograph: its focal lengths, or for an equirectangular panorama its angular
        // rate (width / 2 pi per radian of longitude, height / pi per radian of latitude), which also continues
        // across the left and right edge.
        struct Lens {
            float fx, fy;
            bool wrap;
        };
        static Lens lens(core::CameraModelType model, float fx, float fy, int width, int height);

        // A photograph with alpha: foreground and alpha are warped, then composited over `background`, a [3] colour or
        // a [3, H, W] image read at the destination pixel, so the background is not shifted with the photograph.
        // `stable`, when given, receives the full photograph composited the same way over the [3] colour
        // `stable_background`.
        struct Composite {
            const core::Tensor *alpha, *background, *stable_background;
            core::Tensor* stable;
        };

        // The source maps cover the full photograph. Outputs cover the render tile.
        core::Tensor warp(int slot, float fx, float fy, int x, int y, const core::Tensor& photo,
                          const core::Tensor& render, core::Tensor& mask, core::Tensor& structure,
                          core::Tensor& edge, bool first_tile = true, const Composite* composite = nullptr,
                          bool wrap = false);
        // `gradient` is the loss gradient with respect to the render tile. With the decoupled appearance loss, `raw`
        // holds its contrast and structure part, taken with respect to the raw render. `margin` is the loss's valid
        // padding: pixels that close to the tile's edge have no loss term of their own and add no curvature.
        // `scratch`, when given, holds the summed decoupled gradient for this call instead of a kept buffer.
        void accumulate(const core::Tensor& render, const core::Tensor& gradient, const core::Tensor& weight,
                        float values, int margin = 0, const kernels::DecoupledRawGradient* raw = nullptr,
                        const std::function<char*(size_t)>& scratch = {});
        // Clears a gradient of the last warped tile where the sample falls outside the photograph: the target there
        // is the render itself, and no loss term may push it.
        void clear_outside(core::Tensor& gradient) const;
        // A depth ([H, W]) or unit normal ([3, H, W]) prior of the last warped photograph, at the prior's own
        // resolution, read through the field: bilinear over the valid taps, renormalised, and invalid (zero) where
        // no tap is valid or the sample falls outside.
        core::Tensor follow(const core::Tensor& prior, bool normal) const;
        void update();
        // A report before the start step would only show the zero field.
        static bool report_due(int iteration, int total_iterations, int start_iteration);
        void report(int iteration) const;

        // The returned image and displacement grid are in the photograph's scoring coordinates. The fit uses the
        // training grid and solver with full steps, so it corrects what training can correct, not more.
        static core::Tensor align_for_scoring(const core::Tensor& render, const core::Tensor& photo,
                                              const core::Tensor& mask, core::Tensor& displacement, core::Tensor* raw = nullptr);

        struct Tile {
            int width, height, x, y, w, h, side;
            float sx, sy;
            bool wrap = false;
        };

    private:
        // The share of each solved step an update takes: half in training, the full step in the scoring alignment, which
        // runs a fixed number of iterations.
        static constexpr float training_step = .5f, scoring_step = 1.f;
        CalibrationCorrection(int slots, float step);
        core::Tensor sample_image(int slot, float sx, float sy, int x, int y, const core::Tensor& source,
                                  const core::Tensor& fallback, core::Tensor& mask, core::Tensor& structure,
                                  core::Tensor& edge, bool first_tile, const Composite* composite = nullptr,
                                  bool wrap = false);
        Tile tile_{};
        int slot_ = 0;
        bool first_tile_ = true;
        float step_;
        core::Tensor field_, image_, mask_, structure_, edge_, partials_, gradient_;
    };
} // namespace lfs::training
