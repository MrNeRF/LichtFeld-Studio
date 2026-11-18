// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "core/tensor.hpp"

namespace lfs::training {
    class CalibrationCorrection {
    public:
        static constexpr int grid = 9;
        explicit CalibrationCorrection(int slots);
        core::Tensor& field() { return field_; }
        const core::Tensor& field() const { return field_; }

        // Blob seeds use the same geometry with a fixed background behind source alpha.
        struct Composite {
            const core::Tensor *foreground, *alpha, *background;
            core::Tensor* output;
        };

        // The source maps cover the full photograph. Outputs cover the render tile.
        core::Tensor warp(int slot, float fx, float fy, int x, int y, const core::Tensor& photo,
                          const core::Tensor& render, core::Tensor& mask, core::Tensor& structure,
                          core::Tensor& edge, bool first_tile = true, const Composite* stable = nullptr);
        // Clears the loss gradient at pixels that sample outside the photograph.
        void accumulate(const core::Tensor& render, core::Tensor& gradient,
                        const core::Tensor& weight, float values);
        void update();
        void report(int iteration) const;

        // The returned image and displacement grid are in the photograph's scoring coordinates.
        static core::Tensor align_for_scoring(const core::Tensor& render, const core::Tensor& photo,
                                              const core::Tensor& mask, core::Tensor& displacement, core::Tensor* raw = nullptr);

        struct Tile {
            int width, height, x, y, w, h, side;
            float sx, sy;
        };

    private:
        static constexpr int scoring_grid = 16;
        CalibrationCorrection(int slots, int side);
        core::Tensor sample_image(int slot, float sx, float sy, int x, int y, const core::Tensor& source,
                                  const core::Tensor& fallback, core::Tensor& mask, core::Tensor& structure,
                                  core::Tensor& edge, bool first_tile, const Composite* stable = nullptr);
        Tile tile_{};
        int slot_ = 0;
        bool first_tile_ = true;
        core::Tensor field_, image_, mask_, structure_, edge_, partials_;
    };
} // namespace lfs::training
