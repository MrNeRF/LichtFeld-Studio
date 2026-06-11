/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Reference CPU rasterizer (EWA-style perspective projection of 3D Gaussians,
// depth-sorted front-to-back alpha blending, tiled multithreaded). Not a
// performance path — it exists to validate the LOD scheme: PSNR of LOD cuts
// against the full model, measured contributions-per-pixel (the overdraw the
// quality knob is supposed to control), and temporal popping metrics.
#pragma once

#include "camera.hpp"
#include "splat.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lfs::lod {

    struct RasterInput {
        const SplatCloud* cloud;
        uint32_t offset, count;
        float weight; // cross-fade multiplier on alpha
    };

    struct Image {
        int w = 0, h = 0;
        std::vector<float> rgb; // 3 floats per pixel, row-major

        void init(int width, int height) {
            w = width;
            h = height;
            rgb.assign((size_t)w * h * 3, 0.f);
        }
    };

    struct RasterStats {
        double total_ms = 0, sort_ms = 0, bin_ms = 0, blend_ms = 0;
        uint64_t splats_in = 0, splats_projected = 0;
        double mean_contrib_per_px = 0; // alpha-blend contributions per pixel (overdraw)
    };

    void rasterize(const Camera& cam, const std::vector<RasterInput>& inputs, Image& out,
                   RasterStats& stats, unsigned threads = 0);

    double psnr(const Image& a, const Image& b);
    // Mean absolute per-pixel difference (popping metric between consecutive frames).
    double meanAbsDiff(const Image& a, const Image& b);
    bool writePpm(const std::string& path, const Image& img);

} // namespace lfs::lod
