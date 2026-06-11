/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Synthetic "city block" scene: a ground plane plus a grid of boxes whose
// surfaces are sampled with anisotropic, surface-aligned splats (thin along
// the normal). Walls viewed edge-on exercise the anisotropy path of the LOD
// metric; the regular street grid gives a natural flythrough route.
#pragma once

#include "splat.hpp"

#include <cstdint>

namespace lfs::lod {

    struct SyntheticSceneParams {
        size_t target_splats = 3'000'000;
        int grid = 14;               // grid x grid buildings
        float block_size = 18.f;     // building footprint + street, world units
        float building_fill = 0.55f; // fraction of block occupied by building
        float min_height = 8.f;
        float max_height = 55.f;
        uint64_t seed = 42;
    };

    // Generates splats; scene extent is roughly grid*block_size square, ground at y=0.
    void generateCityScene(const SyntheticSceneParams& p, SplatCloud& out);

} // namespace lfs::lod
