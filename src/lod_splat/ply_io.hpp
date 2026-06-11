/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Binary-little-endian 3DGS PLY load/save, property-compatible with
// LichtFeld's src/io/formats/ply.cpp (x,y,z, f_dc_0..2, scale_0..2,
// rot_0..3, opacity; f_rest_* are skipped on load — this module keeps
// SH degree 0 only, which is also what its coarse LODs ship).
#pragma once

#include "splat.hpp"

#include <string>

namespace lfs::lod {

    // Returns true on success. Skips f_rest_* and nx/ny/nz properties.
    bool loadPly(const std::string& path, SplatCloud& out, std::string* err = nullptr);
    bool savePly(const std::string& path, const SplatCloud& cloud, std::string* err = nullptr);

} // namespace lfs::lod
