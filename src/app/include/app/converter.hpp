/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

namespace lfs::io::args {
    struct VoxelizeMode;
}

namespace lfs::app {

    int run_voxelizer(const lfs::io::args::VoxelizeMode& options);

    int run_converter(const lfs::core::param::ConvertParameters& params);
    int run_mesh2splat(const lfs::core::param::Mesh2SplatParameters& params);

} // namespace lfs::app
