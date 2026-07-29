/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cube_face_projection.hpp"
#include "core/export.hpp"

#include <cstdint>
#include <vector>

namespace lfs::io {

    [[nodiscard]] LFS_IO_API int projected_face_output_size(
        const lfs::core::CubeFaceProjection& projection,
        int resize_factor,
        int max_width);

    [[nodiscard]] LFS_IO_API std::vector<uint8_t> project_cube_face_hwc(
        const uint8_t* source,
        int width,
        int height,
        int channels,
        const lfs::core::CubeFaceProjection& projection,
        int output_size);

} // namespace lfs::io
