/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <vector>

namespace lfs::core {
    class Camera;
}

namespace lfs::training {
    // Canonicalize the scoring order and fingerprint that exact order for resume.
    // Training cameras must have unique UIDs and loaded image dimensions.
    [[nodiscard]] uint64_t sort_and_fingerprint_popspa_cameras(std::vector<core::Camera*>& cameras);
} // namespace lfs::training
