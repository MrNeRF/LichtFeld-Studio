/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace lfs::core {
    class Camera;
}

namespace lfs::training {
    // Canonicalize the scoring order and fingerprint that exact order for resume.
    // Training cameras must have unique UIDs and loaded image dimensions. `identity_uids` replaces the uid of the
    // cameras it holds in the fingerprint, not in the order.
    [[nodiscard]] uint64_t sort_and_fingerprint_popspa_cameras(std::vector<core::Camera*>& cameras,
                                                               const std::unordered_map<int, int>* identity_uids = nullptr);
} // namespace lfs::training
