/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/parameters.hpp"

namespace lfs::preprocessing {
    int run_video_masking(const core::param::VideoMaskParameters& params);
} // namespace lfs::preprocessing
