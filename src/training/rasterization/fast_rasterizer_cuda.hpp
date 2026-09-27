/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "fast_rasterizer.hpp"

#include <rasterization_api.h>

namespace lfs::training {

    [[nodiscard]] fast_lfs::rasterization::FusedAdamSettings fast_adam_settings(
        const lfs::gpu_ops::BackwardAdam& adam);

} // namespace lfs::training
