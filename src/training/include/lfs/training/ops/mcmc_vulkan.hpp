/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/mcmc.hpp"

namespace lfs::training {
    [[nodiscard]] const gpu_ops::McmcOps& vulkan_mcmc_ops();
}
