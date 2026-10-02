/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "lfs/training/ops/session.hpp"

namespace lfs::training {
    const lfs::gpu_ops::SessionOps& vulkan_session_ops();
}
