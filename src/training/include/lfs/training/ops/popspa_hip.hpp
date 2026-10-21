/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/popspa.hpp"

namespace lfs::training {
    const gpu_ops::PopspaOps& hip_popspa_ops();
}
