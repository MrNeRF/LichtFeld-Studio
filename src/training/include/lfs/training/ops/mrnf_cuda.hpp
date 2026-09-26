/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/mrnf.hpp"

namespace lfs::training {
    const lfs::gpu_ops::MrnfOps& cuda_mrnf_ops();
} // namespace lfs::training
