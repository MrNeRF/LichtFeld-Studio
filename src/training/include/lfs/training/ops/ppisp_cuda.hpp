/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "lfs/training/ops/ppisp.hpp"
namespace lfs::training {
    const gpu_ops::PPISPOps& cuda_ppisp_ops();
    const gpu_ops::ControllerOps& cuda_controller_ops();
} // namespace lfs::training
