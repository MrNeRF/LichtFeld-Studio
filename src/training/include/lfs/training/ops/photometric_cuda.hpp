/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/photometric_services.hpp"
namespace lfs::training {
    const lfs::gpu_ops::PhotometricOps& cuda_photometric_ops();
} // namespace lfs::training
