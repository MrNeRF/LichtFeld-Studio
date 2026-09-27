/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/shared_image_ops.hpp"

namespace lfs::core {
    const gpu_ops::SharedImageOps& cuda_shared_image_ops();
}
