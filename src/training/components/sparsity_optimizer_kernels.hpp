/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>

namespace lfs::training {
    void launch_admm_backward_fused(float* grad_opacities, const float* opa_sigmoid,
                                    const float* z, const float* u, float rho,
                                    float grad_loss, size_t n, bool accumulate);
} // namespace lfs::training
