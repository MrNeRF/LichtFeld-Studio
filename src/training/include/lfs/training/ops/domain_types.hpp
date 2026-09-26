/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Relocated unchanged from the fast rasterizer config. Values are the ones
// the fused backward kernels already switch on.
enum class DensificationType : int {
    None = 0,
    MCMC = 1,
    MRNF = 2
};
