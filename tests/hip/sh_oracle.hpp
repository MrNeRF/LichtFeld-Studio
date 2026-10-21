/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <array>
#include <cmath>
namespace hip_sh_oracle {
    inline std::array<double, 16> basis(double x, double y, double z) {
        const double inverse = 1 / std::sqrt(x * x + y * y + z * z);
        x *= inverse;
        y *= inverse;
        z *= inverse;
        return {0.28209479177387814, -0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x,
                1.0925484305920792 * x * y, -1.0925484305920792 * y * z, 0.31539156525252005 * (3 * z * z - 1),
                -1.0925484305920792 * x * z, 0.5462742152960396 * (x * x - y * y),
                -0.5900435899266435 * y * (3 * x * x - y * y), 2.890611442640554 * x * y * z,
                -0.4570457994644658 * y * (5 * z * z - 1), 0.3731763325901154 * z * (5 * z * z - 3),
                -0.4570457994644658 * x * (5 * z * z - 1), 1.445305721320277 * z * (x * x - y * y),
                -0.5900435899266435 * x * (x * x - 3 * y * y)};
    }
} // namespace hip_sh_oracle
