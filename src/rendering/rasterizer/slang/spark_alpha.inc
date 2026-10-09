// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef LFS_SPARK_ALPHA_INC
#define LFS_SPARK_ALPHA_INC

float lfsSparkAlphaFromDensity(float value, float density) {
    // Avoid rounding 1-value to one before exponentiation. Weak Gaussian
    // coverage can still produce visible alpha after the density transform.
    if (value < .01f) {
        // -log1p(-value), through fourth order on [0,.01). The error after
        // the density transform is below FP32 rounding at this boundary.
        const float optical_depth = density * value *
                                    (1.f + value * (.5f + value * (1.f / 3.f + value * .25f)));
        // -expm1(-optical_depth), avoiding the second subtraction near zero.
        // Six terms keep this branch below FP32 rounding through 1/8.
        return optical_depth < .125f
                   ? optical_depth * (1.f + optical_depth * (-.5f + optical_depth * (1.f / 6.f + optical_depth * (-1.f / 24.f + optical_depth * (1.f / 120.f - optical_depth / 720.f)))))
                   : 1.f - exp(-optical_depth);
    }
    return 1.f - pow(max(0.f, 1.f - value), density);
}

#endif
