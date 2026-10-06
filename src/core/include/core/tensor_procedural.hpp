/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
#include <array>
#include <span>

namespace lfs::core {
    // Float32 fields on one device: positions [N,3], scalar parameters [N].
    // Integer-hashed gradient noise, fractional octaves in [0,15], normalized
    // to [0,1]. Detail zero returns zero. CPU and fused GPU use identical arithmetic.
    LFS_CORE_API Tensor procedural_noise(const Tensor& positions, const Tensor& scale, const Tensor& detail,
                                         const Tensor& roughness, const Tensor& distortion, const Tensor& seed);

    enum class GradientType { Linear,
                              Quadratic,
                              Easing,
                              Diagonal,
                              Spherical,
                              Radial };
    LFS_CORE_API Tensor procedural_gradient(const Tensor& positions, GradientType type);

    // Host control data, not element data. Curves are sorted, distinct [x,y,tangent]
    // knots; ramps are sorted [position,r,g,b,a] stops (duplicates allowed).
    using CurveKnot = std::array<float, 3>;
    using ColourStop = std::array<float, 5>;
    enum class RampInterpolation { Constant,
                                   Linear,
                                   Ease };
    LFS_CORE_API Tensor interpolate_curve(const Tensor& values, std::span<const CurveKnot> knots, bool clamp);
    LFS_CORE_API Tensor interpolate_colour_ramp(const Tensor& values, std::span<const ColourStop> stops,
                                                RampInterpolation interpolation, bool alpha);
    // Apply combined then per-channel curves, selection * clamped factor blending.
    // The sh_dc flag interprets/returns SH DC coefficients rather than RGB.
    LFS_CORE_API Tensor interpolate_rgb_curves(const Tensor& colours, const Tensor& selection, const Tensor& factor,
                                               const std::array<std::span<const CurveKnot>, 4>& curves, bool sh_dc);
} // namespace lfs::core
