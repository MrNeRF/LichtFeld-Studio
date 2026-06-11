/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Splat storage with the same field semantics as lfs::core::SplatData
// (src/core/include/core/splat_data.hpp): positions linear, scales log-space,
// quaternions unnormalized (w,x,y,z), opacity logit-space, sh0 = unnormalized
// DC coefficients (color = 0.5 + SH_C0 * sh0). SoA layout, CPU resident.
#pragma once

#include "lod_math.hpp"

#include <cstdint>
#include <vector>

namespace lfs::lod {

    inline constexpr float SH_C0 = 0.28209479177387814f;

    struct SplatCloud {
        std::vector<Vec3> means;
        std::vector<Vec3> log_scales;
        std::vector<Quat> rotations;    // unnormalized, (w,x,y,z)
        std::vector<float> opacity_raw; // logit by default; linear if lod_opacity_encoded
        std::vector<Vec3> sh0;          // DC coefficients (not yet color)
        // LichtFeld/Spark "lodOpacity" encoding: opacity_raw holds display-space
        // alpha directly and may exceed 1.0, so merged splats can conserve the
        // cluster's integrated alpha exactly instead of clamping at 1.
        bool lod_opacity_encoded = false;

        size_t size() const { return means.size(); }

        void resize(size_t n) {
            means.resize(n);
            log_scales.resize(n);
            rotations.resize(n);
            opacity_raw.resize(n);
            sh0.resize(n);
        }
        void reserve(size_t n) {
            means.reserve(n);
            log_scales.reserve(n);
            rotations.reserve(n);
            opacity_raw.reserve(n);
            sh0.reserve(n);
        }
        void push(const Vec3& mean, const Vec3& log_scale, const Quat& rot, float logit_op, const Vec3& dc) {
            means.push_back(mean);
            log_scales.push_back(log_scale);
            rotations.push_back(rot);
            opacity_raw.push_back(logit_op);
            sh0.push_back(dc);
        }

        Vec3 linearScale(size_t i) const {
            const Vec3& ls = log_scales[i];
            return {std::exp(ls.x), std::exp(ls.y), std::exp(ls.z)};
        }
        float opacity(size_t i) const {
            return lod_opacity_encoded ? opacity_raw[i] : sigmoid(opacity_raw[i]);
        }
        Vec3 color(size_t i) const {
            return {0.5f + SH_C0 * sh0[i].x, 0.5f + SH_C0 * sh0[i].y, 0.5f + SH_C0 * sh0[i].z};
        }
        // Mean projected cross-section area of the anisotropic Gaussian, used as
        // the weight base for moment matching and for opacity conservation.
        float meanArea(size_t i) const {
            const Vec3 s = linearScale(i);
            return (s.x * s.y + s.y * s.z + s.z * s.x) * (3.14159265f / 3.f);
        }
    };

    struct Aabb {
        Vec3 mn{1e30f, 1e30f, 1e30f};
        Vec3 mx{-1e30f, -1e30f, -1e30f};

        void grow(const Vec3& p) {
            mn = mn.cwiseMin(p);
            mx = mx.cwiseMax(p);
        }
        void grow(const Aabb& o) {
            mn = mn.cwiseMin(o.mn);
            mx = mx.cwiseMax(o.mx);
        }
        Vec3 center() const { return (mn + mx) * 0.5f; }
        Vec3 extent() const { return mx - mn; }
        // Distance from point to box (0 if inside).
        float distance(const Vec3& p) const {
            const float dx = std::max({mn.x - p.x, 0.f, p.x - mx.x});
            const float dy = std::max({mn.y - p.y, 0.f, p.y - mx.y});
            const float dz = std::max({mn.z - p.z, 0.f, p.z - mx.z});
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    };

} // namespace lfs::lod
