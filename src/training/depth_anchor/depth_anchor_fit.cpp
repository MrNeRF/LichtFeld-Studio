/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/geometry_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lfs::training::kernels {
    namespace {

        constexpr size_t kRansacScoreSubset = 16384;
        constexpr int kRansacIterations = 256;
        constexpr float kMinAnchorCorr = 0.35f;
        constexpr float kMinAnchorInlierFraction = 0.3f;

        struct RobustFit {
            bool usable = false;
            double a = 0.0;
            double b = 0.0;
            double corr = 0.0;
            size_t inliers = 0;
        };

        // RANSAC affine fit y = a*t + b over (t, y) pairs, least-squares refit
        // on the consensus set. Survives the heavily contaminated sparse
        // clouds that break a plain least-squares + trim.
        [[nodiscard]] RobustFit ransac_affine_fit(
            const std::vector<float>& ts,
            const std::vector<float>& ys,
            const uint32_t seed) {
            RobustFit out;
            const size_t n = ts.size();
            if (n < static_cast<size_t>(kMinAnchorSamples)) {
                return out;
            }

            const auto ls_fit = [&](const auto& accept) {
                double st = 0.0, sy = 0.0, stt = 0.0, syy = 0.0, sty = 0.0;
                size_t m = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (!accept(i)) {
                        continue;
                    }
                    const double t = ts[i];
                    const double y = ys[i];
                    st += t;
                    sy += y;
                    stt += t * t;
                    syy += y * y;
                    sty += t * y;
                    ++m;
                }
                RobustFit fit;
                if (m < static_cast<size_t>(kMinAnchorSamples)) {
                    return fit;
                }
                const double inv_m = 1.0 / static_cast<double>(m);
                const double mean_t = st * inv_m;
                const double mean_y = sy * inv_m;
                const double var_t = std::max(stt * inv_m - mean_t * mean_t, 0.0);
                const double var_y = std::max(syy * inv_m - mean_y * mean_y, 0.0);
                const double cov = sty * inv_m - mean_t * mean_y;
                fit.a = cov / (var_t + kDepthLossTargetVarRidge);
                fit.b = mean_y - fit.a * mean_t;
                if (var_t > kDepthLossMinVariance && var_y > kDepthLossMinVariance) {
                    fit.corr = cov / std::sqrt(var_t * var_y);
                }
                fit.inliers = m;
                fit.usable = true;
                return fit;
            };

            const RobustFit initial = ls_fit([](size_t) { return true; });
            if (!initial.usable) {
                return out;
            }

            // Robust residual scale from the median absolute deviation.
            std::vector<float> abs_res(n);
            for (size_t i = 0; i < n; ++i) {
                abs_res[i] = std::fabs(static_cast<float>(ys[i] - (initial.a * ts[i] + initial.b)));
            }
            std::nth_element(abs_res.begin(), abs_res.begin() + n / 2, abs_res.end());
            const double mad_sigma = 1.4826 * abs_res[n / 2];
            const double eps = std::max(3.0 * mad_sigma, 1.0e-6 + 1.0e-4 * std::fabs(initial.b));

            const size_t score_stride = std::max<size_t>(1, n / kRansacScoreSubset);
            uint32_t state = seed | 1u;
            const auto next_rand = [&state]() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                return state;
            };

            double best_a = initial.a;
            double best_b = initial.b;
            size_t best_score = 0;
            for (size_t i = 0; i < n; i += score_stride) {
                if (std::fabs(ys[i] - (initial.a * ts[i] + initial.b)) <= eps) {
                    ++best_score;
                }
            }

            for (int it = 0; it < kRansacIterations; ++it) {
                const size_t i0 = next_rand() % n;
                const size_t i1 = next_rand() % n;
                const double dt = static_cast<double>(ts[i1]) - ts[i0];
                if (std::fabs(dt) < 1.0e-12) {
                    continue;
                }
                const double a = (static_cast<double>(ys[i1]) - ys[i0]) / dt;
                const double b = ys[i0] - a * ts[i0];
                size_t score = 0;
                for (size_t i = 0; i < n; i += score_stride) {
                    if (std::fabs(ys[i] - (a * ts[i] + b)) <= eps) {
                        ++score;
                    }
                }
                if (score > best_score) {
                    best_score = score;
                    best_a = a;
                    best_b = b;
                }
            }

            const double consensus_a = best_a;
            const double consensus_b = best_b;
            RobustFit refined = ls_fit([&](size_t i) {
                return std::fabs(ys[i] - (consensus_a * ts[i] + consensus_b)) <= eps;
            });
            if (!refined.usable) {
                return out;
            }
            const double refined_a = refined.a;
            const double refined_b = refined.b;
            refined = ls_fit([&](size_t i) {
                return std::fabs(ys[i] - (refined_a * ts[i] + refined_b)) <= eps;
            });
            return refined.usable ? refined : out;
        }

    } // namespace

    DepthAnchor fit_depth_anchor_from_samples(const std::vector<lfs::gpu_ops::AnchorSample>& pairs) {
        DepthAnchor anchor;
        const size_t n = pairs.size();
        if (n < static_cast<size_t>(kMinAnchorSamples)) {
            return anchor;
        }
        std::vector<float> ts(n);
        std::vector<float> zs(n);
        for (size_t i = 0; i < n; ++i) {
            ts[i] = pairs[i].x;
            zs[i] = pairs[i].y;
        }

        // Robust floor from the median depth: contaminated sparse clouds have
        // extreme outliers that would wreck a mean-based floor.
        std::vector<float> z_sorted = zs;
        std::nth_element(z_sorted.begin(), z_sorted.begin() + n / 2, z_sorted.end());
        const float median_z = z_sorted[n / 2];
        const float floor_f = std::max(1.0e-8f, kDepthLossFloorFraction * median_z);

        std::vector<float> qs(n);
        for (size_t i = 0; i < n; ++i) {
            qs[i] = 1.0f / (zs[i] + floor_f);
        }

        double mean_t = 0.0;
        for (const float t : ts) {
            mean_t += t;
        }
        mean_t /= static_cast<double>(n);
        double var_t = 0.0;
        for (const float t : ts) {
            var_t += (t - mean_t) * (t - mean_t);
        }
        var_t /= static_cast<double>(n);

        if (var_t <= kDepthLossFlatPriorVar) {
            return anchor;
        }

        const uint32_t seed = 0x9e3779b9u ^ static_cast<uint32_t>(n);
        const RobustFit fit_q = ransac_affine_fit(ts, qs, seed);
        const RobustFit fit_z = ransac_affine_fit(ts, zs, seed * 2654435761u);

        const auto resolvable = [&](const RobustFit& fit) {
            return fit.usable &&
                   fit.inliers >= static_cast<size_t>(kMinAnchorInlierFraction * n) &&
                   std::fabs(fit.corr) >= kMinAnchorCorr;
        };
        const bool q_ok = resolvable(fit_q);
        const bool z_ok = resolvable(fit_z);
        if (q_ok) {
            anchor.disparity.valid = true;
            anchor.disparity.scale = static_cast<float>(fit_q.a);
            anchor.disparity.shift = static_cast<float>(fit_q.b);
            anchor.disparity.corr = static_cast<float>(fit_q.corr);
            anchor.disparity.samples = static_cast<int>(fit_q.inliers);
        }
        if (z_ok) {
            anchor.depth.valid = true;
            anchor.depth.scale = static_cast<float>(fit_z.a);
            anchor.depth.shift = static_cast<float>(fit_z.b);
            anchor.depth.corr = static_cast<float>(fit_z.corr);
            anchor.depth.samples = static_cast<int>(fit_z.inliers);
        }

        if (q_ok || z_ok) {
            const bool use_q = q_ok && (!z_ok || std::fabs(fit_q.corr) >= std::fabs(fit_z.corr));
            const DepthAnchorCandidate& fit = use_q ? anchor.disparity : anchor.depth;
            anchor.valid = true;
            anchor.model = use_q ? 0 : 1;
            anchor.scale = fit.scale;
            anchor.shift = fit.shift;
            anchor.floor = floor_f;
            anchor.corr = fit.corr;
            anchor.samples = fit.samples;
            return anchor;
        }

        return anchor;
    }

} // namespace lfs::training::kernels
