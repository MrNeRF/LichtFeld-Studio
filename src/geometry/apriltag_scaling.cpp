/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "geometry/apriltag_scaling.hpp"
#include "core/utc_time.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <map>
#include <nlohmann/json.hpp>
#include <numbers>
#include <numeric>
#include <random>
#include <set>

namespace lfs::geometry::apriltag {

    namespace {

        constexpr double kEpsilon = 1e-9;

        [[nodiscard]] bool isFiniteVec2(const glm::dvec2& v) noexcept {
            return std::isfinite(v.x) && std::isfinite(v.y);
        }

        [[nodiscard]] bool isFiniteVec3(const glm::dvec3& v) noexcept {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }

        void applyPinholeDistortion(
            const double x,
            const double y,
            const std::span<const double> dist,
            double& dx,
            double& dy) noexcept {
            const double r2 = x * x + y * y;
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;

            const double k1 = dist.size() > 0 ? dist[0] : 0.0;
            const double k2 = dist.size() > 1 ? dist[1] : 0.0;
            const double k3 = dist.size() > 2 ? dist[2] : 0.0;
            const double numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            double radial = numerator;
            if (dist.size() >= 6) {
                const double denominator = 1.0 + dist[3] * r2 + dist[4] * r4 + dist[5] * r6;
                if (std::abs(denominator) > kEpsilon) {
                    radial = numerator / denominator;
                }
            }

            const std::size_t tangential_offset = dist.size() >= 6 ? 6 : 3;
            const double p1 = dist.size() > tangential_offset ? dist[tangential_offset] : 0.0;
            const double p2 = dist.size() > tangential_offset + 1 ? dist[tangential_offset + 1] : 0.0;

            dx = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            dy = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        }

        void applyFisheyeDistortion(
            const double x,
            const double y,
            const std::span<const double> dist,
            double& dx,
            double& dy) noexcept {
            const double r = std::sqrt(x * x + y * y);
            if (r < 1e-8) {
                dx = x;
                dy = y;
                return;
            }

            const double theta = std::atan(r);
            const double theta2 = theta * theta;
            const double theta4 = theta2 * theta2;
            const double theta6 = theta4 * theta2;
            const double theta8 = theta4 * theta4;

            const double k1 = dist.size() > 0 ? dist[0] : 0.0;
            const double k2 = dist.size() > 1 ? dist[1] : 0.0;
            const double k3 = dist.size() > 2 ? dist[2] : 0.0;
            const double k4 = dist.size() > 3 ? dist[3] : 0.0;

            const double theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            const double scale = theta_d / r;
            dx = x * scale;
            dy = y * scale;
        }

        void applyThinPrismFisheyeDistortion(
            const double x,
            const double y,
            const std::span<const double> dist,
            double& dx,
            double& dy) noexcept {
            const double r = std::sqrt(x * x + y * y);
            if (r < 1e-8) {
                dx = x;
                dy = y;
                return;
            }

            const double theta_over_r = std::atan(r) / r;
            const double uu = x * theta_over_r;
            const double vv = y * theta_over_r;

            const double k1 = dist.size() > 0 ? dist[0] : 0.0;
            const double k2 = dist.size() > 1 ? dist[1] : 0.0;
            const double k3 = dist.size() > 2 ? dist[2] : 0.0;
            const double k4 = dist.size() > 3 ? dist[3] : 0.0;
            const double p1 = dist.size() > 4 ? dist[4] : 0.0;
            const double p2 = dist.size() > 5 ? dist[5] : 0.0;
            const double sx1 = dist.size() > 6 ? dist[6] : 0.0;
            const double sx2 = dist.size() > 7 ? dist[7] : 0.0;
            const double sy1 = dist.size() > 8 ? dist[8] : 0.0;
            const double sy2 = dist.size() > 9 ? dist[9] : 0.0;

            const double u2 = uu * uu;
            const double uv = uu * vv;
            const double v2 = vv * vv;
            const double r2 = u2 + v2;
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;
            const double r8 = r4 * r4;

            const double radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
            const double tx = 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) + sx1 * r2 + sx2 * r4;
            const double ty = 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) + sy1 * r2 + sy2 * r4;

            dx = uu + uu * radial + tx;
            dy = vv + vv * radial + ty;
        }

        [[nodiscard]] bool hasNonZeroDistortion(const std::span<const double> coeffs) noexcept {
            for (const double c : coeffs) {
                if (std::abs(c) > 1e-12) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] glm::dvec2 detectionPixelScale(const CameraObservationView& view) noexcept {
            return {
                view.detected_image_width > 0 && view.intrinsics.width > 0
                    ? static_cast<double>(view.detected_image_width) / view.intrinsics.width
                    : 1.0,
                view.detected_image_height > 0 && view.intrinsics.height > 0
                    ? static_cast<double>(view.detected_image_height) / view.intrinsics.height
                    : 1.0,
            };
        }

        struct TagRayObservation {
            const CameraObservationView* view = nullptr;
            glm::dvec2 detected_pixel{0.0};
            Ray3D ray{};
        };

        [[nodiscard]] std::optional<glm::dvec3> intersectRaysLeastSquares(
            const std::span<const TagRayObservation> observations,
            const std::span<const std::size_t> active_indices) {
            glm::dmat3 A(0.0);
            glm::dvec3 b(0.0);

            for (const std::size_t idx : active_indices) {
                const glm::dvec3& d = observations[idx].ray.direction;
                const glm::dvec3& o = observations[idx].ray.origin;
                const glm::dmat3 proj = glm::dmat3(1.0) - glm::outerProduct(d, d);
                A += proj;
                b += proj * o;
            }

            const double det = glm::determinant(A);
            if (!std::isfinite(det) || std::abs(det) < 1e-10) {
                return std::nullopt;
            }

            const glm::dvec3 point = glm::inverse(A) * b;
            if (!isFiniteVec3(point)) {
                return std::nullopt;
            }
            return point;
        }

        [[nodiscard]] double computeMaxParallaxDeg(
            const std::span<const TagRayObservation> observations,
            const std::span<const std::size_t> active_indices) noexcept {
            double max_angle_deg = 0.0;
            for (std::size_t i = 0; i < active_indices.size(); ++i) {
                const glm::dvec3& d0 = observations[active_indices[i]].ray.direction;
                for (std::size_t j = i + 1; j < active_indices.size(); ++j) {
                    const glm::dvec3& d1 = observations[active_indices[j]].ray.direction;
                    const double cos_angle = std::clamp(glm::dot(d0, d1), -1.0, 1.0);
                    const double angle_deg = glm::degrees(std::acos(cos_angle));
                    max_angle_deg = std::max(max_angle_deg, angle_deg);
                }
            }
            return max_angle_deg;
        }

        [[nodiscard]] std::optional<double> computeViewReprojectionErrorPx(
            const TagRayObservation& obs,
            const glm::dvec3& world_point) {
            const auto projected = projectWorldPointToDetectedPixel(*obs.view, world_point);
            if (!projected) {
                return std::nullopt;
            }
            return glm::length(*projected - obs.detected_pixel);
        }

    } // namespace

    glm::dvec2 applyForwardDistortion(
        const glm::dvec2& normalized_xy,
        const core::CameraModelType model,
        const std::span<const double> coeffs) {
        if (coeffs.empty() && model == core::CameraModelType::PINHOLE) {
            return normalized_xy;
        }

        double dx = normalized_xy.x;
        double dy = normalized_xy.y;
        switch (model) {
        case core::CameraModelType::PINHOLE:
            applyPinholeDistortion(normalized_xy.x, normalized_xy.y, coeffs, dx, dy);
            break;
        case core::CameraModelType::FISHEYE:
            applyFisheyeDistortion(normalized_xy.x, normalized_xy.y, coeffs, dx, dy);
            break;
        case core::CameraModelType::THIN_PRISM_FISHEYE:
            applyThinPrismFisheyeDistortion(normalized_xy.x, normalized_xy.y, coeffs, dx, dy);
            break;
        case core::CameraModelType::ORTHO:
        case core::CameraModelType::EQUIRECTANGULAR:
            break;
        }
        return {dx, dy};
    }

    std::optional<glm::dvec2> invertDistortion(
        const glm::dvec2& distorted_xy,
        const core::CameraModelType model,
        const std::span<const double> coeffs) {
        if (!isFiniteVec2(distorted_xy)) {
            return std::nullopt;
        }

        if ((model == core::CameraModelType::PINHOLE && !hasNonZeroDistortion(coeffs)) ||
            model == core::CameraModelType::ORTHO ||
            model == core::CameraModelType::EQUIRECTANGULAR) {
            return distorted_xy;
        }

        glm::dvec2 est = distorted_xy;
        if (model == core::CameraModelType::FISHEYE || model == core::CameraModelType::THIN_PRISM_FISHEYE) {
            const double r_d = glm::length(distorted_xy);
            if (r_d > 1e-8) {
                const double clamped_theta = std::min(r_d, 1.55);
                est = distorted_xy * (std::tan(clamped_theta) / r_d);
            }
        }

        constexpr int kMaxIterations = 25;
        constexpr double kStepEps = 1e-6;
        constexpr double kConvergeEps = 1e-10;
        constexpr double kAcceptEps = 1e-5;

        for (int iter = 0; iter < kMaxIterations; ++iter) {
            const glm::dvec2 f = applyForwardDistortion(est, model, coeffs) - distorted_xy;
            const double err = glm::length(f);
            if (err < kConvergeEps) {
                return est;
            }

            const glm::dvec2 fx = applyForwardDistortion({est.x + kStepEps, est.y}, model, coeffs);
            const glm::dvec2 fy = applyForwardDistortion({est.x, est.y + kStepEps}, model, coeffs);
            const glm::dvec2 df_dx = (fx - (f + distorted_xy)) / kStepEps;
            const glm::dvec2 df_dy = (fy - (f + distorted_xy)) / kStepEps;

            const double det = df_dx.x * df_dy.y - df_dx.y * df_dy.x;
            if (!std::isfinite(det) || std::abs(det) < 1e-12) {
                break;
            }

            glm::dvec2 delta{
                (df_dy.y * f.x - df_dy.x * f.y) / det,
                (-df_dx.y * f.x + df_dx.x * f.y) / det,
            };
            const double delta_len = glm::length(delta);
            if (delta_len > 1.0) {
                delta /= delta_len;
            }

            est -= delta;
            if (!isFiniteVec2(est)) {
                return std::nullopt;
            }
            if (glm::length(delta) < kConvergeEps) {
                break;
            }
        }

        const glm::dvec2 final_residual = applyForwardDistortion(est, model, coeffs) - distorted_xy;
        if (glm::length(final_residual) <= kAcceptEps) {
            return est;
        }
        return std::nullopt;
    }

    std::optional<Ray3D> unprojectPixelToWorldRay(
        const CameraObservationView& view,
        const glm::dvec2& detected_pixel) {
        if (!isFiniteVec2(detected_pixel) ||
            view.intrinsics.fx <= kEpsilon ||
            view.intrinsics.fy <= kEpsilon) {
            return std::nullopt;
        }

        const glm::dvec2 pixel = detected_pixel / detectionPixelScale(view);
        glm::dvec3 cam_dir{0.0, 0.0, 1.0};
        glm::dvec3 cam_origin{0.0};

        if (view.intrinsics.lens_model == core::CameraModelType::EQUIRECTANGULAR) {
            if (view.intrinsics.width <= 0 || view.intrinsics.height <= 0) {
                return std::nullopt;
            }
            const double lon = ((pixel.x / static_cast<double>(view.intrinsics.width)) - 0.5) *
                               2.0 * std::numbers::pi_v<double>;
            const double lat = (0.5 - (pixel.y / static_cast<double>(view.intrinsics.height))) *
                               std::numbers::pi_v<double>;
            cam_dir = glm::dvec3(
                std::cos(lat) * std::sin(lon),
                -std::sin(lat),
                std::cos(lat) * std::cos(lon));
        } else {
            const glm::dvec2 distorted_xy{
                (pixel.x - view.intrinsics.cx) / view.intrinsics.fx,
                (pixel.y - view.intrinsics.cy) / view.intrinsics.fy,
            };
            const int num_dist = std::clamp(
                view.intrinsics.num_distortion,
                0,
                static_cast<int>(view.intrinsics.distortion.size()));
            const std::span<const double> dist_span(
                view.intrinsics.distortion.data(),
                static_cast<std::size_t>(num_dist));

            const auto undistorted_xy = invertDistortion(
                distorted_xy,
                view.intrinsics.lens_model,
                dist_span);
            if (!undistorted_xy) {
                return std::nullopt;
            }
            if (view.intrinsics.lens_model == core::CameraModelType::ORTHO) {
                cam_origin = glm::dvec3(*undistorted_xy, 0.0);
            } else {
                cam_dir = glm::dvec3(*undistorted_xy, 1.0);
            }
        }

        const glm::dmat3 R_c2w = glm::transpose(view.world_to_camera_rotation);
        const glm::dvec3 raw_origin = R_c2w * (cam_origin - view.world_to_camera_translation);
        const glm::dvec3 raw_dir = R_c2w * cam_dir;

        const glm::dvec3 world_origin = glm::dvec3(view.scene_world_transform * glm::dvec4(raw_origin, 1.0));
        const glm::dvec3 world_dir_unnorm = glm::dvec3(view.scene_world_transform * glm::dvec4(raw_dir, 0.0));
        const double dir_len = glm::length(world_dir_unnorm);
        if (!std::isfinite(dir_len) || dir_len <= kEpsilon || !isFiniteVec3(world_origin)) {
            return std::nullopt;
        }

        return Ray3D{
            .origin = world_origin,
            .direction = world_dir_unnorm / dir_len,
        };
    }

    std::optional<glm::dvec2> projectWorldPointToDetectedPixel(
        const CameraObservationView& view,
        const glm::dvec3& world_point) {
        if (!isFiniteVec3(world_point) ||
            view.intrinsics.fx <= kEpsilon ||
            view.intrinsics.fy <= kEpsilon) {
            return std::nullopt;
        }

        const double scene_det = glm::determinant(view.scene_world_transform);
        if (!std::isfinite(scene_det) || std::abs(scene_det) <= 1e-12) {
            return std::nullopt;
        }

        const glm::dmat4 inv_scene = glm::inverse(view.scene_world_transform);
        const glm::dvec3 raw_world = glm::dvec3(inv_scene * glm::dvec4(world_point, 1.0));
        const glm::dvec3 cam_pt =
            view.world_to_camera_rotation * raw_world + view.world_to_camera_translation;

        if (!isFiniteVec3(cam_pt)) {
            return std::nullopt;
        }

        glm::dvec2 intrinsic_pixel{0.0, 0.0};
        if (view.intrinsics.lens_model == core::CameraModelType::EQUIRECTANGULAR) {
            const double norm = glm::length(cam_pt);
            if (norm <= kEpsilon || view.intrinsics.width <= 0 || view.intrinsics.height <= 0) {
                return std::nullopt;
            }
            const glm::dvec3 dir = cam_pt / norm;
            const double lon = std::atan2(dir.x, dir.z);
            const double lat = std::asin(std::clamp(-dir.y, -1.0, 1.0));
            intrinsic_pixel.x = (0.5 + lon / (2.0 * std::numbers::pi_v<double>)) *
                                static_cast<double>(view.intrinsics.width);
            intrinsic_pixel.y = (0.5 - lat / std::numbers::pi_v<double>)*static_cast<double>(view.intrinsics.height);
        } else {
            if (cam_pt.z <= 1e-6) {
                return std::nullopt;
            }
            const glm::dvec2 normalized_xy = view.intrinsics.lens_model == core::CameraModelType::ORTHO
                                                 ? glm::dvec2(cam_pt)
                                                 : glm::dvec2(cam_pt) / cam_pt.z;
            const int num_dist = std::clamp(
                view.intrinsics.num_distortion,
                0,
                static_cast<int>(view.intrinsics.distortion.size()));
            const std::span<const double> dist_span(
                view.intrinsics.distortion.data(),
                static_cast<std::size_t>(num_dist));

            const glm::dvec2 distorted_xy = applyForwardDistortion(
                normalized_xy,
                view.intrinsics.lens_model,
                dist_span);
            intrinsic_pixel.x = view.intrinsics.fx * distorted_xy.x + view.intrinsics.cx;
            intrinsic_pixel.y = view.intrinsics.fy * distorted_xy.y + view.intrinsics.cy;
        }

        if (!isFiniteVec2(intrinsic_pixel)) {
            return std::nullopt;
        }
        return intrinsic_pixel * detectionPixelScale(view);
    }

    TriangulationSummary triangulateAprilTags(
        const std::span<const CameraObservationView> views,
        const TriangulationConfig& config) {
        std::map<int, std::vector<TagRayObservation>> grouped_observations;

        for (const auto& view : views) {
            std::map<int, TagDetection2D> best_per_tag;
            for (const auto& det : view.detections) {
                if (det.id < 0 ||
                    det.hamming > config.max_hamming ||
                    det.decision_margin < config.min_decision_margin) {
                    continue;
                }
                const auto it = best_per_tag.find(det.id);
                if (it == best_per_tag.end() || det.decision_margin > it->second.decision_margin) {
                    best_per_tag[det.id] = det;
                }
            }

            for (const auto& [tag_id, det] : best_per_tag) {
                const auto ray = unprojectPixelToWorldRay(view, det.center);
                if (!ray) {
                    continue;
                }
                grouped_observations[tag_id].push_back(TagRayObservation{
                    .view = &view,
                    .detected_pixel = det.center,
                    .ray = *ray,
                });
            }
        }

        TriangulationSummary summary;
        const int required_views = std::max(2, config.min_views);

        for (const auto& [tag_id, obs_list] : grouped_observations) {
            const int total_obs = static_cast<int>(obs_list.size());
            if (total_obs < required_views) {
                summary.rejected_tags.push_back(RejectedTagDiagnostic{
                    .tag_id = tag_id,
                    .total_observations = total_obs,
                    .max_parallax_angle_deg = 0.0,
                    .reason = "Insufficient camera views",
                });
                continue;
            }

            std::vector<std::size_t> all_indices(obs_list.size());
            std::iota(all_indices.begin(), all_indices.end(), std::size_t{0});

            const double initial_parallax_deg = computeMaxParallaxDeg(obs_list, all_indices);
            if (initial_parallax_deg < config.min_parallax_angle_deg) {
                summary.rejected_tags.push_back(RejectedTagDiagnostic{
                    .tag_id = tag_id,
                    .total_observations = total_obs,
                    .max_parallax_angle_deg = initial_parallax_deg,
                    .reason = "Insufficient parallax angle between views",
                });
                continue;
            }

            std::vector<std::size_t> active_indices;
            if (obs_list.size() >= 3) {
                double best_error_sum = std::numeric_limits<double>::infinity();

                std::vector<std::array<std::size_t, 2>> pairs;
                if (obs_list.size() <= 64) {
                    for (std::size_t i = 0; i < obs_list.size(); ++i) {
                        for (std::size_t j = i + 1; j < obs_list.size(); ++j) {
                            pairs.push_back({i, j});
                        }
                    }
                } else {
                    // Sample across the full list with a fixed seed and bounded work.
                    std::mt19937_64 rng(0);
                    std::uniform_int_distribution<std::size_t> first(0, obs_list.size() - 1);
                    std::uniform_int_distribution<std::size_t> second(0, obs_list.size() - 2);
                    constexpr std::size_t kPairBudget = 64 * 63 / 2;
                    for (std::size_t sample = 0; sample < kPairBudget; ++sample) {
                        const auto i = first(rng);
                        const auto j = second(rng);
                        pairs.push_back({i, j >= i ? j + 1 : j});
                    }
                }
                for (const auto& pair_indices : pairs) {
                    if (computeMaxParallaxDeg(obs_list, pair_indices) < config.min_parallax_angle_deg) {
                        continue;
                    }
                    const auto candidate_pt = intersectRaysLeastSquares(obs_list, pair_indices);
                    if (!candidate_pt) {
                        continue;
                    }

                    std::vector<std::size_t> candidate_inliers;
                    double error_sum = 0.0;
                    for (std::size_t k = 0; k < obs_list.size(); ++k) {
                        const auto err = computeViewReprojectionErrorPx(obs_list[k], *candidate_pt);
                        if (err && *err <= config.max_reprojection_error_px * 1.5) {
                            candidate_inliers.push_back(k);
                            error_sum += *err;
                        }
                    }

                    if (candidate_inliers.size() > active_indices.size() ||
                        (candidate_inliers.size() == active_indices.size() && error_sum < best_error_sum)) {
                        best_error_sum = error_sum;
                        active_indices = std::move(candidate_inliers);
                    }
                }
            }

            if (active_indices.size() < static_cast<std::size_t>(required_views)) {
                active_indices = all_indices;
            }

            std::optional<glm::dvec3> solved_point;
            double mean_reproj_err = 0.0;
            double max_reproj_err = 0.0;

            while (active_indices.size() >= static_cast<std::size_t>(required_views)) {
                solved_point = intersectRaysLeastSquares(obs_list, active_indices);
                if (!solved_point) {
                    break;
                }

                std::size_t worst_pos = 0;
                double worst_err = -1.0;
                double sum_err = 0.0;
                bool any_invalid = false;

                for (std::size_t pos = 0; pos < active_indices.size(); ++pos) {
                    const auto err = computeViewReprojectionErrorPx(obs_list[active_indices[pos]], *solved_point);
                    if (!err) {
                        worst_pos = pos;
                        worst_err = std::numeric_limits<double>::infinity();
                        any_invalid = true;
                        break;
                    }
                    sum_err += *err;
                    if (*err > worst_err) {
                        worst_err = *err;
                        worst_pos = pos;
                    }
                }

                if (!any_invalid && worst_err <= config.max_reprojection_error_px) {
                    mean_reproj_err = sum_err / static_cast<double>(active_indices.size());
                    max_reproj_err = worst_err;
                    break;
                }

                solved_point.reset();
                active_indices.erase(active_indices.begin() + static_cast<std::ptrdiff_t>(worst_pos));
            }

            if (!solved_point) {
                summary.rejected_tags.push_back(RejectedTagDiagnostic{
                    .tag_id = tag_id,
                    .total_observations = total_obs,
                    .max_parallax_angle_deg = initial_parallax_deg,
                    .reason = "Reprojection error exceeded threshold across views",
                });
                continue;
            }

            const double final_parallax_deg = computeMaxParallaxDeg(obs_list, active_indices);
            if (final_parallax_deg < config.min_parallax_angle_deg) {
                summary.rejected_tags.push_back(RejectedTagDiagnostic{
                    .tag_id = tag_id,
                    .total_observations = total_obs,
                    .max_parallax_angle_deg = final_parallax_deg,
                    .reason = "Insufficient parallax angle among inlier views",
                });
                continue;
            }

            TriangulatedTag3D tag_result;
            tag_result.tag_id = tag_id;
            tag_result.position = *solved_point;
            tag_result.total_observations = total_obs;
            tag_result.inlier_observations = static_cast<int>(active_indices.size());
            tag_result.mean_reprojection_error_px = mean_reproj_err;
            tag_result.max_reprojection_error_px = max_reproj_err;
            tag_result.max_parallax_angle_deg = final_parallax_deg;
            tag_result.camera_uids.reserve(active_indices.size());
            tag_result.image_names.reserve(active_indices.size());
            for (const std::size_t idx : active_indices) {
                tag_result.camera_uids.push_back(obs_list[idx].view->camera_uid);
                tag_result.image_names.push_back(obs_list[idx].view->image_name);
            }
            summary.tags.push_back(std::move(tag_result));
        }

        return summary;
    }

    glm::mat4 buildUniformScaleTransform(const glm::vec3& pivot, const float scale_factor) {
        if (!std::isfinite(scale_factor) || scale_factor <= 1e-6f) {
            return glm::mat4(1.0f);
        }
        const glm::mat4 to_pivot = glm::translate(glm::mat4(1.0f), -pivot);
        const glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(scale_factor));
        const glm::mat4 from_pivot = glm::translate(glm::mat4(1.0f), pivot);
        return from_pivot * scale * to_pivot;
    }

    ScaleSolveResult solveGlobalScale(
        const std::span<const TriangulatedTag3D> tags,
        const std::span<const TagDistanceConstraint> constraints,
        const ScaleSolverConfig& config) {
        ScaleSolveResult result;
        result.total_constraints = static_cast<int>(constraints.size());
        result.evaluated_constraints.reserve(constraints.size());

        if (constraints.empty()) {
            result.error_message = "No distance constraints provided";
            return result;
        }

        std::map<int, glm::dvec3> tag_positions;
        for (const auto& tag : tags) {
            if (tag.tag_id >= 0 && isFiniteVec3(tag.position)) {
                tag_positions[tag.tag_id] = tag.position;
            }
        }

        std::vector<std::size_t> valid_indices;
        valid_indices.reserve(constraints.size());

        for (const auto& c : constraints) {
            EvaluatedDistanceConstraint eval;
            eval.tag_a = c.tag_a;
            eval.tag_b = c.tag_b;
            eval.target_distance = c.target_distance;

            if (c.tag_a == c.tag_b) {
                eval.status = "Invalid constraint: tag_a and tag_b must be distinct";
                result.evaluated_constraints.push_back(std::move(eval));
                continue;
            }
            if (!std::isfinite(c.target_distance) || c.target_distance <= 1e-6) {
                eval.status = "Invalid constraint: target distance must be positive";
                result.evaluated_constraints.push_back(std::move(eval));
                continue;
            }

            const auto it_a = tag_positions.find(c.tag_a);
            const auto it_b = tag_positions.find(c.tag_b);
            if (it_a == tag_positions.end() || it_b == tag_positions.end()) {
                eval.status = "Missing triangulated 3D position for one or both tags";
                result.evaluated_constraints.push_back(std::move(eval));
                continue;
            }

            const double measured = glm::length(it_b->second - it_a->second);
            if (!std::isfinite(measured) || measured <= 1e-6) {
                eval.status = "Degenerate triangulated distance between tags";
                result.evaluated_constraints.push_back(std::move(eval));
                continue;
            }

            eval.unscaled_distance = measured;
            eval.candidate_scale = c.target_distance / measured;
            eval.valid = true;
            eval.status = "Valid";
            valid_indices.push_back(result.evaluated_constraints.size());
            result.evaluated_constraints.push_back(std::move(eval));
        }

        result.valid_constraints = static_cast<int>(valid_indices.size());
        const int required_inliers = std::max(1, config.min_inliers);
        if (result.valid_constraints < required_inliers) {
            result.error_message = "Insufficient valid distance constraints";
            return result;
        }

        const double rel_thresh = std::max(1e-6, config.inlier_relative_threshold);
        const double abs_thresh = std::max(0.0, config.inlier_absolute_threshold);

        const auto isConstraintInlier = [&](const double abs_err, const double rel_err) noexcept {
            return rel_err <= rel_thresh || (abs_thresh > 0.0 && abs_err <= abs_thresh);
        };

        std::vector<double> hypotheses;
        hypotheses.reserve(valid_indices.size() * 2);
        for (const std::size_t idx : valid_indices) {
            hypotheses.push_back(result.evaluated_constraints[idx].candidate_scale);
        }
        if (valid_indices.size() <= 64) {
            for (std::size_t i = 0; i < valid_indices.size(); ++i) {
                for (std::size_t j = i + 1; j < valid_indices.size(); ++j) {
                    const double s1 = result.evaluated_constraints[valid_indices[i]].candidate_scale;
                    const double s2 = result.evaluated_constraints[valid_indices[j]].candidate_scale;
                    hypotheses.push_back(0.5 * (s1 + s2));
                }
            }
        }

        double best_truncated_loss = std::numeric_limits<double>::infinity();
        std::vector<std::size_t> best_inliers;

        for (const double candidate_s : hypotheses) {
            if (!std::isfinite(candidate_s) || candidate_s <= 1e-9) {
                continue;
            }

            std::vector<std::size_t> current_inliers;
            double truncated_loss = 0.0;

            for (const std::size_t idx : valid_indices) {
                const auto& eval = result.evaluated_constraints[idx];
                const double abs_err = std::abs(candidate_s * eval.unscaled_distance - eval.target_distance);
                const double rel_err = abs_err / eval.target_distance;

                if (isConstraintInlier(abs_err, rel_err)) {
                    current_inliers.push_back(idx);
                    truncated_loss += rel_err * rel_err;
                } else {
                    truncated_loss += rel_thresh * rel_thresh;
                }
            }

            if (current_inliers.size() > best_inliers.size() ||
                (current_inliers.size() == best_inliers.size() && truncated_loss < best_truncated_loss)) {
                best_truncated_loss = truncated_loss;
                best_inliers = std::move(current_inliers);
            }
        }

        const auto refineScaleWls = [&](const std::span<const std::size_t> inlier_set) -> std::optional<double> {
            double num = 0.0;
            double den = 0.0;
            for (const std::size_t idx : inlier_set) {
                const auto& eval = result.evaluated_constraints[idx];
                const double ratio = eval.unscaled_distance / eval.target_distance;
                num += ratio;
                den += ratio * ratio;
            }
            if (den <= 1e-12 || !std::isfinite(num / den)) {
                return std::nullopt;
            }
            return num / den;
        };

        auto refined_scale = refineScaleWls(best_inliers);
        if (!refined_scale || *refined_scale <= 1e-9) {
            result.error_message = "Failed to refine global scale factor";
            return result;
        }

        std::vector<std::size_t> polished_inliers;
        for (const std::size_t idx : valid_indices) {
            const auto& eval = result.evaluated_constraints[idx];
            const double abs_err = std::abs(*refined_scale * eval.unscaled_distance - eval.target_distance);
            if (isConstraintInlier(abs_err, abs_err / eval.target_distance)) {
                polished_inliers.push_back(idx);
            }
        }
        if (!polished_inliers.empty()) {
            if (const auto second_pass = refineScaleWls(polished_inliers);
                second_pass && *second_pass > 1e-9) {
                refined_scale = second_pass;
            }
        }

        const double final_scale = *refined_scale;
        result.scale_factor = final_scale;
        double sq_err_sum = 0.0;
        double abs_err_sum = 0.0;
        double rel_err_sum = 0.0;
        double max_abs_err = 0.0;
        std::set<int> participating_tags;

        for (const std::size_t idx : valid_indices) {
            auto& eval = result.evaluated_constraints[idx];
            eval.scaled_distance = final_scale * eval.unscaled_distance;
            eval.absolute_error = std::abs(eval.scaled_distance - eval.target_distance);
            eval.relative_error = eval.absolute_error / eval.target_distance;
            eval.inlier = isConstraintInlier(eval.absolute_error, eval.relative_error);
            eval.status = eval.inlier ? "Inlier" : "Outlier";

            if (eval.inlier) {
                ++result.inlier_count;
                sq_err_sum += eval.absolute_error * eval.absolute_error;
                abs_err_sum += eval.absolute_error;
                rel_err_sum += eval.relative_error;
                max_abs_err = std::max(max_abs_err, eval.absolute_error);
                participating_tags.insert(eval.tag_a);
                participating_tags.insert(eval.tag_b);
            }
        }
        result.inlier_ratio = static_cast<double>(result.inlier_count) /
                              static_cast<double>(result.valid_constraints);

        if (result.inlier_count > 0) {
            const double count_d = static_cast<double>(result.inlier_count);
            result.rmse = std::sqrt(sq_err_sum / count_d);
            result.mae = abs_err_sum / count_d;
            result.max_error = max_abs_err;
            result.mean_relative_error = rel_err_sum / count_d;
        }

        glm::dvec3 pivot_sum{0.0};
        std::size_t pivot_count = 0;
        for (const int tag_id : participating_tags) {
            pivot_sum += tag_positions.at(tag_id);
            ++pivot_count;
        }
        if (pivot_count == 0) {
            for (const auto& [tag_id, pos] : tag_positions) {
                (void)tag_id;
                pivot_sum += pos;
                ++pivot_count;
            }
        }
        if (pivot_count > 0) {
            result.pivot = pivot_sum / static_cast<double>(pivot_count);
        }

        result.scale_transform = buildUniformScaleTransform(
            glm::vec3(result.pivot),
            static_cast<float>(final_scale));

        if (result.inlier_count < required_inliers) {
            result.error_message = "Inlier count below minimum required threshold";
            return result;
        }
        if (result.inlier_ratio + 1e-9 < config.min_inlier_ratio) {
            result.error_message = "Inlier ratio below minimum required threshold";
            return result;
        }

        result.success = true;
        return result;
    }

    nlohmann::ordered_json scaleResultToJson(const ScaleSolveResult& result) {
        return {
            {"success", result.success},
            {"error_message", result.error_message},
            {"scale_factor", result.scale_factor},
            {"pivot", {result.pivot.x, result.pivot.y, result.pivot.z}},
            {"total_constraints", result.total_constraints},
            {"valid_constraints", result.valid_constraints},
            {"inlier_count", result.inlier_count},
            {"inlier_ratio", result.inlier_ratio},
            {"rmse", result.rmse},
            {"mae", result.mae},
            {"max_error", result.max_error},
            {"mean_relative_error", result.mean_relative_error},
        };
    }

    nlohmann::ordered_json triangulationToJson(const TriangulationSummary& summary) {
        using json = nlohmann::ordered_json;
        json tags_json = json::array();
        for (const auto& tag : summary.tags) {
            tags_json.push_back({
                {"tag_id", tag.tag_id},
                {"position", {tag.position.x, tag.position.y, tag.position.z}},
                {"total_observations", tag.total_observations},
                {"inlier_observations", tag.inlier_observations},
                {"mean_reprojection_error_px", tag.mean_reprojection_error_px},
                {"max_reprojection_error_px", tag.max_reprojection_error_px},
                {"max_parallax_angle_deg", tag.max_parallax_angle_deg},
                {"camera_uids", tag.camera_uids},
                {"image_names", tag.image_names},
            });
        }

        json rejected_json = json::array();
        for (const auto& rej : summary.rejected_tags) {
            rejected_json.push_back({
                {"tag_id", rej.tag_id},
                {"total_observations", rej.total_observations},
                {"max_parallax_angle_deg", rej.max_parallax_angle_deg},
                {"reason", rej.reason},
            });
        }
        return {{"triangulated_tags", std::move(tags_json)}, {"rejected_tags", std::move(rejected_json)}};
    }

    std::string serializeManifestToJson(
        const ScaleCalibrationManifest& manifest,
        const int indent) {
        using json = nlohmann::ordered_json;

        json root;
        root["schema_version"] = 1;
        root["timestamp_utc"] = manifest.timestamp_utc.empty()
                                    ? core::utc_now()
                                    : manifest.timestamp_utc;
        root["tag_family"] = std::string(aprilTagFamilyName(manifest.family));

        root["triangulation_config"] = {
            {"min_views", manifest.triangulation_config.min_views},
            {"min_parallax_angle_deg", manifest.triangulation_config.min_parallax_angle_deg},
            {"max_reprojection_error_px", manifest.triangulation_config.max_reprojection_error_px},
            {"max_hamming", manifest.triangulation_config.max_hamming},
            {"min_decision_margin", manifest.triangulation_config.min_decision_margin},
        };

        root["solver_config"] = {
            {"inlier_relative_threshold", manifest.solver_config.inlier_relative_threshold},
            {"inlier_absolute_threshold", manifest.solver_config.inlier_absolute_threshold},
            {"min_inliers", manifest.solver_config.min_inliers},
            {"min_inlier_ratio", manifest.solver_config.min_inlier_ratio},
        };

        root["scaling_summary"] = scaleResultToJson(manifest.scale_result);
        root.update(triangulationToJson(manifest.triangulation));

        json constraints_json = json::array();
        for (const auto& eval : manifest.scale_result.evaluated_constraints) {
            constraints_json.push_back({
                {"tag_a", eval.tag_a},
                {"tag_b", eval.tag_b},
                {"target_distance", eval.target_distance},
                {"unscaled_distance", eval.unscaled_distance},
                {"scaled_distance", eval.scaled_distance},
                {"absolute_error", eval.absolute_error},
                {"relative_error", eval.relative_error},
                {"candidate_scale", eval.candidate_scale},
                {"valid", eval.valid},
                {"inlier", eval.inlier},
                {"status", eval.status},
            });
        }
        root["evaluated_constraints"] = std::move(constraints_json);

        return root.dump(indent) + "\n";
    }

} // namespace lfs::geometry::apriltag
