/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Builds without CUDA evaluate GUT's camera models on the host. This transcribes the OpenCV fisheye
// and thin prism fisheye models of src/training/rasterization/gsplat/Cameras.cuh, which
// gut_camera_model_cuda.cu runs on the device, in the same float arithmetic; keep the two in step.

#include "gut_camera_model_cuda.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace gut_camera_model_test {
    namespace {
        constexpr float kPi = 3.14159265358979323846f;
        constexpr int kNewtonIterations = 20;
        constexpr int kUndistortIterations = 5;
        constexpr float kMinNorm = 1e-6f;

        float stable_norm(const float x, const float y) {
            const float lo = std::fmin(std::fabs(x), std::fabs(y));
            const float hi = std::fmax(std::fabs(x), std::fabs(y));
            if (hi <= 0.f)
                return 0.f;
            const float ratio = lo / hi;
            return hi * std::sqrt(1.f + ratio * ratio);
        }

        template <size_t N>
        float horner(const std::array<float, N>& poly, const float x) {
            float y = 0.f;
            for (auto c = poly.rbegin(); c != poly.rend(); ++c)
                y = x * y + *c;
            return y;
        }

        template <size_t N>
        float odd(const std::array<float, N>& poly, const float x) { return x * horner(poly, x * x); }

        template <size_t N>
        float even(const std::array<float, N>& poly, const float x) { return horner(poly, x * x); }

        // eval_poly_inverse_horner_newton: Newton from `x` towards f(x) = y.
        template <class F, class DF>
        float newton(const F& f, const DF& df, float x, const float y, bool& converged) {
            for (int i = 0; i < kNewtonIterations; ++i) {
                const float dfdx = df(x);
                const float dx = (f(x) - y) / dfdx;
                x -= dx;
                if (std::fabs(dx) < 1e-6f) {
                    converged = true;
                    break;
                }
            }
            return x;
        }

        // compute_opencv_fisheye_max_angle: the smallest positive root of 1 + a x + b x^2 + c x^3.
        float first_turn(const float a, const float b, const float c) {
            const float inf = std::numeric_limits<float>::max();
            if (c == 0.0f) {
                if (b == 0.0f)
                    return a >= 0.0f ? inf : -1.0f / a;
                float delta = a * a - 4.0f * b;
                if (delta >= 0.0f) {
                    delta = std::sqrt(delta) - a;
                    if (delta > 0.0f)
                        return 2.0f / delta;
                }
                return inf;
            }
            const float boc = b / c;
            const float boc2 = boc * boc;
            const float t1 = (9.0f * a * boc - 2.0f * b * boc2 - 27.0f) / c;
            const float t2 = 3.0f * a / c - boc2;
            const float delta = t1 * t1 + 4.0f * t2 * t2 * t2;
            if (delta >= 0.0f) {
                const float cube_root = std::cbrt((std::sqrt(delta) + t1) / 2.0f);
                if (cube_root != 0.0f) {
                    const float root = (cube_root - (t2 / cube_root) - boc) / 3.0f;
                    if (root > 0.0f)
                        return root;
                }
                return inf;
            }
            const float theta = std::atan2(std::sqrt(-delta), t1) / 3.0f;
            const float t3 = 2.0f * std::sqrt(-t2);
            float root = inf;
            for (const int i : {-1, 0, 1}) {
                const float s = (t3 * std::cos(theta + i * (2.0f * kPi / 3.0f)) - boc) / 3.0f;
                if (s > 0.0f)
                    root = std::min(root, s);
            }
            return root;
        }

        struct Lens {
            std::array<float, 2> focal, principal;
            std::array<int, 2> resolution;
            std::array<float, 4> prism{}; // p1, p2, sx1, sy1
            bool thin_prism = false;
            std::array<float, 5> forward{};  // odd: theta + k1 theta^3 + ... + k4 theta^9
            std::array<float, 5> dforward{}; // even: its derivative
            std::array<float, 2> backward{}; // line seeding the inverse
            float max_angle = 0.f;

            Lens(const std::array<float, 2>& f, const std::array<float, 2>& c, const std::array<int, 2>& size,
                 const std::array<float, 4>& radial, const std::array<float, 4>& thin, const bool has_prism)
                : focal(f), principal(c), resolution(size), prism(thin), thin_prism(has_prism) {
                const auto [k1, k2, k3, k4] = radial;
                forward = {1.f, k1, k2, k3, k4};
                dforward = {1.f, 3 * k1, 5 * k2, 7 * k3, 9 * k4};
                const float diag_x = std::max(float(resolution[0]) - principal[0], principal[0]);
                const float diag_y = std::max(float(resolution[1]) - principal[1], principal[1]);
                const float max_radius = std::sqrt(diag_x * diag_x + diag_y * diag_y);
                if (k4 == 0) {
                    max_angle = std::sqrt(first_turn(3.f * k1, 5.f * k2, 7.f * k3));
                } else {
                    const std::array<float, 4> ddforward{6 * k1, 20 * k2, 42 * k3, 72 * k4};
                    bool converged = false;
                    max_angle = newton([&](float x) { return even(dforward, x); },
                                       [&](float x) { return odd(ddforward, x); }, 1.57f, 0.f, converged);
                    if (!converged || max_angle <= 0.f)
                        max_angle = std::numeric_limits<float>::max();
                }
                max_angle = std::min(std::min(max_angle, kPi), std::max(max_radius / focal[0], max_radius / focal[1]));
                const float max_normalized = thin_prism ? std::max(resolution[0] / 2.f / focal[0], resolution[1] / 2.f / focal[1])
                                                        : odd(forward, max_angle);
                backward = {0.f, max_angle / max_normalized};
            }

            std::array<float, 2> prism_terms(const float x, const float y) const {
                const auto [p1, p2, sx1, sy1] = prism;
                const float r2 = x * x + y * y;
                return {2.f * p1 * x * y + p2 * (r2 + 2.f * x * x) + sx1 * r2,
                        p1 * (r2 + 2.f * y * y) + 2.f * p2 * x * y + sy1 * r2};
            }

            float inverse(const float distance, bool& converged) const {
                return newton([&](float x) { return odd(forward, x); }, [&](float x) { return even(dforward, x); },
                              horner(backward, distance), distance, converged);
            }

            // camera_ray_to_image_point with a zero margin: [x, y, valid].
            std::array<float, 3> project(const float x, const float y, const float z) const {
                if (z <= 0.f && x == 0.f && y == 0.f)
                    return {0.f, 0.f, 0.f};
                float norm = stable_norm(x, y);
                if (norm <= 0.f)
                    norm = std::numeric_limits<float>::epsilon();
                const float theta_full = std::atan2(norm, z);
                const float theta = theta_full < max_angle ? theta_full : max_angle;
                float u, v;
                if (thin_prism) {
                    const float thetad = odd(forward, theta);
                    if (thetad <= 0.f)
                        return {0.f, 0.f, 0.f};
                    // The tangential and prism terms act on the theta-scaled point.
                    const auto terms = prism_terms(x * (theta / norm), y * (theta / norm));
                    const float scale = thetad / norm;
                    u = focal[0] * (scale * x + terms[0]) + principal[0];
                    v = focal[1] * (scale * y + terms[1]) + principal[1];
                } else {
                    const float delta = odd(forward, theta) / norm;
                    if (delta <= 0.f)
                        return {0.f, 0.f, 0.f};
                    u = focal[0] * delta * x + principal[0];
                    v = focal[1] * delta * y + principal[1];
                }
                const bool valid = 0.f <= u && u < float(resolution[0]) && 0.f <= v && v < float(resolution[1]) &&
                                   theta_full < max_angle;
                return {u, v, valid ? 1.f : 0.f};
            }

            // image_point_to_camera_ray: [x, y, z, valid].
            std::array<float, 4> unproject(const float px, const float py) const {
                const float ux = (px - principal[0]) / focal[0];
                const float uy = (py - principal[1]) / focal[1];
                constexpr std::array<float, 4> invalid{0.f, 0.f, 1.f, 0.f};
                if (!thin_prism) {
                    const float distance = std::sqrt(ux * ux + uy * uy);
                    bool converged = false;
                    const float theta = inverse(distance, converged);
                    if (theta < 0.f || theta >= max_angle || !converged)
                        return invalid;
                    if (distance < kMinNorm)
                        return {0.f, 0.f, 1.f, 1.f};
                    const float scale = std::sin(theta) / distance;
                    return {scale * ux, scale * uy, std::cos(theta), 1.f};
                }
                // Fixed point on the theta-scaled point: uv = radial(point) + prism(point).
                float rx = ux, ry = uy, tx = 0.f, ty = 0.f, theta = 0.f;
                for (int i = 0; i <= kUndistortIterations; ++i) {
                    if (i > 0) {
                        const auto terms = prism_terms(tx, ty);
                        rx = ux - terms[0];
                        ry = uy - terms[1];
                    }
                    const float distance = std::sqrt(rx * rx + ry * ry);
                    bool converged = false;
                    theta = inverse(distance, converged);
                    if (theta < 0.f || theta >= max_angle || !converged)
                        return invalid;
                    tx = distance >= kMinNorm ? rx * (theta / distance) : 0.f;
                    ty = distance >= kMinNorm ? ry * (theta / distance) : 0.f;
                }
                if (theta < kMinNorm)
                    return {0.f, 0.f, 1.f, 1.f};
                const float scale = std::sin(theta) / theta;
                return {scale * tx, scale * ty, std::cos(theta), 1.f};
            }
        };

        Lens lens_of(const ThinPrismCamera& camera) {
            return {camera.focal, camera.principal, camera.resolution, camera.radial, camera.thin_prism, true};
        }

        Lens lens_of(const OpenCvFisheyeCamera& camera) {
            return {camera.focal, camera.principal, camera.resolution, camera.radial, {}, false};
        }

        std::vector<float> project_all(const Lens& lens, const std::vector<float>& rays) {
            std::vector<float> points;
            points.reserve(rays.size());
            for (size_t i = 0; i + 2 < rays.size(); i += 3) {
                const auto point = lens.project(rays[i], rays[i + 1], rays[i + 2]);
                points.insert(points.end(), point.begin(), point.end());
            }
            return points;
        }

        std::vector<float> unproject_all(const Lens& lens, const std::vector<float>& points) {
            std::vector<float> rays;
            rays.reserve(points.size() * 2);
            for (size_t i = 0; i + 1 < points.size(); i += 2) {
                const auto ray = lens.unproject(points[i], points[i + 1]);
                rays.insert(rays.end(), ray.begin(), ray.end());
            }
            return rays;
        }
    } // namespace

    std::vector<float> project(const ThinPrismCamera& camera, const std::vector<float>& rays) {
        return project_all(lens_of(camera), rays);
    }

    std::vector<float> unproject(const ThinPrismCamera& camera, const std::vector<float>& points) {
        return unproject_all(lens_of(camera), points);
    }

    std::vector<float> project(const OpenCvFisheyeCamera& camera, const std::vector<float>& rays) {
        return project_all(lens_of(camera), rays);
    }

    std::vector<float> unproject(const OpenCvFisheyeCamera& camera, const std::vector<float>& points) {
        return unproject_all(lens_of(camera), points);
    }

} // namespace gut_camera_model_test
