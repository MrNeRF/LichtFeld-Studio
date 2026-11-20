/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera_views.hpp"
#include "core/cuda/undistort/distortion_model.cuh"
#include "core/normal_prior_rotation.hpp"
#include "gut_camera_model_cuda.hpp"
#include "preprocessing/lens_priors.hpp"
#include "training/depth_anchor/depth_anchor_cache.hpp"
#include "training/normal_auto_generate.hpp"
#include <algorithm>
#include <cmath>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <numbers>

namespace {
    using namespace lfs::core;
    using gut_camera_model_test::OpenCvFisheyeCamera;
    using gut_camera_model_test::ThinPrismCamera;
    constexpr double degree = std::numbers::pi / 180;

    template <typename Calibration>
    std::shared_ptr<Camera> camera(const Calibration& k) {
        constexpr bool prism = std::is_same_v<Calibration, ThinPrismCamera>;
        Tensor tangential;
        if constexpr (prism)
            tangential = Tensor::from_vector(std::vector<float>(k.thin_prism.begin(), k.thin_prism.end()), {4}, Device::CPU);
        return std::make_shared<Camera>(
            Tensor::from_vector({0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CPU),
            Tensor::from_vector({2.f, -3.f, 1.f}, {3}, Device::CPU),
            k.focal[0], k.focal[1], k.principal[0], k.principal[1],
            Tensor::from_vector(std::vector<float>(k.radial.begin(), k.radial.end()), {4}, Device::CPU), tangential,
            prism ? CameraModelType::THIN_PRISM_FISHEYE : CameraModelType::FISHEYE,
            "sub/lens.png", "lens.png", "mask.png", k.resolution[0], k.resolution[1], 7, 31, "depth.png", "normal.png");
    }

    std::vector<PinholeView> plan_with_native_limit(const Camera& source) {
        const auto lens = lfs::preprocessing::make_lens_camera(source);
        return plan_camera_views(source, lfs::preprocessing::lens_max_angle(lens));
    }

    OpenCvFisheyeCamera equidistant(double reach_degrees) {
        const float f = 128 / (reach_degrees * degree);
        return {{f, f}, {128, 128}, {256, 256}, {}};
    }

    // Construct the ray independently of the planner's rotation matrix.
    glm::dvec3 pixel_ray(const UndistortParams& p, int face, int x, int y) {
        const double u = (x + .5 - p.dst_cx) / p.dst_fx, v = (y + .5 - p.dst_cy) / p.dst_fy;
        const double c = .5, s = std::sqrt(.75);
        switch (face) {
        case 1: return glm::normalize(glm::dvec3{c * u + s, v, c - s * u});
        case 2: return glm::normalize(glm::dvec3{c * u - s, v, c + s * u});
        case 3: return glm::normalize(glm::dvec3{u, c * v + s, c - s * v});
        case 4: return glm::normalize(glm::dvec3{u, c * v - s, c + s * v});
        default: return glm::normalize(glm::dvec3{u, v, 1});
        }
    }

    glm::dvec3 local_ray(const UndistortParams& p, glm::dvec3 ray) {
        const auto* r = p.view_rotation;
        return {r[0] * ray.x + r[3] * ray.y + r[6] * ray.z,
                r[1] * ray.x + r[4] * ray.y + r[7] * ray.z,
                r[2] * ray.x + r[5] * ray.y + r[8] * ray.z};
    }

    bool contains(const UndistortParams& p, glm::dvec3 ray) {
        const auto local = local_ray(p, ray);
        if (local.z <= 0)
            return false;
        const double x = p.dst_fx * local.x / local.z + p.dst_cx;
        const double y = p.dst_fy * local.y / local.z + p.dst_cy;
        return x >= -1e-5 && x <= p.dst_width + 1e-5 && y >= -1e-5 && y <= p.dst_height + 1e-5;
    }

    TEST(FisheyeViews, CoverageAndPixelValidity) {
        for (const double reach : {85., 90., 95.}) {
            const auto k = equidistant(reach);
            const auto plan = plan_camera_views(*camera(k), std::sqrt(2.0) * reach * degree);
            ASSERT_EQ(plan.size(), 5u);
            double worst = 0;
            for (int i = 0; i < 5; ++i) {
                const auto& p = plan[i].params;
                for (int y = 0; y < p.dst_height; ++y) {
                    for (int x = 0; x < p.dst_width; ++x) {
                        const auto ray = pixel_ray(p, i, x, y);
                        const auto mapped = detail::source_position(x + .5f, y + .5f, p);
                        ASSERT_GE(mapped.x, -.5f);
                        ASSERT_LT(mapped.x, p.src_width - .5f);
                        ASSERT_GE(mapped.y, -.5f);
                        ASSERT_LT(mapped.y, p.src_height - .5f);
                        const double theta = std::acos(ray.z);
                        worst = std::max(worst, theta / degree);
                        const double r = std::hypot(ray.x, ray.y);
                        const double scale = r ? theta * k.focal[0] / r : 0;
                        ASSERT_LE(std::abs(scale * ray.x), 128);
                        ASSERT_LE(std::abs(scale * ray.y), 128);
                    }
                }
            }
            EXPECT_LE(worst, reach);
            int sampled = 0, covered = 0, interior_misses = 0;
            for (int t = 0; t <= 360 && t / 4. <= reach; ++t) {
                const double theta = t * degree / 4;
                for (int ph = 0; ph < 720; ++ph) {
                    const double phi = ph * degree / 2;
                    const glm::dvec3 ray{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
                    const bool seen = std::ranges::any_of(plan, [&](const auto& face) { return contains(face.params, ray); });
                    ++sampled;
                    covered += seen;
                    if (!seen && t / 4. <= (reach >= 90 ? 90 : 82))
                        ++interior_misses;
                }
            }
            std::cout << "Reach " << reach << " deg: " << covered << "/" << sampled
                      << " directions covered, outermost pixel " << worst << " deg\n";
            EXPECT_EQ(interior_misses, 0);
        }
    }

    template <typename Calibration>
    void check_content(const Calibration& k) {
        const auto plan = plan_with_native_limit(*camera(k));
        ASSERT_EQ(plan.size(), 5u);
        const int w = k.resolution[0], h = k.resolution[1], n = w * h;
        std::vector<float> coordinates;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                coordinates.insert(coordinates.end(), {x + .5f, y + .5f});
        const auto rays = gut_camera_model_test::unproject(k, coordinates);
        std::vector<float> signal(3 * n);
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c)
                signal[c * n + i] = rays[4 * i + 3] ? .5f + .45f * rays[4 * i + c] : 0.f;
        const auto image = Tensor::from_vector(signal, {3, size_t(h), size_t(w)}, Device::CUDA);
        for (int i = 0; i < 5; ++i) {
            const auto& p = plan[i].params;
            const auto pixels = undistort_image(image, p, nullptr).cpu().to_vector();
            double worst = 0;
            for (int y = 0; y < p.dst_height; ++y)
                for (int x = 0; x < p.dst_width; ++x) {
                    const auto ray = pixel_ray(p, i, x, y);
                    for (int c = 0; c < 3; ++c) {
                        const float value = pixels[(c * p.dst_height + y) * p.dst_width + x];
                        ASSERT_TRUE(std::isfinite(value));
                        worst = std::max(worst, std::abs(value - (.5 + .45 * ray[c])));
                    }
                }
            EXPECT_LT(worst, .006) << "face " << i;
        }
    }

    const OpenCvFisheyeCamera distorted{{62, 60}, {128, 123}, {256, 250}, {.042f, -.004f, .0003f, .00001f}};
    const ThinPrismCamera prism{distorted.focal, distorted.principal, distorted.resolution, distorted.radial, {.008f, -.005f, .003f, -.006f}};

    TEST(FisheyeViews, ResampledPixelsCarryTheirOwnDirections) {
        check_content(distorted);
        check_content(prism);
    }

    template <typename Calibration>
    void check_projection(const Calibration& k) {
        const auto plan = plan_with_native_limit(*camera(k));
        ASSERT_EQ(plan.size(), 5u);
        std::vector<float> rays;
        std::vector<float2> mapped;
        for (int t = 0; t <= 115; ++t)
            for (int ph = 0; ph < 360; ph += 5) {
                const glm::dvec3 ray{std::sin(t * degree) * std::cos(ph * degree),
                                     std::sin(t * degree) * std::sin(ph * degree), std::cos(t * degree)};
                const auto best = std::ranges::max_element(plan, {}, [&](const auto& face) { return local_ray(face.params, ray).z; });
                const auto& p = best->params;
                const auto v = local_ray(p, ray);
                mapped.push_back(detail::source_position(p.dst_cx + p.dst_fx * v.x / v.z, p.dst_cy + p.dst_fy * v.y / v.z, p));
                rays.insert(rays.end(), {float(ray.x), float(ray.y), float(ray.z)});
            }
        const auto projected = gut_camera_model_test::project(k, rays);
        int beyond = 0;
        double worst = 0;
        for (size_t i = 0; i < mapped.size(); ++i) {
            if (!projected[3 * i + 2])
                continue;
            beyond += rays[3 * i + 2] < 0;
            worst = std::max({worst, std::abs(mapped[i].x + .5 - projected[3 * i]), std::abs(mapped[i].y + .5 - projected[3 * i + 1])});
        }
        EXPECT_GT(beyond, 0);
        EXPECT_LT(worst, .0002);
    }

    TEST(FisheyeViews, ProjectionAgreesWithRasterizerAcrossHorizon) {
        check_projection(distorted);
        check_projection(prism);
    }

    TEST(FisheyeViews, NormalConventionProbeUsesPhotographRaysBeyondHorizon) {
        const auto k = equidistant(95);
        const auto source = camera(k);
        source->prepare_undistortion();
        const auto lens = lfs::preprocessing::make_lens_camera(*source);
        const double theta = 92 * degree;
        const auto xy = Tensor::from_vector({128.f, float(128 - k.focal[1] * theta), 128.f, 128.f}, {2, 2}, Device::CUDA);
        auto rays = Tensor::empty({2, 3}, Device::CUDA);
        lfs::preprocessing::lens_rays_at(lens, xy.ptr<float>(), 2, rays.ptr<float>(), rays.stream());
        const auto sampled = rays.to_vector();
        const std::array expected{0., -std::sin(theta), std::cos(theta), 0., 0., 1.};
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(sampled[i], expected[i], 1e-6);
    }

    TEST(FisheyeViews, NarrowCroppedAndAngleLimitedLensesKeepSingleGrid) {
        auto k = equidistant(60);
        const auto narrow = camera(k);
        EXPECT_TRUE(plan_with_native_limit(*narrow).empty());
        EXPECT_EQ(narrow->camera_model_type(), CameraModelType::FISHEYE);
        EXPECT_FALSE(narrow->is_undistort_prepared());
        k = equidistant(100);
        k.resolution[1] = 180;
        EXPECT_TRUE(plan_with_native_limit(*camera(k)).empty());
        k.resolution[1] = 246;
        const auto cropped = plan_with_native_limit(*camera(k));
        ASSERT_EQ(cropped.size(), 5u);
        const auto& side = cropped[1].params;
        // The far edge is one degree inside the reach in this part of the layout.
        const double far_angle = 60 * degree + std::atan((side.dst_width - side.dst_cx) / side.dst_fx);
        EXPECT_NEAR(far_angle + degree, (k.resolution[1] - k.principal[1]) / k.focal[1], 1e-6);
        for (int term : {0, 3}) {
            k = equidistant(120);
            const int power = 2 * (term + 1);
            k.radial[term] = -1 / ((power + 1) * std::pow(84 * degree, power));
            const auto projected = gut_camera_model_test::project(k, {float(std::sin(83 * degree)), 0, float(std::cos(83 * degree)),
                                                                      float(std::sin(85 * degree)), 0, float(std::cos(85 * degree))});
            EXPECT_EQ(projected[2], 1);
            EXPECT_EQ(projected[5], 0);
            EXPECT_TRUE(plan_with_native_limit(*camera(k)).empty());
        }
    }

    // Fails if the planner caps the lens at the image radius over the focal length, which is the reach of an
    // equidistant lens only: a lens that compresses toward its rim reaches further than that.
    TEST(FisheyeViews, CompressingLensReachingTheHorizonGetsFiveViews) {
        // r = f sin(theta) to the ninth order: the image circle touches the border at 90 degrees.
        const OpenCvFisheyeCamera k{{128, 128}, {128, 128}, {256, 256}, {-1.f / 6, 1.f / 120, -1.f / 5040, 1.f / 362880}};
        const auto plan = plan_with_native_limit(*camera(k));
        ASSERT_EQ(plan.size(), 5u);
        for (const auto& face : plan) {
            const auto& p = face.params;
            for (int y = 0; y < p.dst_height; ++y)
                for (int x = 0; x < p.dst_width; ++x) {
                    const auto mapped = detail::source_position(x + .5f, y + .5f, p);
                    ASSERT_GE(mapped.x, -.5f);
                    ASSERT_LT(mapped.x, p.src_width - .5f);
                    ASSERT_GE(mapped.y, -.5f);
                    ASSERT_LT(mapped.y, p.src_height - .5f);
                }
        }
    }

    // Fails if the turning point of the lens is searched for from one Newton seed near 90 degrees: with k4 set, it
    // lands on a later turning point and the planner samples the fold of the lens between the two.
    TEST(FisheyeViews, LensLimitIsTheFirstTurningPoint) {
        // The radial slope 1 + 3 k1 t^2 + 5 k2 t^4 + 7 k3 t^6 + 9 k4 t^8 vanishes at 70 and at 89 degrees.
        const OpenCvFisheyeCamera k{{100, 100}, {128, 128}, {256, 256}, {-.19480130f, -.040408247f, .010150730f, .0019281994f}};
        const auto source = camera(k);
        EXPECT_NEAR(lfs::preprocessing::lens_max_angle(lfs::preprocessing::make_lens_camera(*source)), 70 * degree, 1e-4);
        EXPECT_TRUE(plan_with_native_limit(*source).empty());
    }

    TEST(FisheyeViews, ThresholdFollowsTheMostClippedAzimuth) {
        auto k = prism;
        std::vector<float> rays;
        const double theta = kFisheyeViewThresholdDegrees * degree;
        for (int i = 0; i < 2048; ++i) {
            const double phi = i * 2 * std::numbers::pi / 2048;
            rays.insert(rays.end(), {float(std::sin(theta) * std::cos(phi)), float(std::sin(theta) * std::sin(phi)),
                                     float(std::cos(theta))});
        }
        const auto points = gut_camera_model_test::project(k, rays);
        double scale = std::numeric_limits<double>::infinity();
        size_t limiting_ray = 0;
        for (size_t i = 0; i < rays.size() / 3; ++i) {
            ASSERT_EQ(points[3 * i + 2], 1);
            for (int axis = 0; axis < 2; ++axis) {
                const double delta = points[3 * i + axis] - k.principal[axis];
                if (delta == 0)
                    continue;
                const double border = delta > 0 ? k.resolution[axis] : 0;
                const double at_edge = (border - k.principal[axis]) / delta;
                if (at_edge < scale) {
                    scale = at_edge;
                    limiting_ray = i;
                }
            }
        }
        ASSERT_NE(limiting_ray % 512, 0u);
        for (const float ratio : {.999f, 1.001f}) {
            k.focal = {float(prism.focal[0] * scale * ratio), float(prism.focal[1] * scale * ratio)};
            EXPECT_EQ(plan_with_native_limit(*camera(k)).size(), ratio < 1 ? 5u : 0u);
        }
    }

    TEST(FisheyeViews, CamerasInheritPhotographPoseAndPriors) {
        const auto source = camera(distorted);
        source->precompute_undistortion();
        source->set_split(CameraSplit::Eval);
        source->set_mask_tensor(Tensor::ones({250, 256}, Device::CPU));
        const auto plan = plan_with_native_limit(*source);
        const auto views = make_camera_views(*source, plan, 51);
        ASSERT_EQ(views.size(), 5u);
        const std::array names{"_f", "_px", "_nx", "_py", "_ny"};
        for (int i = 0; i < 5; ++i) {
            const auto& v = *views[i];
            EXPECT_EQ(v.uid(), 51 + i);
            EXPECT_EQ(v.image_name(), "sub/lens" + std::string(names[i]) + ".png");
            EXPECT_EQ(photograph_image_name(v.image_name()), source->image_name());
            EXPECT_EQ(v.camera_id(), source->camera_id());
            EXPECT_EQ(v.split(), source->split());
            EXPECT_EQ(v.mask_path(), source->mask_path());
            EXPECT_EQ(v.raw_mask_tensor().data_ptr(), source->raw_mask_tensor().data_ptr());
            EXPECT_EQ(v.depth_path(), source->depth_path());
            EXPECT_EQ(v.normal_path(), source->normal_path());
            EXPECT_EQ(v.camera_model_type(), CameraModelType::PINHOLE);
            EXPECT_TRUE(lfs::training::depth_camera_projection(v).pinhole_ray_distance);
            EXPECT_FLOAT_EQ(v.focal_x(), std::max(distorted.focal[0], distorted.focal[1]));
            EXPECT_FLOAT_EQ(v.focal_y(), v.focal_x());
            const auto prior_rotation = normal_prior_rotation(v, false, {});
            ASSERT_TRUE(prior_rotation);
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    EXPECT_FLOAT_EQ((*prior_rotation)[3 * row + col], plan[i].params.view_rotation[3 * col + row]);
            const auto rotation = v.R().cpu().to_vector();
            const auto centre = v.cam_position().cpu().to_vector();
            const auto source_centre = source->cam_position().cpu().to_vector();
            for (int a = 0; a < 3; ++a) {
                EXPECT_NEAR(centre[a], source_centre[a], 1e-5);
                for (int b = 0; b < 3; ++b) {
                    double dot = 0;
                    for (int j = 0; j < 3; ++j)
                        dot += rotation[3 * a + j] * rotation[3 * b + j];
                    EXPECT_NEAR(dot, a == b ? 1 : 0, 1e-6);
                }
            }
        }
        const auto photos = lfs::training::prepare_training_prior_maps({}, views, {}, std::array{source});
        ASSERT_TRUE(photos);
        ASSERT_EQ(photos->size(), 1u);
        EXPECT_EQ(photos->front(), source);
    }
} // namespace
