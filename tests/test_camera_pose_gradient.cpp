/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The FastGS camera gradient (grad_w2c, grad_cam_position) chained to a pose offset against finite differences of
// renders from the offset camera.

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include "io/formats/colmap.hpp"
#include "licht_test_support.hpp"
#include "training/components/camera_pose_refinement.hpp"
#include "training/kernels/camera_pose_math.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <numbers>
#include <random>
#include <vector>

namespace {

    using namespace lfs::core;
    using namespace lfs::training;
    namespace cp = lfs::training::camera_pose;
    using Pose = std::array<float, 16>;

    constexpr int kWidth = 64;
    constexpr int kHeight = 48;

    bool cuda_available() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    // Splats of about one scene unit: no pixel of the image crosses the alpha cutoff when the camera moves.
    constexpr float kWideSplats = 2.2f;

    // Eight splats of different size, orientation and view-dependent colour, about three units in front, their
    // log-scales raised by log_scale_offset.
    SplatData make_scene(const float log_scale_offset = 0.0f) {
        constexpr size_t n = 8;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> means, sh0, shN, scales, rotations, opacities;
        for (size_t i = 0; i < n; ++i) {
            means.insert(means.end(), {0.6f * u(rng), 0.45f * u(rng), 0.5f * u(rng)});
            sh0.insert(sh0.end(), {u(rng), u(rng), u(rng)});
            for (int k = 0; k < 9; ++k)
                shN.push_back(0.4f * u(rng));
            scales.insert(scales.end(), {-2.2f + log_scale_offset + 0.4f * u(rng), -2.0f + log_scale_offset + 0.4f * u(rng),
                                         -2.6f + log_scale_offset + 0.3f * u(rng)});
            rotations.insert(rotations.end(), {1.0f, 0.4f * u(rng), 0.4f * u(rng), 0.4f * u(rng)});
            opacities.push_back(0.5f + u(rng));
        }
        return SplatData(1,
                         Tensor::from_vector(means, {n, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(sh0, {n, size_t{1}, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(shN, {n, size_t{3}, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(scales, {n, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(rotations, {n, size_t{4}}, Device::CUDA),
                         Tensor::from_vector(opacities, {n, size_t{1}}, Device::CUDA),
                         1.0f);
    }

    Pose start_pose() {
        const float phi[3] = {0.05f, -0.08f, 0.03f};
        float E[9];
        cp::exp_so3(phi, E);
        return {E[0], E[1], E[2], 0.1f, E[3], E[4], E[5], -0.05f, E[6], E[7], E[8], 3.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    }

    Pose offset_pose(const std::array<float, 6>& off) {
        const Pose start = start_pose();
        Pose w2c;
        float centre[3];
        cp::compose(start.data(), off.data(), w2c.data(), centre);
        return w2c;
    }

    Camera make_camera(const Pose& w2c) {
        auto R = Tensor::from_vector({w2c[0], w2c[1], w2c[2], w2c[4], w2c[5], w2c[6], w2c[8], w2c[9], w2c[10]},
                                     {3, 3}, Device::CPU);
        auto T = Tensor::from_vector({w2c[3], w2c[7], w2c[11]}, {3}, Device::CPU);
        return Camera(R, T, 70.0f, 70.0f, kWidth / 2.0f, kHeight / 2.0f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                      "pose_gradient", "", std::filesystem::path{}, kWidth, kHeight, 0);
    }

    // Low-frequency loss weights [3, H, W]. With per-pixel random weights the finite difference is dominated by
    // pixels crossing the 1/255 alpha cutoff, which the rasterizer's gradient leaves out by design.
    std::vector<float> weights() {
        std::vector<float> w(3 * kWidth * kHeight);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < kHeight; ++y)
                for (int x = 0; x < kWidth; ++x)
                    w[(c * kHeight + y) * kWidth + x] =
                        static_cast<float>(std::sin(2.0 * std::numbers::pi * (x + 0.5) / kWidth + 1.3 * c) *
                                           std::cos(2.0 * std::numbers::pi * (y + 0.5) / kHeight - 0.7 * c));
        return w;
    }

    double dot(const Tensor& image, const std::vector<float>& w) {
        const auto values = image.cpu().contiguous().to_vector();
        double sum = 0.0;
        for (size_t i = 0; i < values.size(); ++i)
            sum += double(values[i]) * w[i];
        return sum;
    }

    // L(offset) = <image, image_w> + <normal, normal_w> rendered from compose(start, offset).
    double render_loss(const std::array<float, 6>& off, const std::vector<float>& image_w,
                       const std::vector<float>& normal_w, const float log_scale_offset) {
        auto model = make_scene(log_scale_offset);
        auto camera = make_camera(offset_pose(off));
        auto bg = Tensor::full({3}, 0.3f, Device::CUDA);
        const bool normals = !normal_w.empty();
        auto forward = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, normals, true, 1.0f, false);
        EXPECT_TRUE(forward.has_value());
        if (!forward)
            return 0.0;
        double loss = image_w.empty() ? 0.0 : dot(forward->first.image, image_w);
        if (normals)
            loss += dot(forward->first.normal, normal_w);
        forward->second.release_forward_context();
        return loss;
    }

    void expect_matches_finite_differences(const std::array<float, 6>& analytic, const std::vector<float>& image_w,
                                           const std::vector<float>& normal_w, const float log_scale_offset = 0.0f) {
        constexpr float h = 1e-3f;
        std::array<double, 6> numeric{};
        double scale = 0.0;
        for (int k = 0; k < 6; ++k) {
            std::array<float, 6> plus{}, minus{};
            plus[k] = h;
            minus[k] = -h;
            numeric[k] = (render_loss(plus, image_w, normal_w, log_scale_offset) -
                          render_loss(minus, image_w, normal_w, log_scale_offset)) /
                         (2.0 * h);
            scale = std::max(scale, std::abs(numeric[k]));
        }
        ASSERT_GT(scale, 1e-3) << "the test scene must respond to the camera pose";
        for (int k = 0; k < 6; ++k)
            EXPECT_NEAR(analytic[k], numeric[k], 0.03 * scale + 0.03 * std::abs(numeric[k])) << "component " << k;
    }

    std::array<float, 6> offset_gradient(const Tensor& grad_w2c, const Tensor& grad_cam_position) {
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto g_w2c = grad_w2c.cpu().to_vector();
        const auto g_c = grad_cam_position.cpu().to_vector();
        const Pose start = start_pose();
        const std::array<float, 6> zero{};
        std::array<float, 6> out{};
        cp::offset_grad(start.data(), zero.data(), g_w2c.data(), g_c.data(), out.data());
        return out;
    }

    // The offset gradient of <image, image_w> from fast_rasterize_camera_backward.
    std::array<float, 6> frozen_image_gradient(SplatData& model, const std::vector<float>& image_w) {
        auto camera = make_camera(start_pose());
        auto bg = Tensor::full({3}, 0.3f, Device::CUDA);
        auto forward = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, false, true, 1.0f, false);
        EXPECT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
        if (!forward)
            return {};
        auto grad_image = Tensor::from_vector(image_w, {size_t{3}, size_t{kHeight}, size_t{kWidth}}, Device::CUDA);
        auto grad_w2c = Tensor::zeros({16}, Device::CUDA);
        auto grad_cam_position = Tensor::zeros({3}, Device::CUDA);
        fast_rasterize_camera_backward(forward->second, grad_image, model, grad_w2c.ptr<float>(),
                                       grad_cam_position.ptr<float>());
        return offset_gradient(grad_w2c, grad_cam_position);
    }

} // namespace

TEST(CameraPoseGradientTest, FrozenImageGradientMatchesFiniteDifferences) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    const auto image_w = weights();
    auto model = make_scene();
    const auto means_before = model.means().cpu().to_vector();
    const auto opacity_before = model.opacity_raw().cpu().to_vector();
    const auto analytic = frozen_image_gradient(model, image_w);

    EXPECT_EQ(model.means().cpu().to_vector(), means_before) << "the camera backward must not move the model";
    EXPECT_EQ(model.opacity_raw().cpu().to_vector(), opacity_before);
    expect_matches_finite_differences(analytic, image_w, {});
}

// The camera position enters only through the colours' view direction (grad_cam_position), and only the translation
// moves it. With the small splats that term stays inside the tolerance; with wide splats no pixel crosses the alpha
// cutoff, the finite differences agree to 1% of the largest component and leaving the term out fails.
TEST(CameraPoseGradientTest, FrozenImageGradientOfWideSplatsMatchesFiniteDifferences) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    const auto image_w = weights();
    auto model = make_scene(kWideSplats);
    expect_matches_finite_differences(frozen_image_gradient(model, image_w), image_w, {}, kWideSplats);
}

// Wide splats here too: with the small ones the finite differences of the normals drift with the step size.
TEST(CameraPoseGradientTest, TrainingBackwardNormalGradientMatchesFiniteDifferences) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    const auto normal_w = weights();
    auto model = make_scene(kWideSplats);
    auto camera = make_camera(start_pose());
    auto bg = Tensor::full({3}, 0.3f, Device::CUDA);
    auto forward = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, true, true, 1.0f, false);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer optimizer(model, cfg);
    optimizer.allocate_gradients();
    optimizer.zero_grad(0);
    auto grad_w2c = Tensor::zeros({16}, Device::CUDA);
    auto grad_cam_position = Tensor::zeros({3}, Device::CUDA);
    FastGSFusedExtraGradients extra;
    extra.grad_w2c = grad_w2c.ptr<float>();
    extra.grad_cam_position = grad_cam_position.ptr<float>();
    auto grad_normal = Tensor::from_vector(normal_w, {size_t{3}, size_t{kHeight}, size_t{kWidth}}, Device::CUDA);
    fast_rasterize_backward(forward->second, Tensor::zeros_like(forward->first.image), model, optimizer, {}, {},
                            DensificationType::None, 1, extra, {}, grad_normal);
    const auto analytic = offset_gradient(grad_w2c, grad_cam_position);

    expect_matches_finite_differences(analytic, {}, normal_w, kWideSplats);
}

namespace {
    void upload_pose_gradient(CameraPoseOffsets& poses, const Pose& gradient, const std::array<float, 3>& centre_gradient) {
        ASSERT_EQ(cudaMemcpy(poses.grad_w2c(), gradient.data(), sizeof(gradient), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(poses.grad_cam_position(), centre_gradient.data(), sizeof(centre_gradient), cudaMemcpyHostToDevice), cudaSuccess);
    }

    void expect_drawn_poses(const std::vector<std::shared_ptr<Camera>>& views, const float* owner_pose,
                            const float* centre, bool committed) {
        const auto shared_centre = views.front()->cam_position().cpu().to_vector();
        for (const auto& camera : views) {
            const auto& rotation = camera->undistort_params().view_rotation;
            const auto device = camera->world_view_transform().cpu().to_vector();
            const auto r = camera->R().to_vector(), t = camera->T().to_vector();
            EXPECT_EQ(camera->cam_position().cpu().to_vector(), shared_centre);
            for (int i = 0; i < 3; ++i) {
                EXPECT_NEAR(shared_centre[i], centre[i], 2e-6f);
                for (int j = 0; j < 4; ++j) {
                    double expected = 0;
                    for (int k = 0; k < 3; ++k)
                        expected += double(rotation[3 * k + i]) * owner_pose[4 * k + j];
                    EXPECT_NEAR(device[4 * i + j], expected, 2e-6f);
                    if (committed)
                        EXPECT_NEAR(j == 3 ? t[i] : r[3 * i + j], expected, 2e-6f);
                }
            }
        }
    }
} // namespace

TEST(CameraPoseOffsetsTest, DrawnViewsShareOneMovingPhotograph) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    const auto ordinary = std::make_shared<Camera>(make_camera(start_pose()));
    for (auto model : {CameraModelType::EQUIRECTANGULAR, CameraModelType::FISHEYE}) {
        const auto photograph = std::make_shared<Camera>(
            ordinary->R(), ordinary->T(), 14, 14, 32, 24, Tensor{}, Tensor{}, model,
            "photo.png", "photo.png", std::filesystem::path{}, 64, 48, 4);
        const auto plan = plan_camera_views(*photograph, std::numbers::pi_v<float>);
        const auto views = make_camera_views(*photograph, plan, 31);
        ASSERT_EQ(views.size(), model == CameraModelType::FISHEYE ? 5u : 6u);
        std::vector<std::shared_ptr<Camera>> draws{ordinary};
        draws.insert(draws.end(), views.begin(), views.end());
        auto groups = group_camera_views(draws, std::array{ordinary, photograph});
        ASSERT_TRUE(groups);
        CameraPoseOffsets poses(std::move(*groups));
        const auto start = photograph->world_view_transform().cpu().to_vector();
        const auto start_centre = photograph->cam_position().cpu().to_vector();
        upload_pose_gradient(poses, Pose{1, 2, -3, 4, 1, -1, 4, 2, -2, 4, 0, 1}, {1, -2, 3});
        cudaStream_t stream;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (int draw : {3, 1}) {
            poses.step(*views[draw], 0.015f, 0.031f, stream);
            const auto state = poses.download();
            ASSERT_EQ(state.steps.size(), 2u);
            EXPECT_EQ(state.steps[0], 0);
            EXPECT_EQ(state.steps[1], draw == 3 ? 1 : 2);
            Pose refined;
            std::array<float, 3> centre;
            cp::compose(start.data(), state.offsets.data() + 6, refined.data(), centre.data());
            expect_drawn_poses(views, refined.data(), centre.data(), false);
        }
        poses.commit();
        const auto pose = photograph->world_view_transform().cpu().to_vector();
        const auto centre = photograph->cam_position().cpu().to_vector();
        const auto r = photograph->R().to_vector(), t = photograph->T().to_vector();
        EXPECT_NE(pose, start);
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(t[i], pose[4 * i + 3]);
            for (int j = 0; j < 3; ++j)
                EXPECT_EQ(r[3 * i + j], pose[4 * i + j]);
        }
        EXPECT_EQ(views.front()->cam_position().cpu().to_vector(), centre);
        expect_drawn_poses(views, pose.data(), centre.data(), true);
        poses.restore_start_poses();
        EXPECT_EQ(photograph->world_view_transform().cpu().to_vector(), start);
        expect_drawn_poses(views, start.data(), start_centre.data(), true);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
}

TEST(CameraPoseOffsetsTest, HeldOutPhotographIsFittedAndRestoredThroughItsViews) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    lfs::test::licht::TemporaryDirectory temp;
    const auto path = temp.path / "photo.png";
    constexpr int w = 96, h = 48;
    std::vector<uint8_t> pixels(w * h * 3, 128);
    ASSERT_TRUE(save_png(path, pixels.data(), w, h, 3, 8, 0));
    const auto base = make_camera(start_pose());
    const auto photo = std::make_shared<Camera>(base.R(), base.T(), 1, 1, w / 2, h / 2, Tensor{}, Tensor{},
                                                CameraModelType::EQUIRECTANGULAR, "photo.png", path,
                                                std::filesystem::path{}, w, h, 1);
    const auto views = make_camera_views(*photo, plan_camera_views(*photo, 0), 10);
    auto groups = group_camera_views(views, std::array{photo});
    ASSERT_TRUE(groups);
    const auto initial = photo->world_view_transform().cpu().to_vector();
    auto scene = make_scene();
    auto background = Tensor::zeros({3}, Device::CUDA);
    auto& cache = lfs::io::CacheLoader::getInstance(true);
    cache.reset_cache();
    cache.update_cache_params(true, static_cast<int>(views.size()), lfs::io::DEFAULT_MIN_FREE_GB,
                              lfs::io::DEFAULT_MIN_FREE_MEMORY_RATIO, false, 1);
    RefittedCameraPoses refit(*groups, scene, background, {.lambda_dssim = 0.0f});
    refit.apply();
    const auto fitted = photo->world_view_transform().cpu().to_vector();
    EXPECT_NE(fitted, initial);
    expect_drawn_poses(views, fitted.data(), photo->cam_position().cpu().to_vector().data(), true);
    refit.restore();
    EXPECT_EQ(photo->world_view_transform().cpu().to_vector(), initial);
    expect_drawn_poses(views, initial.data(), photo->cam_position().cpu().to_vector().data(), true);
    cache.reset_cache();
}

TEST(CameraPoseOffsetsTest, TableAndExportUseTheScenePhotograph) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    lfs::test::licht::TemporaryDirectory temp;
    const auto base = make_camera(start_pose());
    const auto photo = std::make_shared<Camera>(base.R(), base.T(), 1, 1, 48, 24, Tensor{}, Tensor{},
                                                CameraModelType::EQUIRECTANGULAR, "photo.png", temp.path / "photo.png",
                                                std::filesystem::path{}, 96, 48, 0, 1);
    lfs::core::Scene scene;
    scene.addCamera(photo->image_name(), NULL_NODE, photo);
    const auto views = make_camera_views(*photo, plan_camera_views(*photo, 0), 10);
    auto groups = group_camera_views(views, scene.getAllCameras());
    ASSERT_TRUE(groups);
    auto fit = CameraPoseRefinement::create(std::move(*groups), 10);
    ASSERT_TRUE(fit);
    FastGSFusedExtraGradients extra{};
    (*fit)->bind_gradients(extra, 2);
    const Pose gradient{1, -2, 1, 3, 3, 2, -1, 1, 1, 0, 1, -2};
    ASSERT_EQ(cudaMemcpy(extra.grad_w2c, gradient.data(), sizeof(gradient), cudaMemcpyHostToDevice), cudaSuccess);
    (*fit)->step(*views[2], 2, nullptr);
    (*fit)->step(*views[3], 3, nullptr);
    (*fit)->commit();
    EXPECT_NE((*fit)->summary(3).find("1 of 1 cameras updated"), std::string::npos);
    ASSERT_TRUE((*fit)->write_table(temp.path / "pose_refinement.tsv"));
    std::ifstream table(temp.path / "pose_refinement.tsv");
    std::string line;
    ASSERT_TRUE(std::getline(table, line));
    ASSERT_TRUE(std::getline(table, line));
    EXPECT_TRUE(line.starts_with("photo.png\t2\t"));
    EXPECT_FALSE(std::getline(table, line));

    std::ofstream(temp.path / "cameras.txt") << "1 EQUIRECTANGULAR 96 48 48 24\n";
    std::ofstream(temp.path / "images.txt") << "1 1 0 0 0 0 0 0 1 photo.png\n\n";
    std::ofstream(temp.path / "points3D.txt") << "1 0 0 1 0 0 0 0\n";
    const auto output = temp.path / "cameras_refined";
    const auto written_scene = write_scene_cameras_colmap(scene, temp.path, output);
    ASSERT_TRUE(written_scene) << lfs::format_for_developer(written_scene.error());
    const auto exported = lfs::io::read_colmap_cameras_only(output);
    ASSERT_TRUE(exported) << exported.error().format();
    const auto& cameras = std::get<0>(*exported);
    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(cameras.front()->image_name(), photo->image_name());
    const auto written = cameras.front()->world_view_transform().cpu().to_vector();
    const auto committed = photo->world_view_transform().cpu().to_vector();
    for (size_t i = 0; i < committed.size(); ++i)
        EXPECT_NEAR(written[i], committed[i], 2e-6f);
}

TEST(CameraPoseOffsetsTest, RotatedDrawMatchesHandMappedGradient) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    auto owner = std::make_shared<Camera>(make_camera(start_pose()));
    auto ordinary = std::make_shared<Camera>(make_camera(start_pose()));
    UndistortParams p{};
    p.src_width = p.dst_width = kWidth;
    p.src_height = p.dst_height = kHeight;
    p.src_fx = p.src_fy = p.dst_fx = p.dst_fy = 70;
    p.src_cx = p.dst_cx = kWidth / 2;
    p.src_cy = p.dst_cy = kHeight / 2;
    p.model_type = CameraModelType::FISHEYE;
    p.view = true;
    std::mt19937 rng(49);
    std::uniform_real_distribution<float> uniform(-1, 1);
    const float axis_angle[3]{uniform(rng), uniform(rng), uniform(rng)};
    cp::exp_so3(axis_angle, p.view_rotation);
    auto view = owner->make_pinhole_view(p, 7, "draw");
    CameraPoseOffsets grouped({{owner, {view}}}), reference({{ordinary, {ordinary}}});
    Pose gradient{}, mapped{};
    std::array<float, 3> centre_gradient{};
    // At this scale the first Adam update depends on gradient magnitude as well as sign.
    for (float& v : gradient)
        v = 1e-8f * uniform(rng);
    for (float& v : centre_gradient)
        v = 1e-8f * uniform(rng);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k)
                mapped[4 * i + j] += p.view_rotation[3 * i + k] * gradient[4 * k + j];
    upload_pose_gradient(grouped, gradient, centre_gradient);
    upload_pose_gradient(reference, mapped, centre_gradient);
    grouped.step(*view, 0.015f, 0.031f, nullptr);
    reference.step(*ordinary, 0.015f, 0.031f, nullptr);
    const auto actual = grouped.download().offsets, expected = reference.download().offsets;
    ASSERT_EQ(actual.size(), 6u);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(actual[i], expected[i], 1e-8);
}

TEST(CameraPoseOffsetsTest, OrdinaryCameraMatchesOriginalAdamAndCommit) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    const auto cam = std::make_shared<Camera>(make_camera(
        Pose{0, -1, 0, 0.25f, 1, 0, 0, -0.5f, 0, 0, 1, 2, 0, 0, 0, 1}));
    CameraPoseOffsets poses({{cam, {cam}}});
    const std::array<float, 16> grad{0.5f, 0.25f, -0.5f, 0.75f, -0.25f, 1, 0.25f, -1, 0.75f, 0, -0.25f, 0.5f};
    const std::array<float, 3> centre_grad{0.25f, -0.5f, 0.125f};
    upload_pose_gradient(poses, grad, centre_grad);
    poses.step(*cam, 0.015625f, 0.03125f, nullptr);
    // Float bit patterns captured from the original single-camera step and host commit.
    const auto bits = [](const std::vector<float>& values) {
        std::vector<uint32_t> out;
        for (float value : values)
            out.push_back(std::bit_cast<uint32_t>(value));
        return out;
    };
    const std::vector<uint32_t> offsets{
        0x3c7ffe0du, 0x3c7ffe0bu, 0xbc7ffe0bu, 0x3cfffe0bu, 0xbcfffe09u, 0x3cfffe0au};
    const std::vector<uint32_t> device_pose{
        0xbc7df60fu, 0xbf7ff001u, 0xbc80faf2u, 0x3e47a533u,
        0x3f7ff001u, 0xbc80faf2u, 0x3c7df611u, 0xbede8dcdu,
        0xbc80faf3u, 0xbc7df60fu, 0x3f7ff001u, 0x3ffd513bu,
        0u, 0u, 0u, 0x3f800000u};
    EXPECT_EQ(bits(poses.download().offsets), offsets);
    EXPECT_EQ(bits(cam->world_view_transform().cpu().to_vector()), device_pose);
    poses.commit();
    const std::vector<uint32_t> rotation{
        0xbc7df621u, 0xbf7ff001u, 0xbc80fafbu,
        0x3f7ff001u, 0xbc80fafbu, 0x3c7df623u,
        0xbc80fafcu, 0xbc7df621u, 0x3f7ff001u};
    const std::vector<uint32_t> translation{0x3e47a532u, 0xbede8dccu, 0x3ffd513cu};
    EXPECT_EQ(bits(cam->R().to_vector()), rotation);
    EXPECT_EQ(bits(cam->T().to_vector()), translation);
}
