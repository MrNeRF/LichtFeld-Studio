/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera_views.hpp"
#include "core/image_io.hpp"
#include "io/pipelined_image_loader.hpp"
#include "licht_test_support.hpp"
#include "training/dataset.hpp"
#include "training/depth_anchor/depth_anchor_cache.hpp"
#include "training/normal_auto_generate.hpp"
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <gtest/gtest.h>
#include <numbers>

namespace {
    using namespace lfs::core;
    constexpr float pi = std::numbers::pi_v<float>;

    std::shared_ptr<Camera> prior_panorama(const std::filesystem::path& image_path, int width = 128, int height = 64) {
        return std::make_shared<Camera>(
            Tensor::from_vector({0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CPU),
            Tensor::zeros({3}, Device::CPU), 0, 0, 0, 0, Tensor{}, Tensor{},
            CameraModelType::EQUIRECTANGULAR, "source.png", image_path, std::filesystem::path{}, width, height, 0);
    }

    std::vector<PinholeView> panorama_plan(int width, int height) {
        return plan_camera_views(*prior_panorama({}, width, height), 0);
    }

    Tensor direction_panorama(const int width, const int height) {
        std::vector<float> pixels(3 * width * height);
        for (int y = 0; y < height; ++y) {
            const float lat = ((y + 0.5f) / height - 0.5f) * pi;
            for (int x = 0; x < width; ++x) {
                const float lon = ((x + 0.5f) / width - 0.5f) * (2 * pi);
                const glm::vec3 ray{std::cos(lat) * std::sin(lon), std::sin(lat), std::cos(lat) * std::cos(lon)};
                for (int c = 0; c < 3; ++c)
                    pixels[(c * height + y) * width + x] = 0.5f + 0.5f * ray[c];
            }
        }
        return Tensor::from_vector(pixels, {3, size_t(height), size_t(width)}, Device::CUDA);
    }

    TEST(PanoramaUndistort, EachPixelCarriesItsOwnRay) {
        constexpr int w = 256, h = 128;
        const auto source = direction_panorama(w, h);
        const auto views = panorama_plan(w, h);
        for (int i = 0; i < kPanoramaViewCount; ++i) {
            const auto& p = views[i].params;
            const auto pixels = undistort_image(source, p, nullptr).cpu().to_vector();
            float worst = 0;
            for (int y = 0; y < p.dst_height; ++y) {
                for (int x = 0; x < p.dst_width; ++x) {
                    const float u = (x + 0.5f - p.dst_cx) / p.dst_fx;
                    const float v = (y + 0.5f - p.dst_cy) / p.dst_fy;
                    const std::array<glm::vec3, 6> rays{{{1, v, -u}, {-1, v, u}, {-u, 1, v}, {u, -1, v}, {u, v, 1}, {-u, v, -1}}};
                    const auto ray = glm::normalize(rays[i]);
                    for (int c = 0; c < 3; ++c) {
                        const float value = pixels[(c * p.dst_height + y) * p.dst_width + x];
                        ASSERT_TRUE(std::isfinite(value));
                        worst = std::max(worst, std::abs(value - (0.5f + 0.5f * ray[c])));
                    }
                }
            }
            EXPECT_LT(worst, 0.001f) << i;
        }
    }

    TEST(PanoramaUndistort, ConstantSurvivesSeamsAndPoles) {
        const auto source = Tensor::full({1, 64, 128}, 0.625f, Device::CUDA);
        for (const auto& view : panorama_plan(128, 64)) {
            const auto& p = view.params;
            auto face = undistort_image(source, p, nullptr).cpu().to_vector();
            for (float value : face)
                ASSERT_NEAR(value, 0.625f, 2e-6f);
        }
    }

    TEST(PanoramaUndistort, AttenuatesFrequenciesAboveFaceNyquist) {
        constexpr int w = 2048, h = 1024;
        std::vector<float> pixels(w * h);
        for (int x = 0; x < w; ++x) {
            const float lon = ((x + 0.5f) / w - 0.5f) * (2 * pi);
            const float value = 0.5f + 0.5f * std::sin(60 * 2 * pi * lon);
            for (int y = 0; y < h; ++y)
                pixels[y * w + x] = value;
        }
        const auto source = Tensor::from_vector(pixels, {1, h, w}, Device::CUDA);
        const auto p = prepare_undistort_params(panorama_plan(w, h)[4].params, w, h, 1, 64);
        const auto face = undistort_image(source, p, nullptr).cpu().to_vector();
        double squared = 0;
        for (float value : face)
            squared += (value - 0.5) * (value - 0.5);
        EXPECT_LT(std::sqrt(squared / face.size()), 0.10);
    }

    TEST(PanoramaUndistort, MaskFollowsImage) {
        const auto source = direction_panorama(128, 64).slice(0, 0, 1).contiguous();
        for (const auto& view : panorama_plan(128, 64)) {
            const auto& p = view.params;
            const auto image = undistort_image(source, p, nullptr).cpu().to_vector();
            const auto mask = undistort_mask_area(source.squeeze(0), p, nullptr).cpu().to_vector();
            ASSERT_EQ(image.size(), mask.size());
            for (size_t i = 0; i < mask.size(); ++i)
                ASSERT_NEAR(mask[i], image[i], 0.001);
        }
    }

    TEST(PanoramaUndistort, MatchesAngularRateAtFaceCentre) {
        const auto p = panorama_plan(3840, 1920)[0].params;
        EXPECT_EQ(p.dst_width, 1358);
        EXPECT_EQ(panorama_plan(2048, 1024)[0].params.dst_width, 724);
        EXPECT_FLOAT_EQ(focal2fov(p.dst_fx, p.dst_width) * 180 / pi, kPanoramaViewFovDegrees);
    }

    TEST(PanoramaUndistort, CamerasShareCentreAndSourceMetadata) {
        lfs::test::licht::TemporaryDirectory temp;
        auto path = temp.path / "source.png";
        std::vector<uint8_t> pixels(128 * 63);
        ASSERT_TRUE(save_png(path, pixels.data(), 128, 63, 1, 8, 0));
        Camera panorama(Tensor::from_vector({0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CPU),
                        Tensor::from_vector({1.f, 2.f, 3.f}, {3}, Device::CPU), 1, 1, 128, 64, {}, {},
                        CameraModelType::EQUIRECTANGULAR, "dir/source.png", path, "mask.png", 256, 128, 9, 42,
                        "depth.png", "normal.png");
        panorama.set_has_alpha(true);
        panorama.set_split(CameraSplit::Eval);
        panorama.set_mask_tensor(Tensor::ones({63, 128}, Device::CPU));
        panorama.set_sfm_observations({{1, 2, 3, 4, 5}});
        const auto views = make_camera_views(panorama, plan_camera_views(panorama, 0), 70);
        ASSERT_EQ(views.size(), 6u);
        const std::array names{"px", "nx", "py", "ny", "pz", "nz"};
        const auto source_r = panorama.R().to_vector();
        const auto source_t = panorama.T().to_vector();
        for (int i = 0; i < 6; ++i) {
            const auto& view = *views[i];
            EXPECT_EQ(view.uid(), 70 + i);
            EXPECT_EQ(view.camera_id(), 42);
            EXPECT_EQ(view.split(), CameraSplit::Eval);
            EXPECT_EQ(view.image_name(), std::string("dir/source_") + names[i] + ".png");
            EXPECT_EQ(photograph_image_name(view.image_name()), panorama.image_name());
            EXPECT_EQ(view.image_path(), panorama.image_path());
            EXPECT_EQ(view.mask_path(), panorama.mask_path());
            EXPECT_EQ(view.raw_mask_tensor().data_ptr(), panorama.raw_mask_tensor().data_ptr());
            EXPECT_TRUE(view.has_alpha());
            ASSERT_EQ(view.sfm_observation_count(), 1u);
            EXPECT_EQ(view.sfm_observations().front().z, 5);
            EXPECT_EQ(view.depth_path(), panorama.depth_path());
            EXPECT_EQ(view.normal_path(), panorama.normal_path());
            EXPECT_EQ(view.camera_model_type(), CameraModelType::PINHOLE);
            EXPECT_FALSE(view.has_distortion());
            EXPECT_TRUE(view.is_undistort_prepared());
            const auto& p = view.undistort_params();
            EXPECT_TRUE(is_view(p));
            EXPECT_EQ(p.src_width, 256);
            EXPECT_EQ(p.dst_width, 46);
            EXPECT_EQ(view.image_height(), 45);
            EXPECT_FLOAT_EQ(view.FoVx(), 96 * pi / 180);
            EXPECT_EQ(view.cam_position().cpu().to_vector(), panorama.cam_position().cpu().to_vector());
            const auto r = view.R().cpu().to_vector();
            const auto t = view.T().cpu().to_vector();
            const auto w2c = view.world_view_transform().cpu().to_vector();
            for (int j = 0; j < 3; ++j) {
                float translation = 0;
                for (int k = 0; k < 3; ++k)
                    translation += p.view_rotation[3 * k + j] * source_t[k];
                EXPECT_EQ(t[j], translation);
                EXPECT_EQ(w2c[4 * j + 3], translation);
                for (int k = 0; k < 3; ++k) {
                    float dot = 0, rotation = 0;
                    for (int l = 0; l < 3; ++l) {
                        dot += r[3 * j + l] * r[3 * k + l];
                        rotation += p.view_rotation[3 * l + j] * source_r[3 * l + k];
                    }
                    EXPECT_EQ(dot, j == k ? 1 : 0);
                    EXPECT_EQ(r[3 * j + k], rotation);
                    EXPECT_EQ(w2c[4 * j + k], rotation);
                }
            }
            const auto scaled = prepare_undistort_params(p, 128, 63, 2, 20);
            EXPECT_EQ(scaled.dst_width, 20);
            EXPECT_EQ(scaled.dst_height, 20);
            EXPECT_FLOAT_EQ(scaled.src_fx, 128 / (2 * pi));
            EXPECT_FLOAT_EQ(scaled.src_fy, 63 / pi);
            EXPECT_FLOAT_EQ(scaled.dst_fx, p.dst_fx * (20.f / p.dst_width));
        }
    }

    TEST(PanoramaUndistort, EachRotationHasSeparateImageAndSidecarCacheEntries) {
        constexpr int w = 256, h = 128;
        lfs::test::licht::TemporaryDirectory temp;
        const auto path = temp.path / "source.png";
        std::vector<uint8_t> pixels(w * h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                pixels[y * w + x] = x < w / 2 ? 64 : 192;
        ASSERT_TRUE(save_png(path, pixels.data(), w, h, 1, 8, 0));
        const auto views = panorama_plan(w, h);
        lfs::io::PipelinedImageLoader loader;
        lfs::io::ImageRequest request{};
        request.path = path;
        request.depth_path = path;
        request.aux_target_width = request.aux_target_height = views.front().params.dst_width;
        for (const int face : {0, 1, 0}) {
            request.params.undistort = request.undistort = &views[face].params;
            ++request.sequence_id;
            loader.prefetch({request});
            auto ready = loader.get();
            ASSERT_TRUE(ready.error.empty()) << ready.error;
            ASSERT_TRUE(ready.depth.has_value());
            if (ready.depth_ready_event) {
                ASSERT_EQ(cudaEventSynchronize(ready.depth_ready_event), cudaSuccess);
                cudaEventDestroy(ready.depth_ready_event);
            }
            const float expected = face == 0 ? 192.f / 255 : 64.f / 255;
            const auto rgb = ready.tensor.to(DataType::Float32);
            const float scale = ready.tensor.dtype() == DataType::UInt8 ? 255.f : 1.f;
            EXPECT_NEAR(rgb.mean().item<float>() / scale, expected, 0.005);
            EXPECT_NEAR(ready.depth->mean().item<float>(), expected, 0.005);
        }
        EXPECT_GE(loader.get_stats().hot_path_hits, 2u);
    }

    TEST(PanoramaPriors, GeneratesOnceForThePhotographAndUpdatesEveryView) {
        using namespace lfs::training;
        lfs::test::licht::TemporaryDirectory temp;
        const auto images = temp.path / "images";
        std::filesystem::create_directory(images);
        const auto photo = prior_panorama(images / "source.png");
        std::vector<uint8_t> pixels(128 * 64);
        ASSERT_TRUE(save_png(photo->image_path(), pixels.data(), 128, 64, 1, 8, 0));
        const auto views = make_camera_views(*photo, plan_camera_views(*photo, 0), 10);
        param::TrainingParameters params;
        params.dataset.data_path = temp.path;
        params.optimization.undistort = true;
        params.optimization.use_depth_loss = params.optimization.use_normal_loss = true;
        params.optimization.depth_auto_generate = params.optimization.normal_auto_generate = true;
        params.optimization.depth_loss_weight = params.optimization.normal_loss_weight = 1;
        size_t jobs_seen = 0;
        const auto sources = prepare_training_prior_maps(
            params, views, views, std::array{photo},
            [&](std::span<const PriorMapJob> jobs, const PriorMapProgress&) {
                jobs_seen += jobs.size();
                for (const auto& job : jobs) {
                    EXPECT_EQ(job.image_path, photo->image_path());
                    for (const auto& path : {job.depth_output_path, job.normal_output_path}) {
                        std::filesystem::create_directories(path.parent_path());
                        EXPECT_TRUE(save_png(path, pixels.data(), 128, 64, 1, 8, 0));
                    }
                }
                return lfs::Status{}.into_expected();
            });
        ASSERT_TRUE(sources);
        EXPECT_EQ(*sources, std::vector{photo});
        EXPECT_EQ(jobs_seen, 1u);
        ASSERT_TRUE(photo->has_depth() && photo->has_normal());
        for (const auto& view : views) {
            EXPECT_EQ(view->depth_path(), photo->depth_path());
            EXPECT_EQ(view->normal_path(), photo->normal_path());
        }
    }

    TEST(PanoramaPriors, DecodesCameraAndWorldNormalFieldsInEveryViewFrame) {
        using namespace lfs::training;
        lfs::test::licht::TemporaryDirectory temp;
        const auto photo = prior_panorama(temp.path / "source.png");
        constexpr size_t w = 128, h = 64;
        const auto directions = direction_panorama(w, h).cpu().to_vector();
        std::vector<uint16_t> pixels(3 * w * h);
        ASSERT_TRUE(save_png(photo->image_path(), pixels.data(), w, h, 3, 16, 0));
        for (const bool world : {false, true}) {
            for (size_t i = 0; i < w * h; ++i) {
                glm::vec3 n;
                for (int c = 0; c < 3; ++c)
                    n[c] = 1 - 2 * directions[c * w * h + i];
                if (world)
                    n = {n.y, -n.x, n.z}; // Undo the photograph's quarter turn.
                n = {n.x, -n.y, -n.z};
                for (int c = 0; c < 3; ++c)
                    pixels[3 * i + c] = std::lround((0.5f + 0.5f * n[c]) * 65535);
            }
            photo->set_normal_path(temp.path / (world ? "world.png" : "camera.png"));
            ASSERT_TRUE(save_png(photo->normal_path(), pixels.data(), w, h, 3, 16, 0));
            const auto views = make_camera_views(*photo, plan_camera_views(*photo, 0), 10);
            auto dataset = std::make_shared<CameraDataset>(
                views, DatasetConfig{});
            PipelinedAuxiliaryImageConfig aux{.load_normals = true, .normal_flip_yz = true, .normal_world_space = world};
            PipelinedDataLoader loader(dataset, RandomSampler(views.size(), 13), {}, aux);
            for (size_t face = 0; face < views.size(); ++face) {
                auto example = loader.next();
                ASSERT_TRUE(example && example->normal);
                if (example->normal_ready_event) {
                    ASSERT_EQ(cudaEventSynchronize(example->normal_ready_event), cudaSuccess);
                    cudaEventDestroy(example->normal_ready_event);
                }
                auto& camera = *example->data.camera;
                const auto direct = camera.load_and_get_normal(1, 0, {.flip_yz = true, .world_space = world});
                const auto& p = camera.undistort_params();
                for (const auto& map : {*example->normal, direct}) {
                    const auto values = map.cpu().to_vector();
                    ASSERT_EQ(values.size(), 3u * p.dst_width * p.dst_height);
                    for (int y = 0; y < p.dst_height; ++y)
                        for (int x = 0; x < p.dst_width; ++x) {
                            const auto expected = -glm::normalize(glm::vec3{
                                (x + 0.5f - p.dst_cx) / p.dst_fx, (y + 0.5f - p.dst_cy) / p.dst_fy, 1});
                            for (int c = 0; c < 3; ++c)
                                ASSERT_NEAR(values[(c * p.dst_height + y) * p.dst_width + x], expected[c], 0.002f);
                        }
                }
            }
        }
    }

    TEST(PanoramaPriors, FitsAffineRayDistanceAndDifferentiatesTileDepth) {
        using namespace lfs::training;
        using namespace lfs::training::kernels;
        const auto photo = prior_panorama({});
        for (const auto& view : make_camera_views(*photo, plan_camera_views(*photo, 0), 10)) {
            const auto& p = view->undistort_params();
            const auto R = view->R().cpu().to_vector();
            const int w = p.dst_width - 8, h = p.dst_height - 4;
            const auto grid = depth_prior_ray_grid(*view, 7, 3);
            ASSERT_TRUE(grid);
            const auto& tile = *grid;
            ASSERT_FLOAT_EQ(tile.cx, p.dst_cx - 7);
            ASSERT_FLOAT_EQ(tile.cy, p.dst_cy - 3);
            const size_t count = static_cast<size_t>(w) * h;
            const float lo[]{-10, -10, -10}, hi[]{10, 10, 10};
            std::vector<float2> pairs;
            // Several concentric spheres make both affine coefficients observable.
            for (const float encoded : {0.125f, 0.5f, 0.875f}) {
                const float radius = 4 * encoded + 2;
                const auto prior = undistort_depth_area(Tensor::full({64, 128}, encoded, Device::CUDA), p, nullptr);
                std::vector<float> points;
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                        const auto local = radius * glm::normalize(glm::vec3{
                                                        (x + 0.5f - tile.cx) / tile.fx, (y + 0.5f - tile.cy) / tile.fy, 1});
                        for (int c = 0; c < 3; ++c)
                            points.push_back(R[c] * local.x + R[3 + c] * local.y + R[6 + c] * local.z);
                    }
                const auto xyz = Tensor::from_vector(points, {count, 3}, Device::CUDA);
                auto samples = collect_depth_anchor_samples(
                    xyz.ptr<float>(), count, view->world_view_transform_ptr(), p.dst_fx, p.dst_fy, p.dst_cx, p.dst_cy,
                    prior.ptr<float>(), p.dst_width, p.dst_height, 0.01f, lo, hi, nullptr, depth_camera_projection(*view));
                ASSERT_EQ(samples.size(), count);
                pairs.insert(pairs.end(), samples.begin(), samples.end());
            }
            const auto anchor = fit_depth_anchor_from_samples(pairs);
            ASSERT_TRUE(anchor.valid);
            ASSERT_EQ(anchor.model, 1);
            EXPECT_NEAR(anchor.scale, 4, 0.002);
            EXPECT_NEAR(anchor.shift, 2, 0.001);
            const auto alpha = Tensor::full({count}, 0.8f, Device::CUDA);
            auto dz = Tensor::empty({count}, Device::CUDA);
            auto da = Tensor::empty({count}, Device::CUDA);
            auto loss = Tensor::empty({1}, Device::CUDA);
            auto scratch = Tensor::empty({depth_loss_partial_count(count)}, Device::CUDA);
            std::vector<float> lengths(count), accum(count);
            for (size_t i = 0; i < count; ++i)
                lengths[i] = glm::length(glm::vec3{(i % w + 0.5f - tile.cx) / tile.fx,
                                                   (i / w + 0.5f - tile.cy) / tile.fy, 1});
            for (const float encoded : {0.125f, 0.5f, 0.875f}) {
                const float radius = 4 * encoded + 2;
                const auto prior = undistort_depth_area(Tensor::full({64, 128}, encoded, Device::CUDA), p, nullptr)
                                       .slice(0, 3, 3 + h)
                                       .slice(1, 7, 7 + w)
                                       .contiguous();
                const auto run = [&](const DepthRayGrid* grid) {
                    const auto z = Tensor::from_vector(accum, {count}, Device::CUDA);
                    launch_depth_loss(z.ptr<float>(), alpha.ptr<float>(), prior.ptr<float>(), dz.ptr<float>(),
                                      da.ptr<float>(), loss.ptr<float>(), scratch.ptr<float>(), w, h, 1, 0.2f,
                                      1.f / 255, &anchor, nullptr, nullptr, grid);
                    return loss.item<float>();
                };
                for (size_t i = 0; i < count; ++i)
                    accum[i] = 0.8f * radius / lengths[i];
                EXPECT_EQ(run(&tile), 0);
                EXPECT_GT(run(nullptr), 0.01f);

                for (size_t i = 0; i < count; ++i)
                    accum[i] *= i % 3 == 0 ? 1.1f : 0.95f;
                const float tile_loss = run(&tile);
                const auto tile_dz = dz.cpu().to_vector(), tile_da = da.cpu().to_vector();
                for (size_t i = 0; i < count; ++i)
                    accum[i] *= lengths[i];
                EXPECT_NEAR(run(nullptr), tile_loss, 2e-5);
                const auto radial_grad = dz.cpu().to_vector(), radial_alpha = da.cpu().to_vector();
                for (size_t i = 0; i < count; ++i) {
                    ASSERT_NEAR(tile_dz[i], lengths[i] * radial_grad[i], 1e-6);
                    ASSERT_NEAR(tile_da[i], radial_alpha[i], 1e-6);
                }
            }
        }
    }

    TEST(DepthPriors, PinholeAndLensCorrectionKeepCameraZLossAndGradients) {
        using namespace lfs::training;
        using namespace lfs::training::kernels;
        const auto z = Tensor::from_vector({1.5f, 3.f, 2.5f, 4.f}, {2, 2}, Device::CUDA);
        const auto alpha = Tensor::ones({2, 2}, Device::CUDA);
        const auto target = Tensor::from_vector({0.25f, 0.75f, 0.5f, 0.625f}, {2, 2}, Device::CUDA);
        auto dz = Tensor::empty_like(z), da = Tensor::empty_like(z);
        auto loss = Tensor::empty({1}, Device::CUDA);
        auto scratch = Tensor::empty({depth_loss_partial_count(4)}, Device::CUDA);
        const DepthAnchor anchor{.valid = true, .model = 1, .scale = 4, .shift = 1, .floor = 0.25f};
        for (const bool corrected : {false, true}) {
            Camera camera(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                          32, 32, 32, 32, corrected ? Tensor::from_vector({0.05f}, {1}, Device::CPU) : Tensor{},
                          {}, CameraModelType::PINHOLE, "source.png", {}, {}, 64, 64, 0);
            if (corrected)
                camera.prepare_undistortion();
            EXPECT_EQ(camera.is_undistort_prepared(), corrected);
            EXPECT_FALSE(depth_camera_projection(camera).pinhole_ray_distance);
            const auto grid = depth_prior_ray_grid(camera, 3, 2);
            EXPECT_FALSE(grid);
            launch_depth_loss(z.ptr<float>(), alpha.ptr<float>(), target.ptr<float>(), dz.ptr<float>(), da.ptr<float>(),
                              loss.ptr<float>(), scratch.ptr<float>(), 2, 2, 1, 0, 0, &anchor, nullptr, nullptr,
                              grid ? &*grid : nullptr);
            EXPECT_NEAR(loss.item<float>(), 0.043123675f, 2e-6);
            EXPECT_NEAR(dz.cpu().to_vector()[0], -0.104663450f, 2e-6);
            EXPECT_NEAR(da.cpu().to_vector()[3], -0.026876469f, 2e-6);
        }
    }

    TEST(UndistortResampling, PreservesRadialAndFisheyeSamples) {
        std::vector<float> pixels(41 * 23);
        for (size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = ((i * 37 + 11) % 251) / 256.f;
        const auto source = Tensor::from_vector(pixels, {1, 23, 41}, Device::CUDA);
        constexpr std::array indices{0, 30, 97, 294, 588};
        // Reference samples from the original forward kernels at borders and in the interior.
        constexpr float expected[][5] = {{0.072956346f, 0.865408838f, 0.853603005f, 0.465974718f, 0.856731296f},
                                         {0.853362024f, 0.383154958f, 0.241991267f, 0.465973735f, 0.063703813f}};
        for (int i = 0; i < 2; ++i) {
            UndistortParams p{
                .src_fx = 24,
                .src_fy = 21,
                .src_cx = 20.5f,
                .src_cy = 11.5f,
                .dst_fx = 16.5f,
                .dst_fy = 15,
                .dst_cx = 15.5f,
                .dst_cy = 9.5f,
                .src_width = 41,
                .src_height = 23,
                .dst_width = 31,
                .dst_height = 19,
                .model_type = i == 0 ? CameraModelType::PINHOLE : CameraModelType::FISHEYE,
                .distortion = {-.11f, .03f, -.005f},
                .num_distortion = 3,
            };
            const auto output = undistort_image(source, p, nullptr).cpu().to_vector();
            for (size_t k = 0; k < indices.size(); ++k)
                EXPECT_NEAR(output[indices[k]], expected[i][k], 3.e-6f);
        }
    }
} // namespace
