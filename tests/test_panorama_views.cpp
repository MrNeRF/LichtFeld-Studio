/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/panorama_views.hpp"
#include "io/pipelined_image_loader.hpp"
#include "licht_test_support.hpp"
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <gtest/gtest.h>
#include <numbers>

namespace {
    using namespace lfs::core;
    constexpr float pi = std::numbers::pi_v<float>;

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
        const auto views = panorama_view_params(w, h);
        for (int i = 0; i < kPanoramaViewCount; ++i) {
            const auto& p = views[i];
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
        for (const auto& p : panorama_view_params(128, 64)) {
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
        const auto p = prepare_undistort_params(panorama_view_params(w, h)[4], w, h, 1, 64);
        const auto face = undistort_image(source, p, nullptr).cpu().to_vector();
        double squared = 0;
        for (float value : face)
            squared += (value - 0.5) * (value - 0.5);
        EXPECT_LT(std::sqrt(squared / face.size()), 0.10);
    }

    TEST(PanoramaUndistort, MaskFollowsImage) {
        const auto source = direction_panorama(128, 64).slice(0, 0, 1).contiguous();
        for (const auto& p : panorama_view_params(128, 64)) {
            const auto image = undistort_image(source, p, nullptr).cpu().to_vector();
            const auto mask = undistort_mask_area(source.squeeze(0), p, nullptr).cpu().to_vector();
            ASSERT_EQ(image.size(), mask.size());
            for (size_t i = 0; i < mask.size(); ++i)
                ASSERT_NEAR(mask[i], image[i], 0.001);
        }
    }

    TEST(PanoramaUndistort, MatchesAngularRateAtFaceCentre) {
        const auto p = panorama_view_params(3840, 1920)[0];
        EXPECT_EQ(p.dst_width, 1358);
        EXPECT_EQ(panorama_view_params(2048, 1024)[0].dst_width, 724);
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
        const auto views = make_panorama_views(panorama, 70);
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
            EXPECT_EQ(panorama_image_name(view.image_name()), panorama.image_name());
            EXPECT_EQ(view.image_path(), panorama.image_path());
            EXPECT_EQ(view.mask_path(), panorama.mask_path());
            EXPECT_EQ(view.raw_mask_tensor().data_ptr(), panorama.raw_mask_tensor().data_ptr());
            EXPECT_TRUE(view.has_alpha());
            ASSERT_EQ(view.sfm_observation_count(), 1u);
            EXPECT_EQ(view.sfm_observations().front().z, 5);
            EXPECT_TRUE(view.depth_path().empty());
            EXPECT_TRUE(view.normal_path().empty());
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
        const auto views = panorama_view_params(w, h);
        lfs::io::PipelinedImageLoader loader;
        lfs::io::ImageRequest request{};
        request.path = path;
        request.depth_path = path;
        request.aux_target_width = request.aux_target_height = views.front().dst_width;
        for (const int face : {0, 1, 0}) {
            request.params.undistort = request.undistort = &views[face];
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
