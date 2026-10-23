/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "io/pipelined_image_loader.hpp"
#include "training/kernels/depth_loss.hpp"
#include "training/las_prior_sidecar.hpp"

#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;

    class TempDir {
    public:
        TempDir() {
            path_ = fs::temp_directory_path() /
                    ("lfs_las_priors_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            fs::create_directories(path_);
        }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(path_, ec);
        }
        [[nodiscard]] const fs::path& path() const { return path_; }

    private:
        fs::path path_;
    };

    void write_text(const fs::path& path, const std::string& text) {
        std::ofstream file(path, std::ios::binary);
        file << text;
    }

    bool cuda_available() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

} // namespace

TEST(LasPriorSidecar, AbsentFileYieldsNothing) {
    TempDir dir;
    EXPECT_FALSE(lfs::training::read_las_prior_sidecar(dir.path()).has_value());
}

TEST(LasPriorSidecar, ReadsGeosunSupervisionAndAppliesDefaults) {
    TempDir dir;
    write_text(dir.path() / "supervision.json",
               R"({"depth_format": "uint16_png_mm", "depth_type": "ray_distance",
                   "normal_frame": "opencv_camera_facing_camera", "invalid_normal_rgb": [0, 0, 0]})");
    const auto sidecar = lfs::training::read_las_prior_sidecar(dir.path());
    ASSERT_TRUE(sidecar.has_value());
    EXPECT_TRUE(sidecar->depth_millimetres);
    EXPECT_TRUE(sidecar->ray_distance);
    EXPECT_TRUE(sidecar->normals_camera_opencv);

    lfs::core::param::OptimizationParameters opt;
    const auto changes = lfs::training::apply_las_prior_defaults(opt, *sidecar);
    EXPECT_EQ(changes.size(), 4u);
    EXPECT_EQ(opt.depth_loss_mode, "metric");
    EXPECT_FALSE(opt.depth_auto_generate);
    EXPECT_EQ(opt.normal_loss_space, lfs::core::param::NormalLossSpace::CameraOpenCV);
    EXPECT_FALSE(opt.normal_auto_generate);
}

TEST(LasPriorSidecar, KeepsExplicitChoices) {
    lfs::core::param::OptimizationParameters opt;
    opt.depth_loss_mode = "ssi-depth";
    opt.normal_loss_space = lfs::core::param::NormalLossSpace::World;
    opt.depth_auto_generate = false;
    opt.normal_auto_generate = false;
    const auto changes = lfs::training::apply_las_prior_defaults(
        opt, lfs::training::LasPriorSidecar{.depth_millimetres = true, .normals_camera_opencv = true});
    EXPECT_TRUE(changes.empty());
    EXPECT_EQ(opt.depth_loss_mode, "ssi-depth");
    EXPECT_EQ(opt.normal_loss_space, lfs::core::param::NormalLossSpace::World);
}

TEST(LasPriorSidecar, IgnoresUnrelatedOrBrokenJson) {
    TempDir dir;
    write_text(dir.path() / "supervision.json", R"({"depth_format": "png8"})");
    EXPECT_FALSE(lfs::training::read_las_prior_sidecar(dir.path()).has_value());
    write_text(dir.path() / "supervision.json", "{not json");
    EXPECT_FALSE(lfs::training::read_las_prior_sidecar(dir.path()).has_value());
}

TEST(MetricDepthLoss, MatchingRenderHasNoLossAndFartherRenderIsPulledIn) {
    if (!cuda_available()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int height = 16;
    constexpr int width = 16;
    constexpr float step = 1.0f / 65535.0f;
    std::vector<float> target_values(height * width);
    std::vector<float> metres(height * width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            const uint16_t mm = static_cast<uint16_t>(1500 + 40 * x + 25 * y);
            // The last column carries no scan depth.
            target_values[idx] = x == width - 1 ? 0.0f : mm * step;
            metres[idx] = mm * 0.001f;
        }
    }
    const auto anchor = lfs::training::kernels::metric_depth_anchor(0.001f / step);
    ASSERT_TRUE(anchor.valid);

    const auto target = Tensor::from_vector(target_values, {size_t{height}, size_t{width}}, Device::CUDA);
    const auto alpha = Tensor::ones({size_t{height}, size_t{width}}, Device::CUDA);
    const auto run = [&](const float render_scale) {
        std::vector<float> render(metres);
        for (auto& v : render) {
            v *= render_scale;
        }
        const auto depth = Tensor::from_vector(render, {size_t{height}, size_t{width}}, Device::CUDA);
        const cudaStream_t stream = depth.stream();
        auto loss = Tensor::empty({size_t{1}}, Device::CUDA);
        auto grad_depth = Tensor::empty(depth.shape(), Device::CUDA);
        auto grad_alpha = Tensor::empty(alpha.shape(), Device::CUDA);
        auto partials = Tensor::empty(
            {lfs::training::kernels::depth_loss_partial_count(depth.numel())}, Device::CUDA);
        for (auto* t : {&loss, &grad_depth, &grad_alpha, &partials}) {
            t->set_stream(stream);
        }
        lfs::training::kernels::launch_depth_loss(
            depth.ptr<float>(), alpha.ptr<float>(), target.ptr<float>(),
            grad_depth.ptr<float>(), grad_alpha.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(),
            width, height, 1.0f, 1.0f, step, &anchor, stream);
        return std::pair{loss.item<float>(), grad_depth};
    };

    const auto [matched_loss, matched_grad] = run(1.0f);
    EXPECT_NEAR(matched_loss, 0.0f, 1.0e-6f);
    EXPECT_LE(matched_grad.abs().max().item<float>(), 1.0e-6f);

    const auto [far_loss, far_grad] = run(1.1f);
    EXPECT_GT(far_loss, 0.0f);
    EXPECT_GT(far_grad.sum().item<float>(), 0.0f);
    EXPECT_EQ(far_grad.slice(1, width - 1, width).abs().max().item<float>(), 0.0f);
}

TEST(NormalPriorBlank, ZeroRgbDecodesToZeroVector) {
    if (!cuda_available()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    TempDir dir;
    constexpr int size = 8;
    constexpr size_t pixels = size * size;
    std::vector<uint8_t> rgb(pixels * 3, 128);
    std::vector<uint8_t> normal(pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        // Left half has no normal; right half faces the camera (0, 0, -1).
        const bool blank = (i % size) < size / 2;
        normal[i * 3 + 0] = blank ? 0 : 128;
        normal[i * 3 + 1] = blank ? 0 : 128;
        normal[i * 3 + 2] = blank ? 0 : 0;
    }
    const auto rgb_path = dir.path() / "image.png";
    const auto normal_path = dir.path() / "normal.png";
    ASSERT_TRUE(lfs::core::save_png(rgb_path, rgb.data(), size, size, 3, 8, 1));
    ASSERT_TRUE(lfs::core::save_png(normal_path, normal.data(), size, size, 3, 8, 1));

    lfs::io::PipelinedLoaderConfig config;
    config.io_threads = 1;
    config.cold_process_threads = 1;
    config.decoder_pool_size = 1;
    config.output_queue_size = 2;
    config.prefetch_count = 2;
    lfs::io::PipelinedImageLoader loader(config);
    lfs::io::ImageRequest request;
    request.path = rgb_path;
    request.normal_path = normal_path;
    request.sequence_id = 1;
    request.aux_target_width = size;
    request.aux_target_height = size;
    loader.prefetch({request});
    auto ready = loader.try_get_for(std::chrono::seconds(20));
    ASSERT_TRUE(ready.has_value());
    ASSERT_TRUE(ready->normal.has_value());
    if (ready->normal_ready_event) {
        ASSERT_EQ(cudaEventSynchronize(ready->normal_ready_event), cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(ready->normal_ready_event), cudaSuccess);
        ready->normal_ready_event = nullptr;
    }
    const auto values = ready->normal->cpu().to_vector();
    ASSERT_EQ(values.size(), pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        const bool blank = (i % size) < size / 2;
        for (size_t c = 0; c < 3; ++c) {
            const float v = values[c * pixels + i];
            if (blank) {
                EXPECT_EQ(v, 0.0f) << "pixel " << i << " channel " << c;
            } else if (c == 2) {
                EXPECT_LT(v, -0.9f) << "pixel " << i;
            }
        }
    }
}
