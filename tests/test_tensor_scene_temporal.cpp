/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "rendering/rasterizer/tensor/splat_rasterizer.hpp"
#include "rendering/rasterizer/tensor/tensor_scene_temporal.hpp"
#include "visualizer/rendering/scene_motion_reprojection.hpp"
#include "visualizer/rendering/scene_temporal_resolve.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::rendering;
    using namespace lfs::vis;

    template <class T>
    std::vector<T> download(const Tensor& tensor, const std::size_t count) {
        const auto host = tensor.to(Device::CPU);
        std::vector<T> result(count);
        std::memcpy(result.data(), host.data_ptr(), result.size() * sizeof(T));
        return result;
    }

    Tensor upload(const void* data, const std::size_t bytes) {
        return Tensor::from_blob(const_cast<void*>(data), {bytes}, Device::CPU,
                                 DataType::UInt8)
            .to(Device::GPU);
    }

    class TensorSceneTemporal : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GetParam()) || GetParam() == GpuBackend::CUDA)
                GTEST_SKIP();
            scope_ = std::make_unique<GpuBackendScope>(GetParam());
        }
        std::unique_ptr<GpuBackendScope> scope_;
    };

    TEST_P(TensorSceneTemporal, ResolveMatchesCpuReferenceOnRandomInputs) {
        std::mt19937 rng(0x5ce11u);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::uniform_real_distribution<float> signed_value(-1.0f, 1.0f);
        constexpr std::size_t COUNT = 1024;
        std::vector<TensorSceneResolveSample> gpu_samples(COUNT);
        std::vector<SceneTemporalResolveResult> expected(COUNT);

        for (std::size_t i = 0; i < COUNT; ++i) {
            const glm::ivec2 render_extent{48 + int(i % 5) * 8, 36 + int(i % 3) * 8};
            const glm::ivec2 output_extent{96, 72};
            const glm::vec2 pixel{0.5f + float(i % 95), 0.5f + float((i * 37) % 71)};
            const glm::vec4 current{unit(rng), unit(rng), unit(rng), unit(rng)};
            const glm::vec4 history{unit(rng), unit(rng), unit(rng), unit(rng)};
            const glm::vec3 lower{unit(rng) * 0.2f, unit(rng) * 0.2f, unit(rng) * 0.2f};
            const glm::vec3 upper{0.8f + unit(rng) * 0.2f,
                                  0.8f + unit(rng) * 0.2f,
                                  0.8f + unit(rng) * 0.2f};
            const glm::vec3 cross{unit(rng) * 4.0f, unit(rng) * 4.0f, unit(rng) * 4.0f};
            const glm::vec2 motion{signed_value(rng) * 0.2f, signed_value(rng) * 0.2f};
            const glm::vec2 current_jitter{signed_value(rng) * 0.5f,
                                           signed_value(rng) * 0.5f};
            const glm::vec2 previous_jitter{signed_value(rng) * 0.5f,
                                            signed_value(rng) * 0.5f};
            const bool history_valid = i % 11 != 0;
            const bool depth_available = i % 4 != 0;
            const float current_depth = 1.0f + unit(rng) * 20.0f;
            const float history_depth = current_depth + signed_value(rng) * 0.01f;
            SceneTemporalResolveSettings settings{
                .history_weight = 0.25f + unit(rng) * 0.7f,
                .depth_relative_threshold = 0.005f,
                .depth_absolute_threshold = 5e-5f,
                .motion_rejection_pixels = 192.0f,
                .motion_confidence_pixels = 0.3f,
                .current_sharpness = unit(rng) * 0.2f,
            };
            SceneTemporalResolveSample sample{
                .current = current,
                .history = history,
                .neighborhood_min = lower,
                .neighborhood_max = upper,
                .neighborhood_cross_sum = cross,
                .current_pixel_center = pixel,
                .current_to_previous_pixels = motion,
                .current_jitter_pixels = current_jitter,
                .previous_jitter_pixels = previous_jitter,
                .motion_extent = render_extent,
                .output_extent = output_extent,
                .current_linear_depth = current_depth,
                .history_linear_depth = history_depth,
                .depth_far_plane = 100.0f,
                .history_valid = history_valid,
                .depth_available = depth_available,
            };
            expected[i] = resolveSceneTemporalSample(sample, settings);
            gpu_samples[i] = {
                .current = {current.x, current.y, current.z, current.w},
                .history = {history.x, history.y, history.z, history.w},
                .neighborhood_min = {lower.x, lower.y, lower.z, 0.0f},
                .neighborhood_max = {upper.x, upper.y, upper.z, 0.0f},
                .neighborhood_cross_sum = {cross.x, cross.y, cross.z, 0.0f},
                .pixel_motion = {pixel.x, pixel.y, motion.x, motion.y},
                .jitter = {current_jitter.x, current_jitter.y,
                           previous_jitter.x, previous_jitter.y},
                .extents = {std::uint32_t(render_extent.x), std::uint32_t(render_extent.y),
                            std::uint32_t(output_extent.x), std::uint32_t(output_extent.y)},
                .depth = {current_depth, history_depth, 100.0f, history_valid ? 1.0f : 0.0f},
                .settings0 = {settings.history_weight, settings.depth_relative_threshold,
                              settings.depth_absolute_threshold,
                              settings.motion_rejection_pixels},
                .settings1 = {settings.motion_confidence_pixels, settings.current_sharpness,
                              depth_available ? 1.0f : 0.0f, 0.0f},
            };
        }

        TensorSceneTemporalKernels kernels(GetParam());
        Tensor output;
        const auto resolved = kernels.resolveSamples(gpu_samples, output);
        ASSERT_TRUE(resolved) << resolved.error().detail();
        const auto actual = download<TensorSceneResolveResult>(output, COUNT);
        for (std::size_t i = 0; i < COUNT; ++i) {
            SCOPED_TRACE(i);
            const std::array<float, 11> gpu{
                actual[i].color[0], actual[i].color[1], actual[i].color[2], actual[i].color[3],
                actual[i].uv[0], actual[i].uv[1], actual[i].uv[2], actual[i].uv[3],
                actual[i].status[0], actual[i].status[1], actual[i].status[2]};
            const std::array<float, 11> cpu{
                expected[i].color.r, expected[i].color.g, expected[i].color.b, expected[i].color.a,
                expected[i].current_render_uv.x, expected[i].current_render_uv.y,
                expected[i].previous_render_uv.x, expected[i].previous_render_uv.y,
                expected[i].previous_uv.x, expected[i].previous_uv.y,
                expected[i].effective_history_weight};
            for (std::size_t component = 0; component < gpu.size(); ++component)
                EXPECT_NEAR(gpu[component], cpu[component], 3e-5f) << component;
            EXPECT_EQ(std::uint32_t(std::lround(actual[i].status[3])),
                      std::uint32_t(expected[i].rejection));
        }
    }

    TEST_P(TensorSceneTemporal, MotionMatchesCpuReferenceOnRandomDepths) {
        constexpr std::uint32_t WIDTH = 53, HEIGHT = 37;
        std::mt19937 rng(0x9a7u);
        std::uniform_real_distribution<float> depth_distribution(0.02f, 0.98f);
        std::vector<float> depths(std::size_t(WIDTH) * HEIGHT);
        for (auto& depth : depths)
            depth = depth_distribution(rng);
        depths[3] = 1.0f;
        depths[91] = -0.1f;

        const glm::mat4 inverse_current{1.0f};
        glm::mat4 previous{1.0f};
        previous[3][0] = 0.03125f;
        previous[3][1] = -0.0175f;
        previous[2][0] = 0.007f;
        previous[2][1] = -0.004f;
        TensorSceneMotionParameters parameters;
        std::memcpy(parameters.inverse_current_view_projection.data(), glm::value_ptr(inverse_current), 64);
        std::memcpy(parameters.previous_view_projection.data(), glm::value_ptr(previous), 64);
        parameters.render_info = {WIDTH, HEIGHT, 1, 0};
        parameters.depth_info = {0.01f, 1000.0f, 0.0f, 0.0f};
        const auto depth = Tensor::from_blob(depths.data(), {HEIGHT, WIDTH}, Device::CPU,
                                             DataType::Float32)
                               .to(Device::GPU);
        TensorSceneTemporalKernels kernels(GetParam());
        Tensor output;
        const auto motion = kernels.motion(depth, parameters, output);
        ASSERT_TRUE(motion) << motion.error().detail();
        const auto actual = download<glm::vec2>(output, depths.size());
        const SceneMotionReprojectionParams cpu_parameters{
            .inverse_current_view_projection = inverse_current,
            .previous_view_projection = previous,
            .render_extent = {WIDTH, HEIGHT},
            .flip_y = true,
        };
        for (std::uint32_t y = 0; y < HEIGHT; ++y) {
            for (std::uint32_t x = 0; x < WIDTH; ++x) {
                const std::size_t i = std::size_t(y) * WIDTH + x;
                const auto expected = reprojectSceneMotionPixels(
                    cpu_parameters, {float(x) + 0.5f, float(y) + 0.5f}, depths[i]);
                const glm::vec2 value = expected.value_or(glm::vec2(0.0f));
                EXPECT_NEAR(actual[i].x, value.x, 2e-5f) << i;
                EXPECT_NEAR(actual[i].y, value.y, 2e-5f) << i;
            }
        }
    }

    TEST_P(TensorSceneTemporal, SpatialMatchesVulkanReferenceFilter) {
        constexpr std::uint32_t WIDTH = 13, HEIGHT = 11, OUTPUT_WIDTH = 31, OUTPUT_HEIGHT = 27;
        std::mt19937 rng(0x51a7u);
        std::uniform_int_distribution<int> byte(0, 255);
        std::vector<std::uint8_t> pixels(std::size_t(WIDTH) * HEIGHT * 4);
        for (auto& value : pixels)
            value = std::uint8_t(byte(rng));
        const auto current = Tensor::from_blob(pixels.data(), {HEIGHT, WIDTH, 4}, Device::CPU,
                                               DataType::UInt8)
                                 .to(Device::GPU);
        TensorSceneResolveParameters parameters{
            .extents = {WIDTH, HEIGHT, OUTPUT_WIDTH, OUTPUT_HEIGHT},
            .current_layout = {WIDTH, HEIGHT, 4, 0},
            .current_uv = {1.0f, 1.0f,
                           (float(WIDTH) - 0.5f) / float(WIDTH),
                           (float(HEIGHT) - 0.5f) / float(HEIGHT)},
        };
        TensorSceneTemporalKernels kernels(GetParam());
        Tensor output;
        const auto result = kernels.spatial(current, parameters, output);
        ASSERT_TRUE(result) << result.error().detail();
        const auto actual = download<glm::vec4>(output, std::size_t(OUTPUT_WIDTH) * OUTPUT_HEIGHT);

        const auto sample = [&](glm::vec2 uv) {
            uv = glm::clamp(uv, glm::vec2(0.0f),
                            glm::vec2(parameters.current_uv[2], parameters.current_uv[3]));
            const glm::vec2 position = glm::clamp(
                uv * glm::vec2(WIDTH, HEIGHT) - 0.5f, glm::vec2(0.0f),
                glm::vec2(WIDTH - 1, HEIGHT - 1));
            const glm::uvec2 a(glm::floor(position));
            const glm::uvec2 b = glm::min(a + 1u, glm::uvec2(WIDTH - 1, HEIGHT - 1));
            const glm::vec2 f = glm::fract(position);
            const auto load = [&](const glm::uvec2 pixel) {
                const std::size_t at = (std::size_t(pixel.y) * WIDTH + pixel.x) * 4;
                return glm::vec4(pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]) /
                       255.0f;
            };
            return glm::mix(glm::mix(load(a), load({b.x, a.y}), f.x),
                            glm::mix(load({a.x, b.y}), load(b), f.x), f.y);
        };
        const glm::vec2 texel{1.0f / WIDTH, 1.0f / HEIGHT};
        for (std::uint32_t y = 0; y < OUTPUT_HEIGHT; ++y) {
            for (std::uint32_t x = 0; x < OUTPUT_WIDTH; ++x) {
                const glm::vec2 uv = glm::min(
                    (glm::vec2(x, y) + 0.5f) / glm::vec2(OUTPUT_WIDTH, OUTPUT_HEIGHT),
                    glm::vec2(parameters.current_uv[2], parameters.current_uv[3]));
                const glm::vec4 center = sample(uv);
                const glm::vec4 left = sample(uv - glm::vec2(texel.x, 0.0f));
                const glm::vec4 right = sample(uv + glm::vec2(texel.x, 0.0f));
                const glm::vec4 up = sample(uv - glm::vec2(0.0f, texel.y));
                const glm::vec4 down = sample(uv + glm::vec2(0.0f, texel.y));
                constexpr float STRENGTH = 0.18f;
                const glm::vec3 sharpened = glm::vec3(center) * (1.0f + 4.0f * STRENGTH) -
                                             glm::vec3(left + right + up + down) * STRENGTH;
                const glm::vec3 lower = glm::min(glm::vec3(center), glm::min(glm::min(glm::vec3(left), glm::vec3(right)),
                                                                             glm::min(glm::vec3(up), glm::vec3(down))));
                const glm::vec3 upper = glm::max(glm::vec3(center), glm::max(glm::max(glm::vec3(left), glm::vec3(right)),
                                                                             glm::max(glm::vec3(up), glm::vec3(down))));
                const glm::vec4 expected{glm::clamp(sharpened, lower, upper), center.a};
                const auto& value = actual[std::size_t(y) * OUTPUT_WIDTH + x];
                for (int component = 0; component < 4; ++component)
                    EXPECT_NEAR(value[component], expected[component], 3e-5f)
                        << x << ',' << y << ':' << component;
            }
        }
    }

    struct ProjectedSplat {
        std::array<float, 4> mean_depth, conic_opacity, color;
        std::array<std::uint32_t, 4> bounds;
    };
    static_assert(sizeof(ProjectedSplat) == 64);

    float halton(std::uint64_t index, const std::uint64_t base) {
        float value = 0.0f, fraction = 1.0f;
        while (index) {
            fraction /= float(base);
            value += fraction * float(index % base);
            index /= base;
        }
        return value;
    }

    glm::vec2 jitter(const std::uint64_t sequence) {
        return {halton(sequence + 1, 2) - 0.5f,
                halton(sequence + 1, 3) - 0.5f};
    }

    Tensor render_scene(const GpuBackend backend, const std::uint32_t width,
                        const std::uint32_t height, const glm::vec2 offset) {
        struct Source {
            glm::vec2 center;
            float sigma, depth, opacity;
            glm::vec3 color;
        };
        constexpr std::array sources{
            Source{{9.7f, 8.4f}, 0.55f, 2.0f, 0.82f, {0.9f, 0.15f, 0.08f}},
            Source{{20.3f, 9.1f}, 0.8f, 3.0f, 0.72f, {0.08f, 0.75f, 0.18f}},
            Source{{29.2f, 17.8f}, 0.65f, 4.0f, 0.88f, {0.12f, 0.22f, 0.95f}},
            Source{{14.6f, 21.2f}, 1.0f, 5.0f, 0.63f, {0.92f, 0.74f, 0.10f}},
            Source{{24.9f, 14.7f}, 0.45f, 1.5f, 0.78f, {0.70f, 0.12f, 0.82f}},
        };
        constexpr float OUTPUT_WIDTH = 40.0f, OUTPUT_HEIGHT = 30.0f;
        const float scale_x = float(width) / OUTPUT_WIDTH;
        const float scale_y = float(height) / OUTPUT_HEIGHT;
        std::vector<ProjectedSplat> splats;
        for (const auto& source : sources) {
            const float sigma_x = source.sigma * scale_x;
            const float sigma_y = source.sigma * scale_y;
            const float radius = std::ceil(3.0f * std::max(sigma_x, sigma_y));
            const glm::vec2 center{source.center.x * scale_x + offset.x,
                                   source.center.y * scale_y + offset.y};
            const auto clamp_bound = [](const float value, const std::uint32_t limit) {
                return std::uint32_t(std::clamp(value, 0.0f, float(limit)));
            };
            splats.push_back({
                .mean_depth = {center.x, center.y, source.depth, radius},
                .conic_opacity = {1.0f / (sigma_x * sigma_x), 0.0f,
                                  1.0f / (sigma_y * sigma_y), source.opacity},
                .color = {source.color.r, source.color.g, source.color.b,
                          source.depth * source.depth},
                .bounds = {clamp_bound(std::floor(center.x - radius), width),
                           clamp_bound(std::floor(center.y - radius), height),
                           clamp_bound(std::ceil(center.x + radius), width),
                           clamp_bound(std::ceil(center.y + radius), height)},
            });
        }
        const auto projected = upload(splats.data(), splats.size() * sizeof(ProjectedSplat));
        SplatRasterizer rasterizer(backend);
        EXPECT_TRUE(rasterizer.reserve(std::uint32_t(splats.size()), width, height, 100000));
        SplatRasterParameters parameters;
        parameters.count = std::uint32_t(splats.size());
        parameters.width = width;
        parameters.height = height;
        parameters.columns = (width + 15) / 16;
        parameters.tiles = parameters.columns * ((height + 15) / 16);
        parameters.capacity = 100000;
        parameters.mode = std::uint32_t(SplatRasterMode::Gaussian);
        parameters.flags = 128 | 4096;
        parameters.background = {0.0f, 0.0f, 0.0f, 1.0f};
        parameters.intrinsics = {32.0f, 32.0f, width * 0.5f, height * 0.5f};
        parameters.clip = {0.01f, 100.0f, 1.0f, 0.3f};
        parameters.camera = {width, height, 0, 0};
        const auto rasterized = rasterizer.rasterize(
            projected, nullptr, std::uint32_t(splats.size()),
            SplatRasterMode::Gaussian, parameters);
        EXPECT_TRUE(rasterized) << rasterized.error().detail();
        SplatPresentParameters present;
        present.background = {0.0f, 0.0f, 0.0f, 1.0f};
        const auto shown = rasterizer.present(present);
        EXPECT_TRUE(shown) << shown.error().detail();
        return rasterizer.rgba().reshape({int(height), int(width), 4});
    }

    std::vector<float> supersampled_reference(const GpuBackend backend) {
        constexpr std::uint32_t WIDTH = 40, HEIGHT = 30, SCALE = 4;
        const auto high = download<std::uint8_t>(
            render_scene(backend, WIDTH * SCALE, HEIGHT * SCALE, {}),
            std::size_t(WIDTH) * HEIGHT * SCALE * SCALE * 4);
        std::vector<float> result(std::size_t(WIDTH) * HEIGHT * 4);
        for (std::uint32_t y = 0; y < HEIGHT; ++y)
            for (std::uint32_t x = 0; x < WIDTH; ++x)
                for (std::uint32_t c = 0; c < 4; ++c) {
                    float sum = 0.0f;
                    for (std::uint32_t sy = 0; sy < SCALE; ++sy)
                        for (std::uint32_t sx = 0; sx < SCALE; ++sx) {
                            const auto at = (((std::size_t(y) * SCALE + sy) * WIDTH * SCALE +
                                              x * SCALE + sx) * 4 + c);
                            sum += float(high[at]) / 255.0f;
                        }
                    result[(std::size_t(y) * WIDTH + x) * 4 + c] =
                        sum / float(SCALE * SCALE);
                }
        return result;
    }

    float rgb_error(const std::vector<float>& image, const std::vector<float>& reference) {
        double squared = 0.0;
        std::size_t count = 0;
        for (std::size_t i = 0; i < image.size(); i += 4)
            for (std::size_t c = 0; c < 3; ++c) {
                const double difference = image[i + c] - reference[i + c];
                squared += difference * difference;
                ++count;
            }
        return float(std::sqrt(squared / double(count)));
    }

    std::vector<float> convergence_errors(const GpuBackend backend) {
        constexpr std::uint32_t RENDER_WIDTH = 20, RENDER_HEIGHT = 15;
        constexpr std::uint32_t OUTPUT_WIDTH = 40, OUTPUT_HEIGHT = 30;
        const GpuBackendScope scope(backend);
        const auto reference = supersampled_reference(backend);
        const auto motion = Tensor::zeros({RENDER_HEIGHT, RENDER_WIDTH, 2}, Device::GPU,
                                          DataType::Float32);
        TensorSceneTemporalKernels kernels(backend);
        Tensor history;
        glm::vec2 previous_jitter{};
        std::vector<float> errors;
        for (std::uint64_t frame = 0; frame < 16; ++frame) {
            const glm::vec2 current_jitter = jitter(frame);
            const auto current = render_scene(backend, RENDER_WIDTH, RENDER_HEIGHT,
                                              current_jitter);
            TensorSceneResolveParameters parameters{
                .extents = {RENDER_WIDTH, RENDER_HEIGHT, OUTPUT_WIDTH, OUTPUT_HEIGHT},
                .current_layout = {RENDER_WIDTH, RENDER_HEIGHT, 4, 0},
                .control = {frame ? 1.0f : 0.0f,
                            frame ? std::min(0.95f, float(frame) / float(frame + 1)) : 0.0f,
                            192.0f, 0.0f},
                .current_uv = {1.0f, 1.0f,
                               (float(RENDER_WIDTH) - 0.5f) / float(RENDER_WIDTH),
                               (float(RENDER_HEIGHT) - 0.5f) / float(RENDER_HEIGHT)},
                .depth_control = {0.0f, 0.005f, 5e-5f, 100.0f},
                .jitter_pixels = {current_jitter.x, current_jitter.y,
                                  previous_jitter.x, previous_jitter.y},
                .reconstruction = {0.0f, 0.3f, 0.0f, 0.0f},
            };
            Tensor resolved;
            const auto result = kernels.resolve(current, frame ? &history : nullptr,
                                                motion, nullptr, nullptr, parameters, resolved);
            if (!result) {
                ADD_FAILURE() << result.error().detail();
                return {};
            }
            history = resolved;
            errors.push_back(rgb_error(download<float>(history,
                                                       std::size_t(OUTPUT_WIDTH) * OUTPUT_HEIGHT * 4),
                                       reference));
            previous_jitter = current_jitter;
        }
        return errors;
    }

    TEST_P(TensorSceneTemporal, JitteredRasterConvergesTowardFourTimesReference) {
        const auto errors = convergence_errors(GetParam());
        ASSERT_EQ(errors.size(), 16u);
        std::size_t regressions = 0;
        for (std::size_t i = 1; i < errors.size(); ++i)
            regressions += errors[i] > errors[i - 1] * 1.01f;
        std::cout << "temporal convergence backend=" << int(GetParam())
                  << " first_rmse=" << errors.front()
                  << " final_rmse=" << errors.back()
                  << " regressions=" << regressions << '\n';
        EXPECT_LE(regressions, 4u);
        EXPECT_LT(errors.back(), errors.front() * 0.95f);
        EXPECT_LT(errors.back(), 0.025f);
    }

    TEST(TensorSceneTemporalBackends, RasterConvergenceMatchesBetweenMetalAndVulkan) {
        if (!gpu_backend_available(GpuBackend::Metal) ||
            !gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        const auto metal = convergence_errors(GpuBackend::Metal);
        const auto vulkan = convergence_errors(GpuBackend::Vulkan);
        ASSERT_EQ(metal.size(), 16u);
        ASSERT_EQ(vulkan.size(), 16u);
        ASSERT_EQ(metal.size(), vulkan.size());
        for (std::size_t i = 0; i < metal.size(); ++i)
            EXPECT_NEAR(metal[i], vulkan[i], 1e-6f) << i;
    }

    INSTANTIATE_TEST_SUITE_P(
        Backends, TensorSceneTemporal, testing::ValuesIn(kCompiledGpuBackends),
        [](const testing::TestParamInfo<GpuBackend>& parameter) {
            switch (parameter.param) {
            case GpuBackend::Metal: return "Metal";
            case GpuBackend::Vulkan: return "Vulkan";
            case GpuBackend::CUDA: return "CUDA";
            }
            return "Unknown";
        });
} // namespace
