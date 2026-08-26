/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/passes/vulkan_scene_temporal_pipeline.hpp"
#include "visualizer/rendering/passes/vulkan_scene_temporal_resolve_pass.hpp"
#include "visualizer/rendering/scene_temporal_resolve.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace lfs::vis {
    namespace {
        SceneTemporalResolveSample sample() {
            return {
                .current = {0.2f, 0.4f, 0.6f, 0.7f},
                .history = {0.4f, 0.6f, 0.8f, 0.1f},
                .neighborhood_min = {0.1f, 0.2f, 0.3f},
                .neighborhood_max = {0.7f, 0.8f, 0.9f},
                .neighborhood_cross_sum = {0.8f, 1.6f, 2.4f},
                .current_pixel_center = {639.5f, 359.5f},
                .current_to_previous_pixels = {0.0f, 0.0f},
                .motion_extent = {1280, 720},
                .output_extent = {1280, 720},
                .current_linear_depth = 10.0f,
                .history_linear_depth = 10.0f,
                .depth_far_plane = 1000.0f,
                .history_valid = true,
                .depth_available = true,
            };
        }

        VulkanSceneTemporalPipelineRequest pipelineRequest() {
            VulkanSceneTemporalPipelineRequest request;
            request.temporal.requirements = {
                .depth = true,
                .motion = true,
                .jitter = true,
                .history_color = true,
                .history_depth = true};
            request.temporal.render_extent = {640, 360};
            request.temporal.output_extent = {1280, 720};
            request.temporal.frame.view.size = request.temporal.render_extent;
            request.temporal.frame.output_extent = request.temporal.output_extent;
            request.motion.enabled = true;
            request.motion.depth_view = reinterpret_cast<VkImageView>(1);
            request.motion.depth = makeSceneDepthContract(true,
                                                          SceneDepthStorage::VulkanImage,
                                                          SceneDepthEncoding::VulkanNdc,
                                                          {640, 360},
                                                          0.1f,
                                                          1000.0f,
                                                          false,
                                                          false);
            request.motion.render_extent = {640, 360};
            request.resolve.enabled = true;
            request.resolve.current_color_view = reinterpret_cast<VkImageView>(2);
            request.resolve.render_extent = {640, 360};
            request.resolve.output_extent = {1280, 720};
            request.resolve.current_depth.enabled = true;
            request.resolve.current_depth.current_depth_view = request.motion.depth_view;
            request.resolve.current_depth.depth = request.motion.depth;
            return request;
        }

        struct SyntheticImage {
            int width = 0;
            int height = 0;
            std::vector<float> pixels;

            [[nodiscard]] float at(const int x, const int y) const {
                const int clamped_x = std::clamp(x, 0, width - 1);
                const int clamped_y = std::clamp(y, 0, height - 1);
                return pixels[static_cast<std::size_t>(clamped_y) * width + clamped_x];
            }
        };

        float syntheticSignal(const float x, const float y, const float translation = 0.0f) {
            const float shifted_x = x - translation;
            return std::clamp(0.46f + 0.18f * std::sin(0.21f * shifted_x + 0.13f * y) +
                                  0.14f * std::cos(0.09f * shifted_x - 0.25f * y) +
                                  0.10f * std::sin(0.31f * shifted_x + 0.07f * y),
                              0.0f,
                              1.0f);
        }

        float bilinear(const SyntheticImage& image, const float x, const float y) {
            const float clamped_x = std::clamp(x, 0.0f, static_cast<float>(image.width - 1));
            const float clamped_y = std::clamp(y, 0.0f, static_cast<float>(image.height - 1));
            const int x0 = static_cast<int>(std::floor(clamped_x));
            const int y0 = static_cast<int>(std::floor(clamped_y));
            const float tx = clamped_x - static_cast<float>(x0);
            const float ty = clamped_y - static_cast<float>(y0);
            const float top = std::lerp(image.at(x0, y0), image.at(x0 + 1, y0), tx);
            const float bottom = std::lerp(image.at(x0, y0 + 1), image.at(x0 + 1, y0 + 1), tx);
            return std::lerp(top, bottom, ty);
        }

        SyntheticImage syntheticLowResolutionFrame(const glm::ivec2 render_extent,
                                                   const glm::ivec2 output_extent,
                                                   const glm::vec2 jitter_pixels,
                                                   const float translation = 0.0f) {
            SyntheticImage result{render_extent.x,
                                  render_extent.y,
                                  std::vector<float>(static_cast<std::size_t>(render_extent.x) *
                                                     render_extent.y)};
            const glm::vec2 scale = glm::vec2(output_extent) / glm::vec2(render_extent);
            for (int y = 0; y < result.height; ++y) {
                for (int x = 0; x < result.width; ++x) {
                    const glm::vec2 reference =
                        (glm::vec2(x, y) + 0.5f - jitter_pixels) * scale - 0.5f;
                    result.pixels[static_cast<std::size_t>(y) * result.width + x] =
                        syntheticSignal(reference.x, reference.y, translation);
                }
            }
            return result;
        }

        SyntheticImage syntheticResolveCurrent(const SyntheticImage& low_resolution,
                                               const glm::ivec2 output_extent,
                                               const glm::vec2 jitter_pixels) {
            SyntheticImage result{output_extent.x,
                                  output_extent.y,
                                  std::vector<float>(static_cast<std::size_t>(output_extent.x) *
                                                     output_extent.y)};
            const glm::vec2 scale = glm::vec2(low_resolution.width, low_resolution.height) /
                                    glm::vec2(output_extent);
            for (int y = 0; y < result.height; ++y) {
                for (int x = 0; x < result.width; ++x) {
                    const glm::vec2 source =
                        (glm::vec2(x, y) + 0.5f) * scale - 0.5f + jitter_pixels;
                    result.pixels[static_cast<std::size_t>(y) * result.width + x] =
                        bilinear(low_resolution, source.x, source.y);
                }
            }
            return result;
        }

        SyntheticImage syntheticReference(const glm::ivec2 extent,
                                          const float translation = 0.0f) {
            SyntheticImage result{extent.x,
                                  extent.y,
                                  std::vector<float>(static_cast<std::size_t>(extent.x) * extent.y)};
            for (int y = 0; y < extent.y; ++y)
                for (int x = 0; x < extent.x; ++x)
                    result.pixels[static_cast<std::size_t>(y) * extent.x + x] =
                        syntheticSignal(static_cast<float>(x), static_cast<float>(y), translation);
            return result;
        }

        SyntheticImage syntheticSpatialResolve(const SyntheticImage& low_resolution,
                                               const glm::ivec2 output_extent) {
            SyntheticImage result{output_extent.x,
                                  output_extent.y,
                                  std::vector<float>(static_cast<std::size_t>(output_extent.x) *
                                                     output_extent.y)};
            const glm::vec2 scale = glm::vec2(low_resolution.width, low_resolution.height) /
                                    glm::vec2(output_extent);
            constexpr float STRENGTH = 0.18f;
            for (int y = 0; y < result.height; ++y) {
                for (int x = 0; x < result.width; ++x) {
                    const glm::vec2 source =
                        (glm::vec2(x, y) + 0.5f) * scale - 0.5f;
                    const float center = bilinear(low_resolution, source.x, source.y);
                    const float left = bilinear(low_resolution, source.x - 1.0f, source.y);
                    const float right = bilinear(low_resolution, source.x + 1.0f, source.y);
                    const float up = bilinear(low_resolution, source.x, source.y - 1.0f);
                    const float down = bilinear(low_resolution, source.x, source.y + 1.0f);
                    const float sharpened = center * (1.0f + 4.0f * STRENGTH) -
                                            (left + right + up + down) * STRENGTH;
                    const float neighborhood_min =
                        std::min({center, left, right, up, down});
                    const float neighborhood_max =
                        std::max({center, left, right, up, down});
                    result.pixels[static_cast<std::size_t>(y) * result.width + x] =
                        std::clamp(sharpened, neighborhood_min, neighborhood_max);
                }
            }
            return result;
        }

        SyntheticImage syntheticTemporalResolve(const SyntheticImage& low_resolution,
                                                const SyntheticImage& history,
                                                const glm::ivec2 output_extent,
                                                const glm::vec2 current_jitter,
                                                const glm::vec2 previous_jitter,
                                                const glm::vec2 current_to_previous,
                                                const std::uint64_t sequence,
                                                SceneTemporalResolveSettings settings =
                                                    sceneTemporalQualitySettings(
                                                        SceneTemporalQuality::Quality)) {
            const SyntheticImage current =
                syntheticResolveCurrent(low_resolution, output_extent, current_jitter);

            SyntheticImage result{output_extent.x,
                                  output_extent.y,
                                  std::vector<float>(current.pixels.size())};
            settings.history_weight =
                sceneTemporalHistoryWeight(settings.history_weight, sequence);
            const glm::ivec2 render_extent{low_resolution.width, low_resolution.height};
            const glm::vec2 source_scale = glm::vec2(render_extent) / glm::vec2(output_extent);
            for (int y = 0; y < result.height; ++y) {
                for (int x = 0; x < result.width; ++x) {
                    const std::size_t index = static_cast<std::size_t>(y) * result.width + x;
                    const glm::vec2 source =
                        (glm::vec2(x, y) + 0.5f) * source_scale - 0.5f + current_jitter;
                    float neighborhood_min = current.pixels[index];
                    float neighborhood_max = current.pixels[index];
                    float neighborhood_cross_sum = 0.0f;
                    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
                        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
                            const float value = bilinear(low_resolution,
                                                         source.x + offset_x,
                                                         source.y + offset_y);
                            neighborhood_min = std::min(neighborhood_min, value);
                            neighborhood_max = std::max(neighborhood_max, value);
                            if (std::abs(offset_x) + std::abs(offset_y) == 1)
                                neighborhood_cross_sum += value;
                        }
                    }

                    SceneTemporalResolveSample resolve_sample{
                        .current = glm::vec4(current.pixels[index]),
                        .history = glm::vec4(0.0f),
                        .neighborhood_min = glm::vec3(neighborhood_min),
                        .neighborhood_max = glm::vec3(neighborhood_max),
                        .neighborhood_cross_sum = glm::vec3(neighborhood_cross_sum),
                        .current_pixel_center = glm::vec2(x, y) + 0.5f,
                        .current_to_previous_pixels = current_to_previous,
                        .current_jitter_pixels = current_jitter,
                        .previous_jitter_pixels = previous_jitter,
                        .motion_extent = render_extent,
                        .output_extent = output_extent,
                        .history_valid = sequence > 0,
                    };
                    const auto coordinates =
                        resolveSceneTemporalSample(resolve_sample, settings);
                    if (sequence > 0 &&
                        coordinates.rejection == SceneHistoryRejection::None) {
                        const glm::vec2 history_pixel =
                            coordinates.previous_uv * glm::vec2(output_extent) - 0.5f;
                        resolve_sample.history = glm::vec4(
                            bilinear(history, history_pixel.x, history_pixel.y));
                    }
                    result.pixels[index] =
                        resolveSceneTemporalSample(resolve_sample, settings).color.r;
                }
            }
            return result;
        }

        float syntheticPsnr(const SyntheticImage& image, const SyntheticImage& reference) {
            double squared_error = 0.0;
            for (std::size_t i = 0; i < image.pixels.size(); ++i) {
                const double error = static_cast<double>(image.pixels[i]) - reference.pixels[i];
                squared_error += error * error;
            }
            const double mse = squared_error / static_cast<double>(image.pixels.size());
            return static_cast<float>(10.0 * std::log10(1.0 / mse));
        }

        float syntheticSsim(const SyntheticImage& image, const SyntheticImage& reference) {
            double image_mean = 0.0;
            double reference_mean = 0.0;
            for (std::size_t i = 0; i < image.pixels.size(); ++i) {
                image_mean += image.pixels[i];
                reference_mean += reference.pixels[i];
            }
            image_mean /= static_cast<double>(image.pixels.size());
            reference_mean /= static_cast<double>(reference.pixels.size());

            double image_variance = 0.0;
            double reference_variance = 0.0;
            double covariance = 0.0;
            for (std::size_t i = 0; i < image.pixels.size(); ++i) {
                const double image_delta = image.pixels[i] - image_mean;
                const double reference_delta = reference.pixels[i] - reference_mean;
                image_variance += image_delta * image_delta;
                reference_variance += reference_delta * reference_delta;
                covariance += image_delta * reference_delta;
            }
            const double count = static_cast<double>(image.pixels.size());
            image_variance /= count;
            reference_variance /= count;
            covariance /= count;
            constexpr double C1 = 0.01 * 0.01;
            constexpr double C2 = 0.03 * 0.03;
            return static_cast<float>(
                ((2.0 * image_mean * reference_mean + C1) * (2.0 * covariance + C2)) /
                ((image_mean * image_mean + reference_mean * reference_mean + C1) *
                 (image_variance + reference_variance + C2)));
        }
    } // namespace

    TEST(SceneTemporalResolve, QualitySettingsRemainOrderedAndUnknownIsBalanced) {
        const auto performance = sceneTemporalQualitySettings(SceneTemporalQuality::Performance);
        const auto balanced = sceneTemporalQualitySettings(SceneTemporalQuality::Balanced);
        const auto quality = sceneTemporalQualitySettings(SceneTemporalQuality::Quality);
        const auto unknown = sceneTemporalQualitySettings(static_cast<SceneTemporalQuality>(255));
        EXPECT_LT(performance.history_weight, balanced.history_weight);
        EXPECT_LT(balanced.history_weight, quality.history_weight);
        EXPECT_GT(performance.depth_relative_threshold, balanced.depth_relative_threshold);
        EXPECT_GT(balanced.depth_relative_threshold, quality.depth_relative_threshold);
        EXPECT_EQ(unknown.history_weight, balanced.history_weight);
    }

    TEST(SceneTemporalResolve, ConvertsNdcJitterToMotionImagePixels) {
        EXPECT_EQ(sceneTemporalJitterPixels({0.25f, -0.5f}, {640, 360}, false),
                  glm::vec2(80.0f, 90.0f));
        EXPECT_EQ(sceneTemporalJitterPixels({0.25f, -0.5f}, {640, 360}, true),
                  glm::vec2(80.0f, -90.0f));
        EXPECT_EQ(sceneTemporalJitterPixels({0.25f, -0.5f}, {0, 360}, false),
                  glm::vec2(0.0f));

        constexpr glm::ivec2 extent{640, 360};
        constexpr glm::vec2 applied_pixels{0.25f, -0.375f};
        const glm::vec2 round_trip = sceneTemporalJitterPixels(
            temporalJitterNdc(applied_pixels, extent), extent, false);
        EXPECT_NEAR(round_trip.x, applied_pixels.x, 1e-6f);
        EXPECT_NEAR(round_trip.y, applied_pixels.y, 1e-6f);
    }

    TEST(SceneTemporalResolve, WarmupUsesUniformSamplesWithoutExceedingPresetWeight) {
        EXPECT_FLOAT_EQ(sceneTemporalHistoryWeight(0.95f, 0), 0.0f);
        EXPECT_FLOAT_EQ(sceneTemporalHistoryWeight(0.95f, 1), 0.5f);
        EXPECT_NEAR(sceneTemporalHistoryWeight(0.95f, 7), 0.875f, 1e-6f);
        EXPECT_FLOAT_EQ(sceneTemporalHistoryWeight(0.75f, 7), 0.75f);
        EXPECT_FLOAT_EQ(sceneTemporalHistoryWeight(4.0f, 1), 0.5f);
    }

    TEST(SceneTemporalQualityRegression, ProductionResolveMatchesSpatialSharpenOnStaticSignal) {
        const glm::ivec2 OUTPUT{192, 128};
        const glm::ivec2 RENDER{96, 64};
        const auto reference = syntheticReference(OUTPUT);
        const auto spatial = syntheticSpatialResolve(
            syntheticLowResolutionFrame(RENDER, OUTPUT, {}), OUTPUT);

        SyntheticImage temporal;
        glm::vec2 previous_jitter{0.0f};
        for (std::uint64_t sequence = 0; sequence < TemporalConvergenceController::SAMPLE_COUNT;
             ++sequence) {
            const glm::vec2 jitter = temporalJitterPixels(sequence);
            const auto low_resolution =
                syntheticLowResolutionFrame(RENDER, OUTPUT, jitter);
            temporal = syntheticTemporalResolve(low_resolution,
                                                temporal,
                                                OUTPUT,
                                                jitter,
                                                previous_jitter,
                                                glm::vec2(0.0f),
                                                sequence);
            previous_jitter = jitter;
        }

        const float spatial_psnr = syntheticPsnr(spatial, reference);
        const float temporal_psnr = syntheticPsnr(temporal, reference);
        const float spatial_ssim = syntheticSsim(spatial, reference);
        const float temporal_ssim = syntheticSsim(temporal, reference);
        RecordProperty("spatial_psnr_db", spatial_psnr);
        RecordProperty("temporal_psnr_db", temporal_psnr);
        RecordProperty("temporal_minus_spatial_psnr_db", temporal_psnr - spatial_psnr);
        RecordProperty("spatial_ssim", spatial_ssim);
        RecordProperty("temporal_ssim", temporal_ssim);
        EXPECT_GT(temporal_psnr, spatial_psnr - 0.25f);
        EXPECT_GT(temporal_ssim, spatial_ssim - 1e-4f);
    }

    TEST(SceneTemporalQualityRegression, ProductionResolveBoundsLossOnReprojectedMovingSignal) {
        const glm::ivec2 OUTPUT{192, 128};
        const glm::ivec2 RENDER{96, 64};
        constexpr float MOTION_PER_FRAME = 0.25f;
        const auto run_temporal = [&](const SceneTemporalResolveSettings& settings) {
            SyntheticImage temporal;
            glm::vec2 previous_jitter{0.0f};
            for (std::uint64_t sequence = 0;
                 sequence < TemporalConvergenceController::SAMPLE_COUNT;
                 ++sequence) {
                const glm::vec2 jitter = temporalJitterPixels(sequence);
                const float translation = static_cast<float>(sequence) * MOTION_PER_FRAME;
                const auto low_resolution =
                    syntheticLowResolutionFrame(RENDER, OUTPUT, jitter, translation);
                const glm::vec2 output_motion{MOTION_PER_FRAME, 0.0f};
                const glm::vec2 render_motion =
                    output_motion * (glm::vec2(RENDER) / glm::vec2(OUTPUT));
                temporal = syntheticTemporalResolve(low_resolution,
                                                    temporal,
                                                    OUTPUT,
                                                    jitter,
                                                    previous_jitter,
                                                    -render_motion,
                                                    sequence,
                                                    settings);
                previous_jitter = jitter;
            }
            return temporal;
        };

        const auto temporal = run_temporal(
            sceneTemporalQualitySettings(SceneTemporalQuality::Quality));

        const float final_translation =
            static_cast<float>(TemporalConvergenceController::SAMPLE_COUNT - 1) *
            MOTION_PER_FRAME;
        const auto reference = syntheticReference(OUTPUT, final_translation);
        const auto spatial = syntheticSpatialResolve(
            syntheticLowResolutionFrame(RENDER, OUTPUT, {}, final_translation), OUTPUT);
        const float temporal_psnr = syntheticPsnr(temporal, reference);
        const float spatial_psnr = syntheticPsnr(spatial, reference);
        RecordProperty("temporal_psnr_db", temporal_psnr);
        RecordProperty("spatial_psnr_db", spatial_psnr);
        RecordProperty("temporal_minus_spatial_psnr_db", temporal_psnr - spatial_psnr);
        constexpr float MAX_REPROJECTED_MOTION_LOSS_DB = 1.0f;
        RecordProperty("max_reprojected_motion_loss_db",
                       MAX_REPROJECTED_MOTION_LOSS_DB);
        // The moving case bounds accumulated resampling loss independently from
        // the stricter static parity test above. Spatial is a single-frame
        // sharpened control here, not an accumulated moving-history reference;
        // this smooth translated signal therefore does not claim to isolate
        // edge ghosting or disocclusion behavior.
        EXPECT_GT(temporal_psnr, 45.0f);
        EXPECT_GT(temporal_psnr,
                  spatial_psnr - MAX_REPROJECTED_MOTION_LOSS_DB);
    }

    TEST(SceneTemporalResolve, StableSampleUsesClampedHistoryAndPreservesCurrentAlpha) {
        auto stable = sample();
        stable.history = {2.0f, 0.6f, -1.0f, 0.1f};
        const auto result = resolveSceneTemporalSample(stable);
        EXPECT_TRUE(result.usedHistory());
        EXPECT_NEAR(result.color.x, 0.65f, 1e-6f);
        EXPECT_NEAR(result.color.y, 0.58f, 1e-6f);
        EXPECT_NEAR(result.color.z, 0.33f, 1e-6f);
        EXPECT_FLOAT_EQ(result.color.a, stable.current.a);
    }

    TEST(SceneTemporalResolve, PixelCentersMapToUnambiguousHistoryUv) {
        auto center = sample();
        center.current_pixel_center = {0.5f, 0.5f};
        const auto first = resolveSceneTemporalSample(center);
        EXPECT_NEAR(first.previous_uv.x, 0.5f / 1280.0f, 1e-7f);
        EXPECT_NEAR(first.previous_uv.y, 0.5f / 720.0f, 1e-7f);

        center.current_pixel_center = {1279.5f, 719.5f};
        EXPECT_TRUE(resolveSceneTemporalSample(center).usedHistory());
    }

    TEST(SceneTemporalResolve, RejectsAbsentHistoryAndOutOfBoundsReprojection) {
        auto current = sample();
        current.history_valid = false;
        EXPECT_EQ(resolveSceneTemporalSample(current).rejection,
                  SceneHistoryRejection::NoHistory);
        current = sample();
        current.current_pixel_center = {0.5f, 0.5f};
        current.current_to_previous_pixels = {-0.01f, 0.0f};
        EXPECT_EQ(resolveSceneTemporalSample(current).rejection,
                  SceneHistoryRejection::OutsideHistory);
    }

    TEST(SceneTemporalResolve, UsesRelativeLinearDepthAndRejectsFarPlaneForDisocclusion) {
        auto depth = sample();
        depth.history_linear_depth = 10.09f;
        EXPECT_TRUE(resolveSceneTemporalSample(depth).usedHistory());
        depth.history_linear_depth = 10.11f;
        EXPECT_EQ(resolveSceneTemporalSample(depth).rejection,
                  SceneHistoryRejection::Disocclusion);
        depth.history_linear_depth = 0.0f;
        EXPECT_EQ(resolveSceneTemporalSample(depth).rejection,
                  SceneHistoryRejection::Disocclusion);
        depth = sample();
        depth.current_linear_depth = depth.depth_far_plane;
        depth.history_linear_depth = depth.depth_far_plane;
        EXPECT_EQ(resolveSceneTemporalSample(depth).rejection,
                  SceneHistoryRejection::Disocclusion);
    }

    TEST(SceneTemporalResolve, DepthLookupRejectsOutsideRenderGrid) {
        auto depth = sample();
        depth.motion_extent = {640, 360};
        depth.output_extent = {1280, 720};
        depth.current_pixel_center = {1279.5f, 719.5f};
        depth.current_to_previous_pixels = {0.0f, 0.0f};
        depth.previous_jitter_pixels = {0.5f, 0.0f};
        EXPECT_EQ(resolveSceneTemporalSample(depth).rejection,
                  SceneHistoryRejection::Disocclusion);
        depth.depth_available = false;
        EXPECT_TRUE(resolveSceneTemporalSample(depth).usedHistory());
    }

    TEST(SceneTemporalResolve, MotionReducesWeightAndExcessMotionIsRejected) {
        auto moving = sample();
        moving.current_to_previous_pixels = {0.15f, 0.0f};
        const auto result = resolveSceneTemporalSample(moving);
        EXPECT_TRUE(result.usedHistory());
        EXPECT_FLOAT_EQ(result.effective_history_weight, 0.45f);
        moving.current_to_previous_pixels = {129.0f, 0.0f};
        EXPECT_EQ(resolveSceneTemporalSample(moving).rejection,
                  SceneHistoryRejection::InvalidMotion);
    }

    TEST(SceneTemporalResolve, ZeroMotionLimitNeverBlendsHistory) {
        SceneTemporalResolveSettings settings;
        settings.motion_rejection_pixels = 0.0f;
        settings.current_sharpness = 0.0f;
        const auto result = resolveSceneTemporalSample(sample(), settings);
        EXPECT_FALSE(result.usedHistory());
        EXPECT_FLOAT_EQ(result.effective_history_weight, 0.0f);
        EXPECT_EQ(result.color, sample().current);
    }

    TEST(SceneTemporalResolve, RenderPixelMotionIsNormalizedBeforeOutputHistoryLookup) {
        auto scaled = sample();
        scaled.motion_extent = {640, 360};
        scaled.output_extent = {1280, 720};
        scaled.current_pixel_center = {639.5f, 359.5f};
        scaled.current_to_previous_pixels = {32.0f, 18.0f};
        SceneTemporalResolveSettings settings;
        settings.motion_confidence_pixels = 64.0f;
        const auto result = resolveSceneTemporalSample(scaled, settings);
        ASSERT_TRUE(result.usedHistory());
        EXPECT_NEAR(result.previous_uv.x,
                    639.5f / 1280.0f + 32.0f / 640.0f,
                    1e-6f);
        EXPECT_NEAR(result.previous_uv.y,
                    359.5f / 720.0f + 18.0f / 360.0f,
                    1e-6f);
    }

    TEST(SceneTemporalResolve, RemovesCurrentAndPreviousJitterFromStableHistoryLookup) {
        auto jittered = sample();
        jittered.motion_extent = {640, 360};
        jittered.output_extent = {1280, 720};
        jittered.current_pixel_center = {639.5f, 359.5f};
        jittered.current_jitter_pixels = {0.25f, -0.125f};
        jittered.previous_jitter_pixels = {-0.25f, 0.125f};
        jittered.current_to_previous_pixels = {0.0f, 0.0f};
        const auto result = resolveSceneTemporalSample(jittered);
        ASSERT_TRUE(result.usedHistory());
        EXPECT_NEAR(result.current_render_uv.x,
                    639.5f / 1280.0f + 0.25f / 640.0f,
                    1e-6f);
        EXPECT_NEAR(result.current_render_uv.y,
                    359.5f / 720.0f - 0.125f / 360.0f,
                    1e-6f);
        EXPECT_NEAR(result.previous_uv.x, 639.5f / 1280.0f, 1e-6f);
        EXPECT_NEAR(result.previous_uv.y, 359.5f / 720.0f, 1e-6f);
        EXPECT_NEAR(result.previous_render_uv.x,
                    639.5f / 1280.0f - 0.25f / 640.0f,
                    1e-6f);
        EXPECT_NEAR(result.previous_render_uv.y,
                    359.5f / 720.0f + 0.125f / 360.0f,
                    1e-6f);

        jittered.current_jitter_pixels = {-0.4f, 0.3f};
        jittered.previous_jitter_pixels = {0.2f, -0.1f};
        const auto swapped = resolveSceneTemporalSample(jittered);
        ASSERT_TRUE(swapped.usedHistory());
        EXPECT_NEAR(swapped.previous_uv.x, result.previous_uv.x, 1e-6f);
        EXPECT_NEAR(swapped.previous_uv.y, result.previous_uv.y, 1e-6f);
        EXPECT_NEAR(swapped.current_render_uv.x,
                    639.5f / 1280.0f - 0.4f / 640.0f,
                    1e-6f);
        EXPECT_NEAR(swapped.previous_render_uv.x,
                    639.5f / 1280.0f + 0.2f / 640.0f,
                    1e-6f);
    }

    TEST(SceneTemporalResolve, SanitizesWeightsAndRejectsNonFiniteData) {
        auto stable = sample();
        SceneTemporalResolveSettings settings;
        settings.history_weight = 4.0f;
        EXPECT_FLOAT_EQ(resolveSceneTemporalSample(stable, settings).effective_history_weight,
                        1.0f);
        stable.current.x = std::numeric_limits<float>::quiet_NaN();
        const auto invalid_current = resolveSceneTemporalSample(stable);
        EXPECT_EQ(invalid_current.rejection, SceneHistoryRejection::InvalidCurrent);
        EXPECT_EQ(invalid_current.color, glm::vec4(0.0f));
        stable = sample();
        stable.history.x = std::numeric_limits<float>::infinity();
        EXPECT_EQ(resolveSceneTemporalSample(stable).rejection,
                  SceneHistoryRejection::InvalidHistory);
    }

    TEST(VulkanSceneTemporalResolveContract, PingPongSelectionIsDeterministic) {
        EXPECT_TRUE(validTemporalViewId(TemporalViewId::Main));
        EXPECT_TRUE(validTemporalViewId(TemporalViewId::SplitRight));
        EXPECT_FALSE(validTemporalViewId(TemporalViewId::Count));
        EXPECT_EQ(nextTemporalHistoryWriteIndex(false, 0), 0u);
        EXPECT_EQ(nextTemporalHistoryWriteIndex(true, 0), 1u);
        EXPECT_EQ(nextTemporalHistoryWriteIndex(true, 1), 0u);
        EXPECT_EQ(nextTemporalHistoryWriteIndex(true, 99), 0u);
    }

    TEST(VulkanSceneTemporalResolveContract, PaddedCurrentUvNeverSamplesOutsideValidRegion) {
        const auto transform = temporalCurrentUvTransform({1100, 738}, {1152, 768});
        EXPECT_NEAR(transform.x, 1100.0f / 1152.0f, 1e-6f);
        EXPECT_NEAR(transform.y, 738.0f / 768.0f, 1e-6f);
        EXPECT_LT(transform.z, transform.x);
        EXPECT_LT(transform.w, transform.y);
        EXPECT_EQ(temporalCurrentUvTransform({1200, 738}, {1152, 768}), glm::vec4(0.0f));
    }

    TEST(VulkanSceneTemporalResolveContract, DepthRejectionRequiresOwnedCurrentDepth) {
        VulkanSceneTemporalResolveParams params;
        params.render_extent = {1280, 720};
        EXPECT_TRUE(validTemporalDepthInputs(params));
        params.current_depth.enabled = true;
        EXPECT_FALSE(validTemporalDepthInputs(params));
        params.current_depth.current_depth_view = reinterpret_cast<VkImageView>(1);
        params.current_depth.depth = makeSceneDepthContract(true,
                                                            SceneDepthStorage::VulkanImage,
                                                            SceneDepthEncoding::LinearView,
                                                            params.render_extent,
                                                            0.1f,
                                                            1000.0f,
                                                            false,
                                                            false);
        EXPECT_TRUE(validTemporalDepthInputs(params));
    }

    TEST(VulkanSceneTemporalResolveContract, DepthRejectionRejectsUndeclaredLayouts) {
        VulkanSceneTemporalResolveParams params;
        params.render_extent = {1280, 720};
        params.current_depth.enabled = true;
        params.current_depth.current_depth_view = reinterpret_cast<VkImageView>(1);
        params.current_depth.depth = makeSceneDepthContract(true,
                                                            SceneDepthStorage::VulkanImage,
                                                            SceneDepthEncoding::LinearView,
                                                            params.render_extent,
                                                            0.1f,
                                                            1000.0f,
                                                            false,
                                                            false);
        params.current_depth.current_depth_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        EXPECT_FALSE(validTemporalDepthInputs(params));
        params.current_depth.current_depth_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        EXPECT_TRUE(validTemporalDepthInputs(params));
        params.view = TemporalViewId::SplitLeft;
        EXPECT_FALSE(validTemporalDepthInputs(params));
        params.current_depth.view = TemporalViewId::SplitLeft;
        EXPECT_TRUE(validTemporalDepthInputs(params));
    }

    TEST(VulkanSceneTemporalResolveContract, DepthPingPongSlotsArePerViewAndBounded) {
        EXPECT_EQ(temporalDepthHistoryResourceSlot(TemporalViewId::Main, 0), 0u);
        EXPECT_EQ(temporalDepthHistoryResourceSlot(TemporalViewId::Main, 1), 1u);
        EXPECT_EQ(temporalDepthHistoryResourceSlot(TemporalViewId::SplitLeft, 0), 2u);
        EXPECT_EQ(temporalDepthHistoryResourceSlot(TemporalViewId::SplitRight, 1), 5u);
        EXPECT_FALSE(temporalDepthHistoryResourceSlot(TemporalViewId::Main, 2));
        EXPECT_FALSE(temporalDepthHistoryResourceSlot(TemporalViewId::Count, 0));
    }

    TEST(VulkanSceneTemporalPipelineContract, RequiresCoherentViewsExtentsAndInputs) {
        auto request = pipelineRequest();
        EXPECT_TRUE(validVulkanSceneTemporalPipelineRequest(request));
        request.resolve.output_extent.x += 1;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.temporal.frame.output_extent.x -= 1;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.resolve.view = TemporalViewId::SplitLeft;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.resolve.motion_view = reinterpret_cast<VkImageView>(3);
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.resolve.current_depth.current_depth_view = reinterpret_cast<VkImageView>(4);
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request = pipelineRequest();
        request.temporal.frame.view.size.x -= 1;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
    }

    TEST(VulkanSceneTemporalPipelineContract, DepthHistoryRequirementMatchesOwnership) {
        auto request = pipelineRequest();
        EXPECT_TRUE(validVulkanSceneTemporalPipelineRequest(request));
        request.resolve.current_depth.enabled = false;
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
        request.temporal.requirements.history_depth = false;
        EXPECT_TRUE(validVulkanSceneTemporalPipelineRequest(request));
    }

    TEST(VulkanSceneTemporalPipelineContract, JitterRequirementMatchesRenderedInput) {
        auto request = pipelineRequest();
        request.temporal.requirements.jitter = false;
        request.temporal.frame.jitter = {0.0f, 0.0f};
        EXPECT_TRUE(validVulkanSceneTemporalPipelineRequest(request));
    }

    TEST(VulkanSceneTemporalPipelineContract, InactiveRequestRemainsZeroCost) {
        VulkanSceneTemporalPipelineRequest request;
        EXPECT_TRUE(validVulkanSceneTemporalPipelineRequest(request));
        request.temporal.render_extent = {640, 360};
        EXPECT_FALSE(validVulkanSceneTemporalPipelineRequest(request));
    }

} // namespace lfs::vis
