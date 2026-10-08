/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/mesh_offscreen_renderer.hpp"

#include "core/splat_data.hpp"
#include "io/formats/ply.hpp"
#include "rendering/render_constants.hpp"

#include <array>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <utility>
#include <vector>

namespace {

    [[nodiscard]] float projectedDepth(const glm::mat4& projection, const float view_depth) {
        const glm::vec4 clip = projection * glm::vec4(0.0f, 0.0f, -view_depth, 1.0f);
        return clip.z / clip.w;
    }

    TEST(MeshOffscreenDepthTest, LinearizesPerspectiveProjectionDepth) {
        const glm::mat4 projection =
            glm::perspective(glm::radians(55.0f), 16.0f / 9.0f, 0.1f, 250.0f);
        ASSERT_FLOAT_EQ(projection[2][3], -1.0f);

        for (const float expected_depth : std::array{0.1f, 1.0f, 37.0f, 200.0f}) {
            const float z_ndc = projectedDepth(projection, expected_depth);
            EXPECT_NEAR(
                lfs::vis::linearizeMeshViewDepth(z_ndc, projection),
                expected_depth,
                expected_depth * 2e-4f);
        }
    }

    TEST(MeshOffscreenDepthTest, LinearizesOrthographicProjectionDepth) {
        const glm::mat4 projection =
            glm::ortho(-4.0f, 4.0f, -3.0f, 3.0f, 0.25f, 80.0f);
        ASSERT_FLOAT_EQ(projection[2][3], 0.0f);

        for (const float expected_depth : std::array{0.25f, 1.0f, 23.0f, 79.0f}) {
            const float z_ndc = projectedDepth(projection, expected_depth);
            EXPECT_NEAR(
                lfs::vis::linearizeMeshViewDepth(z_ndc, projection),
                expected_depth,
                1e-4f);
        }
    }

    [[nodiscard]] glm::dvec2 pixelOf(const glm::mat4& view_projection, const glm::vec3& point, const glm::ivec2 size) {
        const glm::vec4 clip = view_projection * glm::vec4(point, 1.0f);
        return {(clip.x / clip.w + 1.0) * 0.5 * size.x, (1.0 - clip.y / clip.w) * 0.5 * size.y};
    }

    TEST(MeshExportCropTest, CroppedProjectionKeepsFullImagePixelPositions) {
        auto ply = lfs::io::load_ply(std::filesystem::path(TEST_DATA_DIR) / "kerstbol-isolated-rotated_137502.ply");
        ASSERT_TRUE(ply);
        const auto means = ply->value.means().cpu().contiguous();
        ASSERT_EQ(means.ndim(), 2u);
        ASSERT_EQ(means.size(1), 3u);
        const float* const xyz = means.ptr<float>();
        const std::size_t count = means.size(0);
        glm::vec3 centroid(0.0f);
        for (std::size_t i = 0; i < count; ++i) {
            centroid += glm::vec3(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]) / static_cast<float>(count);
        }
        const glm::mat4 view = glm::lookAt(centroid + glm::vec3(3.0f, 2.0f, 30.0f), centroid, glm::vec3(0.0f, 1.0f, 0.0f));

        const glm::ivec2 full{19200, 17280};
        for (const bool orthographic : {false, true}) {
            const glm::mat4 projection = lfs::rendering::createProjectionMatrix(full, 40.0f, orthographic, 600.0f);
            const glm::mat4 full_view_projection = projection * view;
            glm::dvec2 lo(std::numeric_limits<double>::max());
            glm::dvec2 hi(std::numeric_limits<double>::lowest());
            for (std::size_t i = 0; i < count; i += 7) {
                const glm::dvec2 pixel = pixelOf(full_view_projection, {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]}, full);
                lo = glm::min(lo, pixel);
                hi = glm::max(hi, pixel);
            }
            ASSERT_LT(lo.x, hi.x);
            ASSERT_LT(lo.y, hi.y);
            const glm::ivec2 center((lo + hi) * 0.5);
            const std::array<std::pair<glm::ivec2, glm::ivec2>, 4> rects{{
                {{0, 0}, full},
                {{0, center.y - 256}, {full.x, 512}},
                {center - glm::ivec2(888, 109), {1777, 219}},
                {{0, static_cast<int>(hi.y) - 64}, {full.x, 128}},
            }};
            for (const auto& [origin, size] : rects) {
                const glm::mat4 crop_view_projection =
                    lfs::vis::cropProjectionToRect(projection, full, origin, size) * view;
                std::size_t inside = 0;
                for (std::size_t i = 0; i < count; i += 7) {
                    const glm::vec3 point(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
                    const glm::dvec2 in_rect = pixelOf(full_view_projection, point, full) - glm::dvec2(origin);
                    if (in_rect.x < 0.0 || in_rect.y < 0.0 || in_rect.x >= size.x || in_rect.y >= size.y) {
                        continue;
                    }
                    ++inside;
                    const glm::dvec2 cropped = pixelOf(crop_view_projection, point, size);
                    EXPECT_NEAR(cropped.x, in_rect.x, 0.02) << "splat " << i;
                    EXPECT_NEAR(cropped.y, in_rect.y, 0.02) << "splat " << i;
                }
                EXPECT_GT(inside, 0u) << "rect at " << origin.x << "," << origin.y;
            }
        }
    }

    TEST(MeshExportCompositeTest, MeshPixelsWinOnlyInFrontOfTheSplats) {
        // One row of five mesh pixels at depth 5; the second pixel has no mesh.
        const lfs::vis::MeshLayer layer{
            .rgba = lfs::core::Tensor::from_vector(
                std::vector<float>{0.2f, 0.2f, 0.2f, 0.2f, 1.5f,
                                   0.4f, 0.4f, 0.4f, 0.4f, -0.5f,
                                   0.6f, 0.6f, 0.6f, 0.6f, 0.6f,
                                   1.0f, 0.0f, 1.0f, 1.0f, 1.0f},
                {4, 1, 5},
                lfs::core::Device::CPU),
            .view_depth = lfs::core::Tensor::from_vector(
                std::vector<float>{5.0f, 5.0f, 5.0f, 5.0f, 5.0f}, {1, 5}, lfs::core::Device::CPU),
        };
        // Splats in front, mesh pixel missing, splats behind, empty splat pixel, equal depth.
        const std::vector<float> splat_depth{3.0f, 10.0f, 10.0f, 1e10f, 5.0f, 99.0f};

        for (const std::size_t channels : {std::size_t{3}, std::size_t{4}}) {
            auto image = lfs::core::Tensor::full({2, 7, channels}, 9.0f, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            lfs::vis::compositeMeshLayer(layer, splat_depth.data(), splat_depth.size(), image, {1, 1});
            const auto* const pixels = image.ptr<std::uint8_t>();
            const auto at = [&](const int x, const int y, const std::size_t c) {
                return pixels[(static_cast<std::size_t>(y) * 7 + x) * channels + c];
            };
            const std::array<bool, 5> mesh_wins{false, false, true, true, false};
            for (int x = 0; x < 7; ++x) {
                for (std::size_t c = 0; c < channels; ++c) {
                    EXPECT_EQ(at(x, 0, c), 9) << "row above the layer, x=" << x;
                }
            }
            for (int i = 0; i < 5; ++i) {
                const std::array<std::uint8_t, 3> mesh_rgb{51, 102, 153};
                for (std::size_t c = 0; c < 3; ++c) {
                    EXPECT_EQ(at(1 + i, 1, c), mesh_wins[i] ? mesh_rgb[c] : 9) << "pixel " << i << " channel " << c;
                }
                if (channels == 4) {
                    EXPECT_EQ(at(1 + i, 1, 3), mesh_wins[i] ? 255 : 9) << "pixel " << i;
                }
            }
            EXPECT_EQ(at(0, 1, 0), 9);
            EXPECT_EQ(at(6, 1, 0), 9);

            auto no_splats = lfs::core::Tensor::full({1, 5, channels}, 9.0f, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            lfs::vis::compositeMeshLayer(layer, nullptr, 0, no_splats, {0, 0});
            const auto* const plain = no_splats.ptr<std::uint8_t>();
            EXPECT_EQ(plain[0], 51);
            EXPECT_EQ(plain[channels], 9) << "pixel without a mesh stays";
            EXPECT_EQ(plain[4 * channels], 255) << "mesh colors clamp to 1";
            EXPECT_EQ(plain[4 * channels + 1], 0) << "mesh colors clamp to 0";
        }
    }

} // namespace
