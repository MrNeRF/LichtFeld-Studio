/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "forward.h"
#include "training/rasterization/fast_rasterizer.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;

    struct CullMode {
        const int previous = fast_lfs::rasterization::warp_cull_mode_for_testing();
        explicit CullMode(int mode) { fast_lfs::rasterization::set_warp_cull_mode_for_testing(mode); }
        ~CullMode() { fast_lfs::rasterization::set_warp_cull_mode_for_testing(previous); }
    };

    struct Capture {
        std::array<Tensor, 4> images;
        std::vector<unsigned> indices;
        std::vector<unsigned> last;
        std::vector<std::array<unsigned, 4>> contributors;
        std::vector<uint2> ranges;
    };

    void copy(void* to, const void* from, size_t bytes, cudaStream_t stream) {
        if (bytes && cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            throw std::runtime_error("Could not capture renderer indices");
    }

    Capture capture(Camera& camera, SplatData& model, int mode, bool mip, bool tile) {
        const CullMode culling(mode);
        const int width = tile ? 31 : 65;
        const int height = tile ? 29 : 63;
        const auto background = Tensor::full({3, static_cast<size_t>(height), static_cast<size_t>(width)}, 0.17f, Device::CUDA);
        auto background_color = Tensor::full({3}, 0.1f, Device::CUDA);
        auto result = fast_rasterize_forward(camera, model, background_color,
                                             tile ? 7 : 0, tile ? 5 : 0, tile ? width : 0, tile ? height : 0,
                                             mip, background, true, true);
        if (!result)
            throw std::runtime_error(lfs::format_for_developer(result.error()));
        auto& [output, context] = *result;
        Capture data;
        data.images = {output.image.to(Device::CPU), output.alpha.to(Device::CPU),
                       output.depth.to(Device::CPU), output.normal.to(Device::CPU)};
        const auto& raw = context.forward_ctx;
        const auto view = fast_lfs::rasterization::forward_capture_view(raw, width, height);
        data.indices.resize(raw.n_instances);
        data.last.resize(width * height);
        data.ranges.resize(((width + 15) / 16) * ((height + 15) / 16));
        copy(data.indices.data(), raw.sorted_primitive_indices, data.indices.size() * sizeof(unsigned), raw.stream);
        copy(data.last.data(), view.last_contributors, data.last.size() * sizeof(unsigned), raw.stream);
        copy(data.ranges.data(), view.tile_ranges, data.ranges.size() * sizeof(uint2), raw.stream);
        if (raw.contributor_masks) {
            std::vector<unsigned> positions(raw.n_instances);
            std::vector<unsigned char> masks(raw.n_instances);
            copy(positions.data(), raw.contributor_positions, positions.size() * sizeof(unsigned), raw.stream);
            copy(masks.data(), raw.contributor_masks, masks.size(), raw.stream);
            if (cudaStreamSynchronize(raw.stream) != cudaSuccess)
                throw std::runtime_error("Could not synchronize contributor capture");
            for (size_t tile = 0; tile < data.ranges.size(); ++tile) {
                const auto range = data.ranges[tile];
                for (unsigned i = range.x; i < range.y; ++i) {
                    if (masks[i])
                        data.contributors.push_back({static_cast<unsigned>(tile), positions[i], data.indices[i], masks[i]});
                }
            }
        }
        if (cudaStreamSynchronize(raw.stream) != cudaSuccess)
            throw std::runtime_error("Could not synchronize renderer capture");
        return data;
    }

    TEST(FastGSForwardMask, CulledCompactedRenderMatchesUnculledRender) {
        constexpr size_t count = 259;
        auto means = Tensor::empty({count, 3}, Device::CPU);
        auto rotations = Tensor::zeros({count, 4}, Device::CPU);
        auto scales = Tensor::full({count, 3}, -3.2f, Device::CPU);
        for (size_t i = 0; i < count; ++i) {
            means.ptr<float>()[3 * i] = (static_cast<int>(i % 13) - 6) * 0.1f;
            means.ptr<float>()[3 * i + 1] = (static_cast<int>((i / 13) % 11) - 5) * 0.1f;
            means.ptr<float>()[3 * i + 2] = 0.001f * i;
            rotations.ptr<float>()[4 * i] = 1.0f;
            if (i % 17 == 0)
                scales.ptr<float>()[3 * i] = scales.ptr<float>()[3 * i + 1] = -0.2f;
        }
        SplatData model(0, means.cuda(), Tensor::full({count, 1, 3}, 0.3f, Device::CUDA),
                        Tensor::zeros({count, 0, 3}, Device::CUDA), scales.cuda(), rotations.cuda(),
                        Tensor::full({count}, 1.0f, Device::CUDA), 1.0f);
        auto translation = Tensor::zeros({3}, Device::CPU);
        translation.ptr<float>()[2] = 3.0f;
        Camera camera(Tensor::eye(3, Device::CUDA), translation.cuda(), 60.0f, 61.0f, 32.0f, 31.0f,
                      Tensor{}, Tensor{}, CameraModelType::PINHOLE, "mask", "", std::filesystem::path{}, 65, 63, 0);
        for (bool mip : {false, true}) {
            for (bool tile : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "mip=" << mip << " tile=" << tile);
                const auto culled = capture(camera, model, 0, mip, tile);
                const auto reference = capture(camera, model, 1, mip, tile);
                ASSERT_FALSE(culled.contributors.empty());
                EXPECT_EQ(culled.last, reference.last);
                EXPECT_EQ(culled.contributors, reference.contributors);
                for (size_t channel = 0; channel < culled.images.size(); ++channel) {
                    ASSERT_EQ(culled.images[channel].numel(), reference.images[channel].numel());
                    EXPECT_EQ(std::memcmp(culled.images[channel].data_ptr(), reference.images[channel].data_ptr(),
                                          culled.images[channel].numel() * sizeof(float)),
                              0)
                        << "channel=" << channel;
                }
                // Compaction keeps each tile's surviving entries in their original order.
                for (size_t i = 1; i < culled.contributors.size(); ++i) {
                    const auto& previous = culled.contributors[i - 1];
                    const auto& entry = culled.contributors[i];
                    if (entry[0] == previous[0])
                        EXPECT_LT(previous[1], entry[1]);
                }
                for (const auto& entry : culled.contributors) {
                    ASSERT_LT(entry[2], count);
                    EXPECT_NE(entry[3], 0u);
                }
                const auto negative = capture(camera, model, 2, mip, tile);
                EXPECT_NE(culled.last, negative.last);
                EXPECT_TRUE(std::all_of(negative.last.begin(), negative.last.end(), [](unsigned value) { return value == 0; }));
            }
        }
    }

    TEST(FastGSForwardMask, LiveRenderPreservesIdsAboveTwentyFourBits) {
        constexpr size_t count = (size_t{1} << 24) + 3;
        auto means = Tensor::full({count, 3}, -100.0f, Device::CUDA);
        auto rotations = Tensor::zeros({count, 4}, Device::CUDA);
        const float visible_means[] = {-0.05f, 0.0f, 0.0f, 0.05f, 0.0f, 0.1f};
        const float visible_rotations[] = {1, 0, 0, 0, 1, 0, 0, 0};
        ASSERT_EQ(cudaMemcpy(means.ptr<float>() + 3 * (count - 2), visible_means, sizeof(visible_means), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(rotations.ptr<float>() + 4 * (count - 2), visible_rotations, sizeof(visible_rotations), cudaMemcpyHostToDevice), cudaSuccess);
        SplatData model(0, std::move(means), Tensor::full({count, 1, 3}, 0.3f, Device::CUDA),
                        Tensor::zeros({count, 0, 3}, Device::CUDA), Tensor::full({count, 3}, -2.0f, Device::CUDA), std::move(rotations),
                        Tensor::full({count}, 0.0f, Device::CUDA), 1.0f);
        auto translation = Tensor::zeros({3}, Device::CPU);
        translation.ptr<float>()[2] = 3.0f;
        Camera camera(Tensor::eye(3, Device::CUDA), translation.cuda(), 60.0f, 61.0f, 32.0f, 31.0f,
                      Tensor{}, Tensor{}, CameraModelType::PINHOLE, "large_ids", "", std::filesystem::path{}, 65, 63, 0);
        const auto compact = capture(camera, model, 0, false, false);
        const auto reference = capture(camera, model, 1, false, false);
        ASSERT_FALSE(compact.contributors.empty());
        EXPECT_EQ(compact.contributors, reference.contributors);
        EXPECT_EQ(compact.last, reference.last);
        for (const auto& entry : compact.contributors)
            EXPECT_TRUE(entry[2] == count - 2 || entry[2] == count - 1);
    }
} // namespace
