/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The LPIPS dispatch's pool reduce, with and without a mask's weights, on whichever GPU backend the test binary
// selects, against a reference on the host.

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    constexpr std::size_t kChannels = 64, kHeight = 6, kWidth = 9, kPlane = kHeight * kWidth;

    std::vector<float> pattern(const std::size_t count, const int seed) {
        std::vector<float> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = static_cast<float>(static_cast<int>((i * 17 + seed * 13) % 101) - 50) / 256.f;
        return values;
    }

    std::vector<float> host_values(const Tensor& tensor) { return tensor.to(DataType::Float32).cpu().to_vector(); }

    // What pool reduce adds up for one pixel: the lin-weighted squared distance of the two channel-normalized
    // features.
    double distance(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& lin,
                    const std::size_t pixel) {
        double sx = 0, sy = 0;
        for (std::size_t c = 0; c < kChannels; ++c) {
            sx += double(x[c * kPlane + pixel]) * x[c * kPlane + pixel];
            sy += double(y[c * kPlane + pixel]) * y[c * kPlane + pixel];
        }
        const double ix = 1.0 / std::sqrt(sx + 1e-10), iy = 1.0 / std::sqrt(sy + 1e-10);
        double sum = 0;
        for (std::size_t c = 0; c < kChannels; ++c) {
            const double d = x[c * kPlane + pixel] * ix - y[c * kPlane + pixel] * iy;
            sum += lin[c] * d * d;
        }
        return sum;
    }

} // namespace

// A masked score weighs each pixel of the region by the weights map at (row + weights_y0, column + weights_x0);
// without weights each pixel counts once, times inverse_count.
TEST(LpipsPoolReduce, WeighsEachPixelByTheMask) {
    const auto* ops = lfs::training::training_ops(lfs::core::default_gpu_backend()).lpips;
    if (ops == nullptr)
        GTEST_SKIP() << "No LPIPS dispatch on this backend";
    const auto x = Tensor::from_vector(pattern(kChannels * kPlane, 1), TensorShape({1, kChannels, kHeight, kWidth}),
                                       Device::GPU)
                       .to(DataType::Float16);
    const auto y = Tensor::from_vector(pattern(kChannels * kPlane, 2), TensorShape({1, kChannels, kHeight, kWidth}),
                                       Device::GPU)
                       .to(DataType::Float16);
    std::vector<float> lin_values(kChannels);
    for (std::size_t c = 0; c < kChannels; ++c)
        lin_values[c] = static_cast<float>(c % 7 + 1) / 16.f;
    const auto lin = Tensor::from_vector(lin_values, TensorShape({1, kChannels, 1, 1}), Device::GPU).to(DataType::Float16);
    // The reference reads the fp16 values back, so both sides start from the same inputs.
    const auto hx = host_values(x), hy = host_values(y), hlin = host_values(lin);

    // The region leaves a border on every side; the weights map is larger than the features and offset in them, as
    // a tile's crop is in the whole view's mask. A seventh of its entries are 0, the rest 1/4 to 1.
    constexpr int y0 = 1, y1 = 5, x0 = 2, x1 = 8, map_y0 = 1, map_x0 = 2;
    constexpr std::size_t map_height = kHeight + 2, map_width = kWidth + 3;
    std::vector<float> map(map_height * map_width);
    for (std::size_t i = 0; i < map.size(); ++i)
        map[i] = (i / 5) % 7 == 0 ? 0.f : static_cast<float>(i % 4 + 1) / 4.f;
    const auto weights = Tensor::from_vector(map, TensorShape({map_height, map_width}), Device::GPU);

    double plain = 0, masked = 0;
    for (int row = y0; row < y1; ++row)
        for (int column = x0; column < x1; ++column) {
            const double d = distance(hx, hy, hlin, static_cast<std::size_t>(row) * kWidth + column);
            plain += d;
            masked += d * map[static_cast<std::size_t>(row + map_y0) * map_width + column + map_x0];
        }
    const float inverse_count = 1.f / static_cast<float>((y1 - y0) * (x1 - x0));
    plain *= inverse_count;

    Tensor absent;
    auto plain_score = Tensor::zeros({1}, Device::GPU);
    ops->pool_reduce(x, y, lin, plain_score, absent, absent, {y0, y1, x0, x1, inverse_count});
    auto masked_score = Tensor::zeros({1}, Device::GPU);
    ops->pool_reduce(x, y, lin, masked_score, absent, absent,
                     {y0, y1, x0, x1, 1.f, &weights, static_cast<int>(map_width), map_y0, map_x0});
    const float got_plain = host_values(plain_score)[0], got_masked = host_values(masked_score)[0];
    EXPECT_NEAR(got_plain, plain, 1e-4 * plain) << "unmasked";
    EXPECT_NEAR(got_masked, masked, 1e-4 * masked) << "masked (the unmasked sum is "
                                                   << plain / inverse_count << ")";
}
