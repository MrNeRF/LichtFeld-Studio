/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../external/nanoflann.hpp"
#include "core/cuda/initial_scales.hpp"
#include "io/formats/colmap.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

namespace {
    using lfs::core::Device;
    using lfs::core::Tensor;

    struct Cloud {
        const float* points;
        size_t count;
        [[nodiscard]] size_t kdtree_get_point_count() const { return count; }
        [[nodiscard]] float kdtree_get_pt(const size_t i, const size_t d) const { return points[i * 3 + d]; }
        template <class BBox>
        bool kdtree_get_bbox(BBox&) const { return false; }
    };
    // nanoflann's L2_Simple accumulation with the kernel's explicit fused multiply-adds.
    struct FusedL2 {
        using ElementType = float;
        using DistanceType = float;
        const Cloud& data_source;
        explicit FusedL2(const Cloud& source) : data_source(source) {}
        [[nodiscard]] float evalMetric(const float* a, const uint32_t b, size_t) const {
            const float dx = a[0] - data_source.kdtree_get_pt(b, 0);
            const float dy = a[1] - data_source.kdtree_get_pt(b, 1);
            const float dz = a[2] - data_source.kdtree_get_pt(b, 2);
            return std::fma(dz, dz, std::fma(dy, dy, dx * dx));
        }
        template <typename U, typename V>
        [[nodiscard]] float accum_dist(const U a, const V b, size_t) const { return (a - b) * (a - b); }
    };
    using Tree = nanoflann::KDTreeSingleIndexAdaptor<FusedL2, Cloud, 3>;

    // The CPU kd-tree implementation the GPU kernel replaced, with the kernel's arithmetic.
    std::vector<float> reference_log_scales(const std::vector<float>& points) {
        const size_t count = points.size() / 3;
        float extents[3];
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<float> values;
            for (size_t i = 0; i < count; ++i) {
                if (std::isfinite(points[i * 3 + axis]))
                    values.push_back(points[i * 3 + axis]);
            }
            const size_t len = values.size();
            const auto lower = static_cast<size_t>(0.125f * static_cast<float>(len));
            const auto upper = std::min(len - 1, static_cast<size_t>(0.875f * static_cast<float>(len)));
            std::sort(values.begin(), values.end());
            extents[axis] = (values[upper] - values[lower]) * 0.5f;
        }
        std::sort(extents, extents + 3);
        const float max_scale = std::max(extents[1] * 2.0f, 0.01f) * 0.1f;

        const Cloud cloud{points.data(), count};
        const Tree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        std::vector<float> scales(count);
        for (size_t i = 0; i < count; ++i) {
            size_t indices[3];
            float dists[3];
            nanoflann::KNNResultSet<float> result(3);
            result.init(indices, dists);
            tree.findNeighbors(result, &points[i * 3], nanoflann::SearchParameters(0));
            const float dist = (std::sqrt(dists[1]) + std::sqrt(dists[2])) * 0.25f;
            scales[i] = static_cast<float>(std::log(static_cast<double>(std::clamp(dist, 1e-3f, max_scale))));
        }
        return scales;
    }

    std::vector<float> garden_points() {
        auto cloud = lfs::io::read_colmap_point_cloud(std::filesystem::path(PROJECT_ROOT_PATH) / "data/garden/sparse/0");
        if (!cloud)
            return {};
        auto means = cloud->value.means.cpu().contiguous();
        return {means.ptr<float>(), means.ptr<float>() + means.numel()};
    }

    void expect_matches_reference(const std::vector<float>& points) {
        const size_t count = points.size() / 3;
        auto means = Tensor::from_vector(points, {count, 3}, Device::CPU).cuda();
        auto scaling = Tensor::full({count, 3}, 1234.0f, Device::CUDA);
        lfs::core::cuda::mrnf_knn_log_scales(means, scaling);
        const auto gpu = scaling.cpu().to_vector();
        const auto reference = reference_log_scales(points);

        for (size_t i = 0; i < count; ++i) {
            for (int c = 0; c < 3; ++c)
                ASSERT_EQ(gpu[i * 3 + c], reference[i]) << "point " << i;
        }
    }
} // namespace

// Fails if the GPU tree walk prunes a true neighbour or the scale formula drifts from the CPU kd-tree it replaced.
TEST(InitialScales, MatchesKdTreeOnRealPoints) {
    const auto points = garden_points();
    ASSERT_GT(points.size() / 3, size_t{100000});
    expect_matches_reference(points);
}

// Duplicated points have zero-distance neighbours, so each must find its copy and not count itself twice.
TEST(InitialScales, DuplicatedPointsMatchKdTree) {
    auto points = garden_points();
    ASSERT_FALSE(points.empty());
    points.resize(30000 * 3);
    points.insert(points.end(), points.begin(), points.begin() + 5000 * 3);
    expect_matches_reference(points);
}

TEST(InitialScales, ThreePoints) {
    auto points = garden_points();
    ASSERT_FALSE(points.empty());
    points.resize(9);
    expect_matches_reference(points);
}
