/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../external/nanoflann.hpp"
#include "core/cuda/initial_scales.hpp"
#include "cuda_backend_test.hpp"
#include "io/formats/colmap.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
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
    using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3>;

    float max_scale(const std::vector<float>& points) {
        float extents[3];
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<float> values;
            for (size_t i = axis; i < points.size(); i += 3) {
                if (std::isfinite(points[i]))
                    values.push_back(points[i]);
            }
            const size_t len = values.size();
            const auto lower = static_cast<size_t>(0.125f * static_cast<float>(len));
            const auto upper = std::min(len - 1, static_cast<size_t>(0.875f * static_cast<float>(len)));
            std::sort(values.begin(), values.end());
            extents[axis] = (values[upper] - values[lower]) * 0.5f;
        }
        std::sort(extents, extents + 3);
        return std::max(extents[1] * 2.0f, 0.01f) * 0.1f;
    }

    // The CPU kd-tree implementation the kernel replaced.
    std::vector<float> kd_tree_log_scales(const std::vector<float>& points) {
        const size_t count = points.size() / 3;
        const float limit = max_scale(points);
        const Cloud cloud{points.data(), count};
        const Tree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        std::vector<float> scales(count);
        for (size_t i = 0; i < count; ++i) {
            size_t indices[3];
            float dists[3];
            nanoflann::KNNResultSet<float> result(3);
            result.init(indices, dists);
            tree.findNeighbors(result, &points[i * 3], nanoflann::SearchParameters(0));
            const float dist = (std::sqrt(std::max(dists[1], 0.0f)) + std::sqrt(std::max(dists[2], 0.0f))) * 0.25f;
            scales[i] = std::log(std::clamp(dist, 1e-3f, limit));
        }
        return scales;
    }

    // Exhaustive search with the kernel's arithmetic: fused squared distance and a double-precision log.
    float exhaustive_log_scale(const std::vector<float>& points, const size_t query, const float limit) {
        float best[3] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::infinity()};
        for (size_t j = 0; j < points.size() / 3; ++j) {
            const float dx = points[query * 3] - points[j * 3];
            const float dy = points[query * 3 + 1] - points[j * 3 + 1];
            const float dz = points[query * 3 + 2] - points[j * 3 + 2];
            const float d = std::fma(dz, dz, std::fma(dy, dy, dx * dx));
            if (d < best[2]) {
                best[2] = d;
                std::sort(best, best + 3);
            }
        }
        const float dist = (std::sqrt(best[1]) + std::sqrt(best[2])) * 0.25f;
        return static_cast<float>(std::log(static_cast<double>(std::clamp(dist, 1e-3f, limit))));
    }

    std::vector<float> gpu_log_scales(const std::vector<float>& points) {
        const size_t count = points.size() / 3;
        auto means = Tensor::from_vector(points, {count, 3}, Device::CPU).cuda();
        auto scaling = Tensor::full({count, 3}, 1234.0f, Device::GPU);
        lfs::core::cuda::mrnf_knn_log_scales(means, scaling);
        return scaling.cpu().to_vector();
    }

    std::vector<float> garden_points() {
        auto cloud = lfs::io::read_colmap_point_cloud(std::filesystem::path(PROJECT_ROOT_PATH) / "data/garden/sparse/0");
        if (!cloud)
            return {};
        auto means = cloud->value.means.cpu().contiguous();
        return {means.ptr<float>(), means.ptr<float>() + means.numel()};
    }

    void expect_exhaustive_match(const std::vector<float>& points, const size_t query_stride) {
        const auto gpu = gpu_log_scales(points);
        const float limit = max_scale(points);
        for (size_t query = 0; query < points.size() / 3; query += query_stride) {
            const float expected = exhaustive_log_scale(points, query, limit);
            for (int c = 0; c < 3; ++c)
                ASSERT_EQ(gpu[query * 3 + c], expected) << "point " << query;
        }
    }
} // namespace

using InitialScales = lfs::test::CudaBackendTest;

// Fails if the GPU tree walk prunes a true neighbour: an exhaustive search with the same arithmetic must agree.
TEST_F(InitialScales, MatchesExhaustiveSearchOnRealPoints) {
    const auto points = garden_points();
    ASSERT_GT(points.size() / 3, size_t{100000});
    expect_exhaustive_match(points, 97);
}

// Duplicated points have zero-distance neighbours, so each must find its copy and not count itself twice.
TEST_F(InitialScales, DuplicatedPointsMatchExhaustiveSearch) {
    auto points = garden_points();
    ASSERT_GE(points.size(), size_t{30000 * 3});
    points.resize(30000 * 3);
    const std::vector<float> copies(points.begin(), points.begin() + 5000 * 3);
    points.insert(points.end(), copies.begin(), copies.end());
    expect_exhaustive_match(points, 7);
}

TEST_F(InitialScales, ThreePointsMatchExhaustiveSearch) {
    auto points = garden_points();
    ASSERT_FALSE(points.empty());
    points.resize(9);
    expect_exhaustive_match(points, 1);
}

// Fails if the kernel drifts from the CPU kd-tree it replaced by more than the rounding of the distance metric.
TEST_F(InitialScales, MatchesReplacedKdTree) {
    const auto points = garden_points();
    ASSERT_FALSE(points.empty());
    const auto gpu = gpu_log_scales(points);
    const auto reference = kd_tree_log_scales(points);
    for (size_t i = 0; i < reference.size(); ++i) {
        for (int c = 0; c < 3; ++c)
            ASSERT_NEAR(gpu[i * 3 + c], reference[i], 1e-6f) << "point " << i;
    }
}
