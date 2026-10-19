/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/point_cloud.hpp"
#include "core/splat_data_transform.hpp"
#include "io/formats/colmap.hpp"

#include <algorithm>
#include <filesystem>
#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

namespace {
    std::optional<lfs::core::PointCloud> first_real_point_cloud() {
        std::error_code error;
        std::vector<std::filesystem::path> models;
        for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(PROJECT_ROOT_PATH) / "data", error)) {
            if (std::filesystem::is_regular_file(entry.path() / "sparse/0/points3D.bin", error))
                models.push_back(entry.path() / "sparse/0");
        }
        std::sort(models.begin(), models.end());
        for (const auto& model : models) {
            auto loaded = lfs::io::read_colmap_point_cloud(model);
            if (loaded && loaded->value.size() > 10000)
                return std::move(loaded->value);
        }
        return std::nullopt;
    }
} // namespace

TEST(PercentileBounds, MatchSortedOrderStatisticsOnRealPoints) {
    const auto cloud_points = first_real_point_cloud();
    if (!cloud_points)
        GTEST_SKIP() << "no sparse model with enough points under " << PROJECT_ROOT_PATH << "/data";
    const auto means = cloud_points->means.cpu().contiguous();
    const auto n = static_cast<size_t>(means.size(0));
    const float* data = means.ptr<float>();
    const size_t lo = n / 100;
    const size_t hi = n - 1 - lo;
    constexpr float padding = 0.25f;

    for (const auto device : {lfs::core::Device::CPU, lfs::core::Device::CUDA}) {
        const lfs::core::PointCloud cloud(means.to(device), cloud_points->colors.to(device));
        glm::vec3 min_bounds{0.0f};
        glm::vec3 max_bounds{0.0f};
        ASSERT_TRUE(lfs::core::compute_bounds(cloud, min_bounds, max_bounds, padding, true));
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<float> column(n);
            for (size_t row = 0; row < n; ++row)
                column[row] = data[row * 3 + axis];
            std::sort(column.begin(), column.end());
            EXPECT_EQ(min_bounds[axis], column[lo] - padding) << "axis " << axis;
            EXPECT_EQ(max_bounds[axis], column[hi] + padding) << "axis " << axis;
        }
    }
}
