/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../external/nanoflann.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

namespace {
    struct Cloud {
        const std::vector<float>* points;
        [[nodiscard]] size_t kdtree_get_point_count() const { return points->size() / 3; }
        [[nodiscard]] float kdtree_get_pt(const size_t i, const size_t d) const { return (*points)[i * 3 + d]; }
        template <class BBox>
        bool kdtree_get_bbox(BBox&) const { return false; }
    };
    using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3>;

    std::vector<float> read_colmap_points(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        std::uint64_t count = 0;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        std::vector<float> points;
        points.reserve(count * 3);
        for (std::uint64_t i = 0; i < count && file; ++i) {
            std::array<char, 8 + 24 + 3 + 8> record{};
            file.read(record.data(), record.size());
            std::array<double, 3> xyz{};
            std::memcpy(xyz.data(), record.data() + 8, sizeof(xyz));
            points.insert(points.end(), {static_cast<float>(xyz[0]), static_cast<float>(xyz[1]), static_cast<float>(xyz[2])});
            std::uint64_t track = 0;
            file.read(reinterpret_cast<char*>(&track), sizeof(track));
            file.seekg(static_cast<std::streamoff>(track * 8), std::ios::cur);
        }
        return points;
    }

    std::array<float, 4> brute_force_sq_dists(const std::vector<float>& points, const size_t query) {
        std::array<float, 4> best;
        best.fill(std::numeric_limits<float>::max());
        for (size_t j = 0; j < points.size() / 3; ++j) {
            float d = 0.0f;
            for (size_t k = 0; k < 3; ++k) {
                const float diff = points[query * 3 + k] - points[j * 3 + k];
                d += diff * diff;
            }
            if (d < best[3]) {
                best[3] = d;
                std::sort(best.begin(), best.end());
            }
        }
        return best;
    }
} // namespace

// Fails if the build permutes the cached coordinates out of step with the point indices: the tree then
// splits on the wrong values and exact searches miss true neighbours, single-threaded or concurrent.
TEST(NanoflannBuild, ExactNeighboursOnRealPoints) {
    const auto points = read_colmap_points(std::filesystem::path(PROJECT_ROOT_PATH) / "data/garden/sparse/0/points3D.bin");
    ASSERT_GT(points.size() / 3, size_t{100000});
    const Cloud cloud{&points};
    for (const unsigned threads : {1u, 4u}) {
        SCOPED_TRACE(threads);
        const Tree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10, nanoflann::KDTreeSingleIndexAdaptorFlags::None, threads));
        for (size_t query = 0; query < points.size() / 3; query += 397) {
            std::array<size_t, 4> indices{};
            std::array<float, 4> dists{};
            nanoflann::KNNResultSet<float> result(4);
            result.init(indices.data(), dists.data());
            tree.findNeighbors(result, &points[query * 3], nanoflann::SearchParameters(0));
            ASSERT_EQ(dists, brute_force_sq_dists(points, query)) << "query " << query;
        }
    }
}
