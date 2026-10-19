/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"
#include "io/formats/rad.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numbers>
#include <queue>
#include <utility>
#include <vector>

namespace {

    struct DecodedTree {
        std::vector<float> means;
        std::vector<float> scales;
        std::vector<float> opacity;
        std::vector<std::uint16_t> child_count;
        std::vector<std::uint32_t> child_start;
    };

    std::vector<float> host(const lfs::core::Tensor& tensor) { return tensor.cpu().contiguous().to_vector(); }

    DecodedTree decode(const std::filesystem::path& path) {
        auto loaded = lfs::io::load_rad(path);
        EXPECT_TRUE(loaded.has_value()) << (loaded ? "" : loaded.error());
        if (!loaded || !loaded->lod_tree || !loaded->lod_tree->nodes_in_memory()) {
            ADD_FAILURE() << "RAD export has no in-memory LOD tree: " << path;
            return {};
        }
        return {.means = host(loaded->get_means()),
                .scales = host(loaded->get_scaling()),
                .opacity = host(loaded->opacity_raw()),
                .child_count = loaded->lod_tree->child_count,
                .child_start = loaded->lod_tree->child_start};
    }

    class RadSparkPagingTest : public ::testing::Test {
    protected:
        void SetUp() override {
            const std::filesystem::path ply(LFS_SIMPLIFY_TEST_PLY);
            if (!std::filesystem::exists(ply)) {
                GTEST_SKIP() << "Missing real splat asset: " << ply;
            }
            auto loaded = lfs::io::load_ply(ply);
            ASSERT_TRUE(loaded.has_value()) << loaded.error().detail();
            ASSERT_EQ(loaded->value.size(), 137502u);

            dir_ = std::filesystem::temp_directory_path() / "lfs_rad_spark_paging";
            std::filesystem::remove_all(dir_);
            std::filesystem::create_directories(dir_);
            for (const auto& [name, chunk_size] : {std::pair{"streamed.rad", lfs::io::kRadStreamableChunkSplats},
                                                   std::pair{"native.rad", lfs::io::kRadNativeChunkSplats}}) {
                const auto path = dir_ / name;
                ASSERT_TRUE(lfs::io::save_rad(loaded->value, {.output_path = path, .chunk_size = chunk_size}).has_value());
            }
            streamed_ = decode(dir_ / "streamed.rad");
            native_ = decode(dir_ / "native.rad");
            ASSERT_FALSE(streamed_.child_count.empty());
            ASSERT_EQ(streamed_.child_count.size(), native_.child_count.size());
        }

        void TearDown() override { std::filesystem::remove_all(dir_); }

        std::filesystem::path dir_;
        DecodedTree streamed_;
        DecodedTree native_;
    };

    // Catches a reorder that scrambles attribute rows against tree links: walking
    // both files from the root must meet the same node at every step.
    TEST_F(RadSparkPagingTest, StreamedExportKeepsEveryNodeAndLink) {
        const std::size_t n = streamed_.child_count.size();
        std::queue<std::pair<std::uint32_t, std::uint32_t>> pending;
        pending.emplace(0u, 0u);
        std::size_t visited = 0;
        while (!pending.empty()) {
            const auto [s, t] = pending.front();
            pending.pop();
            ++visited;
            ASSERT_LT(s, n);
            ASSERT_LT(t, n);
            for (int k = 0; k < 3; ++k) {
                ASSERT_EQ(streamed_.means[s * 3 + k], native_.means[t * 3 + k]) << "node " << s;
                ASSERT_FLOAT_EQ(streamed_.scales[s * 3 + k], native_.scales[t * 3 + k]) << "node " << s;
            }
            ASSERT_FLOAT_EQ(streamed_.opacity[s], native_.opacity[t]) << "node " << s;
            ASSERT_EQ(streamed_.child_count[s], native_.child_count[t]) << "node " << s;
            for (std::uint32_t c = 0; c < streamed_.child_count[s]; ++c) {
                ASSERT_GT(streamed_.child_start[s], s) << "children must follow their parent";
                pending.emplace(streamed_.child_start[s] + c, native_.child_start[t] + c);
            }
        }
        EXPECT_EQ(visited, n);
    }

    // Catches a depth-first layout: Spark refines the largest nodes first, and that
    // cut of the whole scene must sit in the first 64K chunk instead of spreading
    // over every chunk the way the first subtree of a depth-first file does.
    TEST_F(RadSparkPagingTest, StreamedExportKeepsTheCoarsestCutInTheFirstChunk) {
        constexpr std::size_t kCut = 16384;
        const auto& tree = streamed_;
        ASSERT_GT(tree.child_count.size(), 2 * lfs::io::kRadStreamableChunkSplats);
        const auto feature_size = [&](const std::uint32_t i) {
            const float max_scale = std::max({tree.scales[i * 3], tree.scales[i * 3 + 1], tree.scales[i * 3 + 2]});
            const float alpha = tree.opacity[i];
            return 2.0f * max_scale * (alpha > 1.0f ? std::sqrt(1.0f + std::numbers::e_v<float> * std::log(alpha)) : 1.0f);
        };
        std::priority_queue<std::pair<float, std::uint32_t>> frontier;
        frontier.emplace(feature_size(0), 0u);
        std::size_t selected = 1;
        std::uint32_t last_node = 0;
        while (!frontier.empty() && selected < kCut) {
            const std::uint32_t parent = frontier.top().second;
            frontier.pop();
            for (std::uint32_t c = 0; c < tree.child_count[parent]; ++c) {
                const std::uint32_t child = tree.child_start[parent] + c;
                last_node = std::max(last_node, child);
                frontier.emplace(feature_size(child), child);
                ++selected;
            }
        }
        EXPECT_LT(last_node, lfs::io::kRadStreamableChunkSplats);
    }

    // Catches a layout LichtFeld's own paged viewer rejects: its node sidecar
    // requires every child block to follow its parent.
    TEST_F(RadSparkPagingTest, StreamedExportOpensInThePagedViewer) {
        const auto path = dir_ / "streamed.rad";
        ASSERT_TRUE(lfs::io::build_rad_meta_sidecar(path).has_value());
        const auto view = lfs::io::open_rad_meta_sidecar(path);
        ASSERT_TRUE(view.has_value()) << view.error();
        EXPECT_EQ(view->node_count, streamed_.child_count.size());
        EXPECT_EQ(view->leaf_count, 137502u);
    }

} // namespace
