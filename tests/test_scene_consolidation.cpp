/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/error.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "core/uuid.hpp"
#include "io/formats/ply.hpp"
#include "io/splat_chapter.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using lfs::core::Device;
using lfs::core::Scene;
using lfs::core::SplatData;
using lfs::core::Tensor;
using lfs::core::Uuid;

namespace {

    // q16 is uniform uint16 over 256-splat [block_min, block_max]. Extract
    // re-blocks independently of the combined model, so this covers two
    // encode/decode passes with different alignments (~2 * range / 65535).
    constexpr float kShNAbsTol = 2e-3f;

    [[nodiscard]] std::filesystem::path bike_ply_path() {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/bike.ply";
    }

    [[nodiscard]] Tensor range_mask(const size_t n, const size_t start, const size_t count,
                                    const Device device) {
        Tensor mask = Tensor::zeros_bool({n}, device);
        if (count > 0) {
            mask.slice(0, start, start + count) = Tensor::ones_bool({count}, device);
        }
        return mask;
    }

    struct CpuAttrs {
        std::vector<float> means;
        std::vector<float> sh0;
        std::vector<float> shN;
        std::vector<float> scaling;
        std::vector<float> rotation;
        std::vector<float> opacity;
        size_t rows = 0;
        int max_sh_degree = 0;
    };

    [[nodiscard]] CpuAttrs snapshot_cpu(const SplatData& model) {
        CpuAttrs attrs;
        attrs.rows = static_cast<size_t>(model.size());
        attrs.max_sh_degree = model.get_max_sh_degree();
        attrs.means = model.means_raw().to_vector();
        attrs.sh0 = model.sh0_raw().to_vector();
        attrs.shN = model.shN_canonical().to_vector();
        attrs.scaling = model.scaling_raw().to_vector();
        attrs.rotation = model.rotation_raw().to_vector();
        attrs.opacity = model.opacity_raw().to_vector();
        return attrs;
    }

    void expect_exact(const std::vector<float>& got, const std::vector<float>& expected,
                      const char* name) {
        ASSERT_EQ(got.size(), expected.size()) << name;
        for (size_t i = 0; i < got.size(); ++i) {
            EXPECT_EQ(got[i], expected[i]) << name << " i=" << i;
        }
    }

    void expect_shN_q16(const std::vector<float>& got, const std::vector<float>& expected) {
        ASSERT_EQ(got.size(), expected.size());
        float max_abs = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(got[i] - expected[i]));
        }
        EXPECT_LE(max_abs, kShNAbsTol) << "shN max abs error " << max_abs;
    }

    void expect_attrs_match(const SplatData& got, const CpuAttrs& expected) {
        ASSERT_EQ(static_cast<size_t>(got.size()), expected.rows);
        EXPECT_EQ(got.get_max_sh_degree(), expected.max_sh_degree);
        expect_exact(got.means_raw().to_vector(), expected.means, "means");
        expect_exact(got.sh0_raw().to_vector(), expected.sh0, "sh0");
        expect_exact(got.scaling_raw().to_vector(), expected.scaling, "scaling");
        expect_exact(got.rotation_raw().to_vector(), expected.rotation, "rotation");
        expect_exact(got.opacity_raw().to_vector(), expected.opacity, "opacity");
        expect_shN_q16(got.shN_canonical().to_vector(), expected.shN);
    }

    struct ThreeNodeScene {
        Scene scene;
        Uuid full_uuid;
        Uuid mid_uuid;
        Uuid last_uuid;
        CpuAttrs full_attrs;
        CpuAttrs mid_attrs;
        CpuAttrs last_attrs;
        size_t full_n = 0;
        size_t mid_n = 0;
        size_t last_n = 0;
    };

} // namespace

class SceneConsolidationExtractTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto path = bike_ply_path();
        if (!std::filesystem::exists(path)) {
            GTEST_SKIP() << "tests/data/bike.ply is not available";
        }
        auto loaded = lfs::io::load_ply(path);
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        bike_ = std::move(loaded->value);
        ASSERT_GE(static_cast<size_t>(bike_.size()), 32u)
            << "tests/data/bike.ply is too small for subset copies";
    }

    void fill_three_node_scene(ThreeNodeScene& built) {
        const size_t n = static_cast<size_t>(bike_.size());
        const auto device = bike_.means_raw().device();
        built.full_n = n;
        built.mid_n = n / 2;
        built.last_n = n / 3;
        ASSERT_GT(built.mid_n, 0u);
        ASSERT_GT(built.last_n, 0u);
        ASSERT_NE(built.mid_n, built.last_n);

        auto mid = lfs::core::extract_by_mask(
            bike_, range_mask(n, 0, built.mid_n, device));
        auto last = lfs::core::extract_by_mask(
            bike_, range_mask(n, n - built.last_n, built.last_n, device));
        ASSERT_EQ(static_cast<size_t>(mid.size()), built.mid_n);
        ASSERT_EQ(static_cast<size_t>(last.size()), built.last_n);

        built.full_attrs = snapshot_cpu(bike_);
        built.mid_attrs = snapshot_cpu(mid);
        built.last_attrs = snapshot_cpu(last);

        const auto full_id =
            built.scene.addSplat("full", std::make_unique<SplatData>(bike_.clone()));
        const auto mid_id =
            built.scene.addSplat("mid", std::make_unique<SplatData>(std::move(mid)));
        const auto last_id =
            built.scene.addSplat("last", std::make_unique<SplatData>(std::move(last)));
        ASSERT_NE(full_id, lfs::core::NULL_NODE);
        ASSERT_NE(mid_id, lfs::core::NULL_NODE);
        ASSERT_NE(last_id, lfs::core::NULL_NODE);
        built.full_uuid = built.scene.getNodeUuid(full_id);
        built.mid_uuid = built.scene.getNodeUuid(mid_id);
        built.last_uuid = built.scene.getNodeUuid(last_id);
    }

    SplatData bike_;
};

TEST_F(SceneConsolidationExtractTest, ExtractsEachNodeMatchingOriginals) {
    ThreeNodeScene built;
    fill_three_node_scene(built);
    ASSERT_EQ(built.scene.consolidateNodeModels(), 3u);
    ASSERT_TRUE(built.scene.isConsolidated());

    auto full = built.scene.extractConsolidatedNodeModel(built.full_uuid);
    auto mid = built.scene.extractConsolidatedNodeModel(built.mid_uuid);
    auto last = built.scene.extractConsolidatedNodeModel(built.last_uuid);
    ASSERT_NE(full, nullptr);
    ASSERT_NE(mid, nullptr);
    ASSERT_NE(last, nullptr);
    expect_attrs_match(*full, built.full_attrs);
    expect_attrs_match(*mid, built.mid_attrs);
    expect_attrs_match(*last, built.last_attrs);

    auto captured = lfs::io::project::SplatChapterPayload::capture(
        *full, lfs::io::project::SplatSourceKind::Generated, false);
    ASSERT_TRUE(captured) << lfs::format_for_developer(captured.error());
    auto hydrated = captured->hydrate();
    ASSERT_TRUE(hydrated) << lfs::format_for_developer(hydrated.error());
    ASSERT_NE(*hydrated, nullptr);
    EXPECT_EQ(static_cast<size_t>((*hydrated)->size()), built.full_n);
}

TEST_F(SceneConsolidationExtractTest, SingleVisibleNodeAliasesWithoutAllocatorUse) {
    Scene scene;
    const auto id = scene.addSplat("single", std::make_unique<SplatData>(bike_.clone()));
    ASSERT_NE(id, lfs::core::NULL_NODE);

    std::atomic<size_t> allocator_calls = 0;
    scene.setCombinedModelAllocator(
        [&allocator_calls](lfs::core::TensorShape shape,
                           size_t,
                           lfs::core::DataType dtype,
                           std::string_view) {
            ++allocator_calls;
            return Tensor::empty(std::move(shape), Device::CUDA, dtype);
        });

    const auto* combined = scene.getCombinedModel();
    ASSERT_NE(combined, nullptr);
    const auto* node = scene.getNodeById(id);
    ASSERT_NE(node, nullptr);
    ASSERT_NE(node->model, nullptr);
    EXPECT_EQ(combined, node->model.get());
    EXPECT_EQ(combined->means_raw().data_ptr(), node->model->means_raw().data_ptr());
    EXPECT_EQ(combined->sh0_raw().data_ptr(), node->model->sh0_raw().data_ptr());
    EXPECT_EQ(combined->scaling_raw().data_ptr(), node->model->scaling_raw().data_ptr());
    EXPECT_EQ(combined->rotation_raw().data_ptr(), node->model->rotation_raw().data_ptr());
    EXPECT_EQ(combined->opacity_raw().data_ptr(), node->model->opacity_raw().data_ptr());
    EXPECT_EQ(allocator_calls.load(), 0u);
}

TEST_F(SceneConsolidationExtractTest, WorkerBuildMatchesSynchronousCombinedModel) {
    auto first = std::make_shared<SplatData>(bike_.clone());
    auto second = std::make_shared<SplatData>(bike_.clone());
    const size_t first_size = static_cast<size_t>(first->size());

    std::vector<size_t> allocation_shape_rows;
    std::vector<size_t> allocation_capacities;
    std::vector<std::string> allocation_names;
    const auto counting_allocator =
        [&allocation_shape_rows, &allocation_capacities, &allocation_names](lfs::core::TensorShape shape,
                                                                            const size_t capacity,
                                                                            lfs::core::DataType dtype,
                                                                            std::string_view name) {
            allocation_shape_rows.push_back(shape[0]);
            allocation_capacities.push_back(capacity);
            allocation_names.emplace_back(name);
            return Tensor::empty(std::move(shape), Device::CUDA, dtype);
        };

    Tensor::trim_memory_pool();
    const auto allocation_snapshot = lfs::core::alloc_counter::snapshot();
    std::optional<Scene::CombinedModelBuild> worker_build;
    std::jthread worker([&] {
        worker_build = Scene::buildCombinedModelCache(
            {{first, true, 0}, {second, true, first_size}},
            first_size + static_cast<size_t>(second->size()),
            counting_allocator);
    });
    worker.join();
    // The direct concatenator may allocate its six result fields, q16 companion
    // storage, indices, and transient SH conversion workspaces. It must not add
    // two full input-model clone sets to that budget.
    EXPECT_LE(lfs::core::alloc_counter::delta_since(allocation_snapshot), 16u);
    ASSERT_TRUE(worker_build.has_value());
    ASSERT_NE(worker_build->model, nullptr);
    ASSERT_GE(allocation_names.size(), 5u);
    const size_t total_size = first_size + static_cast<size_t>(second->size());
    EXPECT_TRUE(std::all_of(allocation_names.begin(), allocation_names.end(),
                            [](const std::string& name) {
                                return name == "SplatData.means" ||
                                       name == "SplatData.sh0" ||
                                       name == "SplatData.shN" ||
                                       name == "SplatData.shN_value_bounds" ||
                                       name == "SplatData.opacity" ||
                                       name == "SplatData.scaling" ||
                                       name == "SplatData.rotation";
                            }));
    for (size_t i = 0; i < allocation_names.size(); ++i) {
        if (allocation_names[i] == "SplatData.means" ||
            allocation_names[i] == "SplatData.sh0" ||
            allocation_names[i] == "SplatData.opacity" ||
            allocation_names[i] == "SplatData.scaling" ||
            allocation_names[i] == "SplatData.rotation") {
            EXPECT_EQ(allocation_capacities[i], total_size);
        }
    }
    EXPECT_TRUE(std::any_of(allocation_shape_rows.begin(), allocation_shape_rows.end(),
                            [total_size](const size_t rows) { return rows == total_size; }));

    Scene synchronous;
    ASSERT_NE(synchronous.addSplat("first", std::make_unique<SplatData>(first->clone())),
              lfs::core::NULL_NODE);
    ASSERT_NE(synchronous.addSplat("second", std::make_unique<SplatData>(second->clone())),
              lfs::core::NULL_NODE);
    const auto* expected = synchronous.getCombinedModel();
    ASSERT_NE(expected, nullptr);

    EXPECT_EQ(worker_build->model->means_raw().to_vector(), expected->means_raw().to_vector());
    EXPECT_EQ(worker_build->model->sh0_raw().to_vector(), expected->sh0_raw().to_vector());
    EXPECT_EQ(worker_build->model->scaling_raw().to_vector(), expected->scaling_raw().to_vector());
    EXPECT_EQ(worker_build->model->rotation_raw().to_vector(), expected->rotation_raw().to_vector());
    EXPECT_EQ(worker_build->model->opacity_raw().to_vector(), expected->opacity_raw().to_vector());
    EXPECT_EQ(worker_build->model->shN_raw().to_vector(), expected->shN_raw().to_vector());
}

TEST_F(SceneConsolidationExtractTest, SceneDestructionJoinsCombinedModelWorker) {
    std::mutex gate_mutex;
    std::condition_variable gate_changed;
    bool allocator_entered = false;
    bool release_allocator = false;

    std::thread releaser([&] {
        std::unique_lock lock(gate_mutex);
        gate_changed.wait(lock, [&] { return allocator_entered; });
        release_allocator = true;
        gate_changed.notify_all();
    });

    {
        Scene scene;
        EXPECT_NE(scene.addSplat("first", std::make_unique<SplatData>(bike_.clone())),
                  lfs::core::NULL_NODE);
        EXPECT_NE(scene.addSplat("second", std::make_unique<SplatData>(bike_.clone())),
                  lfs::core::NULL_NODE);
        scene.setCombinedModelAllocator(
            [&](lfs::core::TensorShape shape,
                size_t,
                lfs::core::DataType dtype,
                std::string_view) {
                std::unique_lock lock(gate_mutex);
                allocator_entered = true;
                gate_changed.notify_all();
                gate_changed.wait(lock, [&] { return release_allocator; });
                return Tensor::empty(std::move(shape), Device::CUDA, dtype);
            });
        scene.requestCombinedModelBuild(true);
    }
    releaser.join();
}

TEST_F(SceneConsolidationExtractTest, StaleWorkerGenerationCannotReplaceNewestCache) {
    Scene scene;
    const auto first_id = scene.addSplat("first", std::make_unique<SplatData>(bike_.clone()));
    const auto second_id = scene.addSplat("second", std::make_unique<SplatData>(bike_.clone()));
    ASSERT_NE(first_id, lfs::core::NULL_NODE);
    ASSERT_NE(second_id, lfs::core::NULL_NODE);

    auto old_snapshot = scene.captureCombinedModelBuild();
    auto old_build = Scene::buildCombinedModelCache(
        old_snapshot.inputs, old_snapshot.full_selection_count,
        old_snapshot.allocator, old_snapshot.generation);
    scene.invalidateCache();
    EXPECT_FALSE(scene.installCombinedModelCache(std::move(old_build)));

    auto newest_snapshot = scene.captureCombinedModelBuild();
    auto newest_build = Scene::buildCombinedModelCache(
        std::move(newest_snapshot.inputs), newest_snapshot.full_selection_count,
        newest_snapshot.allocator, newest_snapshot.generation);
    EXPECT_TRUE(scene.installCombinedModelCache(std::move(newest_build)));
    EXPECT_NE(scene.getCombinedModel(), nullptr);
}

TEST_F(SceneConsolidationExtractTest, OffsetWalksNullSlotAfterMiddleRemove) {
    ThreeNodeScene built;
    fill_three_node_scene(built);
    ASSERT_EQ(built.scene.consolidateNodeModels(), 3u);

    built.scene.removeNodeById(built.scene.getNodeIdByUuid(built.mid_uuid));
    EXPECT_EQ(built.scene.extractConsolidatedNodeModel(built.mid_uuid), nullptr);

    auto last = built.scene.extractConsolidatedNodeModel(built.last_uuid);
    ASSERT_NE(last, nullptr);
    expect_attrs_match(*last, built.last_attrs);

    auto full = built.scene.extractConsolidatedNodeModel(built.full_uuid);
    ASSERT_NE(full, nullptr);
    expect_attrs_match(*full, built.full_attrs);
}

TEST_F(SceneConsolidationExtractTest, SoftDeletedRowsAreExcluded) {
    ThreeNodeScene built;
    fill_three_node_scene(built);
    ASSERT_EQ(built.scene.consolidateNodeModels(), 3u);

    auto* combined = const_cast<SplatData*>(built.scene.getCombinedModel());
    ASSERT_NE(combined, nullptr);
    const size_t combined_n = static_cast<size_t>(combined->size());
    ASSERT_EQ(combined_n, built.full_n + built.mid_n + built.last_n);

    const size_t last_start = built.full_n + built.mid_n;
    const size_t delete_count = std::min<size_t>(7, built.last_n / 2);
    ASSERT_GT(delete_count, 0u);
    const size_t delete_start = last_start + 1;
    ASSERT_LE(delete_start + delete_count, last_start + built.last_n);

    Tensor del = Tensor::zeros_bool({combined_n}, combined->means_raw().device());
    del.slice(0, delete_start, delete_start + delete_count) =
        Tensor::ones_bool({delete_count}, combined->means_raw().device());
    combined->soft_delete(del);
    ASSERT_TRUE(combined->has_deleted_mask());

    auto extracted = built.scene.extractConsolidatedNodeModel(built.last_uuid);
    ASSERT_NE(extracted, nullptr);
    ASSERT_EQ(static_cast<size_t>(extracted->size()), built.last_n - delete_count);

    const size_t drop_lo = 1;
    const size_t drop_hi = 1 + delete_count;
    const auto keep_rows = [&](const std::vector<float>& src, const size_t width) {
        std::vector<float> out;
        out.reserve((built.last_n - delete_count) * width);
        for (size_t row = 0; row < built.last_n; ++row) {
            if (row >= drop_lo && row < drop_hi) {
                continue;
            }
            const float* begin = src.data() + row * width;
            out.insert(out.end(), begin, begin + static_cast<std::ptrdiff_t>(width));
        }
        return out;
    };

    CpuAttrs expected;
    expected.rows = built.last_n - delete_count;
    expected.max_sh_degree = built.last_attrs.max_sh_degree;
    expected.means = keep_rows(built.last_attrs.means, 3);
    expected.sh0 = keep_rows(built.last_attrs.sh0, 3);
    expected.scaling = keep_rows(built.last_attrs.scaling, 3);
    expected.rotation = keep_rows(built.last_attrs.rotation, 4);
    expected.opacity = keep_rows(built.last_attrs.opacity, 1);
    if (!built.last_attrs.shN.empty()) {
        const size_t shN_width = built.last_attrs.shN.size() / built.last_n;
        expected.shN = keep_rows(built.last_attrs.shN, shN_width);
    }
    expect_attrs_match(*extracted, expected);
}

TEST_F(SceneConsolidationExtractTest, ReturnsNullWhenNotConsolidatedOrUnknown) {
    Scene scene;
    EXPECT_EQ(scene.extractConsolidatedNodeModel(lfs::core::generate_uuid_v4()), nullptr);

    const auto id = scene.addSplat("one", std::make_unique<SplatData>(bike_.clone()));
    ASSERT_NE(id, lfs::core::NULL_NODE);
    const auto uuid = scene.getNodeUuid(id);
    EXPECT_FALSE(scene.isConsolidated());
    EXPECT_EQ(scene.extractConsolidatedNodeModel(uuid), nullptr);

    ThreeNodeScene built;
    fill_three_node_scene(built);
    ASSERT_EQ(built.scene.consolidateNodeModels(), 3u);
    EXPECT_EQ(built.scene.extractConsolidatedNodeModel(lfs::core::generate_uuid_v4()),
              nullptr);
    const auto group = built.scene.addGroup("folder");
    ASSERT_NE(group, lfs::core::NULL_NODE);
    EXPECT_EQ(built.scene.extractConsolidatedNodeModel(built.scene.getNodeUuid(group)),
              nullptr);
}
