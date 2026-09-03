/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "operation/undo_history.hpp"
#include "scene/scene_manager.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::PointCloud;
using lfs::core::Tensor;

namespace {
    Tensor makeMeans(const Device device = Device::CPU) {
        return Tensor::from_vector(
            {
                0.0f,
                0.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f,
                2.0f,
                0.0f,
                0.0f,
                3.0f,
                0.0f,
                0.0f,
                4.0f,
                0.0f,
                0.0f,
                5.0f,
                0.0f,
                0.0f,
                6.0f,
                0.0f,
                0.0f,
                7.0f,
                0.0f,
                0.0f,
                8.0f,
                0.0f,
                0.0f,
                9.0f,
                0.0f,
                0.0f,
            },
            {size_t{10}, size_t{3}},
            device);
    }

    Tensor makeColors(const Device device = Device::CPU) {
        auto colors = Tensor::empty({size_t{10}, size_t{3}}, Device::CPU, DataType::UInt8);
        auto* ptr = colors.ptr<uint8_t>();
        for (uint8_t i = 0; i < 10; ++i) {
            ptr[i * 3] = i;
            ptr[i * 3 + 1] = static_cast<uint8_t>(i + 10);
            ptr[i * 3 + 2] = static_cast<uint8_t>(i + 20);
        }
        return device == Device::CPU ? colors : colors.to(device);
    }

    std::shared_ptr<Tensor> makeDeletedMask(const Device device = Device::CPU) {
        return std::make_shared<Tensor>(
            Tensor::from_vector(
                std::vector<bool>{false, true, false, false, true, false, false, true, false, false},
                {size_t{10}},
                device));
    }

    std::shared_ptr<Tensor> makeSelectionMask(const size_t count) {
        return std::make_shared<Tensor>(Tensor::ones({count}, Device::CPU, DataType::UInt8));
    }

    std::shared_ptr<PointCloud> makePointCloud(const size_t count = 4) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> colors(count * 3, 1.0f);
        return std::make_shared<PointCloud>(
            Tensor::from_vector(means, {count, size_t{3}}, Device::CPU),
            Tensor::from_vector(colors, {count, size_t{3}}, Device::CPU));
    }

    bool hasCudaDevice() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }
} // namespace

class PointCloudMaskRenderStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();
        scene_manager_ = std::make_unique<lfs::vis::SceneManager>();
    }

    void TearDown() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        scene_manager_.reset();
        lfs::vis::op::undoHistory().clear();
    }

    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
};

TEST(PointCloudMaskTest, RemoveDeletedPointsCompactsMeansColorsAndOpacity) {
    PointCloud pc(makeMeans(), makeColors());
    pc.opacity = Tensor::from_vector(
        {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
        {size_t{10}, size_t{1}},
        Device::CPU);
    pc.deleted = makeDeletedMask();
    pc.selection = makeSelectionMask(10);

    const auto compacted = lfs::core::remove_deleted_points(pc);

    ASSERT_EQ(compacted.size(), 7);
    EXPECT_EQ(compacted.means.shape()[0], 7);
    EXPECT_EQ(compacted.colors.shape()[0], 7);
    EXPECT_EQ(compacted.opacity.shape()[0], 7);
    EXPECT_EQ(compacted.selection, nullptr);
    EXPECT_EQ(compacted.deleted, nullptr);
    EXPECT_EQ(compacted.means.to_vector(),
              (std::vector<float>{
                  0.0f,
                  0.0f,
                  0.0f,
                  2.0f,
                  0.0f,
                  0.0f,
                  3.0f,
                  0.0f,
                  0.0f,
                  5.0f,
                  0.0f,
                  0.0f,
                  6.0f,
                  0.0f,
                  0.0f,
                  8.0f,
                  0.0f,
                  0.0f,
                  9.0f,
                  0.0f,
                  0.0f,
              }));
    EXPECT_EQ(compacted.colors.to_vector_uint8(),
              (std::vector<uint8_t>{
                  0,
                  10,
                  20,
                  2,
                  12,
                  22,
                  3,
                  13,
                  23,
                  5,
                  15,
                  25,
                  6,
                  16,
                  26,
                  8,
                  18,
                  28,
                  9,
                  19,
                  29,
              }));
    EXPECT_EQ(compacted.opacity.to_vector(),
              (std::vector<float>{0.0f, 2.0f, 3.0f, 5.0f, 6.0f, 8.0f, 9.0f}));
}

TEST(PointCloudMaskTest, RemoveDeletedPointsWithoutDeletedMaskReturnsUnchangedCopy) {
    PointCloud pc(makeMeans(), makeColors());
    pc.selection = makeSelectionMask(10);

    const auto copied = lfs::core::remove_deleted_points(pc);

    EXPECT_EQ(copied.size(), 10);
    EXPECT_EQ(copied.colors.shape()[0], 10);
    EXPECT_EQ(copied.selection, pc.selection);
}

TEST(PointCloudMaskTest, RemoveDeletedPointsSupportsCudaResidentClouds) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "CUDA device required for CUDA-resident point cloud compaction";
    }

    PointCloud pc(makeMeans(Device::CUDA), makeColors(Device::CUDA));
    pc.deleted = makeDeletedMask(Device::CPU);

    const auto compacted = lfs::core::remove_deleted_points(pc);

    ASSERT_EQ(compacted.size(), 7);
    EXPECT_EQ(compacted.means.device(), Device::CUDA);
    EXPECT_EQ(compacted.colors.device(), Device::CUDA);
    EXPECT_EQ(compacted.means.cpu().to_vector(),
              (std::vector<float>{
                  0.0f,
                  0.0f,
                  0.0f,
                  2.0f,
                  0.0f,
                  0.0f,
                  3.0f,
                  0.0f,
                  0.0f,
                  5.0f,
                  0.0f,
                  0.0f,
                  6.0f,
                  0.0f,
                  0.0f,
                  8.0f,
                  0.0f,
                  0.0f,
                  9.0f,
                  0.0f,
                  0.0f,
              }));
}

TEST(PointCloudMaskTest, HasSelectionAndDeletedRejectSizeMismatch) {
    PointCloud pc(makeMeans(), makeColors());
    pc.selection = makeSelectionMask(9);
    pc.deleted = std::make_shared<Tensor>(
        Tensor::from_vector(std::vector<bool>(9, false), {size_t{9}}, Device::CPU));

    EXPECT_FALSE(pc.has_selection());
    EXPECT_FALSE(pc.has_deleted());

    pc.selection = makeSelectionMask(10);
    pc.deleted = makeDeletedMask();
    EXPECT_TRUE(pc.has_selection());
    EXPECT_TRUE(pc.has_deleted());
}

TEST_F(PointCloudMaskRenderStateTest, BuildRenderStateExposesSinglePointCloudMasks) {
    auto& manager = *scene_manager_;
    auto& scene = manager.getScene();
    const auto point_cloud = makePointCloud(4);
    point_cloud->selection = makeSelectionMask(4);
    point_cloud->deleted = std::make_shared<Tensor>(
        Tensor::from_vector(std::vector<bool>{false, true, false, false}, {size_t{4}}, Device::CPU));
    const auto node_id = scene.addPointCloud("PointCloud", point_cloud);

    const auto state = manager.buildRenderState();

    ASSERT_NE(state.point_cloud, nullptr);
    EXPECT_EQ(state.point_cloud_node_id, node_id);
    EXPECT_EQ(state.point_cloud_selection_mask, point_cloud->selection);
    EXPECT_EQ(state.point_cloud_deleted_mask, point_cloud->deleted);
}

TEST_F(PointCloudMaskRenderStateTest, BuildRenderStateSuppressesMismatchedPointCloudMasks) {
    auto& manager = *scene_manager_;
    auto& scene = manager.getScene();
    const auto point_cloud = makePointCloud(4);
    point_cloud->selection = makeSelectionMask(3);
    point_cloud->deleted = std::make_shared<Tensor>(
        Tensor::from_vector(std::vector<bool>(3, false), {size_t{3}}, Device::CPU));
    scene.addPointCloud("PointCloud", point_cloud);

    const auto state = manager.buildRenderState();

    ASSERT_NE(state.point_cloud, nullptr);
    EXPECT_EQ(state.point_cloud_selection_mask, nullptr);
    EXPECT_EQ(state.point_cloud_deleted_mask, nullptr);
}
