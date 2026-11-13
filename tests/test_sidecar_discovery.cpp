/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/events.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "core/uuid.hpp"
#include "io/sidecar_discovery.hpp"

#include <array>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

    constexpr std::array<unsigned char, 70> PNG_1X1_RGB = {
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
        0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
        0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x78,
        0x9C, 0x63, 0x68, 0x68, 0xF8, 0x0F, 0x00, 0x03, 0x83, 0x02, 0x00, 0x00, 0xA5, 0xFB,
        0x89, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82};

    std::shared_ptr<lfs::core::Camera> make_camera(const std::string& name,
                                                   const fs::path& image_path) {
        return std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            name, image_path, fs::path{}, 64, 64, 0);
    }

    void write_png_file(const fs::path& path,
                        const std::span<const unsigned char> bytes = PNG_1X1_RGB) {
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << path;
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(out.good()) << path;
    }

    void write_text_file(const fs::path& path, const std::string& contents = "x") {
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << path;
        out << contents;
        ASSERT_TRUE(out.good()) << path;
    }

    struct ScopedTempDir {
        fs::path path;

        explicit ScopedTempDir(const std::string& name)
            : path(fs::temp_directory_path() / name) {
            std::error_code ec;
            fs::remove_all(path, ec);
            fs::create_directories(path);
        }

        ~ScopedTempDir() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }

        ScopedTempDir(const ScopedTempDir&) = delete;
        ScopedTempDir& operator=(const ScopedTempDir&) = delete;
    };

    class ScopedSceneChangedSubscription {
    public:
        explicit ScopedSceneChangedSubscription(const lfs::event::HandlerId id) : id_(id) {}

        ~ScopedSceneChangedSubscription() {
            if (id_ != 0) {
                lfs::event::EventBridge::instance().unsubscribe(
                    typeid(lfs::core::events::state::SceneChanged), id_);
            }
        }

        ScopedSceneChangedSubscription(const ScopedSceneChangedSubscription&) = delete;
        ScopedSceneChangedSubscription& operator=(const ScopedSceneChangedSubscription&) = delete;

    private:
        lfs::event::HandlerId id_ = 0;
    };

    class SceneSidecarSyncGpuTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
                GTEST_SKIP() << "Requires a CUDA device";
            }
        }
    };

} // namespace

TEST(SidecarDiscoveryTest, ResolveDatasetRootHandlesDirectoriesAndManifestFiles) {
    const ScopedTempDir temp("lfs_sidecar_resolve_root");
    const fs::path manifest = temp.path / "transforms.json";
    write_text_file(manifest, "{}");

    EXPECT_EQ(lfs::io::resolve_dataset_root(temp.path), temp.path);
    EXPECT_EQ(lfs::io::resolve_dataset_root(manifest), temp.path);
    EXPECT_EQ(lfs::io::resolve_dataset_root(temp.path / "transforms_train.json"), temp.path);
    EXPECT_TRUE(lfs::io::resolve_dataset_root(temp.path / "missing_dir").empty());
    EXPECT_TRUE(lfs::io::resolve_dataset_root({}).empty());
}

TEST(SidecarDiscoveryTest, AssociatesColmapNestedAndBlenderSubdirSidecars) {
    const ScopedTempDir temp("lfs_sidecar_colmap_blender");
    const fs::path root = temp.path;

    // COLMAP camera with nested image_name
    const fs::path colmap_img = root / "images" / "cam0" / "view_0001.png";
    const fs::path colmap_mask = root / "masks" / "cam0" / "view_0001.png";
    write_png_file(colmap_img);
    write_png_file(colmap_mask);
    auto colmap_cam = make_camera("cam0/view_0001.png", colmap_img);

    // Blender-style cameras with identical basename under split folders
    const fs::path train_img = root / "images" / "train" / "frame.png";
    const fs::path test_img = root / "images" / "test" / "frame.png";
    const fs::path train_mask = root / "masks" / "train" / "frame.png";
    const fs::path test_mask = root / "masks" / "test" / "frame.png";
    const fs::path train_depth = root / "depths" / "train" / "frame.png";
    write_png_file(train_img);
    write_png_file(test_img);
    write_png_file(train_mask);
    write_png_file(test_mask);
    write_png_file(train_depth);
    auto train_cam = make_camera("frame.png", train_img);
    auto test_cam = make_camera("frame.png", test_img);

    const std::vector<std::shared_ptr<lfs::core::Camera>> cameras{
        colmap_cam, train_cam, test_cam};

    lfs::io::SidecarAssociationOptions opts;
    opts.masks = true;
    opts.depths = true;
    const auto result = lfs::io::associate_dataset_sidecars(cameras, root, opts);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_EQ(result->masks_bound, 3u);
    EXPECT_EQ(result->depths_bound, 1u);
    EXPECT_EQ(result->normals_bound, 0u);
    EXPECT_EQ(result->ambiguous_masks, 0u);

    EXPECT_EQ(colmap_cam->mask_path(), colmap_mask);
    EXPECT_EQ(train_cam->mask_path(), train_mask);
    EXPECT_EQ(test_cam->mask_path(), test_mask);
    EXPECT_EQ(train_cam->depth_path(), train_depth);
    EXPECT_FALSE(test_cam->has_depth());
}

TEST(SidecarDiscoveryTest, PreservesExistingAssetsAndRebindsDeletedMask) {
    const ScopedTempDir temp("lfs_sidecar_preserve_rebind");
    const fs::path root = temp.path;

    write_png_file(root / "images" / "mem.png");
    write_png_file(root / "masks" / "mem.png");
    auto cam_mem = make_camera("mem.png", root / "images" / "mem.png");
    cam_mem->set_mask_tensor(lfs::core::Tensor::zeros({4, 4}, lfs::core::Device::CPU));

    write_png_file(root / "images" / "asset.png");
    write_png_file(root / "masks" / "asset.png");
    auto cam_asset = make_camera("asset.png", root / "images" / "asset.png");
    const fs::path project_mask = root / "project_masks" / "custom.png";
    cam_asset->set_mask_path(project_mask, lfs::core::generate_uuid_v4());

    write_png_file(root / "images" / "keep.png");
    const fs::path keep_mask = root / "masks" / "keep.png";
    write_png_file(keep_mask);
    auto cam_keep = make_camera("keep.png", root / "images" / "keep.png");
    cam_keep->set_mask_path(keep_mask);

    write_png_file(root / "images" / "rebind.png");
    const fs::path seg_mask = root / "segmentation" / "rebind.png";
    write_png_file(seg_mask);
    auto cam_rebind = make_camera("rebind.png", root / "images" / "rebind.png");
    cam_rebind->set_mask_path(root / "masks" / "deleted_rebind.png");
    ASSERT_FALSE(cam_rebind->has_mask());

    const std::vector<std::shared_ptr<lfs::core::Camera>> cameras{
        cam_mem, cam_asset, cam_keep, cam_rebind};

    lfs::io::SidecarAssociationOptions opts;
    opts.masks = true;
    const auto result = lfs::io::associate_dataset_sidecars(cameras, root, opts);
    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_EQ(result->masks_bound, 1u);

    EXPECT_TRUE(cam_mem->has_in_memory_mask());
    EXPECT_TRUE(cam_mem->mask_path().empty());
    EXPECT_EQ(cam_asset->mask_path(), project_mask);
    EXPECT_EQ(cam_keep->mask_path(), keep_mask);
    EXPECT_EQ(cam_rebind->mask_path(), seg_mask);
}

TEST(SidecarDiscoveryTest, AmbiguousMaskHonorsFailFlag) {
    const ScopedTempDir temp("lfs_sidecar_ambiguous");
    const fs::path root = temp.path;

    const fs::path img = root / "images" / "frame.png";
    write_png_file(img);
    write_png_file(root / "masks" / "cam_a" / "frame.png");
    write_png_file(root / "masks" / "cam_b" / "frame.png");
    auto cam = make_camera("frame.png", img);
    const std::vector<std::shared_ptr<lfs::core::Camera>> cameras{cam};

    lfs::io::SidecarAssociationOptions fail_opts;
    fail_opts.masks = true;
    fail_opts.fail_on_ambiguous_mask = true;
    const auto failed = lfs::io::associate_dataset_sidecars(cameras, root, fail_opts);
    EXPECT_FALSE(failed.has_value());
    EXPECT_FALSE(cam->has_mask());

    lfs::io::SidecarAssociationOptions warn_opts;
    warn_opts.masks = true;
    warn_opts.fail_on_ambiguous_mask = false;
    const auto warned = lfs::io::associate_dataset_sidecars(cameras, root, warn_opts);
    ASSERT_TRUE(warned.has_value()) << warned.error();
    EXPECT_EQ(warned->masks_bound, 0u);
    EXPECT_EQ(warned->ambiguous_masks, 1u);
    EXPECT_FALSE(cam->has_mask());
}

TEST_F(SceneSidecarSyncGpuTest, SyncCameraAssetPathsUpdatesNodesAndEmitsTransientMaskMutation) {
    const ScopedTempDir temp("lfs_sidecar_scene_sync");
    const fs::path root = temp.path;

    const fs::path img = root / "images" / "a.png";
    const fs::path mask = root / "masks" / "a.png";
    const fs::path depth = root / "depths" / "a.png";
    write_png_file(img);
    auto cam = make_camera("a.png", img);

    lfs::core::Scene scene;
    const auto group_id = scene.addCameraGroup("Cameras", lfs::core::NULL_NODE, 1);
    const auto node_id = scene.addCamera("a.png", group_id, cam);
    ASSERT_NE(node_id, lfs::core::NULL_NODE);

    const auto* node = scene.getNodeById(node_id);
    ASSERT_NE(node, nullptr);
    EXPECT_TRUE(node->mask_path.empty());
    EXPECT_TRUE(node->depth_path.empty());

    // Sidecars created post-import
    write_png_file(mask);
    write_png_file(depth);
    const std::vector<std::shared_ptr<lfs::core::Camera>> cameras{cam};
    lfs::io::SidecarAssociationOptions opts;
    opts.masks = true;
    opts.depths = true;
    const auto associated = lfs::io::associate_dataset_sidecars(cameras, root, opts);
    ASSERT_TRUE(associated.has_value()) << associated.error();
    EXPECT_EQ(associated->masks_bound, 1u);
    EXPECT_EQ(associated->depths_bound, 1u);

    std::vector<lfs::core::events::state::SceneChanged> events;
    ScopedSceneChangedSubscription sub(
        lfs::core::events::state::SceneChanged::when(
            [&](const auto& event) { events.push_back(event); }));

    EXPECT_EQ(scene.syncCameraAssetPaths(true, true), 1u);
    EXPECT_EQ(node->mask_path, lfs::core::path_to_utf8(mask));
    EXPECT_EQ(node->depth_path, lfs::core::path_to_utf8(depth));
    ASSERT_EQ(events.size(), 1u);
    EXPECT_FALSE(events[0].document_changed);
    EXPECT_NE(events[0].mutation_flags &
                  static_cast<uint32_t>(lfs::core::Scene::MutationType::CAMERA_MASK_CHANGED),
              0u);

    // Calling again without changes is a no-op and emits nothing.
    EXPECT_EQ(scene.syncCameraAssetPaths(true, true), 0u);
    EXPECT_EQ(events.size(), 1u);
}
