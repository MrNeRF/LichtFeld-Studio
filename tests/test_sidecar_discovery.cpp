/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "core/uuid.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loader.hpp"
#include "io/sidecar_discovery.hpp"

#include <array>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
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

    void bump_dir_mtime(const fs::path& dir, const std::chrono::seconds offset) {
        std::error_code ec;
        const auto current = fs::last_write_time(dir, ec);
        if (!ec) {
            fs::last_write_time(dir, current + offset, ec);
        }
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

    class LogHandlerGuard {
    public:
        explicit LogHandlerGuard(lfs::core::LogHandler handler)
            : token_(lfs::core::Logger::get().add_log_handler(std::move(handler))) {}

        ~LogHandlerGuard() {
            lfs::core::Logger::get().remove_log_handler(token_);
        }

        LogHandlerGuard(const LogHandlerGuard&) = delete;
        LogHandlerGuard& operator=(const LogHandlerGuard&) = delete;

    private:
        lfs::core::LogHandlerToken token_;
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
    const fs::path norm_root = lfs::io::normalize_absolute_dir_path(temp.path);
    const fs::path manifest = temp.path / "transforms.json";
    write_text_file(manifest, "{}");

    EXPECT_EQ(lfs::io::resolve_dataset_root(temp.path), norm_root);
    EXPECT_EQ(lfs::io::resolve_dataset_root(manifest), norm_root);
    EXPECT_EQ(lfs::io::resolve_dataset_root(temp.path / "transforms_train.json"), norm_root);
    EXPECT_TRUE(lfs::io::resolve_dataset_root({}).empty());

    auto cam = make_camera("a.png", temp.path / "images/a.png");
    const std::vector<std::shared_ptr<lfs::core::Camera>> cameras{cam};
    const auto missing = lfs::io::associate_dataset_masks(cameras, temp.path / "missing_dir");
    ASSERT_TRUE(missing.has_value());
    EXPECT_EQ(missing->masks_bound, 0u);
    EXPECT_FALSE(cam->has_mask());
}

TEST(SidecarDiscoveryTest, AssociatesColmapNestedAndBlenderSubdirMasks) {
    const ScopedTempDir temp("lfs_sidecar_colmap_blender_masks");
    const fs::path root = temp.path;

    const fs::path colmap_img = root / "images" / "cam0" / "view_0001.png";
    const fs::path colmap_mask = root / "masks" / "cam0" / "view_0001.png";
    write_png_file(colmap_img);
    write_png_file(colmap_mask);
    auto colmap_cam = make_camera("cam0/view_0001.png", colmap_img);

    const fs::path train_img = root / "images" / "train" / "frame.png";
    const fs::path test_img = root / "images" / "test" / "frame.png";
    const fs::path train_mask = root / "masks" / "train" / "frame.png";
    const fs::path test_mask = root / "masks" / "test" / "frame.png";
    write_png_file(train_img);
    write_png_file(test_img);
    write_png_file(train_mask);
    write_png_file(test_mask);
    auto train_cam = make_camera("frame.png", train_img);
    auto test_cam = make_camera("frame.png", test_img);

    const std::vector<std::shared_ptr<lfs::core::Camera>> cameras{
        colmap_cam, train_cam, test_cam};

    const auto result = lfs::io::associate_dataset_masks(cameras, root);
    ASSERT_TRUE(result.has_value()) << lfs::format_for_developer(result.error());
    EXPECT_EQ(result->masks_bound, 3u);
    EXPECT_EQ(result->ambiguous_masks, 0u);
    EXPECT_EQ(colmap_cam->mask_path(), colmap_mask);
    EXPECT_EQ(train_cam->mask_path(), train_mask);
    EXPECT_EQ(test_cam->mask_path(), test_mask);
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

    const auto result = lfs::io::associate_dataset_masks(cameras, root);
    ASSERT_TRUE(result.has_value()) << lfs::format_for_developer(result.error());
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

    const auto failed = lfs::io::associate_dataset_masks(
        cameras, root, {.fail_on_ambiguous_mask = true});
    EXPECT_FALSE(failed.has_value());
    EXPECT_FALSE(cam->has_mask());

    const auto warned = lfs::io::associate_dataset_masks(
        cameras, root, {.fail_on_ambiguous_mask = false});
    ASSERT_TRUE(warned.has_value()) << lfs::format_for_developer(warned.error());
    EXPECT_EQ(warned->masks_bound, 0u);
    EXPECT_EQ(warned->ambiguous_masks, 1u);
    EXPECT_FALSE(cam->has_mask());
}

TEST(SidecarDiscoveryTest, MaskDiscoveryCacheSkipsUnchangedFoldersAndInvalidatesOnChange) {
    const ScopedTempDir temp("lfs_sidecar_mask_cache");
    const fs::path root = temp.path;

    const fs::path img_a = root / "images" / "a.png";
    const fs::path img_b = root / "images" / "b.png";
    const fs::path img_c = root / "images" / "nested" / "c.png";
    const fs::path img_dup = root / "images" / "dup.png";
    for (const auto& img : {img_a, img_b, img_c, img_dup}) {
        write_png_file(img);
    }
    write_png_file(root / "masks" / "a.png");
    write_png_file(root / "masks" / "cam_a" / "dup.png");
    write_png_file(root / "masks" / "cam_b" / "dup.png");

    auto cam_a = make_camera("a.png", img_a);
    auto cam_b = make_camera("b.png", img_b);
    auto cam_c = make_camera("nested/c.png", img_c);
    auto cam_dup = make_camera("dup.png", img_dup);
    const std::vector<std::shared_ptr<lfs::core::Camera>> cameras{
        cam_a, cam_b, cam_c, cam_dup};

    size_t warnings = 0;
    LogHandlerGuard guard([&](const lfs::core::LogLevel level, const lfs::core::SourceSite&,
                              const std::string_view) {
        if (level == lfs::core::LogLevel::Warn) {
            ++warnings;
        }
    });

    lfs::io::MaskDiscoveryCache cache;
    const auto first = lfs::io::associate_dataset_masks(
        cameras, root, {.fail_on_ambiguous_mask = false}, &cache);
    ASSERT_TRUE(first.has_value()) << lfs::format_for_developer(first.error());
    EXPECT_EQ(first->masks_bound, 1u);
    EXPECT_EQ(first->ambiguous_masks, 1u);
    EXPECT_EQ(warnings, 1u);

    // Repeat call with unchanged folders/cameras hits cache: no bound masks and no repeated warning.
    const auto repeat = lfs::io::associate_dataset_masks(
        cameras, root, {.fail_on_ambiguous_mask = false}, &cache);
    ASSERT_TRUE(repeat.has_value()) << lfs::format_for_developer(repeat.error());
    EXPECT_EQ(repeat->masks_bound, 0u);
    EXPECT_EQ(repeat->ambiguous_masks, 0u);
    EXPECT_EQ(warnings, 1u);

    // Adding a file to an existing folder (and advancing folder mtime past FS tick) invalidates.
    write_png_file(root / "masks" / "b.png");
    bump_dir_mtime(root / "masks", std::chrono::seconds(10));
    const auto after_new_file = lfs::io::associate_dataset_masks(
        cameras, root, {.fail_on_ambiguous_mask = false}, &cache);
    ASSERT_TRUE(after_new_file.has_value()) << lfs::format_for_developer(after_new_file.error());
    EXPECT_EQ(after_new_file->masks_bound, 1u);
    EXPECT_EQ(cam_b->mask_path(), root / "masks" / "b.png");

    // Creating a new mask root/subdir invalidates; adding another file in that tracked subdir invalidates too.
    write_png_file(root / "segmentation" / "nested" / "placeholder.png");
    bump_dir_mtime(root, std::chrono::seconds(20));
    const auto after_new_dir = lfs::io::associate_dataset_masks(
        cameras, root, {.fail_on_ambiguous_mask = false}, &cache);
    ASSERT_TRUE(after_new_dir.has_value());
    EXPECT_EQ(after_new_dir->masks_bound, 0u);

    write_png_file(root / "segmentation" / "nested" / "c.png");
    bump_dir_mtime(root / "segmentation" / "nested", std::chrono::seconds(30));
    const auto after_nested = lfs::io::associate_dataset_masks(
        cameras, root, {.fail_on_ambiguous_mask = false}, &cache);
    ASSERT_TRUE(after_nested.has_value());
    EXPECT_EQ(after_nested->masks_bound, 1u);
    EXPECT_EQ(cam_c->mask_path(), root / "segmentation" / "nested" / "c.png");
}

TEST_F(SceneSidecarSyncGpuTest, SyncCameraAssetPathsUpdatesNodesAndEmitsTransientMaskMutation) {
    const ScopedTempDir temp("lfs_sidecar_scene_sync");
    const fs::path root = temp.path;

    const fs::path img = root / "images" / "a.png";
    const fs::path mask = root / "masks" / "a.png";
    write_png_file(img);
    auto cam = make_camera("a.png", img);

    lfs::core::Scene scene;
    const auto group_id = scene.addCameraGroup("Cameras", lfs::core::NULL_NODE, 1);
    const auto node_id = scene.addCamera("a.png", group_id, cam);
    ASSERT_NE(node_id, lfs::core::NULL_NODE);

    const auto* node = scene.getNodeById(node_id);
    ASSERT_NE(node, nullptr);
    EXPECT_TRUE(node->mask_path.empty());

    // Mask created post-import
    write_png_file(mask);
    const std::vector<std::shared_ptr<lfs::core::Camera>> cameras{cam};
    const auto associated = lfs::io::associate_dataset_masks(cameras, root);
    ASSERT_TRUE(associated.has_value()) << lfs::format_for_developer(associated.error());
    EXPECT_EQ(associated->masks_bound, 1u);

    std::vector<lfs::core::events::state::SceneChanged> events;
    ScopedSceneChangedSubscription sub(
        lfs::core::events::state::SceneChanged::when(
            [&](const auto& event) { events.push_back(event); }));

    EXPECT_EQ(scene.syncCameraAssetPaths(true, true), 1u);
    EXPECT_EQ(node->mask_path, lfs::core::path_to_utf8(mask));
    ASSERT_EQ(events.size(), 1u);
    EXPECT_FALSE(events[0].document_changed);
    EXPECT_NE(events[0].mutation_flags &
                  static_cast<uint32_t>(lfs::core::Scene::MutationType::CAMERA_MASK_CHANGED),
              0u);

    // Calling again without changes is a no-op and emits nothing.
    EXPECT_EQ(scene.syncCameraAssetPaths(true, true), 0u);
    EXPECT_EQ(events.size(), 1u);
}

TEST_F(SceneSidecarSyncGpuTest, PostImportMaskDiscoveryMatchesBlenderLoader) {
    const ScopedTempDir temp("lfs_sidecar_blender_mask_agreement");
    const fs::path root = temp.path;

    write_png_file(root / "train" / "r_001.png");
    write_text_file(
        root / "transforms.json",
        R"({"w": 1, "h": 1, "fl_x": 1, "fl_y": 1, "frames": [{"file_path": "train/r_001.png", "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]})");
    write_text_file(
        root / "pointcloud.ply",
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n0 0 -2 255 0 0\n1 0 -2 0 255 0\n");

    auto loader = lfs::io::Loader::create();
    auto post_import = loader->load(root, {});
    ASSERT_TRUE(post_import.has_value()) << post_import.error().format();
    auto& post_cameras = std::get<lfs::io::LoadedScene>(post_import->data).cameras;
    ASSERT_EQ(post_cameras.size(), 1u);

    const fs::path root_mask = root / "masks" / "r_001.png";
    const fs::path nested_mask = root / "masks" / "train" / "r_001.png";
    write_png_file(root_mask);
    write_png_file(nested_mask);

    auto fresh_import = loader->load(root, {.load_masks = true});
    ASSERT_TRUE(fresh_import.has_value()) << fresh_import.error().format();
    const auto& fresh_cameras = std::get<lfs::io::LoadedScene>(fresh_import->data).cameras;
    ASSERT_EQ(fresh_cameras.size(), 1u);

    const auto refreshed = lfs::io::associate_dataset_masks(post_cameras, root);
    ASSERT_TRUE(refreshed.has_value()) << lfs::format_for_developer(refreshed.error());

    EXPECT_EQ(post_cameras[0]->mask_path(), fresh_cameras[0]->mask_path());
    EXPECT_EQ(fresh_cameras[0]->mask_path(), nested_mask);
}
