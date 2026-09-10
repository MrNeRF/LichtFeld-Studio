/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/scene.hpp"
#include "gui/gallery_scene_publication.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "project/session_state.hpp"
#include "rendering/rendering_types.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {

    using lfs::core::ExportFormat;
    using lfs::core::NodeType;
    using lfs::core::PayloadHydrationState;
    using lfs::core::Scene;
    using lfs::core::SplatData;
    using lfs::core::Uuid;
    using lfs::io::project::LazyChunkValue;
    using lfs::io::project::ProjectDocument;
    using lfs::test::licht::fixed_uuid;
    using lfs::test::licht::make_splat;
    using lfs::test::licht::read_file_bytes;
    using lfs::test::licht::require_result;
    using lfs::test::licht::require_result_ptr;
    using lfs::test::licht::require_status;
    using lfs::test::licht::TemporaryDirectory;
    using lfs::vis::gui::copyLazyChunkToFile;
    using lfs::vis::gui::GalleryEncodedAsset;
    using lfs::vis::gui::galleryEncodedAssetReusable;
    using lfs::vis::gui::galleryPublicationExtension;
    using lfs::vis::gui::GalleryScenePublishNode;
    using lfs::vis::gui::GalleryScenePublishRequest;
    using lfs::vis::gui::snapshotGalleryEncodedAsset;
    using lfs::vis::gui::writeGalleryScenePublication;

    std::vector<std::byte> unique_encoded_bytes(const std::uint8_t tag) {
        std::vector<std::byte> bytes(128);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<std::byte>(static_cast<std::uint8_t>(i) ^ tag);
        return bytes;
    }

    GalleryEncodedAsset owned_asset(std::string kind, std::vector<std::byte> bytes, const Uuid& uuid) {
        return GalleryEncodedAsset{
            .source_kind = std::move(kind),
            .bytes = require_result(LazyChunkValue::from_owned(std::move(bytes), uuid)),
        };
    }

    Scene::SplatSnapshot cpu_snapshot(const glm::mat4& transform = glm::mat4{1.0f}) {
        Scene::SplatSnapshot snapshot;
        snapshot.data = std::shared_ptr<SplatData>(make_splat(8).release());
        snapshot.world_transform = transform;
        snapshot.row_count = static_cast<std::size_t>(snapshot.data->size());
        snapshot.active_sh_degree = snapshot.data->get_active_sh_degree();
        return snapshot;
    }

    GalleryScenePublishRequest base_request(const std::filesystem::path& path, const ExportFormat format) {
        GalleryScenePublishRequest request;
        request.path = path;
        request.format = format;
        request.published_render = lfs::vis::project::renderSettingsToProjectJson(lfs::vis::RenderSettings{});
        request.published_camera =
            lfs::vis::project::panelCameraProjectStateToJson("primary", lfs::vis::project::PanelCameraProjectState{});
        return request;
    }

    std::vector<std::byte> read_lazy_bytes(const LazyChunkValue& value) {
        std::vector<std::byte> bytes(static_cast<std::size_t>(value.size()));
        if (!bytes.empty())
            require_status(value.read_at(0, bytes));
        return bytes;
    }

    struct PublishedNode {
        Uuid uuid;
        std::string source_kind;
        std::array<float, 16> local_transform{};
        std::vector<std::byte> dsrc;
        std::string sidecar;
    };

    PublishedNode read_published_node(const std::filesystem::path& directory) {
        auto document = require_result_ptr(ProjectDocument::open(directory / "project.licht"));
        const auto nodes = require_result(document->scene_graph().nodes());
        EXPECT_EQ(nodes.size(), 1u);
        const auto& record = nodes.front();
        EXPECT_TRUE(record.payload.has_value());
        const auto* source = document->find_dataset_source(record.uuid);
        EXPECT_NE(source, nullptr);
        std::ifstream manifest_file(directory / "manifest.json");
        const auto manifest = nlohmann::json::parse(manifest_file);
        PublishedNode published{
            .uuid = record.uuid,
            .source_kind = record.payload->source_kind,
            .local_transform = record.local_transform,
            .dsrc = read_lazy_bytes(*source),
            .sidecar = manifest.at("nodes").at(0).at("path").get<std::string>(),
        };
        return published;
    }

    std::unique_ptr<Scene> restore_splat(const Uuid& uuid, const bool diverged) {
        auto scene = std::make_unique<Scene>();
        Scene::RestoreNodeDesc desc;
        desc.uuid = uuid;
        desc.type = NodeType::SPLAT;
        desc.name = "splat";
        desc.payload_diverged = diverged;
        desc.payload_hydration = PayloadHydrationState::Loaded;
        desc.model = make_splat(8);
        EXPECT_NE(scene->restoreNodeWithUuid(std::move(desc)), lfs::core::NULL_NODE);
        return scene;
    }

} // namespace

TEST(GalleryScenePublicationTest, StudioPreservesCleanCompressedSourceAndExplicitSogMatchesSog) {
    EXPECT_TRUE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SCENE, "sog", false));
    EXPECT_TRUE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SCENE, "ssog", false));
    EXPECT_TRUE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SCENE, "ply", false));
    EXPECT_FALSE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SCENE, "sog", true));
    EXPECT_TRUE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SOG, "sog", false));
    EXPECT_FALSE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SOG, "ply", false));
    EXPECT_FALSE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SOG, "ssog", false));
    EXPECT_TRUE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SSOG, "ssog", false));
    EXPECT_FALSE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SSOG, "sog", false));
    EXPECT_EQ(galleryPublicationExtension(ExportFormat::GALLERY_SCENE, "sog"), "sog");
    EXPECT_EQ(galleryPublicationExtension(ExportFormat::GALLERY_SCENE, std::nullopt), "ply");
    EXPECT_EQ(galleryPublicationExtension(ExportFormat::GALLERY_SOG, std::nullopt), "sog");
}

TEST(GalleryScenePublicationTest, UnchangedEncodedAssetIsByteIdenticalAfterPublication) {
    TemporaryDirectory temporary;
    const auto original = unique_encoded_bytes(0x5a);
    auto request = base_request(temporary.path / "identical.scene", ExportFormat::GALLERY_SOG);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "clean",
        .encoded = owned_asset("sog", original, fixed_uuid(11)),
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "sog");
    EXPECT_EQ(published.sidecar, "0.sog");
    EXPECT_EQ(published.dsrc, original);
    EXPECT_EQ(read_file_bytes(request.path / "0.sog"), original);
}

TEST(GalleryScenePublicationTest, StudioDefaultKeepsCleanSogInsteadOfExpandingToPly) {
    TemporaryDirectory temporary;
    const auto original = unique_encoded_bytes(0x3c);
    auto request = base_request(temporary.path / "studio.scene", ExportFormat::GALLERY_SCENE);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "imported",
        .encoded = owned_asset("sog", original, fixed_uuid(12)),
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "sog");
    EXPECT_EQ(published.sidecar, "0.sog");
    EXPECT_EQ(published.dsrc, original);
    EXPECT_FALSE(std::filesystem::exists(request.path / "0.ply"));
}

TEST(GalleryScenePublicationTest, TransformedCleanNodeReusesEncodedBytesAndKeepsNewTransform) {
    TemporaryDirectory temporary;
    const auto original = unique_encoded_bytes(0xa1);
    const auto transform = glm::translate(glm::mat4{1.0f}, glm::vec3{4.0f, 5.0f, 6.0f});
    auto request = base_request(temporary.path / "transformed.scene", ExportFormat::GALLERY_SOG);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(transform),
        .name = "moved",
        .encoded = owned_asset("sog", original, fixed_uuid(13)),
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.dsrc, original);
    std::array<float, 16> expected{};
    std::memcpy(expected.data(), &transform[0][0], sizeof(expected));
    EXPECT_EQ(published.local_transform, expected);
}

TEST(GalleryScenePublicationTest, EditedNodeExportsChangedDataInsteadOfStaleOriginal) {
    TemporaryDirectory temporary;
    const auto original = unique_encoded_bytes(0x22);
    auto request = base_request(temporary.path / "edited.scene", ExportFormat::GALLERY_SCENE);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "edited",
        .encoded = std::nullopt,
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "ply");
    EXPECT_EQ(published.sidecar, "0.ply");
    EXPECT_NE(published.dsrc, original);
    EXPECT_GE(published.dsrc.size(), 3u);
    EXPECT_EQ(std::memcmp(published.dsrc.data(), "ply", 3), 0);
}

TEST(GalleryScenePublicationTest, SnapshotIgnoresStaleDsrcOncePayloadDiverges) {
    TemporaryDirectory temporary;
    const auto original = unique_encoded_bytes(0x7e);
    auto request = base_request(temporary.path / "source.scene", ExportFormat::GALLERY_SOG);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "source",
        .encoded = owned_asset("sog", original, fixed_uuid(14)),
    });
    writeGalleryScenePublication(request, {}, {});
    auto document = require_result_ptr(ProjectDocument::open(request.path / "project.licht"));
    const auto uuid = require_result(document->scene_graph().nodes()).front().uuid;
    auto clean = restore_splat(uuid, false);
    auto reused = snapshotGalleryEncodedAsset(document.get(), *clean->getNodeByUuid(uuid), ExportFormat::GALLERY_SOG);
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(reused->source_kind, "sog");
    EXPECT_EQ(read_lazy_bytes(reused->bytes), original);

    auto edited = restore_splat(uuid, true);
    EXPECT_FALSE(snapshotGalleryEncodedAsset(document.get(), *edited->getNodeByUuid(uuid), ExportFormat::GALLERY_SOG)
                     .has_value());
}

TEST(GalleryScenePublicationTest, SharedEncodedSourceSurvivesDocumentCloseAndReplacement) {
    TemporaryDirectory temporary;
    const auto original = unique_encoded_bytes(0x91);
    auto request = base_request(temporary.path / "lifetime.scene", ExportFormat::GALLERY_SOG);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "source",
        .encoded = owned_asset("sog", original, fixed_uuid(15)),
    });
    writeGalleryScenePublication(request, {}, {});
    auto document = require_result_ptr(ProjectDocument::open(request.path / "project.licht"));
    const auto uuid = require_result(document->scene_graph().nodes()).front().uuid;
    auto scene = restore_splat(uuid, false);
    auto captured =
        snapshotGalleryEncodedAsset(document.get(), *scene->getNodeByUuid(uuid), ExportFormat::GALLERY_SOG);
    ASSERT_TRUE(captured.has_value());
    document.reset();
    std::filesystem::remove(request.path / "project.licht");
    {
        std::ofstream replacement(request.path / "project.licht", std::ios::binary | std::ios::trunc);
        replacement << "replaced-project-bytes";
    }
    const auto copy = temporary.path / "retained.sog";
    copyLazyChunkToFile(captured->bytes, copy);
    EXPECT_EQ(read_file_bytes(copy), original);
}
