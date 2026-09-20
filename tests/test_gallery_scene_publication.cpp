/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "gui/gallery_scene_publication.hpp"
#include "io/formats/sogs.hpp"
#include "io/formats/spz.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "project/session_state.hpp"
#include "rendering/rendering_types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

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

    // Windows export workers have a small stack. Exercise that constraint on
    // every platform, including Linux where the default stack hides regressions.
    void on_small_stack(const std::function<void()>& operation) {
        struct Work {
            const std::function<void()>& operation;
            std::exception_ptr error;
            void run() noexcept {
                try {
                    operation();
                } catch (...) {
                    error = std::current_exception();
                }
            }
        } work{operation, {}};
        constexpr std::size_t stack_bytes = 512 * 1024;
#ifdef _WIN32
        const auto thread = CreateThread(
            nullptr, stack_bytes,
            [](void* data) -> DWORD {
                static_cast<Work*>(data)->run();
                return 0;
            },
            &work, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
        if (!thread)
            throw std::runtime_error("Could not create publication test worker");
        const auto waited = WaitForSingleObject(thread, INFINITE);
        CloseHandle(thread);
        if (waited != WAIT_OBJECT_0)
            std::terminate(); // Do not unwind while the worker may still reference this stack.
#else
        pthread_attr_t attributes;
        if (pthread_attr_init(&attributes) != 0)
            throw std::runtime_error("Could not initialize publication test worker");
        const auto configured = pthread_attr_setstacksize(&attributes, stack_bytes);
        pthread_t thread;
        const auto created = configured == 0
                                 ? pthread_create(&thread, &attributes, [](void* data) -> void* {
                                       static_cast<Work*>(data)->run();
                                       return nullptr; }, &work)
                                 : configured;
        pthread_attr_destroy(&attributes);
        if (created != 0)
            throw std::runtime_error("Could not create publication test worker");
        if (pthread_join(thread, nullptr) != 0)
            std::terminate(); // Do not unwind while the worker may still reference this stack.
#endif
        if (work.error)
            std::rethrow_exception(work.error);
    }

    std::vector<std::byte> multi_buffer_asset() {
        std::vector<std::byte> bytes(2 * 1024 * 1024 + 37);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<std::byte>((i * 17 + i / 1024) % 251);
        return bytes;
    }

    struct PublishedNode {
        Uuid uuid;
        std::string source_kind;
        std::array<float, 16> local_transform{};
        std::vector<std::byte> dsrc;
        std::string sidecar;
        std::uint64_t publication_count = 0;
    };

    PublishedNode read_published_node(const std::filesystem::path& directory) {
        auto document = require_result_ptr(ProjectDocument::open(directory / "project.licht"));
        const auto nodes = require_result(document->scene_graph().nodes());
        EXPECT_EQ(nodes.size(), 1u);
        const auto& record = nodes.front();
        EXPECT_TRUE(record.payload.has_value());
        const auto* source = document->find_dataset_source(record.uuid);
        EXPECT_NE(source, nullptr);
        const auto element = document->scene_graph().dom().array_find("nodes", record.uuid.to_string());
        EXPECT_TRUE(element.has_value());
        const auto publication = element ? element->get_json("publication") : std::nullopt;
        EXPECT_TRUE(publication.has_value());
        std::ifstream manifest_file(directory / "manifest.json");
        const auto manifest = nlohmann::json::parse(manifest_file);
        PublishedNode published{
            .uuid = record.uuid,
            .source_kind = record.payload->source_kind,
            .local_transform = record.local_transform,
            .dsrc = read_lazy_bytes(*source),
            .sidecar = manifest.at("nodes").at(0).at("path").get<std::string>(),
            .publication_count = publication ? publication->at("count").get<std::uint64_t>() : 0,
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

TEST(GalleryScenePublicationTest, PublishedCameraRecordsWorldExtentFromSettingsScale) {
    Viewport viewport(1920, 1080);
    const float settings_scale = 1080.0f / 6.25f;
    const auto state = lfs::vis::project::capturePanelCameraProjectState(viewport, settings_scale);
    ASSERT_TRUE(state.ortho_extent_world.has_value());
    EXPECT_FLOAT_EQ(*state.ortho_extent_world, 6.25f);
    const auto json = lfs::vis::project::panelCameraProjectStateToJson("primary", state);
    ASSERT_TRUE(json.contains("ortho_extent_world"));
    EXPECT_FLOAT_EQ(json["ortho_extent_world"].get<float>(), 6.25f);
    EXPECT_TRUE(json["ortho_scale"].is_null());

    Viewport secondary(960, 540);
    secondary.ortho_scale_override = 540.0f / 3.0f;
    const auto secondary_state =
        lfs::vis::project::capturePanelCameraProjectState(secondary, settings_scale);
    ASSERT_TRUE(secondary_state.ortho_extent_world.has_value());
    EXPECT_FLOAT_EQ(*secondary_state.ortho_extent_world, 3.0f);
}

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
    EXPECT_TRUE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SPZ, "spz", false));
    EXPECT_FALSE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SPZ, "sog", false));
    EXPECT_TRUE(galleryEncodedAssetReusable(ExportFormat::GALLERY_SCENE, "spz", false));
    EXPECT_EQ(galleryPublicationExtension(ExportFormat::GALLERY_SCENE, "sog"), "sog");
    EXPECT_EQ(galleryPublicationExtension(ExportFormat::GALLERY_SCENE, std::nullopt), "ply");
    EXPECT_EQ(galleryPublicationExtension(ExportFormat::GALLERY_SOG, std::nullopt), "sog");
    EXPECT_EQ(galleryPublicationExtension(ExportFormat::GALLERY_SPZ, std::nullopt), "spz");
    EXPECT_EQ(galleryPublicationExtension(ExportFormat::GALLERY_SPZ, "spz"), "spz");
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

TEST(GalleryScenePublicationStackTest, EncodedCopyFitsWorkerStackAndPreservesEveryByte) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ASSERT_EXIT(
        ([] {
            TemporaryDirectory temporary;
            const auto bytes = multi_buffer_asset();
            const auto source = owned_asset("sog", bytes, fixed_uuid(111));
            const auto destination = temporary.path / "copied.sog";
            on_small_stack([&] { copyLazyChunkToFile(source.bytes, destination); });
            EXPECT_EQ(read_file_bytes(destination), bytes);
        }(),
         std::_Exit(::testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS)),
        ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(GalleryScenePublicationStackTest, PublicationFitsWorkerStackAndPreservesEmbeddedSource) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ASSERT_EXIT(
        ([] {
            TemporaryDirectory temporary;
            const auto bytes = multi_buffer_asset();
            auto request = base_request(temporary.path / "worker.scene", ExportFormat::GALLERY_SOG);
            GalleryScenePublishNode node;
            node.snapshot.row_count = 8;
            node.snapshot.active_sh_degree = 0;
            node.snapshot.world_transform = glm::mat4{1.0f};
            node.name = "encoded";
            node.encoded = owned_asset("sog", bytes, fixed_uuid(112));
            request.nodes.push_back(std::move(node));
            on_small_stack([&] { writeGalleryScenePublication(request, {}, {}); });
            const auto published = read_published_node(request.path);
            EXPECT_EQ(published.dsrc, bytes);
            EXPECT_EQ(read_file_bytes(request.path / "0.sog"), bytes);
            EXPECT_FALSE(request.materialized_payload);
        }(),
         std::_Exit(::testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS)),
        ::testing::ExitedWithCode(EXIT_SUCCESS), "");
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

std::vector<std::byte> ngsp_v4_stub(const std::uint32_t count, const std::uint8_t sh_degree) {
    std::vector<std::byte> bytes(32 + 16, std::byte{0});
    const std::uint32_t magic = 0x5053474e;
    const std::uint32_t version = 4;
    const std::uint8_t streams = 1;
    const std::uint32_t toc = 32;
    std::memcpy(bytes.data(), &magic, 4);
    std::memcpy(bytes.data() + 4, &version, 4);
    std::memcpy(bytes.data() + 8, &count, 4);
    bytes[12] = static_cast<std::byte>(sh_degree);
    bytes[13] = std::byte{12};
    bytes[15] = static_cast<std::byte>(streams);
    std::memcpy(bytes.data() + 16, &toc, 4);
    const std::uint64_t compressed = 0;
    const std::uint64_t uncompressed = 0;
    std::memcpy(bytes.data() + 32, &compressed, 8);
    std::memcpy(bytes.data() + 40, &uncompressed, 8);
    return bytes;
}

TEST(GalleryScenePublicationTest, UnchangedSpzV4AssetIsByteIdenticalAfterPublication) {
    TemporaryDirectory temporary;
    const auto original = ngsp_v4_stub(8, 0);
    auto request = base_request(temporary.path / "identical-spz.scene", ExportFormat::GALLERY_SPZ);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "clean-spz",
        .encoded = owned_asset("spz", original, fixed_uuid(21)),
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "spz");
    EXPECT_EQ(published.sidecar, "0.spz");
    EXPECT_EQ(published.dsrc, original);
    EXPECT_EQ(read_file_bytes(request.path / "0.spz"), original);
}

TEST(GalleryScenePublicationTest, StudioDefaultKeepsCleanSpzInsteadOfExpandingToPly) {
    TemporaryDirectory temporary;
    const auto original = ngsp_v4_stub(8, 1);
    auto request = base_request(temporary.path / "studio-spz.scene", ExportFormat::GALLERY_SCENE);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "imported-spz",
        .encoded = owned_asset("spz", original, fixed_uuid(22)),
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "spz");
    EXPECT_EQ(published.sidecar, "0.spz");
    EXPECT_EQ(published.dsrc, original);
    EXPECT_FALSE(std::filesystem::exists(request.path / "0.ply"));
}

TEST(GalleryScenePublicationTest, StudioOriginalPrecisionReencodesLegacySpzToPly) {
    TemporaryDirectory temporary;
    std::vector<std::byte> gzip_legacy(128, std::byte{0x5a});
    gzip_legacy[0] = std::byte{0x1f};
    gzip_legacy[1] = std::byte{0x8b};
    auto request = base_request(temporary.path / "studio-spz3.scene", ExportFormat::GALLERY_SCENE);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "legacy-spz",
        .encoded = owned_asset("spz", gzip_legacy, fixed_uuid(24)),
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "ply");
    EXPECT_EQ(published.sidecar, "0.ply");
    ASSERT_GE(published.dsrc.size(), 3u);
    EXPECT_EQ(std::memcmp(published.dsrc.data(), "ply", 3), 0);
    EXPECT_FALSE(std::filesystem::exists(request.path / "0.spz"));
}

TEST(GalleryScenePublicationTest, GallerySpzReencodesNonV4SourceViaNativeWriter) {
    TemporaryDirectory temporary;
    auto request = base_request(temporary.path / "legacy-spz.scene", ExportFormat::GALLERY_SPZ);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "legacy",
        .encoded = owned_asset("spz", unique_encoded_bytes(0x11), fixed_uuid(23)),
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "spz");
    EXPECT_EQ(published.sidecar, "0.spz");
    ASSERT_GE(published.dsrc.size(), 8u);
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::memcpy(&magic, published.dsrc.data(), 4);
    std::memcpy(&version, published.dsrc.data() + 4, 4);
    EXPECT_EQ(magic, 0x5053474eu);
    EXPECT_EQ(version, 4u);
    EXPECT_NE(published.dsrc, unique_encoded_bytes(0x11));
}

TEST(GalleryScenePublicationTest, GallerySpzPublicationCountMatchesVisibleAfterSoftDelete) {
    TemporaryDirectory temporary;
    auto snapshot = cpu_snapshot();
    ASSERT_EQ(snapshot.row_count, 8u);
    lfs::core::Tensor del = lfs::core::Tensor::zeros_bool({8}, snapshot.data->means().device());
    del.slice(0, 2, 5) = lfs::core::Tensor::ones_bool({3}, snapshot.data->means().device());
    snapshot.data->soft_delete(del);
    // Snapshot row_count covers stored rows; the deletion mask selects live rows.
    ASSERT_EQ(snapshot.row_count, 8u);
    ASSERT_EQ(snapshot.data->visible_count(), 5u);

    auto request = base_request(temporary.path / "deleted-spz.scene", ExportFormat::GALLERY_SPZ);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = std::move(snapshot),
        .name = "cropped",
        .encoded = std::nullopt,
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "spz");
    EXPECT_EQ(published.sidecar, "0.spz");
    ASSERT_GE(published.dsrc.size(), 16u);
    std::uint32_t count = 0;
    std::memcpy(&count, published.dsrc.data() + 8, 4);
    EXPECT_EQ(count, 5u);

    const auto loaded = lfs::io::load_spz(request.path / "0.spz");
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    EXPECT_EQ(loaded->size(), 5u);
    EXPECT_EQ(loaded->visible_count(), 5u);
}

TEST(GalleryScenePublicationTest, GallerySogPublicationCountMatchesVisibleAfterSoftDelete) {
    TemporaryDirectory temporary;
    auto snapshot = cpu_snapshot();
    ASSERT_EQ(snapshot.row_count, 8u);
    lfs::core::Tensor del = lfs::core::Tensor::zeros_bool({8}, snapshot.data->means().device());
    del.slice(0, 2, 5) = lfs::core::Tensor::ones_bool({3}, snapshot.data->means().device());
    snapshot.data->soft_delete(del);
    ASSERT_EQ(snapshot.data->visible_count(), 5u);

    auto request = base_request(temporary.path / "deleted-sog.scene", ExportFormat::GALLERY_SOG);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = std::move(snapshot),
        .name = "cropped",
        .encoded = std::nullopt,
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "sog");
    EXPECT_EQ(published.sidecar, "0.sog");
    EXPECT_EQ(published.publication_count, 5u);

    const auto loaded = lfs::io::load_sog(request.path / "0.sog");
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    EXPECT_EQ(loaded->size(), 5u);
    EXPECT_EQ(loaded->visible_count(), 5u);
}

TEST(GalleryScenePublicationTest, GallerySpzWritesGenuineV4WhenEncodingFromSplat) {
    TemporaryDirectory temporary;
    auto request = base_request(temporary.path / "encode-spz.scene", ExportFormat::GALLERY_SPZ);
    request.nodes.push_back(GalleryScenePublishNode{
        .snapshot = cpu_snapshot(),
        .name = "encoded",
        .encoded = std::nullopt,
    });
    writeGalleryScenePublication(request, {}, {});
    const auto published = read_published_node(request.path);
    EXPECT_EQ(published.source_kind, "spz");
    EXPECT_EQ(published.sidecar, "0.spz");
    ASSERT_GE(published.dsrc.size(), 16u);
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t count = 0;
    std::memcpy(&magic, published.dsrc.data(), 4);
    std::memcpy(&version, published.dsrc.data() + 4, 4);
    std::memcpy(&count, published.dsrc.data() + 8, 4);
    EXPECT_EQ(magic, 0x5053474eu);
    EXPECT_EQ(version, 4u);
    EXPECT_EQ(count, 8u);
    EXPECT_LE(static_cast<unsigned>(published.dsrc[12]), 3u);
    auto document = require_result_ptr(ProjectDocument::open(request.path / "project.licht"));
    const auto nodes = require_result(document->scene_graph().nodes());
    EXPECT_EQ(nodes.front().payload->source_kind, "spz");
    EXPECT_EQ(nodes.front().payload->fourcc, "DSRC");
}

namespace {
    using lfs::vis::gui::GalleryProjectExportRequest;
    using lfs::vis::gui::prepareGalleryProjectPublication;
    using lfs::vis::gui::verifyGalleryProjectCommit;

    std::filesystem::path portable_fixture(const std::string& kind) {
        return std::filesystem::path(__FILE__).parent_path() / "data" / ("portable-" + kind + ".licht");
    }

    // Deliberately invalid bindings cannot pass ProjectDocument::save validation.
    // Rewrite SCNG through the container writer and prove every other live row
    // unchanged, as in the project document's malformed-fixture helpers.
    void append_unavailable_payload_graph(const std::filesystem::path& path, const ProjectDocument& document) {
        namespace pj = lfs::io::project;
        const auto reader = require_result(pj::ProjectReader::open(path));
        auto writer = require_result(pj::ProjectWriter::append(path));
        require_status(writer.plan_commit());
        const auto bytes = document.scene_graph().to_bytes();
        require_status(writer.preflight(bytes.size()));
        const pj::ChunkKey graph_key{pj::FOURCC_SCNG, document.project_uuid()};
        require_status(writer.write_chunk(graph_key, bytes));
        constexpr std::uint64_t unchanged_epoch = 1;
        for (const auto& row : reader.chunks()) {
            if (!row.is_live() || row.key == graph_key)
                continue;
            const auto proof = require_result(reader.make_clean_proof(row, unchanged_epoch));
            require_status(writer.reuse_if_clean(proof, unchanged_epoch));
        }
        require_status(writer.commit());
    }
} // namespace

TEST(GalleryProjectExportTest, EncodedPortableAssetsAndSessionAreRetainedWithoutTensors) {
    TemporaryDirectory temporary;
    for (const auto& [kind, format] : std::vector<std::pair<std::string, ExportFormat>>{
             {"sog", ExportFormat::GALLERY_SOG},
             {"ssog", ExportFormat::GALLERY_SSOG},
             {"ply", ExportFormat::GALLERY_SCENE}}) {
        const auto path = portable_fixture(kind);
        ASSERT_TRUE(std::filesystem::is_regular_file(path)) << path;
        const auto original = read_file_bytes(path);
        auto source = require_result(ProjectDocument::open(path));
        const auto records = require_result(source.scene_graph().nodes());
        GalleryScenePublishRequest publication;
        std::string commit;
        prepareGalleryProjectPublication({path, temporary.path / (kind + ".scene"), format, ""}, publication, commit);
        ASSERT_EQ(publication.nodes.size(), 1u);
        EXPECT_EQ(commit, source.source_commit_uuid()->to_string());
        ASSERT_TRUE(publication.nodes[0].encoded.has_value());
        EXPECT_FALSE(publication.nodes[0].snapshot.data);
        // Prove writer does not invoke the document decoder for a reusable asset.
        publication.nodes[0].load_payload = []() -> std::shared_ptr<SplatData> {
            throw std::runtime_error("Unexpected tensor hydration on encoded copy path");
        };
        EXPECT_EQ(publication.published_render, *source.view().dom().get_json("render_settings"));
        EXPECT_EQ(publication.published_timeline, *source.sequencer().dom().get_json("timeline"));
        ASSERT_FALSE(publication.environment_source.empty());
        writeGalleryScenePublication(publication, {}, {});
        verifyGalleryProjectCommit(path, commit);
        EXPECT_FALSE(publication.materialized_payload);
        const auto published = read_published_node(publication.path);
        EXPECT_EQ(published.source_kind, kind);
        EXPECT_EQ(published.dsrc, read_lazy_bytes(*source.find_dataset_source(records[0].uuid)));
        EXPECT_EQ(read_file_bytes(publication.path / ("0." + kind)), published.dsrc);
        EXPECT_EQ(read_file_bytes(path), original);
    }
}

TEST(GalleryProjectExportTest, MultiNodeAncestorVisibilityAndWorldTransforms) {
    TemporaryDirectory temporary;
    const auto path = temporary.path / "multi.licht";
    std::filesystem::copy_file(portable_fixture("multi"), path);
    auto source = require_result(ProjectDocument::open(path));
    auto records = require_result(source.scene_graph().nodes());
    ASSERT_EQ(records.size(), 2u);
    // Upserting existing nodes preserves their array positions. Remove the
    // fixture records so the group is inserted before its child, as SCNG requires.
    for (const auto& record : records)
        ASSERT_TRUE(require_result(source.edit_scene_graph().remove_node(record.uuid)));
    const auto parent_transform = glm::translate(glm::mat4(1.0f), glm::vec3(7, 2, -3)) *
                                  glm::rotate(glm::mat4(1.0f), 0.5f, glm::vec3(0, 1, 0));
    lfs::io::project::SceneNodeRecord parent{.uuid = fixed_uuid(80), .type = "group", .name = "parent"};
    std::copy_n(&parent_transform[0][0], 16, parent.local_transform.begin());
    require_status(source.edit_scene_graph().upsert_node(parent));
    const auto child_transform = glm::scale(glm::mat4(1.0f), glm::vec3(2, 3, 4));
    records[0].parent_uuid = parent.uuid;
    std::copy_n(&child_transform[0][0], 16, records[0].local_transform.begin());
    require_status(source.edit_scene_graph().upsert_node(records[0]));
    records[1].child_order = 1;
    require_status(source.edit_scene_graph().upsert_node(records[1]));
    (void)require_result(source.save(path));
    GalleryScenePublishRequest publication;
    std::string commit;
    prepareGalleryProjectPublication({path, temporary.path / "multi.scene", ExportFormat::GALLERY_SCENE, ""}, publication, commit);
    ASSERT_EQ(publication.nodes.size(), 2u);
    const auto expected = parent_transform * child_transform;
    const auto found = std::find_if(publication.nodes.begin(), publication.nodes.end(),
                                    [&](const auto& node) { return node.name == records[0].name; });
    ASSERT_NE(found, publication.nodes.end());
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            EXPECT_FLOAT_EQ(found->snapshot.world_transform[column][row], expected[column][row]);
    parent.visible = false;
    require_status(source.edit_scene_graph().upsert_node(parent));
    (void)require_result(source.save(path));
    GalleryScenePublishRequest hidden;
    prepareGalleryProjectPublication({path, temporary.path / "hidden.scene", ExportFormat::GALLERY_SCENE, ""}, hidden, commit);
    ASSERT_EQ(hidden.nodes.size(), 1u);
    EXPECT_EQ(hidden.nodes.front().name, records[1].name);
    records[1].visible = false;
    require_status(source.edit_scene_graph().upsert_node(records[1]));
    (void)require_result(source.save(path));
    GalleryScenePublishRequest empty;
    try {
        prepareGalleryProjectPublication({path, temporary.path / "empty.scene", ExportFormat::GALLERY_SCENE, ""}, empty, commit);
        FAIL() << "An empty publication must be rejected";
    } catch (const std::runtime_error& error) {
        EXPECT_TRUE(std::string(error.what()).starts_with("gallery_project_no_splats:"));
    }
}

TEST(GalleryProjectExportTest, SavedSpzV4IsByteIdentical) {
    TemporaryDirectory temporary;
    auto original = base_request(temporary.path / "source.scene", ExportFormat::GALLERY_SPZ);
    original.nodes.push_back({.snapshot = cpu_snapshot(), .name = "SPZ fixture"});
    writeGalleryScenePublication(original, {}, {});
    GalleryScenePublishRequest publication;
    std::string commit;
    prepareGalleryProjectPublication({original.path / "project.licht", temporary.path / "copy.scene",
                                      ExportFormat::GALLERY_SPZ, ""},
                                     publication, commit);
    ASSERT_TRUE(publication.nodes.front().encoded.has_value());
    publication.nodes.front().load_payload = []() -> std::shared_ptr<SplatData> {
        throw std::runtime_error("SPZ copy must not hydrate");
    };
    writeGalleryScenePublication(publication, {}, {});
    EXPECT_EQ(read_file_bytes(original.path / "0.spz"), read_file_bytes(publication.path / "0.spz"));
    EXPECT_FALSE(publication.materialized_payload);
}

TEST(GalleryProjectExportTest, PlyPayloadReencodesToRequestedSog) {
    TemporaryDirectory temporary;
    GalleryScenePublishRequest publication;
    std::string commit;
    prepareGalleryProjectPublication({portable_fixture("ply"), temporary.path / "converted.scene",
                                      ExportFormat::GALLERY_SOG, ""},
                                     publication, commit);
    ASSERT_EQ(publication.nodes.size(), 1u);
    EXPECT_FALSE(publication.nodes.front().encoded.has_value());
    ASSERT_TRUE(publication.nodes.front().load_payload);
    writeGalleryScenePublication(publication, {}, {});
    EXPECT_TRUE(publication.materialized_payload);
    EXPECT_EQ(read_published_node(publication.path).source_kind, "sog");
    EXPECT_FALSE(std::filesystem::exists(publication.path / "0.ply"));
}

TEST(GalleryProjectExportTest, CommitMismatchRefusesBeforeStagingAndDetectsLaterSave) {
    TemporaryDirectory temporary;
    const auto path = temporary.path / "source.licht";
    std::filesystem::copy_file(portable_fixture("sog"), path);
    GalleryScenePublishRequest publication;
    std::string commit;
    const auto destination = temporary.path / "refused.scene";
    try {
        prepareGalleryProjectPublication({path, destination, ExportFormat::GALLERY_SOG, fixed_uuid(90).to_string()}, publication, commit);
        FAIL() << "Stale reviewed commit must be refused";
    } catch (const std::runtime_error& error) {
        EXPECT_TRUE(std::string(error.what()).starts_with("gallery_project_commit_mismatch:"));
    }
    EXPECT_FALSE(std::filesystem::exists(destination));
    publication = {};
    prepareGalleryProjectPublication({path, destination, ExportFormat::GALLERY_SOG, ""}, publication, commit);
    auto source = require_result(ProjectDocument::open(path));
    auto node = require_result(source.scene_graph().nodes()).front();
    node.name = "Saved after publication review";
    require_status(source.edit_scene_graph().upsert_node(node));
    const auto saved = require_result(source.save(path));
    EXPECT_NE(saved.commit_uuid.to_string(), commit);
    EXPECT_THROW(verifyGalleryProjectCommit(path, commit), std::runtime_error);
}

TEST(GalleryProjectExportTest, MissingAndExternalPayloadsGiveSpecificFallbackError) {
    TemporaryDirectory temporary;
    for (const bool external : {false, true}) {
        const auto path = temporary.path / (external ? "external.licht" : "missing.licht");
        std::filesystem::copy_file(portable_fixture("sog"), path);
        auto source = require_result(ProjectDocument::open(path));
        auto node = require_result(source.scene_graph().nodes()).front();
        node.payload->instance_uuid = fixed_uuid(91);
        if (external) {
            node.payload->fourcc = "REFS";
            node.payload->reference_uuid = fixed_uuid(91);
        }
        require_status(source.edit_scene_graph().upsert_node(node));
        append_unavailable_payload_graph(path, source);
        GalleryScenePublishRequest publication;
        std::string commit;
        try {
            prepareGalleryProjectPublication({path, temporary.path / "refused.scene", ExportFormat::GALLERY_SOG, ""}, publication, commit);
            FAIL() << "Unavailable payload must be refused";
        } catch (const std::runtime_error& error) {
            EXPECT_TRUE(std::string(error.what()).starts_with("gallery_project_payload_unavailable:")) << error.what();
        }
        EXPECT_FALSE(std::filesystem::exists(temporary.path / "refused.scene"));
    }
}
