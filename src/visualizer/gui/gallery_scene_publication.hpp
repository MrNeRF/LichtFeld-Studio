/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/export.hpp"
#include "core/scene.hpp"
#include "io/project_document.hpp"
#include "project/session_state.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis::gui {

    struct GalleryEncodedAsset {
        std::string source_kind;
        lfs::io::project::LazyChunkValue bytes;
    };

    struct GalleryScenePublishNode {
        core::Scene::SplatSnapshot snapshot;
        std::string name;
        std::optional<GalleryEncodedAsset> encoded;
    };

    struct GalleryScenePublishRequest {
        std::filesystem::path path;
        core::ExportFormat format = core::ExportFormat::GALLERY_SCENE;
        std::vector<GalleryScenePublishNode> nodes;
        project::SessionJson published_render;
        project::SessionJson published_camera;
        project::SessionJson published_timeline;
        std::string published_loop_mode = "once";
        float published_playback_speed = 1.0f;
        std::filesystem::path environment_source;
        bool created_directory = false;
    };

    // Studio/GALLERY_SCENE keeps a clean ply/sog/ssog asset. Explicit SOG/SSOG
    // reuse only when the requested compression already matches the source.
    [[nodiscard]] LFS_VIS_API bool galleryEncodedAssetReusable(
        core::ExportFormat requested, std::string_view source_kind,
        bool payload_diverged) noexcept;

    [[nodiscard]] LFS_VIS_API std::string galleryPublicationExtension(
        core::ExportFormat requested,
        const std::optional<std::string>& reused_kind);

    // Captures an independent owner of the node's current DSRC bytes. Never
    // reads the live document after this returns.
    [[nodiscard]] LFS_VIS_API std::optional<GalleryEncodedAsset>
    snapshotGalleryEncodedAsset(const lfs::io::project::ProjectDocument* document,
                                const core::SceneNode& node,
                                core::ExportFormat requested);

    LFS_VIS_API void copyLazyChunkToFile(
        const lfs::io::project::LazyChunkValue& source,
        const std::filesystem::path& destination,
        const std::function<bool()>& canceled = {});

    LFS_VIS_API void writeGalleryScenePublication(
        GalleryScenePublishRequest& request,
        const std::function<bool(float, const std::string&)>& report,
        const std::function<bool()>& canceled);

} // namespace lfs::vis::gui
