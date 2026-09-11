/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gallery_scene_publication.hpp"

#include "core/logger.hpp"
#include "core/provenance.hpp"
#include "core/uuid.hpp"
#include "io/exporter.hpp"
#include "io/selection_chapter.hpp"
#include "rendering/environment_image.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <utility>

namespace lfs::vis::gui {
    namespace {

        using ExportFormat = lfs::core::ExportFormat;

        [[nodiscard]] bool isEncodedSplatKind(const std::string_view kind) noexcept {
            return kind == "ply" || kind == "sog" || kind == "ssog" || kind == "spz";
        }

        [[nodiscard]] const char* requestedFallbackExtension(const ExportFormat format) noexcept {
            switch (format) {
            case ExportFormat::GALLERY_SOG:
                return "sog";
            case ExportFormat::GALLERY_SSOG:
                return "ssog";
            case ExportFormat::GALLERY_SPZ:
                return "spz";
            default:
                return "ply";
            }
        }

        // Bounded header sniff: NGSP v4 magic/version/count/SH. Does not decompress ZSTD.
        [[nodiscard]] bool spzEncodedAssetLooksLikeV4(const lfs::io::project::LazyChunkValue& bytes,
                                                      const std::uint64_t expected_count) {
            if (bytes.size() < 16)
                return false;
            std::array<std::byte, 16> header{};
            const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(header.size(), bytes.size()));
            if (!bytes.read_at(0, std::span<std::byte>(header.data(), n)))
                return false;
            std::uint32_t magic = 0;
            std::uint32_t version = 0;
            std::uint32_t num_points = 0;
            std::memcpy(&magic, header.data(), 4);
            std::memcpy(&version, header.data() + 4, 4);
            std::memcpy(&num_points, header.data() + 8, 4);
            const auto sh_degree = static_cast<std::uint8_t>(header[12]);
            constexpr std::uint32_t kNgspMagic = 0x5053474e; // 'NGSP'
            return magic == kNgspMagic && version == 4 && num_points == expected_count && sh_degree <= 3;
        }

        void throwIfCanceled(const std::function<bool()>& canceled, const char* message) {
            if (canceled && canceled())
                throw std::runtime_error(message);
        }

    } // namespace

    bool galleryEncodedAssetReusable(const ExportFormat requested, const std::string_view source_kind,
                                     const bool payload_diverged) noexcept {
        if (payload_diverged || !isEncodedSplatKind(source_kind))
            return false;
        switch (requested) {
        case ExportFormat::GALLERY_SOG:
            return source_kind == "sog";
        case ExportFormat::GALLERY_SSOG:
            return source_kind == "ssog";
        case ExportFormat::GALLERY_SPZ:
            return source_kind == "spz";
        case ExportFormat::GALLERY_SCENE:
            return true;
        default:
            return false;
        }
    }

    std::string galleryPublicationExtension(const ExportFormat requested,
                                            const std::optional<std::string>& reused_kind) {
        if (reused_kind && isEncodedSplatKind(*reused_kind))
            return *reused_kind;
        return requestedFallbackExtension(requested);
    }

    std::optional<GalleryEncodedAsset> snapshotGalleryEncodedAsset(
        const lfs::io::project::ProjectDocument* document, const core::SceneNode& node,
        const ExportFormat requested) {
        if (!document || node.type != core::NodeType::SPLAT || node.payload_diverged)
            return std::nullopt;
        if (node.payload_hydration == core::PayloadHydrationState::Unloaded ||
            node.payload_hydration == core::PayloadHydrationState::Hydrating ||
            node.payload_hydration == core::PayloadHydrationState::Failed)
            return std::nullopt;
        if (node.model && node.model->has_deleted_mask())
            return std::nullopt;
        const auto records = document->scene_graph().nodes();
        if (!records)
            return std::nullopt;
        const lfs::io::project::SceneNodeRecord* binding_record = nullptr;
        for (const auto& record : *records) {
            if (record.uuid == node.uuid) {
                binding_record = &record;
                break;
            }
        }
        if (!binding_record || !binding_record->payload)
            return std::nullopt;
        const auto& binding = *binding_record->payload;
        if (binding.fourcc != "DSRC" || binding.instance_uuid != node.uuid || binding.reference_uuid ||
            !galleryEncodedAssetReusable(requested, binding.source_kind, false))
            return std::nullopt;
        const auto* source = document->find_dataset_source(node.uuid);
        if (!source)
            return std::nullopt;
        auto owned = source->share();
        if (!owned) {
            LOG_WARN("Gallery publication could not retain encoded asset {}: {}", node.uuid.to_string(),
                     owned.error().user_message());
            return std::nullopt;
        }
        return GalleryEncodedAsset{.source_kind = binding.source_kind, .bytes = std::move(*owned)};
    }

    void copyLazyChunkToFile(const lfs::io::project::LazyChunkValue& source,
                             const std::filesystem::path& destination,
                             const std::function<bool()>& canceled) {
        std::ofstream output(destination, std::ios::binary | std::ios::trunc);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        const auto copied = source.visit_stream([&](std::istream& input, const std::uint64_t size) -> lfs::Result<void> {
            std::array<char, 1024 * 1024> buffer{};
            std::uint64_t offset = 0;
            while (offset < size) {
                throwIfCanceled(canceled, "Scene preparation canceled.");
                const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), size - offset));
                input.read(buffer.data(), static_cast<std::streamsize>(count));
                if (input.gcount() != static_cast<std::streamsize>(count)) {
                    throw std::runtime_error("Encoded scene asset is incomplete.");
                }
                output.write(buffer.data(), static_cast<std::streamsize>(count));
                offset += count;
            }
            return {};
        });
        if (!copied)
            throw std::runtime_error(std::string(copied.error().user_message()));
        output.close();
    }

    void writeGalleryScenePublication(GalleryScenePublishRequest& request,
                                      const std::function<bool(float, const std::string&)>& report,
                                      const std::function<bool()>& canceled) {
        throwIfCanceled(canceled, "Scene preparation canceled.");
        if (!std::filesystem::create_directory(request.path))
            throw std::runtime_error("The gallery preparation directory already exists.");
        request.created_directory = true;
        std::filesystem::permissions(request.path, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace);
        auto nodes = nlohmann::json::array();
        std::vector<std::pair<core::Uuid, std::filesystem::path>> embedded_files;
        namespace pj = lfs::io::project;
        const auto project_id = core::generate_uuid_v4();
        auto document_result = pj::ProjectDocument::create(project_id);
        if (!document_result)
            throw std::runtime_error(std::string(document_result.error().user_message()));
        auto document = std::move(*document_result);
        const auto checked = [](auto result) {
            if (!result)
                throw std::runtime_error(std::string(result.error().user_message()));
        };
        if (!request.published_timeline.is_null())
            checked(document.edit_sequencer().dom().set_json("timeline", request.published_timeline));
        checked(document.edit_sequencer().dom().set_json("loop_mode", request.published_loop_mode));
        checked(document.edit_sequencer().dom().set_json("playback_speed", request.published_playback_speed));
        auto render = request.published_render.is_null() ? project::SessionJson::object() : request.published_render;
        render["environment_reference_uuid"] = nullptr;
        checked(document.edit_view().dom().set_json("render_settings", render));
        auto published_camera =
            request.published_camera.is_null() ? project::SessionJson{{"panel", "primary"}} : request.published_camera;
        auto right_camera = published_camera;
        right_camera["panel"] = "secondary";
        checked(document.edit_view().dom().set_json("panel_cameras",
                                                    project::SessionJson::array({published_camera, right_camera})));
        const auto add_reference = [&](const std::string& file, const std::string& kind) {
            const auto id = core::generate_uuid_v4();
            auto fingerprint = pj::fingerprint_path(request.path / file, true);
            if (!fingerprint)
                throw std::runtime_error(std::string(fingerprint.error().user_message()));
            checked(document.edit_references().upsert(pj::ReferenceRecord{
                .uuid = id,
                .key = file,
                .kind = kind,
                .locator = {.preferred = id.to_string() + ".lfsenv", .base = pj::LocatorBase::Project},
                .fingerprint = *fingerprint}));
            return id;
        };
        const auto node_count = request.nodes.size();
        const float geometry_span = request.environment_source.empty() ? 1.0f : 0.9f;
        for (size_t i = 0; i < node_count; ++i) {
            if (report && !report(static_cast<float>(i) / static_cast<float>(node_count) * geometry_span,
                                  "Preparing scene for upload"))
                throw std::runtime_error("Scene preparation canceled.");
            auto& published = request.nodes[i];
            std::optional<std::string> reused_kind;
            if (published.encoded) {
                reused_kind = published.encoded->source_kind;
                if (reused_kind == "spz" &&
                    !spzEncodedAssetLooksLikeV4(published.encoded->bytes, published.snapshot.row_count))
                    reused_kind.reset();
            }
            const std::string extension = galleryPublicationExtension(request.format, reused_kind);
            std::uint64_t published_count = 0;
            if (published.encoded && reused_kind) {
                if (published.snapshot.row_count == 0)
                    continue;
                published_count = published.snapshot.row_count;
                copyLazyChunkToFile(published.encoded->bytes,
                                    request.path / (std::to_string(nodes.size()) + "." + extension), canceled);
            } else {
                auto data = published.snapshot.materialize();
                if (data->visible_count() == 0)
                    continue;
                published_count = data->visible_count();
                const auto filename = std::to_string(nodes.size()) + "." + extension;
                const io::PlySaveOptions options{
                    .output_path = request.path / filename,
                    .progress_callback =
                        [&](float progress, const std::string&) {
                            return !report ||
                                   report((static_cast<float>(i) + progress) /
                                              static_cast<float>(node_count) * geometry_span,
                                          "Preparing scene for upload");
                        },
                    .provenance = core::make_minimal_provenance_stamp()};
                const auto result =
                    extension == "sog"
                        ? io::save_sog(*data, {.output_path = options.output_path,
                                               .progress_callback = options.progress_callback,
                                               .provenance = options.provenance})
                    : extension == "ssog"
                        ? io::save_ssog(*data, {.output_path = options.output_path,
                                                .progress_callback = options.progress_callback,
                                                .provenance = options.provenance})
                    : extension == "spz"
                        ? io::save_spz(*data, {.output_path = options.output_path,
                                               .version = 4,
                                               .progress_callback = options.progress_callback,
                                               .provenance = options.provenance})
                        : io::save_ply(*data, options);
                if (!result)
                    throw std::runtime_error(result.error().message);
            }
            const auto filename = std::to_string(nodes.size()) + "." + extension;
            auto transform = nlohmann::json::array();
            for (int row = 0; row < 4; ++row) {
                auto values = nlohmann::json::array();
                for (int column = 0; column < 4; ++column)
                    values.push_back(published.snapshot.world_transform[column][row]);
                transform.push_back(values);
            }
            const auto reference_id = core::generate_uuid_v4();
            embedded_files.emplace_back(reference_id, request.path / filename);
            pj::SceneNodeRecord record{
                .uuid = reference_id,
                .type = "splat",
                .name = published.name,
                .child_order = static_cast<uint32_t>(nodes.size()),
                .training_enabled = false,
                .payload = pj::PayloadBinding{.fourcc = "DSRC", .instance_uuid = reference_id, .source_kind = extension}};
            std::copy_n(&published.snapshot.world_transform[0][0], 16, record.local_transform.begin());
            checked(document.edit_scene_graph().upsert_node(record));
            auto published_node = document.edit_scene_graph().dom().array_find("nodes", record.uuid.to_string());
            checked(published_node->set_json(
                "publication",
                project::SessionJson{{"count", published_count}, {"sh_degree", published.snapshot.active_sh_degree}}));
            nodes.push_back(
                {{"path", filename}, {"transform", transform}, {"shDegree", published.snapshot.active_sh_degree}});
        }
        if (nodes.empty())
            throw std::runtime_error("There are no visible splats to upload.");
        throwIfCanceled(canceled, "Scene preparation canceled.");
        auto metadata = nlohmann::json{{"version", 1}, {"nodes", nodes}};
        if (!request.environment_source.empty()) {
            if (report && !report(0.95f, "Preparing HDR background"))
                throw std::runtime_error("Scene preparation canceled.");
            const auto loaded = rendering::loadEnvironmentImageShared(request.environment_source);
            if (!loaded)
                throw std::runtime_error("The HDR background could not be read. Choose another background and retry.");
            const auto& image = **loaded;
            if (!image.valid() || image.width > 8192 || image.height > 8192 ||
                uint64_t(image.width) * image.height > 8'388'608)
                throw std::runtime_error("Choose an HDR background with at most 8 million pixels for gallery upload.");
            static_assert(std::endian::native == std::endian::little);
            std::ofstream background(request.path / "environment.lfsenv", std::ios::binary);
            background.exceptions(std::ios::badbit | std::ios::failbit);
            background.write("LFSENV1\0", 8);
            const uint32_t width = image.width, height = image.height;
            background.write(reinterpret_cast<const char*>(&width), 4);
            background.write(reinterpret_cast<const char*>(&height), 4);
            for (int row = 0; row < image.height; ++row) {
                throwIfCanceled(canceled, "Scene preparation canceled.");
                const auto* pixels = image.pixels.data() + size_t(row) * width * 3;
                if (!std::all_of(pixels, pixels + width * 3, [](float x) { return std::isfinite(x); }))
                    throw std::runtime_error("The HDR background contains invalid pixel values.");
                background.write(reinterpret_cast<const char*>(pixels), size_t(width) * 12);
            }
            background.close();
            metadata["environment"] = "environment.lfsenv";
            const auto environment_id = add_reference("environment.lfsenv", "environment_map");
            embedded_files.emplace_back(environment_id, request.path / "environment.lfsenv");
            checked(document.edit_view().dom().set_json("render_settings.environment_reference_uuid",
                                                        environment_id.to_string()));
        }
        // A fresh native project contains only the published scene and default empty
        // session chapters. Never copy the user's project or its history.
        auto writer = pj::ProjectWriter::create(request.path / "project.licht",
                                                {.project_uuid = project_id,
                                                 .file_uuid = core::generate_uuid_v4(),
                                                 .index_compression = pj::IndexCompression::StoredForDeterministicTests});
        if (!writer)
            throw std::runtime_error(std::string(writer.error().user_message()));
        pj::CommitOptions commit_options;
        commit_options.extra_reader_capabilities.set(pj::ENCODED_SCENE_ASSETS);
        commit_options.extra_writer_capabilities.set(pj::ENCODED_SCENE_ASSETS);
        checked(writer->plan_commit(commit_options));
        uint64_t planned_bytes = 4 * 1024 * 1024;
        for (const auto& [id, file] : embedded_files)
            planned_bytes += std::filesystem::file_size(file);
        checked(writer->preflight(planned_bytes));
        const auto chapter = [&](pj::Fourcc type, const auto& value) {
            const auto bytes = value.to_bytes();
            if constexpr (std::is_same_v<std::decay_t<decltype(bytes)>, std::vector<std::byte>>)
                checked(writer->write_chunk({type, project_id}, bytes));
            else {
                if (!bytes)
                    throw std::runtime_error(std::string(bytes.error().user_message()));
                checked(writer->write_chunk({type, project_id}, *bytes));
            }
        };
        chapter(pj::FOURCC_PROJ, document.project());
        chapter(pj::FOURCC_REFS, document.references());
        chapter(pj::FOURCC_SCNG, document.scene_graph());
        chapter(pj::FOURCC_PRMS, document.parameters());
        const auto selection = pj::encode_selection_chapter(document.selection());
        if (!selection)
            throw std::runtime_error(std::string(selection.error().user_message()));
        checked(writer->write_chunk({pj::FOURCC_SELM, project_id}, *selection));
        chapter(pj::FOURCC_GUIL, document.gui_layout());
        chapter(pj::FOURCC_VIEW, document.view());
        chapter(pj::FOURCC_EDTR, document.editor());
        chapter(pj::FOURCC_SEQR, document.sequencer());
        chapter(pj::FOURCC_METR, document.metrics());
        for (const auto& [id, file] : embedded_files) {
            const auto bytes = std::filesystem::file_size(file);
            auto target = writer->begin_chunk({pj::FOURCC_DSRC, id}, {.expected_stream_bytes = bytes});
            if (!target)
                throw std::runtime_error(std::string(target.error().user_message()));
            std::ifstream input(file, std::ios::binary);
            input.exceptions(std::ios::badbit);
            std::array<char, 1024 * 1024> buffer{};
            uint64_t copied = 0;
            while (input) {
                throwIfCanceled(canceled, "Project preparation canceled.");
                input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                (*target)->write(buffer.data(), input.gcount());
                copied += static_cast<uint64_t>(input.gcount());
            }
            if (copied != bytes)
                throw std::runtime_error("Project asset changed during preparation.");
            checked(writer->end_chunk());
        }
        checked(writer->commit());
        auto verified_project = pj::ProjectDocument::open(request.path / "project.licht");
        if (!verified_project)
            throw std::runtime_error(std::string(verified_project.error().user_message()));
        std::ofstream manifest(request.path / "manifest.json.tmp", std::ios::binary | std::ios::trunc);
        manifest.exceptions(std::ios::badbit | std::ios::failbit);
        manifest << metadata.dump();
        manifest.close();
        std::filesystem::rename(request.path / "manifest.json.tmp", request.path / "manifest.json");
    }

} // namespace lfs::vis::gui
