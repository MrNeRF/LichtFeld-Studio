/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/video_reconstruction.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <ranges>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

namespace lfs::io::video {

    namespace {

        constexpr std::size_t MAX_SELECTION_JSON_BYTES = 8u * 1024u;
        constexpr std::size_t MAX_IDENTIFIER_BYTES = 128u;

        [[nodiscard]] bool validIdentifier(const std::string_view id) noexcept {
            if (id.empty() || id.size() > MAX_IDENTIFIER_BYTES)
                return false;
            if ((id.front() < 'a' || id.front() > 'z') &&
                (id.front() < '0' || id.front() > '9'))
                return false;
            return std::ranges::all_of(id, [](const char value) {
                return (value >= 'a' && value <= 'z') ||
                       (value >= '0' && value <= '9') ||
                       value == '.' || value == '_' || value == '-';
            });
        }

        [[nodiscard]] VideoReconstructionBackendDescriptor nativeDescriptor() {
            return {
                .backend_id = "native",
                .provider_id = "lichtfeld",
                .available = true,
                .supports_offline_export = true,
                .supports_perspective = true,
                .supports_equirectangular = true,
                .required_resources = resourceMask(VideoReconstructionResource::None),
                .presets = {{.preset_id = "native", .input_scale = 1.0f}},
            };
        }

        class NativeVideoReconstructionCatalog final : public VideoReconstructionCatalog {
        public:
            std::optional<VideoReconstructionBackendDescriptor> findBackend(
                const std::string_view backend_id) const override {
                if (backend_id == "native")
                    return nativeDescriptor();
                return std::nullopt;
            }
        };

        [[nodiscard]] std::optional<std::string> validateDescriptor(
            const VideoReconstructionBackendDescriptor& descriptor,
            const std::string_view requested_backend_id) {
            if (!validIdentifier(descriptor.backend_id) ||
                descriptor.backend_id != requested_backend_id) {
                return "The catalog returned a descriptor with a mismatched or invalid backend id";
            }
            if (descriptor.presets.empty())
                return "The reconstruction backend does not declare any presets";

            std::unordered_set<std::string> preset_ids;
            preset_ids.reserve(descriptor.presets.size());
            for (const auto& preset : descriptor.presets) {
                if (!validIdentifier(preset.preset_id))
                    return "The reconstruction backend declares an invalid preset id";
                if (!preset_ids.emplace(preset.preset_id).second)
                    return "The reconstruction backend declares a duplicate preset id";
                if (!std::isfinite(preset.input_scale) ||
                    preset.input_scale <= 0.0f || preset.input_scale > 1.0f) {
                    return "The reconstruction backend declares an invalid input scale";
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] std::pair<int, int> scaledExtent(
            const int output_width,
            const int output_height,
            const float input_scale) {
            const auto scale = [input_scale](const int value) {
                const double scaled = std::round(
                    static_cast<double>(value) * static_cast<double>(input_scale));
                if (scaled >= static_cast<double>(std::numeric_limits<int>::max()))
                    return std::numeric_limits<int>::max();
                int result = std::max(2, static_cast<int>(scaled));
                if ((result & 1) != 0)
                    ++result;
                return result;
            };
            return {scale(output_width), scale(output_height)};
        }

    } // namespace

    VideoReconstructionPlan::VideoReconstructionPlan(
        const int output_width,
        const int output_height,
        const int input_width,
        const int input_height,
        const float input_scale,
        const VideoReconstructionProjection projection,
        const VideoReconstructionResourceMask required_resources,
        const VideoReconstructionFallback fallback_policy,
        VideoReconstructionProvenance provenance)
        : output_width_(output_width),
          output_height_(output_height),
          input_width_(input_width),
          input_height_(input_height),
          input_scale_(input_scale),
          projection_(projection),
          required_resources_(required_resources),
          fallback_policy_(fallback_policy),
          provenance_(std::move(provenance)) {}

    const VideoReconstructionCatalog& nativeVideoReconstructionCatalog() {
        static const NativeVideoReconstructionCatalog catalog;
        return catalog;
    }

    std::expected<void, VideoReconstructionSelectionError> validateVideoReconstructionSelection(
        const VideoReconstructionSelection& selection) {
        if (!validIdentifier(selection.backend_id)) {
            return std::unexpected(VideoReconstructionSelectionError{
                .issue = VideoReconstructionSelectionIssue::InvalidBackendId,
                .message = "Video reconstruction backend id is invalid",
            });
        }
        if (!validIdentifier(selection.preset_id)) {
            return std::unexpected(VideoReconstructionSelectionError{
                .issue = VideoReconstructionSelectionIssue::InvalidPresetId,
                .message = "Video reconstruction preset id is invalid",
            });
        }
        switch (selection.fallback) {
        case VideoReconstructionFallback::Abort:
        case VideoReconstructionFallback::Native:
            break;
        default:
            return std::unexpected(VideoReconstructionSelectionError{
                .issue = VideoReconstructionSelectionIssue::InvalidFallback,
                .message = "Video reconstruction fallback is invalid",
            });
        }
        if (selection.backend_id == "native" && selection.preset_id != "native") {
            return std::unexpected(VideoReconstructionSelectionError{
                .issue = VideoReconstructionSelectionIssue::InvalidNativePreset,
                .message = "Native video reconstruction requires the native preset",
            });
        }
        return {};
    }

    std::expected<VideoReconstructionPlan, VideoReconstructionResolutionError>
    resolveVideoReconstructionPlan(
        const VideoReconstructionRequest& request,
        const VideoReconstructionCatalog& catalog) {
        if (const auto valid = validateVideoReconstructionSelection(request.selection); !valid) {
            return std::unexpected(VideoReconstructionResolutionError{
                .issue = VideoReconstructionResolutionIssue::InvalidSelection,
                .message = valid.error().message,
            });
        }
        if (request.output_width <= 0 || request.output_height <= 0 ||
            (request.output_width & 1) != 0 || (request.output_height & 1) != 0) {
            return std::unexpected(VideoReconstructionResolutionError{
                .issue = VideoReconstructionResolutionIssue::InvalidOutputExtent,
                .message = "Video reconstruction output dimensions must be positive and even",
            });
        }

        const auto make_plan = [&request](
                                   const VideoReconstructionBackendDescriptor& effective,
                                   const VideoReconstructionPresetDescriptor& effective_preset,
                                   const VideoReconstructionBackendDescriptor* requested,
                                   const VideoReconstructionResolutionIssue issue,
                                   std::string unavailable_reason_id) {
            const auto [input_width, input_height] = scaledExtent(
                request.output_width, request.output_height, effective_preset.input_scale);
            VideoReconstructionProvenance provenance{
                .requested_backend_id = request.selection.backend_id,
                .requested_preset_id = request.selection.preset_id,
                .effective_backend_id = effective.backend_id,
                .effective_preset_id = effective_preset.preset_id,
                .effective_provider_id = effective.provider_id,
                .effective_provider_version = effective.provider_version,
                .effective_provider_digest = effective.provider_digest,
                .resolution_issue = issue,
                .unavailable_reason_id = std::move(unavailable_reason_id),
            };
            if (requested) {
                provenance.requested_provider_id = requested->provider_id;
                provenance.requested_provider_version = requested->provider_version;
                provenance.requested_provider_digest = requested->provider_digest;
            }
            return VideoReconstructionPlan(
                request.output_width,
                request.output_height,
                input_width,
                input_height,
                effective_preset.input_scale,
                request.projection,
                effective.required_resources | effective_preset.additional_resources,
                request.selection.fallback,
                std::move(provenance));
        };
        const auto fail_or_use_native =
            [&request, &make_plan](
                const VideoReconstructionBackendDescriptor* requested,
                const VideoReconstructionResolutionIssue issue,
                std::string message,
                std::string unavailable_reason_id = {})
            -> std::expected<VideoReconstructionPlan, VideoReconstructionResolutionError> {
            if (request.selection.fallback != VideoReconstructionFallback::Native) {
                return std::unexpected(VideoReconstructionResolutionError{
                    .issue = issue,
                    .message = std::move(message),
                });
            }
            const auto native = nativeDescriptor();
            return make_plan(
                native,
                native.presets.front(),
                requested,
                issue,
                std::move(unavailable_reason_id));
        };

        if (request.selection.backend_id == "native") {
            const auto native = nativeDescriptor();
            return make_plan(
                native,
                native.presets.front(),
                &native,
                VideoReconstructionResolutionIssue::None,
                {});
        }

        const auto descriptor = catalog.findBackend(request.selection.backend_id);
        if (!descriptor) {
            return fail_or_use_native(
                nullptr,
                VideoReconstructionResolutionIssue::BackendNotFound,
                std::format(
                    "Video reconstruction backend '{}' was not found",
                    request.selection.backend_id));
        }
        if (const auto invalid = validateDescriptor(*descriptor, request.selection.backend_id)) {
            return fail_or_use_native(
                &*descriptor,
                VideoReconstructionResolutionIssue::InvalidDescriptor,
                *invalid);
        }
        if (!descriptor->supports_offline_export) {
            return fail_or_use_native(
                &*descriptor,
                VideoReconstructionResolutionIssue::OfflineExportUnsupported,
                std::format(
                    "Video reconstruction backend '{}' does not support offline export",
                    request.selection.backend_id));
        }
        if (!descriptor->available) {
            return fail_or_use_native(
                &*descriptor,
                VideoReconstructionResolutionIssue::BackendUnavailable,
                std::format(
                    "Video reconstruction backend '{}' is unavailable{}{}",
                    request.selection.backend_id,
                    descriptor->unavailable_reason_id.empty() ? "" : ": ",
                    descriptor->unavailable_reason_id),
                descriptor->unavailable_reason_id);
        }

        const bool projection_supported =
            request.projection == VideoReconstructionProjection::Perspective
                ? descriptor->supports_perspective
                : descriptor->supports_equirectangular;
        if (!projection_supported) {
            return fail_or_use_native(
                &*descriptor,
                VideoReconstructionResolutionIssue::ProjectionUnsupported,
                std::format(
                    "Video reconstruction backend '{}' does not support the requested projection",
                    request.selection.backend_id));
        }

        const auto preset = std::ranges::find(
            descriptor->presets,
            request.selection.preset_id,
            &VideoReconstructionPresetDescriptor::preset_id);
        if (preset == descriptor->presets.end()) {
            return fail_or_use_native(
                &*descriptor,
                VideoReconstructionResolutionIssue::PresetNotFound,
                std::format(
                    "Video reconstruction preset '{}' was not found for backend '{}'",
                    request.selection.preset_id,
                    request.selection.backend_id));
        }
        return make_plan(
            *descriptor,
            *preset,
            &*descriptor,
            VideoReconstructionResolutionIssue::None,
            {});
    }

    std::expected<std::string, VideoReconstructionSelectionError>
    serializeVideoReconstructionSelection(
        const VideoReconstructionSelection& selection) {
        if (const auto valid = validateVideoReconstructionSelection(selection); !valid)
            return std::unexpected(valid.error());

        nlohmann::ordered_json json{
            {"version", VIDEO_RECONSTRUCTION_SELECTION_VERSION},
            {"backend_id", selection.backend_id},
            {"preset_id", selection.preset_id},
            {"fallback", videoReconstructionFallbackId(selection.fallback)},
        };
        return json.dump();
    }

    std::expected<VideoReconstructionSelection, VideoReconstructionSelectionError>
    deserializeVideoReconstructionSelection(
        const std::optional<std::string_view> serialized) {
        if (!serialized)
            return VideoReconstructionSelection{};
        if (serialized->size() > MAX_SELECTION_JSON_BYTES) {
            return std::unexpected(VideoReconstructionSelectionError{
                .issue = VideoReconstructionSelectionIssue::SizeLimitExceeded,
                .message = "Video reconstruction selection exceeds the size limit",
            });
        }

        try {
            const auto json = nlohmann::json::parse(*serialized);
            if (!json.is_object()) {
                return std::unexpected(VideoReconstructionSelectionError{
                    .issue = VideoReconstructionSelectionIssue::InvalidShape,
                    .message = "Video reconstruction selection must be a JSON object",
                });
            }
            const auto version = json.find("version");
            const bool supported_version =
                version != json.end() &&
                ((version->is_number_unsigned() &&
                  version->get<std::uint64_t>() == VIDEO_RECONSTRUCTION_SELECTION_VERSION) ||
                 (version->is_number_integer() &&
                  version->get<std::int64_t>() == VIDEO_RECONSTRUCTION_SELECTION_VERSION));
            if (!supported_version) {
                return std::unexpected(VideoReconstructionSelectionError{
                    .issue = VideoReconstructionSelectionIssue::UnsupportedVersion,
                    .message = "Unsupported video reconstruction selection version",
                });
            }
            if (!json.contains("backend_id") || !json["backend_id"].is_string() ||
                !json.contains("preset_id") || !json["preset_id"].is_string() ||
                !json.contains("fallback") || !json["fallback"].is_string()) {
                return std::unexpected(VideoReconstructionSelectionError{
                    .issue = VideoReconstructionSelectionIssue::MissingField,
                    .message =
                        "Video reconstruction selection requires backend_id, preset_id, and fallback",
                });
            }

            const auto fallback = videoReconstructionFallbackFromId(
                json["fallback"].get_ref<const std::string&>());
            if (!fallback) {
                return std::unexpected(VideoReconstructionSelectionError{
                    .issue = VideoReconstructionSelectionIssue::InvalidFallback,
                    .message = "Video reconstruction fallback is invalid",
                });
            }

            VideoReconstructionSelection selection{
                .backend_id = json["backend_id"].get<std::string>(),
                .preset_id = json["preset_id"].get<std::string>(),
                .fallback = *fallback,
            };
            if (const auto valid = validateVideoReconstructionSelection(selection); !valid)
                return std::unexpected(valid.error());
            return selection;
        } catch (const nlohmann::json::exception& error) {
            return std::unexpected(VideoReconstructionSelectionError{
                .issue = VideoReconstructionSelectionIssue::InvalidJson,
                .message = std::format(
                    "Invalid video reconstruction selection JSON: {}", error.what()),
            });
        }
    }

    std::string_view videoReconstructionFallbackId(
        const VideoReconstructionFallback fallback) noexcept {
        switch (fallback) {
        case VideoReconstructionFallback::Abort:
            return "abort";
        case VideoReconstructionFallback::Native:
            return "native";
        }
        return "unknown";
    }

    std::optional<VideoReconstructionFallback> videoReconstructionFallbackFromId(
        const std::string_view id) noexcept {
        if (id == "abort")
            return VideoReconstructionFallback::Abort;
        if (id == "native")
            return VideoReconstructionFallback::Native;
        return std::nullopt;
    }

    std::string_view videoReconstructionResolutionIssueId(
        const VideoReconstructionResolutionIssue issue) noexcept {
        switch (issue) {
        case VideoReconstructionResolutionIssue::None:
            return "none";
        case VideoReconstructionResolutionIssue::InvalidSelection:
            return "invalid_selection";
        case VideoReconstructionResolutionIssue::InvalidOutputExtent:
            return "invalid_output_extent";
        case VideoReconstructionResolutionIssue::BackendNotFound:
            return "backend_not_found";
        case VideoReconstructionResolutionIssue::BackendUnavailable:
            return "backend_unavailable";
        case VideoReconstructionResolutionIssue::OfflineExportUnsupported:
            return "offline_export_unsupported";
        case VideoReconstructionResolutionIssue::ProjectionUnsupported:
            return "projection_unsupported";
        case VideoReconstructionResolutionIssue::PresetNotFound:
            return "preset_not_found";
        case VideoReconstructionResolutionIssue::InvalidDescriptor:
            return "invalid_descriptor";
        }
        return "unknown";
    }

    std::string_view videoReconstructionSelectionIssueId(
        const VideoReconstructionSelectionIssue issue) noexcept {
        switch (issue) {
        case VideoReconstructionSelectionIssue::InvalidBackendId:
            return "invalid_backend_id";
        case VideoReconstructionSelectionIssue::InvalidPresetId:
            return "invalid_preset_id";
        case VideoReconstructionSelectionIssue::InvalidFallback:
            return "invalid_fallback";
        case VideoReconstructionSelectionIssue::InvalidNativePreset:
            return "invalid_native_preset";
        case VideoReconstructionSelectionIssue::SizeLimitExceeded:
            return "size_limit_exceeded";
        case VideoReconstructionSelectionIssue::InvalidJson:
            return "invalid_json";
        case VideoReconstructionSelectionIssue::InvalidShape:
            return "invalid_shape";
        case VideoReconstructionSelectionIssue::UnsupportedVersion:
            return "unsupported_version";
        case VideoReconstructionSelectionIssue::MissingField:
            return "missing_field";
        }
        return "unknown";
    }

} // namespace lfs::io::video
