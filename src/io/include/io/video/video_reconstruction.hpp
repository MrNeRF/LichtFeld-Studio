/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::io::video {

    inline constexpr std::uint32_t VIDEO_RECONSTRUCTION_SELECTION_VERSION = 1;

    enum class VideoReconstructionFallback : std::uint8_t {
        Abort = 0,
        Native,
    };

    enum class VideoReconstructionProjection : std::uint8_t {
        Perspective = 0,
        Equirectangular,
    };

    enum class VideoReconstructionResource : std::uint32_t {
        None = 0,
        Color = 1u << 0u,
        Depth = 1u << 1u,
        MotionVectors = 1u << 2u,
        Jitter = 1u << 3u,
        History = 1u << 4u,
        Exposure = 1u << 5u,
        ReactiveMask = 1u << 6u,
    };

    using VideoReconstructionResourceMask = std::uint32_t;

    [[nodiscard]] constexpr VideoReconstructionResourceMask resourceMask(
        const VideoReconstructionResource resource) noexcept {
        return static_cast<VideoReconstructionResourceMask>(resource);
    }

    struct VideoReconstructionSelection {
        std::string backend_id = "native";
        std::string preset_id = "native";
        VideoReconstructionFallback fallback = VideoReconstructionFallback::Abort;

        bool operator==(const VideoReconstructionSelection&) const = default;
    };

    enum class VideoReconstructionSelectionIssue : std::uint8_t {
        InvalidBackendId = 0,
        InvalidPresetId,
        InvalidFallback,
        InvalidNativePreset,
        SizeLimitExceeded,
        InvalidJson,
        InvalidShape,
        UnsupportedVersion,
        MissingField,
        InvalidVersion,
    };

    struct VideoReconstructionSelectionError {
        VideoReconstructionSelectionIssue issue =
            VideoReconstructionSelectionIssue::InvalidJson;
        std::string message;

        bool operator==(const VideoReconstructionSelectionError&) const = default;
    };

    struct VideoReconstructionPresetDescriptor {
        std::string preset_id;
        float input_scale = 1.0f;
        VideoReconstructionResourceMask additional_resources =
            resourceMask(VideoReconstructionResource::None);

        bool operator==(const VideoReconstructionPresetDescriptor&) const = default;
    };

    struct VideoReconstructionBackendDescriptor {
        std::string backend_id;
        std::string provider_id;
        std::string provider_version;
        std::string provider_digest;
        bool available = false;
        bool supports_offline_export = false;
        bool supports_perspective = false;
        bool supports_equirectangular = false;
        VideoReconstructionResourceMask required_resources =
            resourceMask(VideoReconstructionResource::None);
        std::vector<VideoReconstructionPresetDescriptor> presets;
        std::string unavailable_reason_id;

        bool operator==(const VideoReconstructionBackendDescriptor&) const = default;
    };

    // This interface is intentionally metadata-only. Implementations must not load
    // plugin code, create GPU objects, or probe a device while answering a lookup.
    class VideoReconstructionCatalog {
    public:
        virtual ~VideoReconstructionCatalog() = default;

        [[nodiscard]] virtual std::optional<VideoReconstructionBackendDescriptor>
        findBackend(std::string_view backend_id) const = 0;
    };

    struct VideoReconstructionRequest {
        VideoReconstructionSelection selection{};
        int output_width = 0;
        int output_height = 0;
        VideoReconstructionProjection projection = VideoReconstructionProjection::Perspective;
    };

    enum class VideoReconstructionResolutionIssue : std::uint8_t {
        None = 0,
        InvalidSelection,
        InvalidOutputExtent,
        BackendNotFound,
        BackendUnavailable,
        OfflineExportUnsupported,
        ProjectionUnsupported,
        PresetNotFound,
        InvalidDescriptor,
        InvalidProjection,
    };

    struct VideoReconstructionResolutionError {
        VideoReconstructionResolutionIssue issue = VideoReconstructionResolutionIssue::None;
        std::string message;

        bool operator==(const VideoReconstructionResolutionError&) const = default;
    };

    struct VideoReconstructionProvenance {
        std::string requested_backend_id;
        std::string requested_preset_id;
        std::string requested_provider_id;
        std::string requested_provider_version;
        std::string requested_provider_digest;
        std::string effective_backend_id;
        std::string effective_preset_id;
        std::string effective_provider_id;
        std::string effective_provider_version;
        std::string effective_provider_digest;
        VideoReconstructionResolutionIssue resolution_issue =
            VideoReconstructionResolutionIssue::None;
        std::string unavailable_reason_id;

        bool operator==(const VideoReconstructionProvenance&) const = default;
    };

    class VideoReconstructionPlan {
    public:
        VideoReconstructionPlan(const VideoReconstructionPlan&) = default;
        VideoReconstructionPlan(VideoReconstructionPlan&&) noexcept = default;
        VideoReconstructionPlan& operator=(const VideoReconstructionPlan&) = default;
        VideoReconstructionPlan& operator=(VideoReconstructionPlan&&) noexcept = default;

        [[nodiscard]] int outputWidth() const noexcept { return output_width_; }
        [[nodiscard]] int outputHeight() const noexcept { return output_height_; }
        [[nodiscard]] int inputWidth() const noexcept { return input_width_; }
        [[nodiscard]] int inputHeight() const noexcept { return input_height_; }
        [[nodiscard]] float inputScale() const noexcept { return input_scale_; }
        [[nodiscard]] VideoReconstructionProjection projection() const noexcept { return projection_; }
        [[nodiscard]] VideoReconstructionResourceMask requiredResources() const noexcept {
            return required_resources_;
        }
        [[nodiscard]] VideoReconstructionFallback fallbackPolicy() const noexcept {
            return fallback_policy_;
        }
        [[nodiscard]] bool fellBack() const noexcept {
            return provenance_.resolution_issue != VideoReconstructionResolutionIssue::None;
        }
        [[nodiscard]] bool requiresReconstruction() const noexcept {
            return provenance_.effective_backend_id != "native";
        }
        [[nodiscard]] const VideoReconstructionProvenance& provenance() const noexcept {
            return provenance_;
        }

        bool operator==(const VideoReconstructionPlan&) const = default;

    private:
        VideoReconstructionPlan(
            int output_width,
            int output_height,
            int input_width,
            int input_height,
            float input_scale,
            VideoReconstructionProjection projection,
            VideoReconstructionResourceMask required_resources,
            VideoReconstructionFallback fallback_policy,
            VideoReconstructionProvenance provenance);

        int output_width_ = 0;
        int output_height_ = 0;
        int input_width_ = 0;
        int input_height_ = 0;
        float input_scale_ = 1.0f;
        VideoReconstructionProjection projection_ = VideoReconstructionProjection::Perspective;
        VideoReconstructionResourceMask required_resources_ =
            resourceMask(VideoReconstructionResource::None);
        VideoReconstructionFallback fallback_policy_ = VideoReconstructionFallback::Abort;
        VideoReconstructionProvenance provenance_;

        friend std::expected<VideoReconstructionPlan, VideoReconstructionResolutionError>
        resolveVideoReconstructionPlan(
            const VideoReconstructionRequest& request,
            const VideoReconstructionCatalog& catalog);
    };

    [[nodiscard]] const VideoReconstructionCatalog& nativeVideoReconstructionCatalog();

    [[nodiscard]] std::expected<void, VideoReconstructionSelectionError>
    validateVideoReconstructionSelection(
        const VideoReconstructionSelection& selection);

    [[nodiscard]] std::expected<VideoReconstructionPlan, VideoReconstructionResolutionError>
    resolveVideoReconstructionPlan(
        const VideoReconstructionRequest& request,
        const VideoReconstructionCatalog& catalog = nativeVideoReconstructionCatalog());

    [[nodiscard]] std::expected<std::string, VideoReconstructionSelectionError>
    serializeVideoReconstructionSelection(
        const VideoReconstructionSelection& selection);

    // A missing value is the migration path for projects written before this
    // selection existed and resolves to the compatible native default.
    [[nodiscard]] std::expected<
        VideoReconstructionSelection,
        VideoReconstructionSelectionError>
    deserializeVideoReconstructionSelection(std::optional<std::string_view> serialized);

    [[nodiscard]] std::string_view videoReconstructionFallbackId(
        VideoReconstructionFallback fallback) noexcept;
    [[nodiscard]] std::optional<VideoReconstructionFallback> videoReconstructionFallbackFromId(
        std::string_view id) noexcept;
    [[nodiscard]] std::string_view videoReconstructionResolutionIssueId(
        VideoReconstructionResolutionIssue issue) noexcept;
    [[nodiscard]] std::string_view videoReconstructionSelectionIssueId(
        VideoReconstructionSelectionIssue issue) noexcept;

} // namespace lfs::io::video
