// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/export.hpp"
#include "media/file_frame_sink.hpp"
#include "media/source_exif.hpp"
#include "media/source_metadata.hpp"
#include <functional>
#include <string_view>

namespace lfs::media {
    enum class SelectionMode { FPS,
                               Interval };
    enum class ResizeMode { Original,
                            Scale,
                            Custom };
    enum class SharpnessMethod { Laplacian,
                                 Tenengrad,
                                 Combined };
    struct Selection {
        SelectionMode mode = SelectionMode::FPS;
        double fps = 1;
        int interval = 1;
    };
    struct Geometry {
        ResizeMode mode = ResizeMode::Original;
        float scale = 1;
        int width = 0;
        int height = 0;
        int clockwise_rotation = 0;
    };
    struct Sharpness {
        bool enabled = false;
        SharpnessMethod method = SharpnessMethod::Combined;
        double threshold = 0;
        int window_candidates = 10;
        bool window = false;
    };
    struct IngestProgress {
        int processed = 0;
        int estimated = 0;
        int discarded = 0;
    };
    struct IngestRequest {
        std::filesystem::path input;
        Selection selection;
        Geometry geometry;
        Sharpness sharpness;
        double start_seconds = 0;
        double end_seconds = -1;
        bool convert_hdr_to_sdr = false;
        bool allow_hardware_decode = true;
        std::function<void(const IngestProgress&)> progress;
        std::function<bool()> cancelled;
        FramePixelFormat output_format = FramePixelFormat::RGB8;
        // Unspecified means use source metadata. Overrides are explicit assumptions.
        FrameColor input_color;
        FloatColorProfile float_color_profile = FloatColorProfile::Sdr;
    };
    struct FileExtraction {
        FileFrameSinkOptions files;
        bool write_metadata = false;
        // Opt-in verified image-sequence replay. Existing output without a
        // compatible checkpoint is never adopted or overwritten.
        bool resume = false;
        // Opt-in static demuxer observation in the shared extraction manifest.
        // Includes retained tags/opaque side data, potentially GPS/identifiers.
        bool preserve_source_metadata = false;
        // Opt-in XMP beside each image (full image filename + .xmp). The
        // checkpoint is used only with resume; a pair is not atomic.
        bool write_source_metadata_sidecars = false;
    };
    struct IngestReport {
        std::size_t frames_accepted = 0;
        int discarded = 0;
    };
    struct IngestCapabilities {
        bool software_decode = true;
        bool rgb8 = true;
        bool png = true;
        bool jpeg = true;
        bool hardware_decode = false;
        bool hdr_to_sdr = false;
        bool rgb_float_sdr = true;
        bool exr = true;
        bool rgb_float_pq = true;
        std::string_view float_pq_profile = "Opt-in PQ to linear cd/m2; declared BT709/BT2020 primaries preserved; encoded RGB clamped to [0,1]; no tone mapping, mastering rescale, HLG, LOG or Dolby Vision";
        bool rgb_float_hlg = true;
        std::string_view float_hlg_profile = "Opt-in HLG inverse OETF to relative scene-linear [0,1]; declared BT709/BT2020 primaries preserved; encoded RGB clamped to [0,1]; no display OOTF, absolute nits, tone mapping, mastering rescale, LOG or Dolby Vision";
        bool tiff16 = true;
        // Stable capability description shared by CLI, Python and MCP.
        std::string_view float_sdr_profile = "Integer RGB or planar/semiplanar YUV 8-16-bit, including subsampled chroma; Linear/sRGB/BT709 transfer; BT709/BT2020 primaries; explicit metadata or overrides; shared CPU reference and optional tensor backend; no HDR or alpha video";
        bool jxl = true;
        bool avif = false;
    };
    struct CodecBuildInfo {
        std::string ffmpeg_version;
        std::string ffmpeg_license;
        std::string ffmpeg_configuration;
    };
    // Synchronous API; optional Studio backends are registered by the host. Failed
    // results retain accepted count as an error field; sinks retain partial data.
    class LFS_MEDIA_API MediaIngest {
    public:
        [[nodiscard]] static IngestCapabilities capabilities() noexcept;
        [[nodiscard]] static CodecBuildInfo codecBuildInfo();
        [[nodiscard]] static Result<MediaDescription> probe(const std::filesystem::path&, const ProbeOptions& = {});
        [[nodiscard]] static Result<ImageExifDescription> sourceExif(const std::filesystem::path&);
        // Explicit metadata-only collection; never called implicitly by training or probe.
        [[nodiscard]] static Result<SourceMetadataCatalog> sourceMetadata(
            const std::filesystem::path&, const SourceMetadataIndexLimits& = {}, const core::image_codecs::XmpLimits& = {});
        [[nodiscard]] static Result<SourceMetadataCatalog> sourceContainerMetadata(
            const std::filesystem::path&, const ProbeOptions& = {}, const SourceMetadataIndexLimits& = {});
        [[nodiscard]] static Result<IngestReport> extract(const IngestRequest&, FrameSink&);
        [[nodiscard]] static Result<IngestReport> extractFiles(const IngestRequest&, const FileExtraction&);
    };
} // namespace lfs::media
