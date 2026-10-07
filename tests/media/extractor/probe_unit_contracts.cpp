// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/decoded_video_frame_ffmpeg.hpp"
#include "media/media_probe_ffmpeg.hpp"
#include "media/video_player.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/display.h>
}
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace {
    void require(bool condition, std::string_view message) {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }
} // namespace

int runProbeUnitContracts() {
    using namespace lfs::media;
    require(MediaProbe::inspect({}).error().code() == lfs::ErrorCode::InvalidArgument, "empty path rejected");
    const auto optionFailure = MediaProbe::inspect({});
    require(!optionFailure && !optionFailure.error().native(), "option failure has no native FFmpeg status");
    const auto copied = optionFailure;
    require(copied.error().domain() == lfs::ErrorDomain::IO &&
                !copied.error().frames().empty() && !copied.error().detail().empty(),
            "structured error owns diagnostics and detection site after copy");
    auto nulPath = std::filesystem::path("bad").native();
    nulPath.push_back(0);
    nulPath += std::filesystem::path("suffix").native();
    require(MediaProbe::inspect(std::filesystem::path(nulPath)).error().code() == lfs::ErrorCode::InvalidArgument, "embedded NUL rejected before path conversion");
    ProbeOptions invalidOptions;
    invalidOptions.depth = static_cast<ProbeDepth>(-1);
    require(MediaProbe::inspect("unused", invalidOptions).error().code() == lfs::ErrorCode::InvalidArgument, "invalid depth rejected");
    auto release = [](AVFormatContext* context) { avformat_free_context(context); };
    std::unique_ptr<AVFormatContext, decltype(release)> context(avformat_alloc_context(), release);
    require(context != nullptr, "allocate test context");
    auto* audio = avformat_new_stream(context.get(), nullptr);
    auto* video = avformat_new_stream(context.get(), nullptr);
    auto* second = avformat_new_stream(context.get(), nullptr);
    require(audio && video && second, "allocate test streams");
    audio->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    audio->codecpar->codec_id = AV_CODEC_ID_PCM_S16LE;
    for (auto* stream : {video, second}) {
        stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
        stream->codecpar->codec_id = AV_CODEC_ID_FFV1;
        stream->codecpar->width = 64;
        stream->codecpar->height = 32;
        stream->time_base = {1, 1000};
        stream->start_time = AV_NOPTS_VALUE;
        stream->duration = AV_NOPTS_VALUE;
        stream->avg_frame_rate = {0, 0};
        stream->r_frame_rate = {0, 1};
    }
    context->duration = AV_NOPTS_VALUE;
    auto description = detail::describeContext(context.get());
    require(description.streams.size() == 3 && description.selected_video_stream == 1, "all streams and first usable video");
    require(!description.duration && !description.streams[1].duration && !description.streams[1].start, "unknown timestamps stay unknown");
    require(!description.streams[1].average_frame_rate && !description.streams[1].nominal_frame_rate, "no invented FPS");
    require(!description.streams[1].color.component_depth && !description.streams[1].color.transfer, "no invented pixel depth or transfer");
    require(!description.streams[1].orientation.clockwise_degrees, "absent rotation is unknown");
    require(audio->discard == AVDISCARD_DEFAULT && second->discard == AVDISCARD_DEFAULT, "description has no discard side effects");
    video->start_time = -40;
    video->duration = 0;
    auto stream = detail::describeStream(video);
    require(stream.start && stream.start->ticks == -40 && stream.start->time_base.denominator == 1000, "signed raw start preserved");
    require(stream.duration && stream.duration->ticks == 0, "zero duration is known");
    video->time_base = {1, 0};
    require(!detail::describeStream(video).start, "invalid timebase is unknown");
    video->time_base = {1, 1000};

    int32_t matrix[9];
    av_display_rotation_set(matrix, 90);
    auto* side = av_packet_side_data_new(&video->codecpar->coded_side_data,
                                         &video->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX, sizeof(matrix), 0);
    require(side && side->data, "allocate display matrix");
    std::memcpy(side->data, matrix, sizeof(matrix));
    stream = detail::describeStream(video);
    require(stream.orientation.display_matrix && legacyQuarterTurn(stream.orientation) == 90, "clockwise display matrix rotation");
    require(stream.orientation.reflected == false, "ordinary rotation is not reflected");
    av_display_matrix_flip(matrix, 1, 0);
    auto reflected = detail::describeDisplayMatrix(reinterpret_cast<unsigned char*>(matrix), sizeof(matrix));
    require(reflected.reflected == true && reflected.display_matrix, "mirror and raw matrix preserved");
    auto truncated = detail::describeDisplayMatrix(reinterpret_cast<unsigned char*>(matrix), sizeof(matrix) - 1);
    require(!truncated.display_matrix && !truncated.clockwise_degrees, "truncated matrix never read");
    std::memset(matrix, 0, sizeof(matrix));
    auto degenerate = detail::describeDisplayMatrix(reinterpret_cast<unsigned char*>(matrix), sizeof(matrix));
    require(!degenerate.clockwise_degrees && !degenerate.reflected, "degenerate matrix produces no angle");
    av_dict_set(&video->metadata, "rotate", "-90", 0);
    stream = detail::describeStream(video);
    require(stream.orientation.source == OrientationSource::RotateTag && stream.orientation.rotate_tag == "-90" &&
                stream.orientation.display_matrix && legacyQuarterTurn(stream.orientation) == 270,
            "tag precedence preserves matrix and negative angle");
    for (const std::string invalid : {std::string("90junk"), std::string("nan"), std::string("inf"), std::string(400, '9')}) {
        av_dict_set(&video->metadata, "rotate", invalid.c_str(), 0);
        require(!detail::describeStream(video).orientation.clockwise_degrees, "invalid tag is unknown");
    }
    av_dict_set(&video->metadata, "rotate", " +90.9 ", 0);
    stream = detail::describeStream(video);
    require(stream.orientation.clockwise_degrees == 90.9 && legacyQuarterTurn(stream.orientation) == 90,
            "valid signed/spaced fractional tags preserve legacy truncation");
    av_dict_set(&video->metadata, "rotate", "45", 0);
    stream = detail::describeStream(video);
    require(stream.orientation.clockwise_degrees == 45 && legacyQuarterTurn(stream.orientation) == 0, "raw non-quarter angle retained without unsupported preview rotation");
    detail::discardNonVideoStreams(context.get(), 1);
    require(audio->discard == AVDISCARD_ALL && second->discard == AVDISCARD_ALL && video->discard == AVDISCARD_DEFAULT, "legacy discard policy");
    require(detail::findUsableHeaderVideoStream(nullptr) == -1, "null context");
    require(detail::describeContext(nullptr).streams.empty(), "null description");
    require(!detail::describeDecodedVideoFrame(nullptr), "missing decoded frame rejected");
    auto releaseFrame = [](AVFrame* f) { av_frame_free(&f); };
    std::unique_ptr<AVFrame, decltype(releaseFrame)> decoded(av_frame_alloc(), releaseFrame);
    require(decoded != nullptr, "allocate descriptor frame");
    decoded->format = AV_PIX_FMT_YUV420P10LE;
    decoded->width = 65;
    decoded->height = 33;
    decoded->color_primaries = AVCOL_PRI_BT2020;
    decoded->color_trc = AVCOL_TRC_SMPTE2084;
    decoded->colorspace = AVCOL_SPC_BT2020_NCL;
    decoded->color_range = AVCOL_RANGE_MPEG;
    require(av_frame_get_buffer(decoded.get(), 64) >= 0, "allocate padded descriptor planes");
    auto* master = av_mastering_display_metadata_create_side_data(decoded.get());
    require(master != nullptr, "allocate mastering metadata");
    master->has_luminance = 1;
    master->min_luminance = {1, 1000};
    master->max_luminance = {1000, 1};
    auto* plus = av_dynamic_hdr_plus_create_side_data(decoded.get());
    require(plus != nullptr, "allocate HDR10+ metadata");
    plus->num_windows = 1;
    plus->params[0].maxscl[0] = {1, 10};
    plus->params[0].maxscl[1] = {1, 5};
    plus->params[0].maxscl[2] = {3, 10};
    plus->params[0].average_maxrgb = {1, 100};
    auto view = detail::describeDecodedVideoFrame(decoded.get(), video);
    require(view.has_value(), "describe padded HDR frame");
    require(view->plane_count == 3 && view->component_count == 3 && view->components[0].depth == 10, "component description");
    require(view->planes[1].width == 33 && view->planes[1].height == 17 && view->planes[0].pitch == decoded->linesize[0], "odd chroma extent and byte pitches preserved");
    require(view->planes[0].data == decoded->data[0] && view->color_trc == ColorTransfer::Pq && view->color_primaries == ColorPrimaries::Bt2020, "borrowed storage and independent colour enums");
    require(view->frame_hdr.mastering && view->frame_hdr.mastering->max_luma == 1000.f && view->frame_hdr.mastering->min_luma == .001f, "mastering luminance precision");
    require(view->frame_hdr.hdr10_plus && view->frame_hdr.hdr10_plus->maxscl == std::array{.1, .2, .3} && view->frame_hdr.hdr10_plus->average_maxrgb == .01, "HDR10+ normalized values");
    auto invalid_view = *view;
    invalid_view.planes[1].pitch = 1;
    auto invalid_storage = validateDecodedVideoFrame(invalid_view);
    require(!invalid_storage && invalid_storage.error().detail().find("plane=1") != std::string_view::npos && invalid_storage.error().detail().find("pitch=1") != std::string_view::npos && invalid_storage.error().detail().find("row_bytes=66") != std::string_view::npos, "stride diagnostic includes plane, observed pitch and row bytes");
    invalid_view = *view;
    invalid_view.components[1].plane = 4;
    require(!validateDecodedVideoFrame(invalid_view), "invalid component indices rejected before indexing");
    invalid_view = *view;
    invalid_view.frame_hdr.hdr10_plus->num_anchors = 16;
    require(!validateDecodedVideoFrame(invalid_view), "HDR10+ count cannot exceed metadata storage");
    invalid_view = *view;
    invalid_view.planes[0].data += (invalid_view.planes[0].height - 1) * invalid_view.planes[0].pitch;
    invalid_view.planes[0].pitch *= -1;
    require(validateDecodedVideoFrame(invalid_view).has_value(), "negative software pitch remains representable");
    const auto invalid_format = decoded->format;
    decoded->format = -100;
    require(!detail::describeDecodedVideoFrame(decoded.get()), "invalid pixel format rejected");
    decoded->format = invalid_format;
    // Metadata-only hardware fixtures: no device or decoder is initialized.
    std::unique_ptr<AVFrame, decltype(releaseFrame)> hardware(av_frame_alloc(), releaseFrame);
    require(hardware != nullptr, "allocate hardware descriptor fixture");
    hardware->format = AV_PIX_FMT_VIDEOTOOLBOX;
    hardware->width = 64;
    hardware->height = 32;
    hardware->hw_frames_ctx = av_buffer_allocz(sizeof(AVHWFramesContext));
    require(hardware->hw_frames_ctx != nullptr, "allocate hardware metadata fixture");
    auto* hardware_context = reinterpret_cast<AVHWFramesContext*>(hardware->hw_frames_ctx->data);
    hardware_context->sw_format = AV_PIX_FMT_NV12;
    std::uint8_t first_slot = 0, pixel_buffer = 0;
    hardware->data[0] = &first_slot;
    hardware->data[3] = &pixel_buffer;
    auto hw_view = detail::describeDecodedVideoFrame(hardware.get());
    require(hw_view && hw_view->hardware && hw_view->hardware_handle == &pixel_buffer,
            "VideoToolbox identity comes from data[3], even when data[0] is present");
    require(hw_view->plane_count == 2 && hw_view->format_name == "nv12",
            "hardware view retains logical software layout");
    require(hw_view->planes[0].data == nullptr && hw_view->planes[1].data == nullptr &&
                hw_view->planes[0].pitch == 0 && hw_view->planes[1].pitch == 0,
            "hardware handles never masquerade as CPU plane storage");
    hardware->data[3] = nullptr;
    require(!detail::describeDecodedVideoFrame(hardware.get()),
            "VideoToolbox without pixel buffer is rejected instead of falling back to data[0]");
    hardware->format = AV_PIX_FMT_CUDA;
    hw_view = detail::describeDecodedVideoFrame(hardware.get());
    require(hw_view && hw_view->hardware_handle == &first_slot,
            "CUDA identity retains its data[0] location");
    hardware->data[0] = nullptr;
    require(!detail::describeDecodedVideoFrame(hardware.get()), "missing CUDA handle rejected");
    return 0;
}
