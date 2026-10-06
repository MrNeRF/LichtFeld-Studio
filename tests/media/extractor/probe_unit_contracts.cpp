// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "io/media_probe_ffmpeg.hpp"
#include "io/video_player.hpp"
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
    return 0;
}
