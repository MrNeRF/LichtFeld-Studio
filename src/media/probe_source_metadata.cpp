// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_probe_ffmpeg.hpp"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
}
#include <memory>
#include <optional>
#include <utility>

namespace lfs::media::detail {
    namespace {
        enum class CopyStatus { Complete,
                                Limited,
                                Invalid };
        struct Collector {
            ContainerMetadataLimits limits;
            std::size_t records = 0;
            std::size_t bytes = 0;
            std::size_t next_bytes = 0;
            std::optional<std::size_t> length(const char* text, std::size_t available) const {
                if (!text)
                    return std::nullopt;
                for (std::size_t i = 0;; ++i) {
                    if (!text[i])
                        return i;
                    if (i == available)
                        return std::nullopt;
                }
            }
            bool dictionary(const AVDictionary* dictionary, std::vector<MetadataTextEntry>& output) {
                const AVDictionaryEntry* entry = nullptr;
                while ((entry = av_dict_get(dictionary, "", entry, AV_DICT_IGNORE_SUFFIX))) {
                    if (records >= limits.records)
                        return false;
                    const auto key = length(entry->key, limits.bytes - bytes);
                    if (!key)
                        return false;
                    const auto value = length(entry->value, limits.bytes - bytes - *key);
                    if (!value)
                        return false;
                    output.push_back({std::string(entry->key, *key), std::string(entry->value, *value)});
                    bytes += *key + *value;
                    ++records;
                }
                return true;
            }
            CopyStatus retain(const std::uint8_t* data, std::size_t size, std::vector<std::uint8_t>& output, bool record = true) {
                next_bytes = size;
                if (size && !data)
                    return CopyStatus::Invalid;
                if ((record && records >= limits.records) || size > limits.bytes - bytes)
                    return CopyStatus::Limited;
                if (size)
                    output.assign(data, data + size);
                bytes += size;
                if (record)
                    ++records;
                return CopyStatus::Complete;
            }
            CopyStatus sideData(const AVPacketSideData* data, int count, std::vector<MetadataSideData>& output) {
                if (count < 0 || (count && !data))
                    return CopyStatus::Invalid;
                for (int i = 0; i < count; ++i) {
                    MetadataSideData block;
                    block.type = static_cast<int>(data[i].type);
                    const auto copied = retain(data[i].data, data[i].size, block.bytes);
                    if (copied != CopyStatus::Complete)
                        return copied;
                    output.push_back(std::move(block));
                }
                return CopyStatus::Complete;
            }
        };
    } // namespace

    MetadataProviderInfo metadataProviderInfo() {
        return {"ffmpeg", av_version_info(), avcodec_version()};
    }
    void collectContextMetadata(const AVFormatContext* context, MediaDescription& description, const ContainerMetadataLimits& limits) {
        description.source_metadata.reset();
        if (!context)
            return;
        description.source_metadata.emplace();
        auto& output = *description.source_metadata;
        output.provider = metadataProviderInfo();
        Collector collector{limits};
        if (!collector.dictionary(context->metadata, output.tags)) {
            output.limited = true;
            return;
        }
        for (unsigned int i = 0; i < context->nb_streams; ++i) {
            if (output.streams.size() >= limits.streams) {
                output.limited = true;
                return;
            }
            const auto* stream = context->streams[i];
            if (!stream) {
                output.limited = true;
                return;
            }
            output.streams.push_back({stream->index, {}, {}});
            auto& metadata = output.streams.back();
            if (!collector.dictionary(stream->metadata, metadata.tags) ||
                (stream->codecpar && collector.sideData(stream->codecpar->coded_side_data, stream->codecpar->nb_coded_side_data, metadata.coded_side_data) != CopyStatus::Complete)) {
                output.limited = true;
                return;
            }
        }
    }
    Result<TimedMetadataObservation> collectTimedMetadata(AVFormatContext* context, int index, const TimedMetadataLimits& limits) {
        if (static_cast<unsigned>(index) >= context->nb_streams || !context->streams[index] || !context->streams[index]->codecpar)
            return probeInvalidOptions("Timed metadata stream index is outside the input stream table (index=" + std::to_string(index) + ", nb_streams=" + std::to_string(context->nb_streams) + ")");
        auto* stream = context->streams[index];
        const auto* parameters = stream->codecpar;
        if (parameters->codec_type != AVMEDIA_TYPE_DATA && parameters->codec_type != AVMEDIA_TYPE_SUBTITLE)
            return probeInvalidOptions("Timed metadata collection requires a data or subtitle stream (codec_type=" + std::to_string(parameters->codec_type) + ")");
        if (stream->time_base.num <= 0 || stream->time_base.den <= 0)
            return make_error({.code = ErrorCode::DataLoss, .domain = ErrorDomain::IO, .detail = "Timed metadata stream has no valid time base (time_base=" + std::to_string(stream->time_base.num) + "/" + std::to_string(stream->time_base.den) + ")", .detection = LFS_SOURCE_SITE_CURRENT()});
        keepOnlyStream(context, index);
        const auto interrupted = [&] {
            return context->interrupt_callback.callback && context->interrupt_callback.callback(context->interrupt_callback.opaque);
        };
        Collector collector{{.records = limits.records, .bytes = limits.bytes}};
        const auto exhausted = [&] {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Timed metadata budget exceeded (retained_bytes=" + std::to_string(collector.bytes) + "/" + std::to_string(limits.bytes) + ", next_bytes=" + std::to_string(collector.next_bytes) + ", records=" + std::to_string(collector.records) + "/" + std::to_string(limits.records) + ")", .detection = LFS_SOURCE_SITE_CURRENT()});
        };
        TimedMetadataObservation observation;
        observation.provider = metadataProviderInfo();
        observation.stream_index = index;
        observation.codec_tag = parameters->codec_tag;
        const auto extra = parameters->extradata_size < 0 ? CopyStatus::Invalid : collector.retain(parameters->extradata, static_cast<std::size_t>(parameters->extradata_size), observation.extradata, false);
        if (extra == CopyStatus::Invalid)
            return make_error({.code = ErrorCode::DataLoss, .domain = ErrorDomain::IO, .detail = "Timed metadata codec configuration is invalid (extradata_size=" + std::to_string(parameters->extradata_size) + ")", .detection = LFS_SOURCE_SITE_CURRENT()});
        if (extra == CopyStatus::Limited)
            return exhausted();
        auto release = [](AVPacket* packet) { av_packet_free(&packet); };
        std::unique_ptr<AVPacket, decltype(release)> packet(av_packet_alloc(), release);
        if (!packet)
            return probeFailure(AVERROR(ENOMEM), "Allocate metadata packet", false);
        while (true) {
            if (interrupted())
                return probeFailure(AVERROR_EXIT, "Read timed metadata", true);
            const auto read = av_read_frame(context, packet.get());
            if (read == AVERROR_EOF) {
                if (context->pb && context->pb->error < 0 && context->pb->error != AVERROR_EOF)
                    return probeFailure(context->pb->error, "Read timed metadata", interrupted());
                // MOV may report EOF for a missing selected sample. Use its
                // parsed index, including fragments, without reparsing boxes or
                // equating nb_frames with the number of returned packets.
                const auto input_size = context->pb ? avio_size(context->pb) : AVERROR(EIO);
                if (input_size < 0)
                    return probeFailure(static_cast<int>(input_size), "Determine timed metadata input size", interrupted());
                const auto entries = avformat_index_get_entries_count(stream);
                for (int i = 0; i < entries; ++i) {
                    if (interrupted())
                        return probeFailure(AVERROR_EXIT, "Validate timed metadata index", true);
                    const auto* entry = avformat_index_get_entry(stream, i);
                    if (!entry || entry->pos < 0 || entry->size < 0 || entry->pos > input_size || entry->size > input_size - entry->pos)
                        return make_error({.code = ErrorCode::DataLoss, .domain = ErrorDomain::IO, .detail = "Timed metadata sample exceeds input bounds (stream=" + std::to_string(index) + ", index=" + std::to_string(i) + ", position=" + std::to_string(entry ? entry->pos : -1) + ", bytes=" + std::to_string(entry ? entry->size : -1) + ", input_bytes=" + std::to_string(input_size) + ")", .detection = LFS_SOURCE_SITE_CURRENT()});
                }
                if (interrupted())
                    return probeFailure(AVERROR_EXIT, "Read timed metadata", true);
                break;
            }
            if (read < 0)
                return probeFailure(read, "Read timed metadata", interrupted());
            if (packet->stream_index != index) {
                av_packet_unref(packet.get());
                continue;
            }
            const auto packetError = [&] {
                const auto operation = "Read timed metadata packet (size=" + std::to_string(packet->size) + ", flags=" + std::to_string(packet->flags) + ", side_data_count=" + std::to_string(packet->side_data_elems) + ", duration=" + std::to_string(packet->duration) + ")";
                return probeFailure(AVERROR_INVALIDDATA, operation.c_str(), false);
            };
            if (packet->duration < 0 || packet->size < 0 || (packet->flags & AV_PKT_FLAG_CORRUPT))
                return packetError();
            TimedMetadataPacket sample;
            sample.pts = probeTimestamp(packet->pts, stream->time_base);
            sample.dts = probeTimestamp(packet->dts, stream->time_base);
            if (packet->duration > 0)
                sample.duration = probeTimestamp(packet->duration, stream->time_base);
            if (packet->pos >= 0)
                sample.source_offset = packet->pos;
            sample.flags = packet->flags;
            const auto payload = collector.retain(packet->data, static_cast<std::size_t>(packet->size), sample.bytes);
            if (payload == CopyStatus::Invalid)
                return packetError();
            if (payload == CopyStatus::Limited)
                return exhausted();
            const auto side = collector.sideData(packet->side_data, packet->side_data_elems, sample.side_data);
            if (side == CopyStatus::Invalid)
                return packetError();
            if (side == CopyStatus::Limited)
                return exhausted();
            observation.packets.push_back(std::move(sample));
            av_packet_unref(packet.get());
        }
        return observation;
    }
} // namespace lfs::media::detail
