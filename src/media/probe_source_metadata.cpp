// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_probe_ffmpeg.hpp"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
}
#include <optional>
#include <utility>

namespace lfs::media::detail {
    namespace {
        struct Collector {
            ContainerMetadataLimits limits;
            std::size_t records = 0;
            std::size_t bytes = 0;
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
            bool sideData(const AVCodecParameters* parameters, std::vector<MetadataSideData>& output) {
                if (!parameters)
                    return true;
                for (int i = 0; i < parameters->nb_coded_side_data; ++i) {
                    const auto& data = parameters->coded_side_data[i];
                    if (records >= limits.records || data.size > limits.bytes - bytes || (data.size && !data.data))
                        return false;
                    MetadataSideData block;
                    block.type = static_cast<int>(data.type);
                    if (data.size)
                        block.bytes.assign(data.data, data.data + data.size);
                    output.push_back(std::move(block));
                    bytes += data.size;
                    ++records;
                }
                return true;
            }
        };
    } // namespace

    void collectContextMetadata(const AVFormatContext* context, MediaDescription& description, const ContainerMetadataLimits& limits) {
        description.source_metadata.reset();
        if (!context)
            return;
        description.source_metadata.emplace();
        auto& output = *description.source_metadata;
        output.provider = MetadataProviderInfo{"ffmpeg", av_version_info(), avcodec_version()};
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
                !collector.sideData(stream->codecpar, metadata.coded_side_data)) {
                output.limited = true;
                return;
            }
        }
    }
} // namespace lfs::media::detail
