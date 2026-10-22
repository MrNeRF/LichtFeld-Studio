// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_json.hpp"
#include "media/media_probe_ffmpeg.hpp"
#include "media/source_metadata.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
}
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

int runMetadataObservationContracts();

namespace {
    using namespace lfs::media;
    using namespace lfs::core::image_codecs;
    void require(bool condition, const char* text) {
        if (!condition)
            throw std::runtime_error(text);
    }
    ImageExifDescription description() {
        ImageExifDescription result;
        result.original.status = MetadataStatus::Limited;
        result.original.scope = "test observation";
        result.original.diagnostic = "retained partial source";
        result.original.blocks.resize(2);
        result.original.blocks[0].bytes = {0, 255, 0, 42};
        result.interpretations.resize(2);
        result.interpretations[0].status = ExifParseStatus::Parsed;
        result.interpretations[1].status = ExifParseStatus::Unsupported;
        result.interpretations[1].diagnostic = "opaque source block";
        ExifEntry entry;
        entry.id = 0xc7fe;
        entry.type = 5;
        entry.rational_values.push_back({42, 0});
        result.interpretations[0].entries.push_back(entry);
        entry.rational_values[0] = {43, -2};
        result.interpretations[0].entries.push_back(entry);
        result.original.tiff.emplace();
        TiffSourceDirectory directory;
        directory.offset = 1234;
        directory.kind = "root";
        TiffSourceTag tag;
        tag.id = 0xc7fe;
        tag.type = 2;
        tag.bytes = {0, 255, 65};
        directory.tags.push_back(tag);
        directory.tags.push_back(tag);
        result.original.tiff->directories.push_back(directory);
        return result;
    }
} // namespace

namespace {
    void containerContracts() {
        using namespace lfs::media;
        const auto release = [](AVFormatContext* value) { avformat_free_context(value); };
        std::unique_ptr<AVFormatContext, decltype(release)> context(avformat_alloc_context(), release);
        require(bool(context), "allocate static metadata context");
        require(av_dict_set(&context->metadata, "x", "y", 0) >= 0, "set container metadata");
        auto* stream = avformat_new_stream(context.get(), nullptr);
        require(stream != nullptr, "allocate metadata stream");
        stream->index = 7;
        stream->time_base = {1, 90000};
        stream->codecpar->codec_type = AVMEDIA_TYPE_DATA;
        require(av_dict_set(&stream->metadata, "k", "v", 0) >= 0, "set stream metadata");
        auto* side = av_packet_side_data_new(&stream->codecpar->coded_side_data,
                                             &stream->codecpar->nb_coded_side_data,
                                             AV_PKT_DATA_STRINGS_METADATA, 3, 0);
        require(side != nullptr, "allocate metadata side data");
        side->data[0] = 0;
        side->data[1] = 255;
        side->data[2] = 42;
        auto observed = detail::describeContext(context.get());
        require(!observed.source_metadata && !catalogContainerMetadata(observed),
                "legacy descriptions do not implicitly collect raw source metadata");
        detail::collectContextMetadata(context.get(), observed, {.streams = 1, .records = 3, .bytes = 7});
        require(observed.source_metadata && !observed.source_metadata->limited &&
                    observed.source_metadata->streams[0].stream_index == 7,
                "exact collection budget retains native stream identity");
        MediaDescription limited_side;
        detail::collectContextMetadata(context.get(), limited_side, {.streams = 1, .records = 3, .bytes = 6});
        require(limited_side.source_metadata->limited && limited_side.source_metadata->streams[0].tags.size() == 1 &&
                    limited_side.source_metadata->streams[0].coded_side_data.empty(),
                "side-data byte limit never copies a partial payload");
        detail::collectContextMetadata(context.get(), limited_side, {.streams = 1, .records = 2, .bytes = 7});
        require(limited_side.source_metadata->limited && limited_side.source_metadata->streams[0].coded_side_data.empty(),
                "container tags, stream tags and side data share one record budget");
        require(observed.source_metadata->provider && observed.source_metadata->provider->name == "ffmpeg" &&
                    !observed.source_metadata->provider->version.empty() && observed.source_metadata->provider->codec_api_version != 0,
                "side-data namespace is identified by its producing provider version");
        auto catalog = catalogContainerMetadata(std::move(observed));
        require(bool(catalog) && catalog->records().size() == 4, "container tags, stream properties, stream tags and side data indexed");
        require(catalog->containerTag({0})->value == "y" && catalog->streamTag({0, 0})->key == "k" &&
                    catalog->streamDescription({0})->index == 7 && catalog->streamDescription({0})->time_base->denominator == 90000,
                "static metadata uses existing stream properties and timebase without fabricated samples");
        context.reset();
        require(catalog->streamSideData({0, 0})->bytes == std::vector<uint8_t>({0, 255, 42}),
                "side data ownership survives destruction of the demuxer");
        const auto json = json_detail::sourceMetadata(*catalog);
        require(json.at("scope") == "container_stream" && !json.contains("image_exif") &&
                    json.at("source_metadata")["streams"][0]["coded_side_data"][0]["bytes"] == "00ff2a",
                "container serializer preserves opaque codec side data");
        const auto huge = std::numeric_limits<std::size_t>::max();
        require(!catalog->containerTag({huge}) && !catalog->streamTag({huge, 0}) &&
                    !catalog->streamTag({0, huge}) && !catalog->streamSideData({0, huge}) &&
                    !catalog->streamSideData({huge, 0}) && !catalog->streamDescription({huge}),
                "all container reference kinds reject invalid indices");
        auto copy = *catalog;
        require(copy.containerTag({0}) == catalog->containerTag({0}), "copies share one immutable payload without duplicating storage");
        auto moved = std::move(copy);
        require(moved.containerTag({0})->value == "y", "moved container references resolve their new owner");
        auto limited = catalogContainerMetadata(*catalog->container(), {.records = 3});
        require(bool(limited) && limited->indexStatus() == MetadataIndexStatus::Limited &&
                    limited->container()->source_metadata->streams[0].coded_side_data.size() == 1,
                "index limit preserves unindexed collected payloads");
        context.reset(avformat_alloc_context());
        require(bool(context), "allocate limit context");
        require(av_dict_set(&context->metadata, "x", "y", 0) >= 0, "set limit fixture tag");
        MediaDescription bounded;
        detail::collectContextMetadata(context.get(), bounded, {.streams = 0, .records = 0, .bytes = 0});
        require(bounded.source_metadata->limited && bounded.source_metadata->tags.empty(), "zero record budget stops before retaining tags");
        detail::collectContextMetadata(context.get(), bounded, {.streams = 0, .records = 1, .bytes = 1});
        require(bounded.source_metadata->limited && bounded.source_metadata->tags.empty(), "key and value share the same byte budget");
        detail::collectContextMetadata(context.get(), bounded, {.streams = 0, .records = 1, .bytes = 2});
        require(!bounded.source_metadata->limited && bounded.source_metadata->tags.size() == 1, "exact text byte boundary accepted");
        require(avformat_new_stream(context.get(), nullptr) != nullptr, "add stream for stream budget test");
        detail::collectContextMetadata(context.get(), bounded, {.streams = 0, .records = 1, .bytes = 2});
        require(bounded.source_metadata->limited && bounded.source_metadata->streams.empty(), "zero stream budget reports omitted stream metadata");
        const auto partial = catalogContainerMetadata(bounded);
        require(bool(partial) && partial->indexStatus() == MetadataIndexStatus::Complete && partial->container()->source_metadata->limited,
                "index completeness does not erase collection truncation");
        detail::collectContextMetadata(nullptr, bounded);
        require(!bounded.source_metadata, "null context clears stale metadata");
        const auto binary_text = json_detail::metadataText({"x", std::string("a\0\xff", 3)});
        require(binary_text.at("value_bytes") == "6100ff", "text transport keeps NUL and invalid UTF8 bytes in hex");
    }
} // namespace

int runSourceMetadataContracts() {
    containerContracts();
    SourceXmp xmp;
    xmp.status = MetadataStatus::Collected;
    XmpBlock packet;
    packet.bytes = {0, 255, 42};
    packet.payload_offset = 1;
    xmp.blocks.push_back(packet);
    xmp.blocks.push_back(packet);
    const auto* xmp_bytes = xmp.blocks.front().bytes.data();
    auto with_xmp = catalogSourceMetadata({}, {}, std::nullopt, std::move(xmp));
    require(bool(with_xmp) && with_xmp->records().size() == 2 && with_xmp->xmpBlock({0})->bytes.data() == xmp_bytes,
            "XMP occurrences transfer payload ownership without copying");
    const auto handle = with_xmp->bind(XmpBlockReference{1});
    require(handle && with_xmp->resolve(*handle) && !with_xmp->xmpBlock({2}) &&
                !with_xmp->xmpBlock({std::numeric_limits<std::size_t>::max()}),
            "XMP snapshot references check occurrence bounds");
    auto other_xmp = catalogSourceMetadata({});
    require(!other_xmp->resolve(*handle), "XMP handles cannot cross source observations");
    const auto xmp_json = json_detail::sourceMetadata(*with_xmp);
    require(xmp_json.at("image_xmp").at("blocks")[0].at("bytes") == "00ff2a" &&
                xmp_json.at("image_xmp").at("blocks")[0].at("payload_bytes") == "ff2a" &&
                xmp_json.at("records")[1].at("block") == 1,
            "XMP serialization preserves binary originals and duplicate ordinals");
    auto limited_xmp = catalogSourceMetadata({}, {.records = 0}, std::nullopt, with_xmp->imageXmp());
    require(limited_xmp && limited_xmp->indexStatus() == MetadataIndexStatus::Limited &&
                limited_xmp->records().empty() && limited_xmp->imageXmp().blocks.size() == 2 &&
                limited_xmp->imageExifIndexStatus() == MetadataIndexStatus::Complete,
            "XMP retention is independent of the reference index budget");
    auto kept_xmp = *with_xmp;
    with_xmp = SourceMetadataCatalog{};
    require(kept_xmp.resolve(*handle).has_value(), "XMP handles survive catalog reassignment");
    auto tiff_description = description();
    auto& tiff_tag = tiff_description.original.tiff->directories[0].tags[0];
    tiff_tag.id = 700;
    tiff_tag.type = 1;
    tiff_tag.count = tiff_tag.bytes.size();
    const auto* tiff_bytes = tiff_tag.bytes.data();
    SourceXmp tiff_xmp;
    tiff_xmp.status = MetadataStatus::Collected;
    XmpBlock tiff_block;
    tiff_block.envelope = XmpEnvelope::TiffTag;
    tiff_block.tiff_tag = TiffXmpReference{0, 0};
    tiff_xmp.blocks.push_back(tiff_block);
    auto tiff_catalog = catalogSourceMetadata(std::move(tiff_description), {}, {}, tiff_xmp);
    require(bool(tiff_catalog) && tiff_catalog->imageXmp().blocks[0].bytes.empty() &&
                tiff_catalog->xmpBytes({0})->data() == tiff_bytes,
            "TIFF XMP references the existing tag bytes without another payload allocation");
    const auto tiff_handle = tiff_catalog->bind(XmpBlockReference{0});
    auto tiff_copy = *tiff_catalog;
    tiff_catalog = SourceMetadataCatalog{};
    auto tiff_moved = std::move(tiff_copy);
    require(tiff_handle && tiff_moved.resolve(*tiff_handle) && tiff_moved.xmpBytes({0})->data() == tiff_bytes,
            "TIFF XMP reference and borrowed bytes survive catalog copy/move and original destruction");
    require(!tiff_moved.xmpBytes({1}) && !tiff_moved.xmpBytes({std::numeric_limits<size_t>::max()}),
            "XMP byte lookup rejects invalid ordinals");
    auto invalid_tiff = description();
    require(!catalogSourceMetadata(invalid_tiff, {}, {}, tiff_xmp), "TIFF XMP cannot reference a different tag ID");
    tiff_xmp.blocks[0].tiff_tag->directory = std::numeric_limits<size_t>::max();
    require(!catalogSourceMetadata(invalid_tiff, {}, {}, tiff_xmp), "TIFF XMP rejects invalid directory ordinals");
    auto original = description();
    const auto* payload = original.original.blocks[0].bytes.data();
    auto catalog = catalogSourceMetadata(std::move(original));
    require(bool(catalog), "existing EXIF description is accepted");
    require(catalog->imageExif().original.blocks[0].bytes.data() == payload, "original bytes move without a second payload allocation");
    require(catalog->records().size() == 4 && catalog->indexStatus() == MetadataIndexStatus::Complete,
            "complete index covers all retained EXIF entries and TIFF tags");
    require(catalog->imageExif().original.status == MetadataStatus::Limited &&
                catalog->imageExif().interpretations[1].status == ExifParseStatus::Unsupported,
            "index completeness never upgrades collection or interpretation status");
    const auto first = std::get<ExifEntryReference>(catalog->records()[0]);
    const auto second = std::get<ExifEntryReference>(catalog->records()[1]);
    require(catalog->exifEntry(first)->rational_values[0].denominator == 0 &&
                catalog->exifEntry(second)->rational_values[0].numerator == 43,
            "duplicate tag occurrences and invalid rationals remain distinct and unnormalized");
    const auto tiff = std::get<TiffTagReference>(catalog->records()[2]);
    require(catalog->tiffTag(tiff)->bytes == std::vector<uint8_t>({0, 255, 65}), "opaque TIFF bytes are unchanged");
    const auto serialized = json_detail::sourceMetadata(*catalog);
    require(serialized.at("scope") == "static_image_exif" && serialized.at("index_status") == "complete" &&
                serialized.at("image_exif") == json_detail::sourceExif(catalog->imageExif()),
            "common transport reuses the original EXIF serializer without changing its contract");
    require(serialized.at("records")[1].at("entry") == 1 &&
                serialized.at("records")[2].at("kind") == "tiff_tag" &&
                serialized.at("records")[3].at("tag") == 1,
            "serialized references preserve duplicate occurrences in source order");
    auto copy = *catalog;
    require(copy.exifEntry(first) == catalog->exifEntry(first) && copy.exifEntry(first)->id == 0xc7fe,
            "copied catalog shares immutable source storage");
    const auto owning_handle = copy.bind(first);
    require(bool(owning_handle), "bind an owning immutable snapshot handle");
    const auto* borrowed_entry = copy.exifEntry(first);
    auto moved = std::move(copy);
    require(moved.tiffTag(tiff) && moved.exifEntry(second), "moved catalog retains valid references");
    copy = SourceMetadataCatalog{};
    require(borrowed_entry->id == 0xc7fe && moved.resolve(*owning_handle).has_value(), "owning handles retain immutable storage across catalog assignment");
    const auto huge = std::numeric_limits<std::size_t>::max();
    require(!moved.exifEntry({huge, 0}) && !moved.exifEntry({0, huge}) &&
                !moved.tiffTag({huge, 0}) && !moved.tiffTag({0, huge}),
            "foreign and invalid references never access out of bounds");
    SourceMetadataCatalog empty;
    require(!empty.exifEntry({}) && !empty.tiffTag({}), "empty catalog lookups are safe");
    for (std::size_t count = 0; count <= 5; ++count) {
        auto limited = catalogSourceMetadata(description(), {.records = count});
        require(bool(limited) && limited->records().size() == std::min(count, std::size_t{4}), "reference limit enforced at exact boundary");
        require((limited->indexStatus() == MetadataIndexStatus::Limited) == (count < 4), "limit reported only when an entry is omitted");
        require(limited->imageExifIndexStatus() == limited->indexStatus(), "omitting an EXIF or TIFF reference marks the image sub-index limited");
        require(limited->imageExif().interpretations[0].entries.size() == 2 &&
                    limited->imageExif().original.tiff->directories[0].tags.size() == 2,
                "limited index never truncates already collected values");
    }
    auto xmp_tail = catalogSourceMetadata(description(), {.records = 4}, std::nullopt, kept_xmp.imageXmp());
    require(xmp_tail && xmp_tail->indexStatus() == MetadataIndexStatus::Limited &&
                xmp_tail->imageExifIndexStatus() == MetadataIndexStatus::Complete,
            "XMP-only index omission preserves complete EXIF/TIFF sub-index status");
    auto copied_index = *xmp_tail;
    auto moved_index = std::move(copied_index);
    require(moved_index.imageExifIndexStatus() == MetadataIndexStatus::Complete &&
                SourceMetadataCatalog{}.imageExifIndexStatus() == MetadataIndexStatus::Complete,
            "sub-index status follows immutable ownership and an empty catalog is complete");
    auto no_data = catalogSourceMetadata({}, {.records = 0});
    require(bool(no_data) && no_data->indexStatus() == MetadataIndexStatus::Complete,
            "empty unexamined source is not falsely reported limited or collected");
    auto mismatch = description();
    mismatch.interpretations.pop_back();
    require(!catalogSourceMetadata(std::move(mismatch)), "missing interpretation ordinal is rejected");
    mismatch = description();
    mismatch.interpretations.emplace_back();
    require(!catalogSourceMetadata(std::move(mismatch)), "extra interpretation ordinal is rejected");
    return runMetadataObservationContracts();
}
