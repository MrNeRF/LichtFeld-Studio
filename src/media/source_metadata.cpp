// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/source_metadata.hpp"
#include "core/path_utils.hpp"
#include "media/media_ingest.hpp"
#include "media/source_exif_internal.hpp"
#include "source_metadata_internal.hpp"
#include <algorithm>
#include <new>
#include <type_traits>
#include <utility>

namespace lfs::media {
    struct SourceMetadataStorage : MetadataObservationIdentity {
        std::optional<MetadataSourceLocator> locator;
        ImageExifDescription image_exif;
        core::image_codecs::SourceXmp image_xmp;
        std::optional<MediaDescription> container;
        MetadataIndexStatus index_status = MetadataIndexStatus::Complete;
        MetadataIndexStatus image_exif_index_status = MetadataIndexStatus::Complete;
        std::vector<SourceMetadataReference> records;
    };
    const std::optional<MetadataSourceLocator>& SourceMetadataCatalog::sourceLocator() const noexcept {
        static const std::optional<MetadataSourceLocator> empty;
        return storage_ ? storage_->locator : empty;
    }
    const ImageExifDescription& SourceMetadataCatalog::imageExif() const noexcept {
        static const ImageExifDescription empty;
        return storage_ ? storage_->image_exif : empty;
    }
    const core::image_codecs::SourceXmp& SourceMetadataCatalog::imageXmp() const noexcept {
        static const core::image_codecs::SourceXmp empty;
        return storage_ ? storage_->image_xmp : empty;
    }
    const core::image_codecs::XmpBlock* SourceMetadataCatalog::xmpBlock(XmpBlockReference reference) const noexcept {
        return storage_ && reference.block < storage_->image_xmp.blocks.size() ? &storage_->image_xmp.blocks[reference.block] : nullptr;
    }
    std::optional<std::span<const std::uint8_t>> SourceMetadataCatalog::xmpBytes(XmpBlockReference reference) const noexcept {
        const auto* block = xmpBlock(reference);
        return block ? core::image_codecs::source_xmp_bytes(imageExif().original, *block) : std::nullopt;
    }
    std::optional<std::span<const std::uint8_t>> SourceMetadataCatalog::xmpPayloadBytes(XmpBlockReference reference) const noexcept {
        const auto* block = xmpBlock(reference);
        const auto original = xmpBytes(reference);
        if (!block || block->status != core::image_codecs::MetadataStatus::Collected || !original || block->payload_offset > original->size())
            return std::nullopt;
        return block->inflated ? std::span<const std::uint8_t>(*block->inflated) : original->subspan(block->payload_offset);
    }
    const std::optional<MediaDescription>& SourceMetadataCatalog::container() const noexcept {
        static const std::optional<MediaDescription> empty;
        return storage_ ? storage_->container : empty;
    }
    MetadataIndexStatus SourceMetadataCatalog::indexStatus() const noexcept {
        return storage_ ? storage_->index_status : MetadataIndexStatus::Complete;
    }
    MetadataIndexStatus SourceMetadataCatalog::imageExifIndexStatus() const noexcept {
        return storage_ ? storage_->image_exif_index_status : MetadataIndexStatus::Complete;
    }
    const std::vector<SourceMetadataReference>& SourceMetadataCatalog::records() const noexcept {
        static const std::vector<SourceMetadataReference> empty;
        return storage_ ? storage_->records : empty;
    }
    namespace {
        bool validLocator(const std::optional<MetadataSourceLocator>& locator) noexcept {
            if (!locator)
                return true;
            const auto& bytes = locator->native_bytes;
            if (bytes.empty())
                return false;
            switch (locator->encoding) {
            case MetadataPathEncoding::PosixBytes:
                return bytes.find('\0') == std::string::npos;
            case MetadataPathEncoding::Utf16LE:
                if (bytes.size() % 2)
                    return false;
                for (std::size_t i = 0; i < bytes.size(); i += 2)
                    if (!bytes[i] && !bytes[i + 1])
                        return false;
                return true;
            }
            return false;
        }
    } // namespace

    MetadataSourceLocator detail::sourceMetadataLocator(const std::filesystem::path& path) {
        MetadataSourceLocator result;
        result.display = core::path_to_utf8(path);
#ifdef _WIN32
        result.encoding = MetadataPathEncoding::Utf16LE;
        for (const auto unit : path.native()) {
            const auto value = static_cast<std::uint16_t>(unit);
            result.native_bytes.push_back(static_cast<char>(value & 255));
            result.native_bytes.push_back(static_cast<char>(value >> 8));
        }
#else
        result.encoding = MetadataPathEncoding::PosixBytes;
        result.native_bytes = path.native();
#endif
        return result;
    }

    std::optional<MetadataRecordHandle> SourceMetadataCatalog::bind(SourceMetadataReference reference) const {
        if (!storage_)
            return std::nullopt;
        MetadataRecordHandle handle{storage_, std::move(reference)};
        return resolve(handle) ? std::optional(std::move(handle)) : std::nullopt;
    }
    std::optional<MetadataRecordView> SourceMetadataCatalog::resolve(const MetadataRecordHandle& handle) const noexcept {
        if (!storage_ || handle.observation.get() != static_cast<const MetadataObservationIdentity*>(storage_.get()))
            return std::nullopt;
        return std::visit([&](const auto& reference) -> std::optional<MetadataRecordView> {
            using T = std::decay_t<decltype(reference)>;
            if constexpr (std::is_same_v<T, ExifEntryReference>) {
                if (const auto* value = exifEntry(reference))
                    return MetadataRecordView(value);
            } else if constexpr (std::is_same_v<T, TiffTagReference>) {
                if (const auto* value = tiffTag(reference))
                    return MetadataRecordView(value);
            } else if constexpr (std::is_same_v<T, ContainerTagReference>) {
                if (const auto* value = containerTag(reference))
                    return MetadataRecordView(value);
            } else if constexpr (std::is_same_v<T, StreamDescriptionReference>) {
                if (const auto* value = streamDescription(reference))
                    return MetadataRecordView(value);
            } else if constexpr (std::is_same_v<T, StreamTagReference>) {
                if (const auto* value = streamTag(reference))
                    return MetadataRecordView(value);
            } else if constexpr (std::is_same_v<T, StreamSideDataReference>) {
                if (const auto* value = streamSideData(reference))
                    return MetadataRecordView(value);
            } else if constexpr (std::is_same_v<T, XmpBlockReference>) {
                if (const auto* value = xmpBlock(reference))
                    return MetadataRecordView(value);
            }
            return std::nullopt;
        },
                          handle.reference);
    }

    const ExifEntry* SourceMetadataCatalog::exifEntry(ExifEntryReference reference) const noexcept {
        if (reference.block >= imageExif().original.blocks.size() || reference.block >= imageExif().interpretations.size())
            return nullptr;
        const auto& entries = imageExif().interpretations[reference.block].entries;
        return reference.entry < entries.size() ? &entries[reference.entry] : nullptr;
    }

    const core::image_codecs::TiffSourceTag* SourceMetadataCatalog::tiffTag(TiffTagReference reference) const noexcept {
        if (!imageExif().original.tiff || reference.directory >= imageExif().original.tiff->directories.size())
            return nullptr;
        const auto& tags = imageExif().original.tiff->directories[reference.directory].tags;
        return reference.tag < tags.size() ? &tags[reference.tag] : nullptr;
    }

    const MetadataTextEntry* SourceMetadataCatalog::containerTag(ContainerTagReference reference) const noexcept {
        if (!container() || !container()->source_metadata || reference.entry >= container()->source_metadata->tags.size())
            return nullptr;
        return &container()->source_metadata->tags[reference.entry];
    }
    const StreamDescription* SourceMetadataCatalog::streamDescription(StreamDescriptionReference reference) const noexcept {
        return container() && reference.stream < container()->streams.size() ? &container()->streams[reference.stream] : nullptr;
    }
    const MetadataTextEntry* SourceMetadataCatalog::streamTag(StreamTagReference reference) const noexcept {
        if (!container() || !container()->source_metadata || reference.stream >= container()->source_metadata->streams.size())
            return nullptr;
        const auto& tags = container()->source_metadata->streams[reference.stream].tags;
        return reference.entry < tags.size() ? &tags[reference.entry] : nullptr;
    }
    const MetadataSideData* SourceMetadataCatalog::streamSideData(StreamSideDataReference reference) const noexcept {
        if (!container() || !container()->source_metadata || reference.stream >= container()->source_metadata->streams.size())
            return nullptr;
        const auto& blocks = container()->source_metadata->streams[reference.stream].coded_side_data;
        return reference.block < blocks.size() ? &blocks[reference.block] : nullptr;
    }

    Result<SourceMetadataCatalog> catalogContainerMetadata(MediaDescription description, const SourceMetadataIndexLimits& limits, std::optional<MetadataSourceLocator> locator) {
        if (!validLocator(locator))
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Metadata source locator has an invalid native encoding or filename", .detection = LFS_SOURCE_SITE_CURRENT()});
        if (!description.source_metadata)
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Container catalog requires explicitly collected source metadata", .detection = LFS_SOURCE_SITE_CURRENT()});
        try {
            const auto& source_streams = description.source_metadata->streams;
            if (source_streams.size() > description.streams.size())
                return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Source metadata stream nodes exceed the probe stream inventory", .detection = LFS_SOURCE_SITE_CURRENT()});
            if (!description.source_metadata->limited && source_streams.size() != description.streams.size())
                return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Incomplete source metadata stream inventory requires a limited status", .detection = LFS_SOURCE_SITE_CURRENT()});
            std::vector<int> stream_ids;
            stream_ids.reserve(description.streams.size());
            for (std::size_t i = 0; i < description.streams.size(); ++i) {
                const auto id = description.streams[i].index;
                if (id < 0 || (i < source_streams.size() && source_streams[i].stream_index != id))
                    return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Source metadata stream identity does not match its probe description", .detection = LFS_SOURCE_SITE_CURRENT()});
                stream_ids.push_back(id);
            }
            std::sort(stream_ids.begin(), stream_ids.end());
            if (std::adjacent_find(stream_ids.begin(), stream_ids.end()) != stream_ids.end())
                return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Probe stream identities must be unique within a metadata observation", .detection = LFS_SOURCE_SITE_CURRENT()});
            auto staged = std::make_shared<SourceMetadataStorage>();
            auto& result = *staged;
            result.locator = std::move(locator);
            result.container = std::move(description);
            const auto append = [&](SourceMetadataReference reference) {
                if (result.records.size() >= limits.records) {
                    result.index_status = MetadataIndexStatus::Limited;
                    return false;
                }
                result.records.push_back(reference);
                return true;
            };
            const auto& metadata = *result.container->source_metadata;
            for (std::size_t i = 0; i < metadata.tags.size(); ++i)
                if (!append(ContainerTagReference{i}))
                    return SourceMetadataCatalog(std::move(staged));
            for (std::size_t i = 0; i < result.container->streams.size(); ++i)
                if (!append(StreamDescriptionReference{i}))
                    return SourceMetadataCatalog(std::move(staged));
            for (std::size_t stream = 0; stream < metadata.streams.size(); ++stream) {
                for (std::size_t entry = 0; entry < metadata.streams[stream].tags.size(); ++entry)
                    if (!append(StreamTagReference{stream, entry}))
                        return SourceMetadataCatalog(std::move(staged));
                for (std::size_t block = 0; block < metadata.streams[stream].coded_side_data.size(); ++block)
                    if (!append(StreamSideDataReference{stream, block}))
                        return SourceMetadataCatalog(std::move(staged));
            }
            return SourceMetadataCatalog(std::move(staged));
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Container metadata index allocation failed", .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }

    Result<SourceMetadataCatalog> catalogSourceMetadata(ImageExifDescription description, const SourceMetadataIndexLimits& limits, std::optional<MetadataSourceLocator> locator, core::image_codecs::SourceXmp xmp) {
        if (!validLocator(locator))
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Metadata source locator has an invalid native encoding or filename", .detection = LFS_SOURCE_SITE_CURRENT()});
        // A caller-built description must not silently associate an interpretation
        // with a missing or different block. Empty/failed interpretations still
        // occupy their original ordinal, as in MediaIngest::sourceExif.
        if (description.original.blocks.size() != description.interpretations.size())
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "EXIF interpretation count must match the original block count", .detection = LFS_SOURCE_SITE_CURRENT()});
        for (const auto& block : description.original.blocks) {
            if ((block.envelope == core::image_codecs::ExifEnvelope::JxlOffset) != block.jxl_box.has_value() ||
                (block.jxl_box && (block.image_item || block.metadata_item)))
                return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "EXIF box provenance does not match its carrier", .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        for (const auto& block : xmp.blocks)
            if (!core::image_codecs::source_xmp_bytes(description.original, block))
                return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "XMP byte reference does not match its source observation", .detection = LFS_SOURCE_SITE_CURRENT()});
        try {
            auto staged = std::make_shared<SourceMetadataStorage>();
            auto& result = *staged;
            result.locator = std::move(locator);
            result.image_exif = std::move(description);
            result.image_xmp = std::move(xmp);
            const auto append = [&](SourceMetadataReference reference) {
                if (result.records.size() >= limits.records) {
                    result.index_status = MetadataIndexStatus::Limited;
                    return false;
                }
                result.records.push_back(reference);
                return true;
            };
            for (std::size_t block = 0; block < result.image_exif.interpretations.size(); ++block) {
                const auto& interpretation = result.image_exif.interpretations[block];
                for (std::size_t entry = 0; entry < interpretation.entries.size(); ++entry) {
                    if (!append(ExifEntryReference{block, entry})) {
                        result.image_exif_index_status = MetadataIndexStatus::Limited;
                        return SourceMetadataCatalog(std::move(staged));
                    }
                }
            }
            if (result.image_exif.original.tiff) {
                const auto& directories = result.image_exif.original.tiff->directories;
                for (std::size_t directory = 0; directory < directories.size(); ++directory) {
                    for (std::size_t tag = 0; tag < directories[directory].tags.size(); ++tag) {
                        if (!append(TiffTagReference{directory, tag})) {
                            result.image_exif_index_status = MetadataIndexStatus::Limited;
                            return SourceMetadataCatalog(std::move(staged));
                        }
                    }
                }
            }
            for (std::size_t block = 0; block < result.image_xmp.blocks.size(); ++block)
                if (!append(XmpBlockReference{block}))
                    return SourceMetadataCatalog(std::move(staged));
            return SourceMetadataCatalog(std::move(staged));
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Source metadata index allocation failed", .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }

    Result<SourceMetadataCatalog> MediaIngest::sourceMetadata(const std::filesystem::path& path, const SourceMetadataIndexLimits& limits, const core::image_codecs::XmpLimits& xmp_limits) {
        if (path.empty() || path.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos)
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Metadata source path must not be empty or contain a null byte", .detection = LFS_SOURCE_SITE_CURRENT()});
        try {
            auto collected = core::image_codecs::read_source_metadata(path, {}, xmp_limits);
            auto description = detail::describeSourceExif(std::move(collected.exif));
            return catalogSourceMetadata(std::move(description), limits, detail::sourceMetadataLocator(path), std::move(collected.xmp));
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Source metadata collection allocation failed", .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
    Result<SourceMetadataCatalog> MediaIngest::sourceContainerMetadata(const std::filesystem::path& path, const ProbeOptions& options, const SourceMetadataIndexLimits& limits) {
        try {
            auto requested = options;
            requested.collect_source_metadata = true;
            auto description = MediaProbe::inspect(path, requested);
            if (!description)
                return description.error();
            return catalogContainerMetadata(std::move(*description), limits, detail::sourceMetadataLocator(path));
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Container metadata collection allocation failed", .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
} // namespace lfs::media
