// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/error.hpp"
#include "media/media_probe.hpp"
#include "media/source_exif.hpp"
#include <cstddef>
#include <memory>
#include <utility>
#include <variant>

namespace lfs::media {
    struct ExifEntryReference {
        std::size_t block = 0;
        std::size_t entry = 0;
    };
    struct XmpBlockReference {
        std::size_t block = 0;
    };
    struct TiffTagReference {
        std::size_t directory = 0;
        std::size_t tag = 0;
    };
    struct ContainerTagReference {
        std::size_t entry = 0;
    };
    struct StreamDescriptionReference {
        std::size_t stream = 0;
    };
    struct StreamTagReference {
        std::size_t stream = 0;
        std::size_t entry = 0;
    };
    struct StreamSideDataReference {
        std::size_t stream = 0;
        std::size_t block = 0;
    };
    // References identify occurrences, not tag IDs: duplicates stay distinct.
    // These are static source observations, never timed video samples.
    using SourceMetadataReference = std::variant<ExifEntryReference, TiffTagReference, ContainerTagReference, StreamDescriptionReference, StreamTagReference, StreamSideDataReference, XmpBlockReference>;
    // Observation identity is deliberately independent of file names and hashes.
    // It prevents mixing catalog-local ordinals, not detection of file changes.
    struct MetadataObservationIdentity {};
    struct MetadataRecordHandle {
        std::shared_ptr<const MetadataObservationIdentity> observation;
        SourceMetadataReference reference;
    };
    using MetadataRecordView = std::variant<const ExifEntry*, const core::image_codecs::TiffSourceTag*,
                                            const MetadataTextEntry*, const StreamDescription*, const MetadataSideData*, const core::image_codecs::XmpBlock*>;
    enum class MetadataIndexStatus { Complete,
                                     Limited };
    struct SourceMetadataIndexLimits {
        // Bounds only the additional reference index. Existing collector/parser
        // budgets govern originals and decoded values independently.
        std::size_t records = 4096;
    };
    enum class MetadataPathEncoding { PosixBytes,
                                      Utf16LE };
    struct MetadataSourceLocator {
        std::string display;
        std::string native_bytes;
        MetadataPathEncoding encoding = MetadataPathEncoding::PosixBytes;
    };
    struct SourceMetadataStorage;
    struct SourceMetadataCatalog {
        SourceMetadataCatalog() = default;
        [[nodiscard]] LFS_MEDIA_API const std::optional<MetadataSourceLocator>& sourceLocator() const noexcept;
        [[nodiscard]] LFS_MEDIA_API const ImageExifDescription& imageExif() const noexcept;
        [[nodiscard]] LFS_MEDIA_API const core::image_codecs::SourceXmp& imageXmp() const noexcept;
        [[nodiscard]] LFS_MEDIA_API const std::optional<MediaDescription>& container() const noexcept;
        [[nodiscard]] LFS_MEDIA_API MetadataIndexStatus indexStatus() const noexcept;
        [[nodiscard]] LFS_MEDIA_API const std::vector<SourceMetadataReference>& records() const noexcept;
        [[nodiscard]] LFS_MEDIA_API std::optional<MetadataRecordHandle> bind(SourceMetadataReference) const;
        [[nodiscard]] LFS_MEDIA_API std::optional<MetadataRecordView> resolve(const MetadataRecordHandle&) const noexcept;

        // Checked lookup; returned pointers borrow immutable snapshot storage.
        // Keep an owning catalog or handle alive. Copies/moves retain index meaning.
        [[nodiscard]] LFS_MEDIA_API const MetadataTextEntry* containerTag(ContainerTagReference) const noexcept;
        [[nodiscard]] LFS_MEDIA_API const StreamDescription* streamDescription(StreamDescriptionReference) const noexcept;
        [[nodiscard]] LFS_MEDIA_API const MetadataTextEntry* streamTag(StreamTagReference) const noexcept;
        [[nodiscard]] LFS_MEDIA_API const MetadataSideData* streamSideData(StreamSideDataReference) const noexcept;
        [[nodiscard]] LFS_MEDIA_API const ExifEntry* exifEntry(ExifEntryReference) const noexcept;
        [[nodiscard]] LFS_MEDIA_API const core::image_codecs::TiffSourceTag* tiffTag(TiffTagReference) const noexcept;
        [[nodiscard]] LFS_MEDIA_API const core::image_codecs::XmpBlock* xmpBlock(XmpBlockReference) const noexcept;

    private:
        // Payload and identity form one immutable shared observation.
        explicit SourceMetadataCatalog(std::shared_ptr<const SourceMetadataStorage> storage) : storage_(std::move(storage)) {}
        std::shared_ptr<const SourceMetadataStorage> storage_;
        friend LFS_MEDIA_API Result<SourceMetadataCatalog> catalogSourceMetadata(ImageExifDescription, const SourceMetadataIndexLimits&, std::optional<MetadataSourceLocator>, core::image_codecs::SourceXmp);
        friend LFS_MEDIA_API Result<SourceMetadataCatalog> catalogContainerMetadata(MediaDescription, const SourceMetadataIndexLimits&, std::optional<MetadataSourceLocator>);
    };
    // Consumes an already collected description without rereading/reparsing it.
    // Pass an rvalue to transfer ownership without copying the source payloads.
    // A limited index still retains all already collected originals and values.
    // Transfer ownership without retaining mutable aliases to moved source vectors.
    [[nodiscard]] LFS_MEDIA_API Result<SourceMetadataCatalog> catalogSourceMetadata(
        ImageExifDescription, const SourceMetadataIndexLimits& = {}, std::optional<MetadataSourceLocator> = std::nullopt, core::image_codecs::SourceXmp = {});
    // Requires a probe description explicitly collected with source metadata.
    [[nodiscard]] LFS_MEDIA_API Result<SourceMetadataCatalog> catalogContainerMetadata(
        MediaDescription, const SourceMetadataIndexLimits& = {}, std::optional<MetadataSourceLocator> = std::nullopt);
} // namespace lfs::media
