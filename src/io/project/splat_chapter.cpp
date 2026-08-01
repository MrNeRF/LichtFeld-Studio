/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/splat_chapter.hpp"

#include <cstring>
#include <exception>
#include <format>
#include <sstream>
#include <utility>

namespace lfs::io::project {

    namespace {

        constexpr std::uint32_t LFSP_MAGIC = 0x4c465350;

        lfs::Error splat_error(const lfs::ErrorCode code, std::string message,
                               std::string detail) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = {},
                .native = std::nullopt,
            });
        }

    } // namespace

    lfs::Result<SplatChapterPayload> SplatChapterPayload::from_lfsp(
        const std::span<const std::byte> bytes) {
        if (bytes.size() < 8) {
            return splat_error(
                lfs::ErrorCode::DataLoss, "The embedded splat payload is truncated.",
                std::format("SPLT payload has {} bytes; LFSP header needs 8", bytes.size()));
        }
        std::uint32_t magic = 0;
        std::uint32_t version = 0;
        std::memcpy(&magic, bytes.data(), sizeof(magic));
        std::memcpy(&version, bytes.data() + sizeof(magic), sizeof(version));
        if (magic != LFSP_MAGIC) {
            return splat_error(
                lfs::ErrorCode::DataLoss, "The embedded splat payload has invalid magic.",
                std::format("SPLT payload magic is 0x{:08x}, expected LFSP", magic));
        }
        if (version != 3 && version != 4) {
            return splat_error(
                lfs::ErrorCode::Unsupported,
                "This embedded splat payload version is not supported.",
                std::format("LFSP version {} is unsupported", version));
        }
        SplatChapterPayload result;
        result.bytes_.assign(bytes.begin(), bytes.end());
        result.lfsp_version_ = version;
        return result;
    }

    lfs::Result<SplatChapterPayload> SplatChapterPayload::capture(
        const lfs::core::SplatData& model, const SplatSourceKind source_kind,
        const bool is_training_model) {
        if (is_training_model) {
            return splat_error(
                lfs::ErrorCode::FailedPrecondition,
                "The training model cannot be written as a SPLT chapter.",
                "Training model state is authoritative only in CKPT");
        }
        if (must_reference_external(source_kind)) {
            return splat_error(
                lfs::ErrorCode::FailedPrecondition,
                "A live RAD node cannot be embedded in the project.",
                "Live RAD nodes remain external REFS records until explicitly baked");
        }
        try {
            std::ostringstream stream(std::ios::binary);
            model.serialize(stream);
            if (!stream) {
                return splat_error(
                    lfs::ErrorCode::Unavailable,
                    "The splat payload could not be serialized.",
                    "LFSP serialization stream failed");
            }
            const std::string serialized = std::move(stream).str();
            return from_lfsp(std::as_bytes(
                std::span(serialized.data(), serialized.size())));
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): LFSP's legacy stream API throws;
            // normalize it into the chapter Result surface.
            return splat_error(
                lfs::ErrorCode::DataLoss,
                "The splat payload could not be serialized.",
                std::format("LFSP serialization failed: {}", error.what()));
        }
    }

    lfs::Result<std::unique_ptr<lfs::core::SplatData>>
    SplatChapterPayload::hydrate(
        lfs::core::SplatTensorAllocator tensor_allocator) const {
        try {
            auto result = std::make_unique<lfs::core::SplatData>();
            const std::string_view view(
                reinterpret_cast<const char*>(bytes_.data()), bytes_.size());
            std::istringstream stream(std::string(view), std::ios::binary);
            result->deserialize(stream, std::move(tensor_allocator));
            if (!stream || stream.peek() != std::char_traits<char>::eof()) {
                return splat_error(
                    lfs::ErrorCode::DataLoss,
                    "The embedded splat payload has trailing or unread bytes.",
                    "LFSP deserializer did not consume the bounded SPLT payload exactly");
            }
            return result;
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): LFSP's legacy stream API throws;
            // normalize it into the chapter Result surface.
            return splat_error(
                lfs::ErrorCode::DataLoss,
                "The embedded splat payload could not be restored.",
                std::format("LFSP deserialization failed: {}", error.what()));
        }
    }

    bool SplatChapterPayload::must_reference_external(
        const SplatSourceKind source_kind) noexcept {
        return source_kind == SplatSourceKind::LiveRad;
    }

} // namespace lfs::io::project
