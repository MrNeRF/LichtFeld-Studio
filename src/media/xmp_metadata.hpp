// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/metadata_observation.hpp"
#include <array>
#include <optional>
#include <span>
namespace lfs::media {
    // Adobe Extended XMP identifier. Unavailable under providers without MD5;
    // preserves the caller's OpenSSL error queue and propagates allocation failure.
    std::optional<std::array<char, 32>> xmpPacketDigest(std::span<const std::uint8_t>);
    // Internal projection of already-collected packets, sharing summary budgets.
    void appendXmpMetadata(const SourceMetadataCatalog&, MetadataSummary&, std::size_t max_facts,
                           std::size_t max_text_bytes, std::size_t& charged);
} // namespace lfs::media
