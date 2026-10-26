// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/metadata_observation.hpp"
namespace lfs::media {
    // Internal projection of already-collected packets, sharing summary budgets.
    void appendXmpMetadata(const SourceMetadataCatalog&, MetadataSummary&, std::size_t max_facts,
                           std::size_t max_text_bytes, std::size_t& charged);
} // namespace lfs::media
