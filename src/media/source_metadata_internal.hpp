// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/source_metadata.hpp"

namespace lfs::media::detail {
    inline constexpr char source_metadata_preservation_policy[] = "preserve_static_container_v1";
    [[nodiscard]] MetadataSourceLocator sourceMetadataLocator(const std::filesystem::path&);
} // namespace lfs::media::detail
