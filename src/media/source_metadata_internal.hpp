// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/source_metadata.hpp"

namespace lfs::media::detail {
    [[nodiscard]] MetadataSourceLocator sourceMetadataLocator(const std::filesystem::path&);
}
