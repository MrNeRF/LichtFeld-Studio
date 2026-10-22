// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/source_exif.hpp"

namespace lfs::media::detail {
    // Consume an already collected snapshot without another source read.
    ImageExifDescription describeSourceExif(core::image_codecs::SourceExif);
} // namespace lfs::media::detail
