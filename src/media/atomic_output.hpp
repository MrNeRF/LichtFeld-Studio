// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/image_output.hpp"
#include <string_view>
namespace lfs::media::detail {
    [[nodiscard]] SinkResult writeAtomicText(const std::filesystem::path&, std::string_view, const ExrOutputOptions&);
}
