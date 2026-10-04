/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <expected>
#include <string>

namespace lfs::vis {
    struct ModifierError {
        std::string message;
    };
    using ModifierResult = std::expected<void, ModifierError>;
} // namespace lfs::vis
