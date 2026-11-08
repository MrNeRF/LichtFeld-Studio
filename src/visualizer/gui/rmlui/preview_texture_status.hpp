/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace lfs::vis::gui {
    // Encoded grayscale mask PNG, resolved on the graphics thread. Queued work must not retain a discarded
    // proposal; only an active decoder acquires ownership of its encoded bytes.
    using PreviewMemoryProvider = std::function<std::weak_ptr<const std::vector<std::byte>>(std::string_view)>;
    enum class PreviewTextureStatus : std::uint8_t {
        Unknown,
        Pending,
        Ready,
        Error
    };
} // namespace lfs::vis::gui
