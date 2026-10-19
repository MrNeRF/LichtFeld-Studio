/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "diagnostics/export.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lfs::diagnostics {

    enum class VramOwner : std::uint8_t {
        Model,
        Optimizer,
        Rasterizer,
        LossStep,
        Densification,
        ImageIo,
        Viewer,
        Slack,
        Context,
        Unattributed,
        Count,
    };

    constexpr std::size_t kVramOwnerCount = static_cast<std::size_t>(VramOwner::Count);

    struct VramOwnerRow {
        std::string scope;
        std::string label;
        VramOwner owner = VramOwner::Unattributed;
        std::size_t bytes = 0;
        std::size_t peak_bytes = 0;
    };

    struct LFS_DIAGNOSTICS_API VramOwnerBreakdown {
        std::array<std::int64_t, kVramOwnerCount> bytes{};
        std::vector<VramOwnerRow> rows;
        std::size_t process_bytes = 0;
        bool process_valid = false;
        bool over_attributed = false;
        std::int64_t signed_residual_bytes = 0;
        std::int64_t context_inferred_bytes = 0;
        std::array<std::int64_t, 5> unattributed_roots{};

        [[nodiscard]] std::int64_t sum() const noexcept;
    };

    [[nodiscard]] LFS_DIAGNOSTICS_API VramOwnerBreakdown
    buildVramOwnerBreakdown(const VramProfilerSnapshot& snapshot,
                            bool arena_external_backing = false);

    [[nodiscard]] LFS_DIAGNOSTICS_API std::array<std::int64_t, kVramOwnerCount>
    displayVramOwnerBytes(const VramOwnerBreakdown& breakdown) noexcept;

    [[nodiscard]] LFS_DIAGNOSTICS_API const char* vramOwnerName(VramOwner owner) noexcept;

} // namespace lfs::diagnostics
