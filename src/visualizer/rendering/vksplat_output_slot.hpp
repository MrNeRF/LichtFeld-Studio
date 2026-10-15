/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace lfs::vis {

    // Shared by enum validation and output storage. Preview is not the last
    // slot: the VR eyes must pass the same checks as desktop outputs.
    enum class VksplatOutputSlot : std::size_t {
        Main = 0,
        SplitLeft = 1,
        SplitRight = 2,
        Preview = 3,
        VrLeft = 4,
        VrRight = 5,
        Count,
    };

    inline constexpr std::size_t kVksplatOutputSlotCount =
        static_cast<std::size_t>(VksplatOutputSlot::Count);

    [[nodiscard]] inline std::size_t vksplatOutputSlotIndex(const VksplatOutputSlot slot) {
        const auto index = static_cast<std::size_t>(slot);
        if (index >= kVksplatOutputSlotCount) [[unlikely]] {
            throw std::out_of_range(
                "VkSplat output-slot enum is outside the output arrays (observed_enum_value=" +
                std::to_string(index) + ", output_slot_count=" +
                std::to_string(kVksplatOutputSlotCount) + ")");
        }
        return index;
    }

} // namespace lfs::vis
