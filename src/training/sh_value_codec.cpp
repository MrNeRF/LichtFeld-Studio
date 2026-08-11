/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/sh_value_codec.hpp"

#include "core/sh_value_quant.hpp"

namespace lfs::training::sh_value {

    void set_sh_value_quant_enabled_for_testing(const std::optional<bool> enabled) {
        core::sh_value_quant::set_enabled_for_testing(enabled);
    }

    bool sh_value_quant_enabled() {
        return core::sh_value_quant::enabled();
    }

} // namespace lfs::training::sh_value
