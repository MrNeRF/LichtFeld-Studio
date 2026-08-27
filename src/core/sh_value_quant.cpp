/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/sh_value_quant.hpp"

#include <atomic>

namespace lfs::core::sh_value_quant {
    namespace {
        std::atomic<int>& override_flag() {
            static std::atomic<int> g{-1};
            return g;
        }
    } // namespace

    void set_enabled_for_testing(const std::optional<bool> enabled) {
        if (!enabled.has_value()) {
            override_flag().store(-1, std::memory_order_relaxed);
            return;
        }
        override_flag().store(*enabled ? 1 : 0, std::memory_order_relaxed);
    }

    bool enabled() {
        const int o = override_flag().load(std::memory_order_relaxed);
        if (o >= 0)
            return o != 0;
        return true; // production default: always ON
    }
} // namespace lfs::core::sh_value_quant
