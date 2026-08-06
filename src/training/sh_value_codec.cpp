/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/sh_value_codec.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace lfs::training::sh_value {
    namespace {
        // -1 = use env, 0 = force fp32, 1 = force quant
        std::atomic<int> g_override{-1};

        [[nodiscard]] bool env_quant_enabled() {
            // Default OFF until FastGS decode/encode + densify/export dequant
            // paths are fully wired (see perf_campaign/ISSUES.md ISS-2.1).
            // Opt-in: LFS_SH_VALUE_QUANT=1. Force fp32: LFS_SH_VALUE_FP32=1.
            const char* force_fp32 = std::getenv("LFS_SH_VALUE_FP32");
            if (force_fp32 != nullptr && force_fp32[0] != '\0' &&
                !(force_fp32[0] == '0' && force_fp32[1] == '\0') &&
                std::strcmp(force_fp32, "false") != 0 &&
                std::strcmp(force_fp32, "off") != 0) {
                return false;
            }
            const char* opt_in = std::getenv("LFS_SH_VALUE_QUANT");
            if (opt_in == nullptr || opt_in[0] == '\0')
                return false; // default OFF (infrastructure landed; wiring pending)
            if (opt_in[0] == '0' && opt_in[1] == '\0')
                return false;
            if (std::strcmp(opt_in, "false") == 0 || std::strcmp(opt_in, "off") == 0)
                return false;
            return true; // any other value → quant ON
        }
    } // namespace

    void set_sh_value_quant_enabled_for_testing(const std::optional<bool> enabled) {
        if (!enabled.has_value()) {
            g_override.store(-1, std::memory_order_relaxed);
            return;
        }
        g_override.store(*enabled ? 1 : 0, std::memory_order_relaxed);
    }

    bool sh_value_quant_enabled() {
        const int o = g_override.load(std::memory_order_relaxed);
        if (o >= 0)
            return o != 0;
        return env_quant_enabled();
    }

} // namespace lfs::training::sh_value
