/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/joint_adam_codec.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace lfs::training::joint_adam {
    namespace {
        // -1 = use env, 0 = force legacy, 1 = force joint
        std::atomic<int> g_override{-1};

        [[nodiscard]] bool env_joint_enabled() {
            const char* env = std::getenv("LFS_ADAM_LEGACY_CODEC");
            if (env == nullptr || env[0] == '\0') {
                return true; // default: joint ON
            }
            if (env[0] == '0' && env[1] == '\0')
                return true;
            if (std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0)
                return true;
            return false; // any other value → legacy
        }
    } // namespace

    void set_joint_codec_enabled_for_testing(const std::optional<bool> enabled) {
        if (!enabled.has_value()) {
            g_override.store(-1, std::memory_order_relaxed);
            return;
        }
        g_override.store(*enabled ? 1 : 0, std::memory_order_relaxed);
    }

    bool joint_codec_enabled() {
        const int o = g_override.load(std::memory_order_relaxed);
        if (o >= 0)
            return o != 0;
        return env_joint_enabled();
    }

} // namespace lfs::training::joint_adam
