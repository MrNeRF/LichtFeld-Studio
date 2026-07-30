/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "memory_arena.hpp"

namespace lfs::core {

    GlobalArenaManager& global_arena_manager() {
        static GlobalArenaManager manager;
        return manager;
    }

    void shutdown_global_arena_manager() {
        auto& manager = global_arena_manager();
        std::lock_guard<std::mutex> lock(manager.init_mutex_);
        if (manager.shutdown_) {
            return;
        }

        manager.shutdown_ = true;
        if (!manager.arena_) {
            return;
        }

        manager.arena_->full_reset();
        manager.arena_.reset();
    }

} // namespace lfs::core
