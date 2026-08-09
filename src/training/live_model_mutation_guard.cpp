/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/live_model_mutation_guard.hpp"

#include "core/logger.hpp"

#include <cassert>
#include <format>
#include <stdexcept>

namespace lfs::training {
    namespace {
        thread_local int t_depth = 0;
        thread_local bool t_externally_held = false;
    } // namespace

    void mark_live_model_mutation_lock_held(bool held) { t_externally_held = held; }

    bool live_model_mutation_lock_held_by_this_thread() {
        return t_externally_held || t_depth > 0;
    }

    void assert_live_model_mutation_lock_held(std::string_view site,
                                              const std::source_location& loc) {
#ifndef NDEBUG
        const bool held = live_model_mutation_lock_held_by_this_thread();
        if (!held) {
            const auto msg = std::format(
                "Live-model mutation without exclusive guard at {} ({}:{}) — "
                "call site must be under LiveModelMutationGuard or refining block",
                site,
                loc.file_name(),
                loc.line());
            LOG_ERROR("{}", msg);
            assert(false && "Live-model mutation lock not held");
            throw std::runtime_error(msg);
        }
#else
        (void)site;
        (void)loc;
#endif
    }

    LiveModelMutationGuard::LiveModelMutationGuard(std::string_view site) {
        ++t_depth;
        (void)site;
    }

    LiveModelMutationGuard::~LiveModelMutationGuard() {
        if (t_depth <= 0) {
            return;
        }
        --t_depth;
    }

} // namespace lfs::training
