/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/live_model_mutation_guard.hpp"

#include "core/logger.hpp"

#include <cassert>
#include <format>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace lfs::training {
    namespace {
        std::mutex g_hooks_mu;
        LiveModelMutationHooks g_hooks{};
        bool g_hooks_set = false;

        thread_local int t_depth = 0;
        thread_local bool t_held_by_refining = false;
        thread_local bool t_externally_held = false;
    } // namespace

    void register_live_model_mutation_hooks(LiveModelMutationHooks hooks) {
        std::lock_guard lock(g_hooks_mu);
        g_hooks = std::move(hooks);
        g_hooks_set = true;
    }

    void clear_live_model_mutation_hooks() {
        std::lock_guard lock(g_hooks_mu);
        g_hooks = {};
        g_hooks_set = false;
    }

    bool live_model_mutation_hooks_registered() {
        std::lock_guard lock(g_hooks_mu);
        return g_hooks_set;
    }

    void mark_live_model_mutation_lock_held(bool held) { t_externally_held = held; }

    bool live_model_mutation_lock_held_by_this_thread() {
        return t_externally_held || t_depth > 0;
    }

    int live_model_mutation_depth() { return t_depth; }

    void assert_live_model_mutation_lock_held(std::string_view site,
                                              const std::source_location& loc) {
#ifndef NDEBUG
        const bool held = live_model_mutation_lock_held_by_this_thread() ||
                          (g_hooks.mutation_lock_held_by_this_thread &&
                           g_hooks.mutation_lock_held_by_this_thread());
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
        if (t_depth > 0) {
            ++t_depth;
            return; // re-entrant no-op
        }

        LiveModelMutationHooks hooks;
        {
            std::lock_guard lock(g_hooks_mu);
            hooks = g_hooks;
            if (!g_hooks_set) {
                // Headless / unit: no trainer wired — mutation is single-threaded.
                ++t_depth;
                return;
            }
        }

        if (hooks.viewer_present && hooks.viewer_present() &&
            !hooks.acquire_render_exclusive) {
#ifndef NDEBUG
            LOG_ERROR(
                "LiveModelMutationGuard: viewer present but no mutation hooks "
                "wired (site={}) — this is a bug",
                site);
            assert(false && "viewer present without mutation hooks");
#endif
        }

        // Already inside trainer refining exclusive — no-op acquire, still count depth.
        if (t_externally_held ||
            (hooks.mutation_lock_held_by_this_thread &&
             hooks.mutation_lock_held_by_this_thread())) {
            t_held_by_refining = true;
            ++t_depth;
            return;
        }

        if (hooks.acquire_render_exclusive) {
            took_render_ = hooks.acquire_render_exclusive();
        }
        if (hooks.acquire_combined) {
            took_combined_ = hooks.acquire_combined();
        }
        if (hooks.wait_for_model_readers) {
            hooks.wait_for_model_readers();
        }
        if (hooks.begin_densify_barrier) {
            took_barrier_ = hooks.begin_densify_barrier();
        }
        acquired_ = true;
        ++t_depth;
        (void)site;
    }

    LiveModelMutationGuard::~LiveModelMutationGuard() {
        if (t_depth <= 0) {
            return;
        }
        --t_depth;
        if (t_depth > 0) {
            return; // nested
        }

        if (t_held_by_refining) {
            t_held_by_refining = false;
            return;
        }
        if (!acquired_) {
            return;
        }

        LiveModelMutationHooks hooks;
        {
            std::lock_guard lock(g_hooks_mu);
            hooks = g_hooks;
        }
        if (took_barrier_ && hooks.end_densify_barrier) {
            hooks.end_densify_barrier();
        }
        if (took_combined_ && hooks.release_combined) {
            hooks.release_combined();
        }
        if (took_render_ && hooks.release_render_exclusive) {
            hooks.release_render_exclusive();
        }
    }

} // namespace lfs::training
