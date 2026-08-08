/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file live_model_mutation_guard.hpp
 * @brief RAII guard that enforces live-model mutation exclusion INSIDE helpers.
 *
 * Lesson 6 / rock-solid bar: locks enforced only at call sites rot. Acquire the
 * refining-block composition (render_mutex exclusive + Scene combined-model +
 * waitForModelReaders + exportable densify barrier) from inside mutation helpers
 * (ensure/commit/degree/compact) so off-spine callers (crop, python, stop_refine)
 * cannot reallocate live model storage under a concurrent viewer.
 *
 * Depth-counted per thread: nested helpers no-op on re-entry. When the trainer's
 * refining block already holds the locks, the guard detects that and no-ops so
 * steady-state refining steps keep identical lock ordering.
 */

#include <cstdint>
#include <functional>
#include <source_location>
#include <string_view>

#ifndef LFS_STRINGIFY
#define LFS_STRINGIFY_INNER(x) #x
#define LFS_STRINGIFY(x)       LFS_STRINGIFY_INNER(x)
#endif

namespace lfs::training {

    /// Callbacks installed by Trainer (or TrainerManager) so core mutation
    /// helpers stay free of a hard trainer dependency.
    struct LiveModelMutationHooks {
        /// Take render_mutex exclusive if not already held by this thread.
        /// Returns true if this call acquired (must release on exit).
        std::function<bool()> acquire_render_exclusive;
        std::function<void()> release_render_exclusive;
        /// Combined Scene model lock (optional when no scene).
        std::function<bool()> acquire_combined;
        std::function<void()> release_combined;
        std::function<void()> wait_for_model_readers;
        std::function<bool()> begin_densify_barrier;
        std::function<void()> end_densify_barrier;
        /// True when this thread already holds the mutation exclusive (refining block).
        std::function<bool()> mutation_lock_held_by_this_thread;
        /// True when a viewer/GUI is wired (debug assert if hooks missing).
        std::function<bool()> viewer_present;
    };

    /// Process-wide hook registration (null = headless unit, guard is no-op).
    void register_live_model_mutation_hooks(LiveModelMutationHooks hooks);
    void clear_live_model_mutation_hooks();
    [[nodiscard]] bool live_model_mutation_hooks_registered();

    /// Trainer refining block: mark this thread as already holding mutation exclusive
    /// so nested LiveModelMutationGuard no-ops (preserves lock ordering).
    void mark_live_model_mutation_lock_held(bool held);
    [[nodiscard]] bool live_model_mutation_lock_held_by_this_thread();

    /// Per-thread depth of LiveModelMutationGuard (0 = not in a guarded mutation).
    [[nodiscard]] int live_model_mutation_depth();

    /// Debug-only: hard-assert the mutation exclusive is held (by guard or refining block).
    void assert_live_model_mutation_lock_held(
        std::string_view site,
        const std::source_location& loc = std::source_location::current());

#define LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD()         \
    ::lfs::training::assert_live_model_mutation_lock_held( \
        std::string_view{__FILE__ ":" LFS_STRINGIFY(__LINE__)})

    /// RAII: acquire full mutation exclusion on first entry for this thread.
    class LiveModelMutationGuard {
    public:
        explicit LiveModelMutationGuard(std::string_view site = "mutation");
        ~LiveModelMutationGuard();

        LiveModelMutationGuard(const LiveModelMutationGuard&) = delete;
        LiveModelMutationGuard& operator=(const LiveModelMutationGuard&) = delete;

        [[nodiscard]] bool acquired() const { return acquired_; }

    private:
        bool acquired_ = false;
        bool took_render_ = false;
        bool took_combined_ = false;
        bool took_barrier_ = false;
    };

} // namespace lfs::training
