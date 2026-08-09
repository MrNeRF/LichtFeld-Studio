/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file live_model_mutation_guard.hpp
 * @brief RAII marker for live-model mutation scopes.
 *
 * Mutation helpers are depth-counted per thread so debug assertions can verify
 * that a helper runs inside either a guarded scope or a trainer-owned exclusive
 * section marked with mark_live_model_mutation_lock_held().
 */

#include <source_location>
#include <string_view>

#ifndef LFS_STRINGIFY
#define LFS_STRINGIFY_INNER(x) #x
#define LFS_STRINGIFY(x)       LFS_STRINGIFY_INNER(x)
#endif

namespace lfs::training {

    /// Trainer refining block: mark this thread as already holding mutation exclusive
    /// so nested assertions recognize the external lock.
    void mark_live_model_mutation_lock_held(bool held);
    [[nodiscard]] bool live_model_mutation_lock_held_by_this_thread();

    /// Debug-only: hard-assert the mutation exclusive is held (by guard or refining block).
    void assert_live_model_mutation_lock_held(
        std::string_view site,
        const std::source_location& loc = std::source_location::current());

#define LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD()         \
    ::lfs::training::assert_live_model_mutation_lock_held( \
        std::string_view{__FILE__ ":" LFS_STRINGIFY(__LINE__)})

    /// RAII marker for a depth-counted mutation scope on this thread.
    class LiveModelMutationGuard {
    public:
        explicit LiveModelMutationGuard(std::string_view site = "mutation");
        ~LiveModelMutationGuard();

        LiveModelMutationGuard(const LiveModelMutationGuard&) = delete;
        LiveModelMutationGuard& operator=(const LiveModelMutationGuard&) = delete;
    };

} // namespace lfs::training
