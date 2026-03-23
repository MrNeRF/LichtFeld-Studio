/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <mutex>

namespace lfs::core {

    struct IGSPlusFailureSnapshot {
        bool valid = false;
        int iter = -1;
        int64_t size_before = 0;
        int64_t active_before = 0;
        int64_t free_before = 0;
        int64_t budget = 0;
        int64_t budget_for_alloc = 0;
        int64_t candidate_budget = 0;
        int64_t selectable = 0;
        int64_t selected = 0;
        int64_t num_filled = 0;
        int64_t num_appended = 0;
        int64_t active_after = 0;
        int64_t free_after = 0;
        float sampled_scale_p95 = 0.0f;
        float sampled_scale_max = 0.0f;
        float sampled_scale_exp_max = 0.0f;
    };

    inline std::mutex& igs_plus_failure_snapshot_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    inline IGSPlusFailureSnapshot& igs_plus_failure_snapshot_storage() {
        static IGSPlusFailureSnapshot snapshot;
        return snapshot;
    }

    inline void update_igs_plus_failure_snapshot(const IGSPlusFailureSnapshot& snapshot) {
        std::lock_guard<std::mutex> lock(igs_plus_failure_snapshot_mutex());
        igs_plus_failure_snapshot_storage() = snapshot;
    }

    inline bool try_get_igs_plus_failure_snapshot(IGSPlusFailureSnapshot& out) {
        std::lock_guard<std::mutex> lock(igs_plus_failure_snapshot_mutex());
        const auto& snapshot = igs_plus_failure_snapshot_storage();
        if (!snapshot.valid) {
            return false;
        }
        out = snapshot;
        return true;
    }

} // namespace lfs::core
