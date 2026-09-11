/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <utility>

namespace lfs::vis {

    // An episode is the pending viewport refresh since the last publication.
    // Camera/model changes coalesce into that request: changing attempt ids,
    // error text or a continuously moving camera must not restart its budget.
    class StaleFrameGuard {
    public:
        static constexpr std::uint32_t kMaxCachedDeferrals = 30;

        // True exactly once, on the attempt that exhausts the cache budget.
        [[nodiscard]] bool onDeferral() {
            if (deferrals_ == kMaxCachedDeferrals) {
                return false;
            }
            if (++deferrals_ == kMaxCachedDeferrals) {
                recovery_pending_ = true;
                return true;
            }
            return false;
        }

        [[nodiscard]] bool canUseCachedFrame() const {
            return deferrals_ < kMaxCachedDeferrals;
        }

        // Consume before acquiring trainer locks: the existing scratch reset
        // waits for an idle arena and must not run while holding those locks.
        [[nodiscard]] bool takeRecoveryRequest() {
            return std::exchange(recovery_pending_, false);
        }

        void onSuccess() {
            deferrals_ = 0;
            recovery_pending_ = false;
        }

    private:
        std::uint32_t deferrals_ = 0;
        bool recovery_pending_ = false;
    };

} // namespace lfs::vis
