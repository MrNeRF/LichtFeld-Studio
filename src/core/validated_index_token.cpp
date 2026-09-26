/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/device_fault.hpp"

namespace lfs::core {
    bool ValidatedIndexToken::matches(const void* storage_identity,
                                      const std::uint64_t mutation_version,
                                      const int device_ordinal,
                                      const std::uint64_t producer_event_or_range) const noexcept {
        return storage_identity_ == storage_identity &&
               mutation_version_ == mutation_version &&
               device_ordinal_ == device_ordinal &&
               producer_event_or_range_ == producer_event_or_range;
    }

    ValidatedIndexToken issue_validated_index_token(
        const void* storage_identity,
        const std::uint64_t mutation_version,
        const int device_ordinal,
        const std::uint64_t producer_event_or_range) {
        return ValidatedIndexToken(storage_identity, mutation_version, device_ordinal,
                                   producer_event_or_range);
    }
} // namespace lfs::core
