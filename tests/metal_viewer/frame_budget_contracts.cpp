/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "frame_budget.hpp"
#include <cstdio>
#include <limits>

using namespace lfs::rendering::metal;
int main() {
    constexpr uint64_t gib = uint64_t{1} << 30;
    static_assert(frameFitsWorkingSet(400, 400, 1000));
    static_assert(!frameFitsWorkingSet(400, 401, 1000));
    static_assert(!frameFitsWorkingSet(801, 0, 1000));
    static_assert(frameFitsWorkingSet(std::numeric_limits<uint64_t>::max(), 1, 0));
    static_assert(!frameFitsWorkingSet(std::numeric_limits<uint64_t>::max(), 1, 1000));
    constexpr auto maximum = frameReservationBytes(16384, 16384,
                                                   std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(), false);
    static_assert(maximum > 100 * gib);
    static_assert(!frameFitsWorkingSet(0, maximum, 32 * gib));
    static_assert(frameReservationBytes(1920, 1080, 1000000, 16000000, true) <
                  frameReservationBytes(1920, 1080, 1000000, 16000000, false));
    static_assert(scanReservationBytes(257) > scanReservationBytes(256));
    static_assert(scanReservationBytes(65537, 4) < scanReservationBytes(65537, 8));
    // Narrow post-admission histograms do not narrow the reservation arithmetic.
    static_assert(scanReservationBytes(std::numeric_limits<uint32_t>::max(), 4) > 65536);
    std::puts("Metal frame working-set admission and overflow contracts passed.");
}
