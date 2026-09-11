/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/size_bucketed_pool.hpp"

namespace lfs::core {

    SizeBucketedPool& SizeBucketedPool::instance() {
        static SizeBucketedPool pool;
        return pool;
    }

} // namespace lfs::core
