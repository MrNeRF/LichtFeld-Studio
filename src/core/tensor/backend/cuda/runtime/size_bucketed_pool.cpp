/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor/backend/cuda/runtime/size_bucketed_pool.hpp"

namespace lfs::core {

    SizeBucketedPool& SizeBucketedPool::instance() {
        static SizeBucketedPool pool;
        return pool;
    }

} // namespace lfs::core
