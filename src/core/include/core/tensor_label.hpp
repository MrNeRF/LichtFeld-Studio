/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/tensor/internal/memory_pool.hpp>

#define LFS_TENSOR_LABEL_CONCAT_INNER(a, b) a##b
#define LFS_TENSOR_LABEL_CONCAT(a, b)       LFS_TENSOR_LABEL_CONCAT_INNER(a, b)

#define LFS_LABEL_SCOPE(name)                                        \
    ::lfs::core::CudaMemoryPool::LabelGuard LFS_TENSOR_LABEL_CONCAT( \
        _lfs_label_guard_, __LINE__)(name)

#define LFS_TENSOR_LABEL(name, expr)                                           \
    ([&]() {                                                                   \
        ::lfs::core::CudaMemoryPool::LabelGuard _lfs_tensor_label_guard(name); \
        auto _lfs_tensor_label_value = (expr);                                 \
        _lfs_tensor_label_value.set_name(name);                                \
        return _lfs_tensor_label_value;                                        \
    }())
