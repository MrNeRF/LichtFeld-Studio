/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_exportable_storage.hpp"

namespace lfs::core {
    void* resolve_exportable_device_ptr(const Tensor& tensor) {
        return const_cast<void*>(tensor.is_valid() ? tensor.data_ptr() : nullptr);
    }
} // namespace lfs::core
