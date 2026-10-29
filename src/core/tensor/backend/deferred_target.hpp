/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "backend_driver.hpp"

namespace lfs::core::internal {
    inline BackendDriver& deferred_target_driver(GpuBackend backend) {
        // Drivers have process lifetime; activation and native thread state stay
        // in the driver. Only a non-null deferred home needs this cache.
        thread_local BackendRegistry drivers{};
        auto& driver = drivers[static_cast<size_t>(backend)];
        if (!driver)
            driver = &backend_driver(backend);
        return *driver;
    }
} // namespace lfs::core::internal
