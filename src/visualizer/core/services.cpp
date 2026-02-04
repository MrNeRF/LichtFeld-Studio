/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/services.hpp"

namespace lfs::vis {

    Services& Services::instance() {
        static Services s;
        return s;
    }

} // namespace lfs::vis
