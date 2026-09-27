/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::gpu_ops {

    struct SessionOps {
        void (*profile)(bool start);
    };

} // namespace lfs::gpu_ops
