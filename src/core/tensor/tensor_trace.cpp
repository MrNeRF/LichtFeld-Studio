/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_trace.hpp"

namespace lfs::core::debug {

    TensorOpTracer& TensorOpTracer::instance() {
        static TensorOpTracer inst;
        return inst;
    }

} // namespace lfs::core::debug
