/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/splat_decimate.hpp"

namespace lfs::io {
    Result<lfs::core::SplatData> decimate_splats(const lfs::core::SplatData&,
                                                 const DecimateOptions&) {
        return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                          "Splat decimation requires CUDA, which is unavailable in this build");
    }
} // namespace lfs::io
