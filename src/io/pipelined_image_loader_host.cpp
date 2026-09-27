/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/pipelined_image_loader.hpp"

namespace lfs::io {

    bool PipelinedImageLoader::attach_cuda_decode_stage() { return false; }

    void PipelinedImageLoader::start_cuda_decode_workers() {}

    void PipelinedImageLoader::release_cuda_decode_stage() {}

} // namespace lfs::io
