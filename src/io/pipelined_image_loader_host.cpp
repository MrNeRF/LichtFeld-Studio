/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/pipelined_image_loader.hpp"

namespace lfs::io {

    bool PipelinedImageLoader::decodes_on_gpu(lfs::core::GpuBackend) { return false; }

    bool PipelinedImageLoader::attach_cuda_decode_stage() { return false; }

    void PipelinedImageLoader::start_cuda_decode_workers() {}

    void PipelinedImageLoader::release_cuda_decode_stage() {}

    // Without nvImageCodec there is no decoder to warm up.
    ImageDecoderWarmup::ImageDecoderWarmup(const size_t decoder_pool_size)
        : decoder_pool_size_(decoder_pool_size) {}

    ImageDecoderWarmup::~ImageDecoderWarmup() = default;

} // namespace lfs::io
