/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "image_generic_encoder.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <stdexcept>
#include <unordered_set>
#include "default_executor.h"
#include "encode_state_batch.h"
#include "icode_stream.h"
#include "icodec.h"
#include "icodec_registry.h"
#include "iimage.h"
#include "iimage_encoder.h"
#include "iimage_encoder_factory.h"
#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/copy_image.h"
#include "imgproc/exception.h"
#include "imgproc/type_utils.h"
#include "log.h"
#include "processing_results.h"
#include "user_executor.h"

namespace nvimgcodec {

void ImageGenericEncoder::canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
    const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* processing_status, int force_format) noexcept
{
    try {
        canProcess(code_streams, images, processing_status, force_format, params);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during canEncode: " << e.what());
        std::fill(processing_status, processing_status + code_streams.size(), NVIMGCODEC_PROCESSING_STATUS_FAIL);
    }
}

ProcessingResultsPromise::FutureImpl ImageGenericEncoder::encode(
    const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcodecEncodeParams_t* params) noexcept
{
    try {
        return process(code_streams, images, params);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during encode: " << e.what());
    } catch (...) {
        NVIMGCODEC_LOG_ERROR(logger_, "Unknown exception during encode");
    }

    ProcessingResultsPromise promise(static_cast<int>(code_streams.size()));
    for (size_t i = 0; i < code_streams.size(); i++) {
        // coverity[fun_call_w_exception : FALSE]
        promise.set(i, ProcessingResult::failure(NVIMGCODEC_PROCESSING_STATUS_FAIL));
    }
    return promise.getFuture();
}


namespace {

// nvTIFF, and potentially other codecs, can only consume interleaved samples. These two helpers
// describe the one layout conversion the core is willing to perform on the caller's behalf.
// Restricted to planar RGB/BGR: LaunchConvertNormKernel has no YUV to RGB path, and P_Y is
// already identical to its interleaved form.
bool isPlanarRgbLike(const nvimgcodecImageInfo_t& info)
{
    if (info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_RGB && info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_BGR) {
        return false;
    }
    // Exactly three single-channel planes. The tag alone is not enough: the BMP parser reports
    // P_RGB for any num_planes != 1, so a 32bpp BMP arrives as a 4-plane P_RGB image. Feeding
    // that to makeInterleaved() would build an I_RGB descriptor with 4 channels, which is not a
    // valid I_RGB image.
    if (info.num_planes != 3) {
        return false;
    }
    for (uint32_t p = 0; p < info.num_planes; ++p) {
        if (info.plane_info[p].num_channels != 1) {
            return false;
        }
    }
    return true;
}

nvimgcodecImageInfo_t makeInterleaved(const nvimgcodecImageInfo_t& in)
{
    nvimgcodecImageInfo_t out = in;
    out.sample_format = in.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB ? NVIMGCODEC_SAMPLEFORMAT_I_RGB
                                                                          : NVIMGCODEC_SAMPLEFORMAT_I_BGR;
    out.num_planes = 1;
    out.plane_info[0] = in.plane_info[0];
    out.plane_info[0].num_channels = in.num_planes;
    out.plane_info[0].row_stride = static_cast<size_t>(out.plane_info[0].width) *
                                   out.plane_info[0].num_channels * TypeSize(out.plane_info[0].sample_type);
    return out;
}

} // namespace

nvimgcodecProcessingStatus_t ImageGenericEncoder::canProcessImpl(Entry& sample, ProcessorEntry* processor, int tid) noexcept
{
    NVIMGCODEC_LOG_TRACE(this->logger_, tid << ": " << processor->id_ << " canEncode #" << sample.sample_idx);
    auto ask = [&](ProcessorEntry* proc) {
        nvimgcodecProcessingStatus_t s = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        bool can_encode = proc->instance_->canEncode(sample.code_stream->getCodeStreamDesc(), sample.getImageDesc(),
            static_cast<const nvimgcodecEncodeParams_t*>(sample.params), &s, tid);
        // A refusal must not be mistaken for acceptance, even if the encoder left no reason for it.
        if (!can_encode && (s == NVIMGCODEC_PROCESSING_STATUS_SUCCESS || s == NVIMGCODEC_PROCESSING_STATUS_UNKNOWN))
            s = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        return s;
    };
    try {
        // Once a sample has been committed to the interleaved layout, only a GPU encoder may take
        // it: the interleaving runs on the GPU and hands the encoder a device buffer. The sample
        // can still be moved to a fallback after this processor accepted (completeSetup() reassigns
        // over sample_count_hint_ in image_generic_codec.h), and the tiff chain really does fall
        // back from nvTIFF to the CPU-only opencv encoder -- so declining here is what keeps that
        // reassignment from happening, rather than catching it later.
        if (sample.convert_to_interleaved && processor->backend_kind_ == NVIMGCODEC_BACKEND_KIND_CPU_ONLY) {
            return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }
        nvimgcodecProcessingStatus_t status = ask(processor);
        NVIMGCODEC_LOG_DEBUG(
            this->logger_, tid << ": " << processor->id_ << " canEncode #" << sample.sample_idx << " returned " << status);

        // If the encoder turned the sample down over its layout, offer the interleaved equivalent
        // and let the core interleave before encode(). Only the layout is renegotiated; any other
        // reason for rejection is returned untouched so encoder selection still sees it.
        //
        // Each reason is one distinctive bit ORed with bit 0, which every non-UNKNOWN status
        // carries -- FAIL is 0x3, CODEC_UNSUPPORTED 0xb, and so on. A plain `status & layout_bits`
        // would therefore match nearly every rejection, so test that the distinctive bits
        // themselves are present.
        const bool layout_rejected = (status & NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED) ==
                                         NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED ||
                                     (status & NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED) ==
                                         NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        if (status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS && layout_rejected &&
            processor->backend_kind_ != NVIMGCODEC_BACKEND_KIND_CPU_ONLY && isPlanarRgbLike(sample.image_info)) {
            const auto planar_info = sample.image_info;
            sample.image_info = makeInterleaved(planar_info);
            const nvimgcodecProcessingStatus_t retry = ask(processor);
            if (retry == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                // Committed rather than probed-and-reverted: later fallback probes for this sample
                // then evaluate the same layout, so whichever encoder is finally selected agrees
                // with convert_to_interleaved. Every encoder that accepts planar RGB also accepts
                // interleaved RGB, so committing cannot lose an otherwise viable candidate.
                sample.convert_to_interleaved = true;
                NVIMGCODEC_LOG_DEBUG(this->logger_,
                    tid << ": " << processor->id_ << " #" << sample.sample_idx << " accepted after interleaving");
                return retry;
            }
            sample.image_info = planar_info;
        }
        return status;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during canEncode: " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

bool ImageGenericEncoder::processImpl(Entry& sample, int tid) noexcept
{
    try {
        copyToTempBuffers(sample, tid);
        bool encode_ret = sample.processor->instance_->encode(
            sample.code_stream->getCodeStreamDesc(), sample.getImageDesc(),
            static_cast<const nvimgcodecEncodeParams_t*>(sample.params), &sample.status, tid);

        assert(sample.status != NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        bool encode_success = encode_ret && sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        return encode_success;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during processImpl: " << e.what());
        sample.status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        return false;
    }
}

void ImageGenericEncoder::sortSamples()
{
}

bool ImageGenericEncoder::processBatchImpl(ProcessorEntry& processor) noexcept
{
    return true;
}

bool ImageGenericEncoder::copyToTempBuffers(Entry& sample, int tid)
{
    nvtx3::scoped_range marker{"copyToTempBuffers"};
    auto& output_info = sample.image_info;
    auto& processor = sample.processor;
    const auto& input_info = sample.orig_image_info;
    bool d2h = processor->backend_kind_ == NVIMGCODEC_BACKEND_KIND_CPU_ONLY &&
               input_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    bool h2d =
        processor->backend_kind_ != NVIMGCODEC_BACKEND_KIND_CPU_ONLY && input_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

    // Layout conversion negotiated in canProcessImpl(): interleave into a dedicated device buffer
    // and hand the encoder that instead. output_info already describes the interleaved layout, so
    // only its buffer has to be filled in. Never taken for a CPU-only backend -- canProcessImpl()
    // does not offer the conversion there, since LaunchConvertNormKernel is CUDA-only.
    if (sample.convert_to_interleaved) {
        // Belt and braces for the guard in canProcessImpl(). Not an assert: with NDEBUG this would
        // otherwise fall through and hand a device buffer to a CPU-only encoder. Throwing rather
        // than returning false, because processImpl() ignores this function's return value and
        // calls encode() regardless -- false has always meant "no copy was needed". processImpl()
        // catches this and reports NVIMGCODEC_PROCESSING_STATUS_FAIL for the sample.
        if (processor->backend_kind_ == NVIMGCODEC_BACKEND_KIND_CPU_ONLY) {
            throw std::logic_error("CPU-only encoder selected for a sample requiring GPU interleaving");
        }
        auto& t = per_thread_[tid];
        nvimgcodecImageInfo_t src_info = input_info;
        if (h2d) {
            // The kernel reads through a device pointer, so a host input is staged first. This
            // uses the regular ring buffer; the interleaved result goes to convert_buffer.
            auto& device_buffer = t.device_buffers[t.device_buffer_idx];
            t.device_buffer_idx = (t.device_buffer_idx + 1) % t.device_buffers.size();
            for (unsigned int c = 0; c < src_info.num_planes; ++c) {
                auto& pi = src_info.plane_info[c];
                pi.row_stride = static_cast<size_t>(pi.width) * TypeSize(pi.sample_type) * pi.num_channels;
            }
            device_buffer.resize(GetBufferSize(src_info), output_info.cuda_stream);
            src_info.buffer = device_buffer.data;
            src_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
            CopyImage(src_info, input_info, cudaMemcpyHostToDevice, output_info.cuda_stream);
        }
        t.convert_buffer.resize(GetBufferSize(output_info), output_info.cuda_stream);
        output_info.buffer = t.convert_buffer.data;
        output_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        NVIMGCODEC_LOG_DEBUG(logger_, "interleaving planar input before encode");
        LaunchConvertNormKernel(output_info, src_info, output_info.cuda_stream);
        return true;
    }

    // restore original stride in case we don't need to copy
    for (unsigned int c = 0; c < output_info.num_planes; ++c) {
        output_info.plane_info[c].row_stride = input_info.plane_info[c].row_stride;
    }
    output_info.buffer = input_info.buffer;
    output_info.buffer_kind = input_info.buffer_kind;

    if (!h2d && !d2h)
        return false;

    output_info.buffer = nullptr;
    for (unsigned int c = 0; c < output_info.num_planes; ++c) {
        auto& plane_info = output_info.plane_info[c];
        size_t bpp = TypeSize(plane_info.sample_type);
        plane_info.row_stride = static_cast<size_t>(plane_info.width) * bpp * plane_info.num_channels;
    }

    assert(num_threads_ + 1 == per_thread_.size());
    assert(tid >= 0 && tid <= static_cast<int>(num_threads_));
    auto& t = per_thread_[tid];
    if (h2d) {
        auto& device_buffer = t.device_buffers[t.device_buffer_idx];
        t.device_buffer_idx = (t.device_buffer_idx + 1) % t.device_buffers.size();
        device_buffer.resize(GetBufferSize(output_info), output_info.cuda_stream);
        output_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        output_info.buffer = device_buffer.data;
        assert(GetBufferSize(output_info) == device_buffer.size);
        assert(output_info.cuda_stream == device_buffer.stream);

    } else if (d2h) {
        auto& pinned_buffer = t.pinned_buffers[t.pinned_buffer_idx];
        t.pinned_buffer_idx = (t.pinned_buffer_idx + 1) % t.pinned_buffers.size();
        pinned_buffer.resize(GetBufferSize(output_info), output_info.cuda_stream);
        output_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        output_info.buffer = pinned_buffer.data;
        assert(GetBufferSize(output_info) == pinned_buffer.size);
        assert(output_info.cuda_stream == pinned_buffer.stream);
    } else {
        assert(false); // should not happen
        return false;
    }
    auto copy_direction = d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

    // output is always continuous (just allocated above); CopyImage collapses to
    // a single cudaMemcpyAsync when both sides are contiguous and falls back to
    // per-plane cudaMemcpy2DAsync otherwise.
    NVIMGCODEC_LOG_DEBUG(logger_, "CopyImage " << (d2h ? "D2H" : "H2D") << " stream=" << output_info.cuda_stream);
    CopyImage(output_info, input_info, copy_direction, output_info.cuda_stream);

    if (d2h) {
        CHECK_CUDA(cudaStreamSynchronize(output_info.cuda_stream));
    }
    return true;
}

} // namespace nvimgcodec
