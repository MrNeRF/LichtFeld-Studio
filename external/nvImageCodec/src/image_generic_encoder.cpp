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
#include "default_executor.h"
#include "encode_state_batch.h"
#include "icode_stream.h"
#include "icodec.h"
#include "icodec_registry.h"
#include "iimage.h"
#include "iimage_encoder.h"
#include "iimage_encoder_factory.h"
#include "imgproc/copy_image.h"
#include "imgproc/exception.h"
#include "imgproc/type_utils.h"
#include "log.h"
#include "processing_results.h"
#include "user_executor.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <unordered_set>

namespace nvimgcodec {

    void ImageGenericEncoder::canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
                                        const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* processing_status, int force_format) noexcept {
        try {
            canProcess(code_streams, images, processing_status, force_format, params);
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(logger_, "Exception during canEncode: " << e.what());
            std::fill(processing_status, processing_status + code_streams.size(), NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
    }

    ProcessingResultsPromise::FutureImpl ImageGenericEncoder::encode(
        const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcodecEncodeParams_t* params) noexcept {
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

    nvimgcodecProcessingStatus_t ImageGenericEncoder::canProcessImpl(Entry& sample, ProcessorEntry* processor, int tid) noexcept {
        NVIMGCODEC_LOG_TRACE(this->logger_, tid << ": " << sample.processor->id_ << " canEncode #" << sample.sample_idx);
        nvimgcodecProcessingStatus_t status;
        try {
            sample.processor->instance_->canEncode(
                sample.code_stream->getCodeStreamDesc(), sample.getImageDesc(),
                static_cast<const nvimgcodecEncodeParams_t*>(sample.params), &status, tid);
            NVIMGCODEC_LOG_DEBUG(
                this->logger_, tid << ": " << sample.processor->id_ << " canEncode #" << sample.sample_idx << " returned " << status);
            return status;
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(logger_, "Exception during canEncode: " << e.what());
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
    }

    bool ImageGenericEncoder::processImpl(Entry& sample, int tid) noexcept {
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

    void ImageGenericEncoder::sortSamples() {
    }

    bool ImageGenericEncoder::processBatchImpl(ProcessorEntry& processor) noexcept {
        return true;
    }

    bool ImageGenericEncoder::copyToTempBuffers(Entry& sample, int tid) {
        nvtx3::scoped_range marker{"copyToTempBuffers"};
        auto& output_info = sample.image_info;
        auto& processor = sample.processor;
        const auto& input_info = sample.orig_image_info;
        bool d2h = processor->backend_kind_ == NVIMGCODEC_BACKEND_KIND_CPU_ONLY &&
                   input_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        bool h2d =
            processor->backend_kind_ != NVIMGCODEC_BACKEND_KIND_CPU_ONLY && input_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

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
