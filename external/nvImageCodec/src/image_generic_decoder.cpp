/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "image_generic_decoder.h"
#include "decode_state_batch.h"
#include "default_executor.h"
#include "icode_stream.h"
#include "icodec.h"
#include "icodec_registry.h"
#include "iimage.h"
#include "iimage_decoder.h"
#include "iimage_decoder_factory.h"
#include "imgproc/copy_image.h"
#include "imgproc/device_guard.h"
#include "imgproc/exception.h"
#include "imgproc/type_utils.h"
#include "log.h"
#include "processing_results.h"
#include "user_executor.h"
#include <algorithm>
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

    nvimgcodecStatus_t ImageGenericDecoder::getMetadata(ICodeStream* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) noexcept {
        nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
        try {
            NVIMGCODEC_LOG_INFO(logger_, "ImageGenericDecoder::getMetadata");
            DeviceGuard device_guard(exec_params_.device_id);
            auto codec = code_stream->getCodec();
            auto* processor = initProcessorsAndGetFirstForCodec(codec);

            while (processor) {
                auto cs_desc = code_stream->getCodeStreamDesc();
                auto status = processor->instance_->getMetadata(cs_desc, metadata, metadata_count);
                if (status != NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED) {
                    ret = status;
                }
                if (status == NVIMGCODEC_STATUS_SUCCESS) {
                    break;
                }
                processor = processor->fallback_;
            }
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(logger_, "Exception during getMetadata: " << e.what());
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
        return ret;
    }

    void ImageGenericDecoder::canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
                                        const nvimgcodecDecodeParams_t* params, nvimgcodecProcessingStatus_t* processing_status, int force_format) noexcept {
        try {
            canProcess(code_streams, images, processing_status, force_format, params);
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(logger_, "Exception during canDecode: " << e.what());
            std::fill(processing_status, processing_status + code_streams.size(), NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
    }

    ProcessingResultsPromise::FutureImpl ImageGenericDecoder::decode(
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcodecDecodeParams_t* params) noexcept {
        try {
            NVIMGCODEC_LOG_INFO(logger_, "ImageGenericDecoder::decode num_samples=" << code_streams.size());
            return process(code_streams, images, params);
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(logger_, "Exception during decode: " << e.what());
        } catch (...) {
            NVIMGCODEC_LOG_ERROR(logger_, "Unknown exception during decode");
        }

        ProcessingResultsPromise promise(static_cast<int>(code_streams.size()));
        for (size_t i = 0; i < code_streams.size(); i++) {
            // coverity[fun_call_w_exception : FALSE]
            promise.set(i, ProcessingResult::failure(NVIMGCODEC_PROCESSING_STATUS_FAIL));
        }
        return promise.getFuture();
    }

    void ImageGenericDecoder::sortSamples() {
        nvtx3::scoped_range marker{"sort samples"};
        assert(curr_order_.size() == code_streams_.size());
        int batch_size = code_streams_.size();
        curr_order_.clear();
        curr_order_.resize(batch_size);
        std::iota(curr_order_.begin(), curr_order_.end(), 0);

        subsampling_score_.resize(batch_size);
        area_.resize(batch_size);
        auto subsamplingScore = [](nvimgcodecChromaSubsampling_t subsampling) {
            switch (subsampling) {
            case NVIMGCODEC_SAMPLING_444:
                return 8;
            case NVIMGCODEC_SAMPLING_422:
                return 7;
            case NVIMGCODEC_SAMPLING_420:
                return 6;
            case NVIMGCODEC_SAMPLING_440:
                return 5;
            case NVIMGCODEC_SAMPLING_411:
                return 4;
            case NVIMGCODEC_SAMPLING_410:
                return 3;
            case NVIMGCODEC_SAMPLING_GRAY:
                return 2;
            case NVIMGCODEC_SAMPLING_410V:
            default:
                return 1;
            }
        };
        for (int i = 0; i < batch_size; i++) {
            nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            code_streams_[i]->getImageInfo(&cs_image_info);
            subsampling_score_[i] = subsamplingScore(cs_image_info.chroma_subsampling);
            area_[i] = cs_image_info.plane_info[0].height * cs_image_info.plane_info[0].width;
        }

        // Sort in descending order
        std::sort(curr_order_.begin(), curr_order_.end(), [&](int lhs, int rhs) {
            if (subsampling_score_[lhs] != subsampling_score_[rhs])
                return subsampling_score_[lhs] > subsampling_score_[rhs]; // descending order
            if (area_[lhs] != area_[rhs])
                return area_[lhs] > area_[rhs]; // descending order
            else
                return lhs < rhs; // ascending order
        });
    }

    nvimgcodecProcessingStatus_t ImageGenericDecoder::canProcessImpl(Entry& sample, ProcessorEntry* processor, int tid) noexcept {
        NVIMGCODEC_LOG_TRACE(this->logger_, tid << ": " << processor->id_ << " canDecode #" << sample.sample_idx);
        nvimgcodecProcessingStatus_t status;
        try {
            processor->instance_->canDecode(
                sample.getImageDesc(), sample.code_stream->getCodeStreamDesc(),
                static_cast<const nvimgcodecDecodeParams_t*>(sample.params), &status, tid);
            NVIMGCODEC_LOG_DEBUG(
                this->logger_, tid << ": " << processor->id_ << " canDecode #" << sample.sample_idx << " returned " << status);
            return status;
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(logger_, "Exception during canDecode: " << e.what());
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
    }

    bool ImageGenericDecoder::processImpl(Entry& sample, int tid) noexcept {
        try {
            sample.should_copy = allocateTempBuffers(sample, tid);
            auto decode_ret =
                sample.processor->instance_->decode(sample.getImageDesc(), sample.code_stream->getCodeStreamDesc(),
                                                    static_cast<const nvimgcodecDecodeParams_t*>(sample.params), tid);
            assert(sample.status != NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
            bool decode_success = decode_ret && sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

            if (sample.should_copy && decode_success) {
                nvtx3::scoped_range marker{"copyToOutputBuffer #" + std::to_string(sample.sample_idx)};
                copyToOutputBuffer(sample.orig_image_info, sample.image_info, tid);
            }
            return decode_success;
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(logger_, "Exception during decode: " << e.what());
            sample.status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
            return false;
        }
    }

    bool ImageGenericDecoder::processBatchImpl(ProcessorEntry& processor) noexcept {
        try {
            nvtx3::scoped_range marker{processor.id_ + " decodeBatch"};
            NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ << " decodeBatch");
            int tid = num_threads_; // the last entry in per_thread_ is the main thread
            auto& t = per_thread_[tid];
            if (t.pinned_buffers.size() < (batched_processed_.size() * 2)) {
                t.pinned_buffers.resize(batched_processed_.size() * 2);
                t.device_buffers.resize(batched_processed_.size() * 2);
            }
            assert(t.pinned_buffers.size() >= (batched_processed_.size() * 2));
            for (auto& sample : batched_processed_) {
                sample->should_copy = allocateTempBuffers(*sample, tid);
            }
            // All samples in this submission share the same params (set together
            // in initState()), so any sample carries the correct pointer.
            assert(!batched_processed_.empty());
            auto* batch_params = static_cast<const nvimgcodecDecodeParams_t*>(batched_processed_.front()->params);
            auto ret = processor.instance_->decodeBatch(
                batched_image_descs_.data(), batched_code_stream_descs_.data(), batch_params, batched_processed_.size(), 0);
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return false;

            // Now handle the copies after decode
            for (auto* sample : batched_processed_) {
                if (ret == NVIMGCODEC_STATUS_SUCCESS && sample->status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                    if (sample->should_copy) {
                        nvtx3::scoped_range marker{"copyToOutputBuffer " + std::to_string(sample->sample_idx)};
                        copyToOutputBuffer(sample->orig_image_info, sample->image_info, tid);
                    }
                }
            }
            return true;
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(logger_, "Exception during decodeBatch: " << e.what());
            for (auto* sample : batched_processed_) {
                sample->status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
            return false;
        }
    }

    bool ImageGenericDecoder::allocateTempBuffers(Entry& sample, int tid) {
        auto backend_kind = sample.processor->backend_kind_;
        const auto& orig_info = sample.orig_image_info;
        auto& info = sample.image_info;
        bool h2d = orig_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE &&
                   backend_kind == NVIMGCODEC_BACKEND_KIND_CPU_ONLY;
        bool d2h = orig_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST && backend_kind != NVIMGCODEC_BACKEND_KIND_CPU_ONLY;

        assert(num_threads_ + 1 == per_thread_.size());
        assert(tid >= 0 && tid <= static_cast<int>(num_threads_));
        auto& t = per_thread_[tid];
        auto& s = per_stream_[t.stream_idx];
        auto& pinned_buffers = t.pinned_buffers;
        auto& device_buffers = t.device_buffers;
        auto& pinned_buffer_idx = t.pinned_buffer_idx;
        auto& device_buffer_idx = t.device_buffer_idx;

        // restore original stride in case we don't need to copy
        for (unsigned int c = 0; c < info.num_planes; ++c) {
            info.plane_info[c].row_stride = orig_info.plane_info[c].row_stride;
        }
        info.buffer = orig_info.buffer;
        info.buffer_kind = orig_info.buffer_kind;

        if (!h2d && !d2h)
            return false;

        info.buffer = nullptr;
        for (unsigned int c = 0; c < info.num_planes; ++c) {
            auto& plane_info = info.plane_info[c];
            size_t bpp = TypeSize(plane_info.sample_type);
            plane_info.row_stride = static_cast<size_t>(plane_info.width) * bpp * plane_info.num_channels;
        }

        if (h2d) {
            nvtx3::scoped_range marker{"allocateTempBuffers h2d"};
            assert(pinned_buffers.size() > pinned_buffer_idx);
            auto& pinned_buffer = pinned_buffers[pinned_buffer_idx];
            pinned_buffer_idx = (pinned_buffer_idx + 1) % pinned_buffers.size();
            assert(pinned_buffer.size == 0 || pinned_buffer.stream == s.stream); // sanity check
            CHECK_CUDA(cudaEventSynchronize(t.event));                           // wait for the previous work to complete
            pinned_buffer.resize(GetBufferSize(info), s.stream);
            CHECK_CUDA(cudaEventRecord(t.event, s.stream));
            CHECK_CUDA(cudaEventSynchronize(t.event)); // wait for the allocation to complete
            info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            info.buffer = pinned_buffer.data;
            assert(GetBufferSize(info) == pinned_buffer.size);
            assert(info.cuda_stream == pinned_buffer.stream);
        } else if (d2h) {
            nvtx3::scoped_range marker{"allocateTempBuffers d2h"};
            assert(device_buffers.size() > device_buffer_idx);
            auto& device_buffer = device_buffers[device_buffer_idx];
            device_buffer_idx = (device_buffer_idx + 1) % device_buffers.size();
            device_buffer.resize(GetBufferSize(info), s.stream);
            info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
            info.buffer = device_buffer.data;
            assert(GetBufferSize(info) == device_buffer.size);
            assert(info.cuda_stream == device_buffer.stream);
        } else {
            assert(false); // should not happen
            return false;
        }
        return true;
    }

    void ImageGenericDecoder::copyToOutputBuffer(const nvimgcodecImageInfo_t& output_info, const nvimgcodecImageInfo_t& info, int tid) {
        nvtx3::scoped_range marker{"copyToOutputBuffer"};
        bool d2h = output_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST &&
                   info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        bool h2d = output_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE &&
                   info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        assert((static_cast<int>(d2h) + static_cast<int>(h2d)) == 1);
        (void)h2d;
        auto copy_direction = d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        auto& t = per_thread_[tid];
        auto& s = per_stream_[t.stream_idx];

        // input is always continuous (we specify it in that way in allocateTempBuffers());
        // CopyImage collapses to a single cudaMemcpyAsync when both sides are
        // contiguous and falls back to per-plane cudaMemcpy2DAsync otherwise.
        NVIMGCODEC_LOG_DEBUG(logger_, "CopyImage " << (d2h ? "D2H" : "H2D") << " stream=" << s.stream);
        CopyImage(output_info, info, copy_direction, s.stream);
        CHECK_CUDA(cudaEventRecord(t.event, s.stream)); // record event for the next iteration to wait for
        if (d2h) {
            CHECK_CUDA(cudaStreamSynchronize(s.stream));
        }
    }

} // namespace nvimgcodec
