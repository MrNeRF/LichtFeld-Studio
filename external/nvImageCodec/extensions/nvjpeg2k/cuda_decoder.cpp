/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define NOMINMAX

#include "cuda_decoder.h"
#include "error_handling.h"
#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/device_buffer.h"
#include "imgproc/image_info_checks.h"
#include "imgproc/out_of_bound_roi_fill.h"
#include "imgproc/region_orientation.h"
#include "imgproc/safe_arithmetic.h"
#include "imgproc/sample_format_utils.h"
#include "log.h"
#include "nvjpeg2k_utils.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <future>
#include <imgproc/stream_device.h>
#include <iostream>
#include <memory>
#include <nvimgcodec.h>
#include <nvjpeg2k.h>
#include <nvtx3/nvtx3.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

using nvimgcodec::DeviceBuffer;

namespace nvjpeg2k {

    struct DecoderImpl;

    struct PerThreadResources {
        const nvimgcodecFrameworkDesc_t* framework_;
        const char* plugin_id_;
        nvjpeg2kHandle_t handle_;
        nvimgcodec::DeviceBuffer device_buffer_;

        cudaEvent_t event_;
        nvjpeg2kDecodeState_t state_;
        nvjpeg2kStream_t nvjpeg2k_stream_;
        std::optional<uint64_t> parsed_stream_id_;

        PerThreadResources(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, const nvimgcodecExecutionParams_t* exec_params,
                           nvjpeg2kHandle_t handle)
            : framework_(framework), plugin_id_(plugin_id), handle_(handle), device_buffer_(exec_params ? exec_params->device_allocator : nullptr) {
            XM_CHECK_CUDA(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
            XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle_, &state_));
            XM_CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&nvjpeg2k_stream_));
        }

        ~PerThreadResources() {
            if (event_) {
                XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
            }
            if (state_) {
                XM_NVJPEG2K_LOG_DESTROY(nvjpeg2kDecodeStateDestroy(state_));
            }
            if (nvjpeg2k_stream_) {
                XM_NVJPEG2K_LOG_DESTROY(nvjpeg2kStreamDestroy(nvjpeg2k_stream_));
            }
        }
    };

    struct PerTileResources {
        cudaStream_t stream_;
        cudaEvent_t event_;
        nvjpeg2kDecodeState_t state_;
    };

    struct PerTileResourcesPool {
        std::vector<PerTileResources> res_;
        std::queue<PerTileResources*> free_;
        std::mutex mtx_;
        std::condition_variable cv_;

        size_t size() const {
            return res_.size();
        }

        PerTileResources* Acquire() {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [&]() { return !free_.empty(); });
            auto res_ptr = free_.front();
            free_.pop();
            return res_ptr;
        }

        void Release(PerTileResources* res_ptr) {
            std::lock_guard<std::mutex> lock(mtx_);
            free_.push(res_ptr);
            cv_.notify_one();
        }
    };

    struct DecoderImpl {
        DecoderImpl(
            const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options);
        ~DecoderImpl();

        static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);

        static nvimgcodecStatus_t static_get_metadata(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) {
            try {
                XM_CHECK_NULL(decoder);
                auto handle = reinterpret_cast<DecoderImpl*>(decoder);
                return handle->getMetadata(code_stream, metadata, metadata_count);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
            }
        }
        static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
                                                              const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx) {
            try {
                XM_CHECK_NULL(decoder);
                auto handle = reinterpret_cast<DecoderImpl*>(decoder);
                return handle->canDecode(image, code_stream, params, thread_idx);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
        }

        nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const;
        nvimgcodecProcessingStatus_t canDecode(
            const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);

        nvimgcodecStatus_t decode(
            const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
        static nvimgcodecStatus_t static_decode_sample(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
                                                       const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx) {
            try {
                XM_CHECK_NULL(decoder);
                auto handle = reinterpret_cast<DecoderImpl*>(decoder);
                return handle->decode(image, code_stream, params, thread_idx);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                ;
            }
        }

        void parseOptions(const char* options);

        const char* plugin_id_;
        nvjpeg2kHandle_t handle_;
        nvjpeg2kDeviceAllocatorV2_t device_allocator_;
        nvjpeg2kPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcodecFrameworkDesc_t* framework_;
        const nvimgcodecExecutionParams_t* exec_params_;
        int num_parallel_tiles_;
        std::optional<size_t> device_mem_padding_;
        std::optional<size_t> pinned_mem_padding_;

        std::vector<PerThreadResources> per_thread_;
        PerTileResourcesPool per_tile_res_;

        Nvjpeg2kVersion nvjpeg2k_version_;
    };

    NvJpeg2kDecoderPlugin::NvJpeg2kDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
        : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg2k",
                        NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create, DecoderImpl::static_destroy, DecoderImpl::static_get_metadata, DecoderImpl::static_can_decode,
                        DecoderImpl::static_decode_sample, nullptr},
          framework_(framework) {
    }

    nvimgcodecDecoderDesc_t* NvJpeg2kDecoderPlugin::getDecoderDesc() {
        return &decoder_desc_;
    }

    nvimgcodecStatus_t DecoderImpl::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    nvimgcodecProcessingStatus_t DecoderImpl::canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
                                                        const nvimgcodecDecodeParams_t* params, int thread_idx) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");

            nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

            XM_CHECK_NULL(code_stream);
            nvimgcodecTileGeometryInfo_t tile_geometry{NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr};
            nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &tile_geometry};
            auto ret = code_stream->getImageInfo(code_stream->instance, &cs_image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

            bool is_jpeg2k = strcmp(cs_image_info.codec_name, "jpeg2k") == 0;
            if (!is_jpeg2k)
                return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;

            XM_CHECK_NULL(image);
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

            if (tile_geometry.tile_offset_x > 0 || tile_geometry.tile_offset_y > 0) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Non-zero tile offsets are not supported");
                status |= NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED;
            }

            XM_CHECK_NULL(params);

            nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
            ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

            if (codestream_info.code_stream_view) {
                if (codestream_info.code_stream_view->image_idx != 0) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED;
                }

                const auto& region = codestream_info.code_stream_view->region;
                if (region.ndim == 0) {
                    // no ROI, okay
                } else if (region.ndim == 2) {
                    if (nvimgcodec::is_region_out_of_bounds_effective(region, image_info.orientation,
                                                                      cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height,
                                                                      params->apply_exif_orientation)) {
                        if (auto err_message = nvimgcodec::verify_region_fill_support(region, image_info); !err_message.empty()) {
                            status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                            NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, err_message);
                        }
                    }
                } else {
                    status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Region decoding is supported only for 2 dimensions.");
                }
            }

            static const std::set<nvimgcodecColorSpec_t> supported_color_space{
                NVIMGCODEC_COLORSPEC_UNCHANGED, NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC};
            if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
                status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_YUV) {
                static const std::set<nvimgcodecChromaSubsampling_t> supported_css{
                    NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420};
                if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
                }
            }

            static const std::set<nvimgcodecSampleFormat_t> supported_sample_format{
                NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED,
                NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
                NVIMGCODEC_SAMPLEFORMAT_P_Y,
                NVIMGCODEC_SAMPLEFORMAT_I_Y,
                NVIMGCODEC_SAMPLEFORMAT_P_RGB,
                NVIMGCODEC_SAMPLEFORMAT_I_RGB,
                NVIMGCODEC_SAMPLEFORMAT_P_YUV,
                NVIMGCODEC_SAMPLEFORMAT_I_YUV,
                NVIMGCODEC_SAMPLEFORMAT_P_RGBA,
                NVIMGCODEC_SAMPLEFORMAT_I_RGBA,
            };
            if (supported_sample_format.find(image_info.sample_format) == supported_sample_format.end()) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }

            if (image_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC && cs_image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Conversion from RGB to YCC when decoding is not supported.");
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }

            constexpr Nvjpeg2kVersion nvjpeg2k_ycc_rgb_444_version_fix = {0, 11, 0};
            if (cs_image_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC && image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB &&
                cs_image_info.chroma_subsampling == NVIMGCODEC_SAMPLING_444 && nvjpeg2k_version_ < nvjpeg2k_ycc_rgb_444_version_fix) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Conversion from YCC to RGB with 444 subsampling is not supported with nvJPEG2k < 0.11. Please update to a newer version.");
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }

            static const std::set<nvimgcodecSampleDataType_t> supported_sample_type{
                NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, NVIMGCODEC_SAMPLE_DATA_TYPE_INT16};
            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                auto sample_type = image_info.plane_info[p].sample_type;
                if (supported_sample_type.find(sample_type) == supported_sample_type.end()) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                }
            }
            if (!nvimgcodec::check_planes_consistency(framework_, plugin_id_, image_info)) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
            return status;
        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg2k can decode - " << e.info());
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
    }

    void DecoderImpl::parseOptions(const char* options) {
        num_parallel_tiles_ = 16; // default 16 tiles in parallel for all threads
        std::istringstream iss(options ? options : "");
        std::string token;
        while (std::getline(iss, token, ' ')) {
            std::string::size_type colon = token.find(':');
            std::string::size_type equal = token.find('=');
            if (colon == std::string::npos || equal == std::string::npos || colon > equal)
                continue;
            std::string module = token.substr(0, colon);
            if (module != "" && module != "nvjpeg2k_cuda_decoder")
                continue;
            std::string option = token.substr(colon + 1, equal - colon - 1);
            std::string value_str = token.substr(equal + 1);

            std::istringstream value(value_str);
            if (option == "num_parallel_tiles") {
                value >> num_parallel_tiles_;
            } else if (option == "device_memory_padding") {
                size_t padding_value = 0;
                value >> padding_value;
                device_mem_padding_ = padding_value;
            } else if (option == "host_memory_padding") {
                size_t padding_value = 0;
                value >> padding_value;
                pinned_mem_padding_ = padding_value;
            }
        }
    }

    DecoderImpl::DecoderImpl(
        const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
        : plugin_id_(id), device_allocator_{nullptr, nullptr, nullptr}, pinned_allocator_{nullptr, nullptr, nullptr}, framework_(framework), exec_params_(exec_params), nvjpeg2k_version_(get_nvjpeg2k_version()) {
        if (!nvjpeg2k_version_) {
            throw NvJpeg2kException(NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED, "Failed to get nvJPEG2000 version");
        }

        parseOptions(options);
        if (exec_params_->device_allocator && exec_params_->device_allocator->device_malloc && exec_params_->device_allocator->device_free) {
            device_allocator_.device_ctx = exec_params_->device_allocator->device_ctx;
            device_allocator_.device_malloc = exec_params_->device_allocator->device_malloc;
            device_allocator_.device_free = exec_params_->device_allocator->device_free;
        }

        if (exec_params_->pinned_allocator && exec_params_->pinned_allocator->pinned_malloc && exec_params_->pinned_allocator->pinned_free) {
            pinned_allocator_.pinned_ctx = exec_params_->pinned_allocator->pinned_ctx;
            pinned_allocator_.pinned_malloc = exec_params_->pinned_allocator->pinned_malloc;
            pinned_allocator_.pinned_free = exec_params_->pinned_allocator->pinned_free;
        }

        if (device_allocator_.device_malloc && device_allocator_.device_free && pinned_allocator_.pinned_malloc &&
            pinned_allocator_.pinned_free) {
            XM_CHECK_NVJPEG2K(nvjpeg2kCreateV2(NVJPEG2K_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, &handle_));
        } else {
            XM_CHECK_NVJPEG2K(nvjpeg2kCreateSimple(&handle_));
        }

        if (!device_mem_padding_.has_value() && exec_params_->device_allocator && exec_params_->device_allocator->device_mem_padding != 0)
            device_mem_padding_ = exec_params_->device_allocator->device_mem_padding;
        if (!pinned_mem_padding_.has_value() && exec_params_->device_allocator && exec_params_->pinned_allocator->pinned_mem_padding != 0)
            pinned_mem_padding_ = exec_params_->pinned_allocator->pinned_mem_padding;

        if (device_mem_padding_.has_value() && device_mem_padding_.value() > 0) {
            XM_CHECK_NVJPEG2K(nvjpeg2kSetDeviceMemoryPadding(device_mem_padding_.value(), handle_));
        }
        if (pinned_mem_padding_.has_value() && pinned_mem_padding_.value() > 0) {
            XM_CHECK_NVJPEG2K(nvjpeg2kSetPinnedMemoryPadding(pinned_mem_padding_.value(), handle_));
        }

        auto executor = exec_params_->executor;
        int num_threads = executor->getNumThreads(executor->instance);
        per_thread_.reserve(num_threads);
        for (int i = 0; i < num_threads; i++) {
            per_thread_.emplace_back(framework_, plugin_id_, exec_params_, handle_);
        }

        if (num_parallel_tiles_ >= num_threads) {
            per_tile_res_.res_.resize(num_parallel_tiles_);
            for (auto& tile_res : per_tile_res_.res_) {
                XM_CHECK_CUDA(cudaStreamCreateWithFlags(&tile_res.stream_, cudaStreamNonBlocking));
                XM_CHECK_CUDA(cudaEventCreateWithFlags(&tile_res.event_, cudaEventDisableTiming));
                XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle_, &tile_res.state_));
                per_tile_res_.free_.push(&tile_res);
            }
        } else {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_,
                                "num_parallel_tiles(" << num_parallel_tiles_ << ") < num_threads(" << num_threads
                                                      << "). Per-tile parallelization will be disabled");
        }
    }

    nvimgcodecStatus_t NvJpeg2kDecoderPlugin::create(
        nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_create");
            XM_CHECK_NULL(decoder);
            XM_CHECK_NULL(exec_params);
            if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_, framework_, exec_params, options));
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg2k decoder - " << e.info());
            return e.nvimgcodecStatus();
        }
    }

    nvimgcodecStatus_t NvJpeg2kDecoderPlugin::static_create(
        void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        if (!instance) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }

        auto handle = reinterpret_cast<NvJpeg2kDecoderPlugin*>(instance);
        return handle->create(decoder, exec_params, options);
    }

    DecoderImpl::~DecoderImpl() {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_destroy");

            for (auto& tile_res : per_tile_res_.res_) {
                if (tile_res.event_) {
                    XM_CUDA_LOG_DESTROY(cudaEventDestroy(tile_res.event_));
                }
                if (tile_res.stream_) {
                    XM_CUDA_LOG_DESTROY(cudaStreamDestroy(tile_res.stream_));
                }
                if (tile_res.state_) {
                    XM_NVJPEG2K_LOG_DESTROY(nvjpeg2kDecodeStateDestroy(tile_res.state_));
                }
            }

            per_thread_.clear();

            if (handle_)
                XM_CHECK_NVJPEG2K(nvjpeg2kDestroy(handle_));
        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg2k decoder - " << e.info());
        }
    }

    nvimgcodecStatus_t DecoderImpl::static_destroy(nvimgcodecDecoder_t decoder) {
        try {
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            delete handle;
        } catch (const NvJpeg2kException& e) {
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t DecoderImpl::decode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx) {
        if (!code_stream->io_stream) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "No io stream");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        void* encoded_stream_data = nullptr;
        size_t encoded_stream_data_size = 0;
        if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size) !=
            NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        assert(encoded_stream_data != nullptr);
        assert(encoded_stream_data_size > 0);

        auto& t = per_thread_[thread_idx];
        auto& per_tile_res = per_tile_res_;
        auto jpeg2k_state = t.state_;

        try {
            XM_CHECK_NULL(image);
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected buffer kind");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);
            // io_stream is null-checked at function entry, before any deref.
            if (code_stream->io_stream->id != t.parsed_stream_id_) {
                nvtx3::scoped_range marker{"nvjpegJpegStreamParse"};
                XM_CHECK_NVJPEG2K(nvjpeg2kStreamParse(handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size,
                                                      false, false, t.nvjpeg2k_stream_));
                t.parsed_stream_id_ = code_stream->io_stream->id;
            }

            nvjpeg2kImageInfo_t jpeg2k_info;
            XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(t.nvjpeg2k_stream_, &jpeg2k_info));

            nvjpeg2kImageComponentInfo_t comp;
            XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(t.nvjpeg2k_stream_, &comp, 0));
            auto bpp = comp.precision;
            auto sgn = comp.sgn;
            auto num_components = jpeg2k_info.num_components;
            if (bpp > 16) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected bitdepth");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            nvjpeg2kImage_t output_image;

            nvimgcodecSampleDataType_t out_data_type = image_info.plane_info[0].sample_type;
            size_t out_bytes_per_sample = static_cast<unsigned int>(out_data_type) >> (8 + 3);
            for (size_t p = 1; p < image_info.num_planes; p++) {
                if (image_info.plane_info[p].sample_type != out_data_type) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same data type");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }

            nvimgcodecSampleDataType_t orig_data_type;
            if (sgn) {
                if ((bpp > 8) && (bpp <= 16)) {
                    output_image.pixel_type = NVJPEG2K_INT16;
                    orig_data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
                } else {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "unsupported bit depth for a signed type. It must be 8 > bpp <= 16");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            } else {
                if (bpp <= 8) {
                    output_image.pixel_type = NVJPEG2K_UINT8;
                    orig_data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
                } else if (bpp <= 16) {
                    output_image.pixel_type = NVJPEG2K_UINT16;
                    orig_data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
                } else {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "bit depth not supported");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }
            size_t in_bytes_per_sample = static_cast<int>(orig_data_type) >> (8 + 3);
            size_t in_bits_per_sample = in_bytes_per_sample << 3;

            nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            ret = code_stream->getImageInfo(code_stream->instance, &cs_image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            bool convert_dtype =
                image_info.plane_info[0].sample_type != orig_data_type || (bpp != in_bits_per_sample && image_info.plane_info[0].precision != bpp);
            // Can decode directly to interleaved
            bool native_interleaved =
                !convert_dtype && ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB && num_components == 3) || (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED && num_components > 1) || (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGBA && num_components == 4) || (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV && num_components == 3));

            // Will decode to planar then convert
            bool convert_interleaved = !native_interleaved && (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGBA || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV);
            bool convert_gray =
                (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_Y) &&
                (cs_image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_Y && cs_image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_Y);

            bool needs_convert = convert_gray || convert_interleaved || convert_dtype;
            bool planar_subset = image_info.num_planes > 1 && image_info.num_planes < num_components;

            if (convert_dtype && out_data_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Only original dtype or conversion to uint8 is allowed");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            nvjpeg2kDecodeParams_t decode_params;
            nvjpeg2kDecodeParamsCreate(&decode_params);
            std::unique_ptr<std::remove_pointer<nvjpeg2kDecodeParams_t>::type, decltype(&nvjpeg2kDecodeParamsDestroy)> decode_params_raii(
                decode_params, &nvjpeg2kDecodeParamsDestroy);

            int rgb_output = image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB || image_info.color_spec == NVIMGCODEC_COLORSPEC_UNKNOWN ||
                             (image_info.color_spec == NVIMGCODEC_COLORSPEC_UNCHANGED && cs_image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB);
            XM_CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, rgb_output));
            XM_CHECK_NVJPEG2K(
                nvjpeg2kDecodeParamsSetOutputFormat(decode_params, native_interleaved ? NVJPEG2K_FORMAT_INTERLEAVED : NVJPEG2K_FORMAT_PLANAR));

            if (image_info.num_planes > 1 && num_components < image_info.num_planes) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of planes");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            } else if (convert_interleaved && (num_components < image_info.plane_info[0].num_channels && num_components != 1)) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of channels");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            } else if (native_interleaved && image_info.plane_info[0].num_channels != num_components) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of channels");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            for (size_t p = 1; p < image_info.num_planes; p++) {
                if (image_info.plane_info[p].sample_type != image_info.plane_info[0].sample_type) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same data type");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }
            nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
            ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream information");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            if (jpeg2k_info.image_width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                jpeg2k_info.image_height > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Image dimensions exceeds int32, nvjpeg2k decode is not supported.");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            bool has_roi = false;
            bool has_oob_roi = false;
            int roi_y_begin = 0;
            int roi_x_begin = 0;
            int roi_y_end = static_cast<int>(jpeg2k_info.image_height);
            int roi_x_end = static_cast<int>(jpeg2k_info.image_width);

            if (codestream_info.code_stream_view) {
                const auto& region = codestream_info.code_stream_view->region;
                has_roi = region.ndim != 0;
                if (has_roi) {
                    has_oob_roi = nvimgcodec::is_region_out_of_bounds(region, jpeg2k_info.image_width, jpeg2k_info.image_height);
                    roi_y_begin = region.start[0];
                    roi_x_begin = region.start[1];
                    roi_y_end = region.end[0];
                    roi_x_end = region.end[1];
                }
            }

            NVIMGCODEC_LOG_DEBUG(
                framework_,
                plugin_id_,
                "Input ROI: y=[" << roi_y_begin << ", " << roi_y_end << "); x=[" << roi_x_begin << ", " << roi_x_end << ")");

            // nvjpeg2k support only decoding of ROI in image bounds. If roi exits we will clip it to image bounds below
            int clipped_roi_y_begin = std::max(0, roi_y_begin);
            int clipped_roi_x_begin = std::max(0, roi_x_begin);
            int clipped_roi_y_end = std::min(static_cast<int>(jpeg2k_info.image_height), roi_y_end);
            int clipped_roi_x_end = std::min(static_cast<int>(jpeg2k_info.image_width), roi_x_end);
            NVIMGCODEC_LOG_DEBUG(
                framework_,
                plugin_id_,
                "ROI After clipping: y=[" << clipped_roi_y_begin << ", " << clipped_roi_y_end << "); x=["
                                          << clipped_roi_x_begin << ", " << clipped_roi_x_end << ")");

            if (clipped_roi_x_end < clipped_roi_x_begin || clipped_roi_y_end < clipped_roi_y_begin) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "invalid ROI");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            if (has_roi) {
                // we only use that if roi was specified, otherwise we just decode whole image, which will be slightly faster
                // if this function is not used
                XM_CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetDecodeArea(
                    decode_params, clipped_roi_x_begin, clipped_roi_x_end, clipped_roi_y_begin, clipped_roi_y_end));
            }

            int roi_height = roi_y_end - roi_y_begin;
            int roi_width = roi_x_end - roi_x_begin;

            if (roi_width <= 0 || roi_height <= 0 || num_components == 0) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_,
                                     "Invalid ROI or component count: roi=" << roi_width << "x" << roi_height
                                                                            << ", components=" << num_components);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            for (size_t p = 0; p < image_info.num_planes; p++) {
                if (static_cast<int>(image_info.plane_info[p].height) != roi_height ||
                    static_cast<int>(image_info.plane_info[p].width) != roi_width) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected plane info dimensions");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }

            // roi_width/roi_height/num_components/bps all derive from the codestream
            // header and are therefore attacker-controlled. The individual products
            // below are computed in size_t (because in/out_bytes_per_sample are size_t),
            // but the *result* can still overflow size_t — e.g. a SIZ marker claiming
            // image_width == image_height == 2^31 wraps planar_in_plane_nbytes and
            // produces an undersized t.device_buffer_.resize() at the needs_convert /
            // planar_subset branches, after which the decode writes overrun.
            size_t planar_in_row_nbytes = 0;
            size_t planar_in_plane_nbytes = 0;
            size_t planar_out_row_nbytes = 0;
            size_t decode_buffer_nbytes = 0;
            const size_t roi_width_sz = static_cast<size_t>(roi_width);
            const size_t roi_height_sz = static_cast<size_t>(roi_height);
            const size_t num_components_sz = static_cast<size_t>(num_components);
            if (!nvimgcodec::SafeMulSizeT(in_bytes_per_sample, roi_width_sz, planar_in_row_nbytes) ||
                !nvimgcodec::SafeMulSizeT(planar_in_row_nbytes, roi_height_sz, planar_in_plane_nbytes) ||
                !nvimgcodec::SafeMulSizeT(roi_width_sz, out_bytes_per_sample, planar_out_row_nbytes) ||
                !nvimgcodec::SafeMulSizeT(planar_in_plane_nbytes, num_components_sz, decode_buffer_nbytes)) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_,
                                     "Image dimensions overflow size_t: roi=" << roi_width << "x" << roi_height
                                                                              << ", components=" << num_components
                                                                              << ", in_bps=" << in_bytes_per_sample
                                                                              << ", out_bps=" << out_bytes_per_sample);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            uint8_t* decode_buffer = nullptr;
            std::vector<unsigned char*> decode_output(num_components);
            std::vector<size_t> pitch_in_bytes(num_components);

            if (native_interleaved) {
                assert(planar_in_row_nbytes == planar_out_row_nbytes);
                decode_buffer = device_buffer;
                decode_output[0] = decode_buffer;
                pitch_in_bytes[0] = image_info.plane_info[0].row_stride;
            } else if (needs_convert) {
                // If there are conversions needed, we decode to a temporary buffer first
                t.device_buffer_.resize(decode_buffer_nbytes, image_info.cuda_stream);
                decode_buffer = reinterpret_cast<uint8_t*>(t.device_buffer_.data);
                for (size_t p = 0; p < num_components; p++) {
                    decode_output[p] = decode_buffer + p * planar_in_plane_nbytes;
                    pitch_in_bytes[p] = planar_in_row_nbytes;
                }
            } else {
                assert(planar_in_row_nbytes == planar_out_row_nbytes);
                uint32_t p = 0;
                decode_buffer = device_buffer;
                for (; p < image_info.num_planes; ++p) {
                    decode_output[p] = device_buffer + p * image_info.plane_info[p].row_stride * roi_height;
                    pitch_in_bytes[p] = image_info.plane_info[p].row_stride;
                }

                if (planar_subset) {
                    // If there are more components than we want, we allocate temp memory for the planes we don't need
                    t.device_buffer_.resize((num_components - image_info.num_planes) * planar_in_plane_nbytes, image_info.cuda_stream);
                    for (; p < num_components; ++p) {
                        decode_output[p] = reinterpret_cast<uint8_t*>(t.device_buffer_.data) + (p - image_info.num_planes) * planar_in_plane_nbytes;
                        pitch_in_bytes[p] = planar_in_row_nbytes;
                    }
                }
            }

            // offset pixel data pointer in case we have oob ROI
            if (has_roi) {
                int num_planes = native_interleaved ? 1 : num_components;
                if (roi_x_begin < 0) {
                    size_t num_components_in_plane = native_interleaved ? num_components : 1u;
                    size_t offset = (-roi_x_begin) * num_components_in_plane * in_bytes_per_sample;
                    for (int i = 0; i < num_planes; ++i) {
                        decode_output[i] += offset;
                    }
                }

                if (roi_y_begin < 0) {
                    for (int i = 0; i < num_planes; ++i) {
                        decode_output[i] += (-roi_y_begin) * pitch_in_bytes[i];
                    }
                }
            }

            output_image.num_components = num_components;
            output_image.pixel_data = reinterpret_cast<void**>(decode_output.data());
            output_image.pitch_in_bytes = pitch_in_bytes.data();

            // Waits for GPU stage from previous iteration
            XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

            bool tiled = (jpeg2k_info.num_tiles_y > 1 || jpeg2k_info.num_tiles_x > 1);
            // If per-tile parallelization is smaller than per-thread parallelization, it is not worth to decode tile-by-tile.
            if (!tiled || per_tile_res.size() <= 1 || image_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC) {
                nvtx3::scoped_range marker{"nvjpeg2kDecodeImage"};
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpeg2kDecodeImage");
                XM_CHECK_NVJPEG2K(
                    nvjpeg2kDecodeImage(handle_, jpeg2k_state, t.nvjpeg2k_stream_, decode_params_raii.get(), &output_image, image_info.cuda_stream));
            } else {
                std::vector<uint8_t*> tile_decode_output(jpeg2k_info.num_components, nullptr);
                for (uint32_t tile_y = 0; tile_y < jpeg2k_info.num_tiles_y; tile_y++) {
                    for (uint32_t tile_x = 0; tile_x < jpeg2k_info.num_tiles_x; tile_x++) {
                        int tile_y_begin = tile_y * jpeg2k_info.tile_height;
                        int tile_y_end = std::min<int>(tile_y_begin + jpeg2k_info.tile_height, jpeg2k_info.image_height);
                        int tile_x_begin = tile_x * jpeg2k_info.tile_width;
                        int tile_x_end = std::min<int>(tile_x_begin + jpeg2k_info.tile_width, jpeg2k_info.image_width);
                        int offset_y = tile_y_begin > roi_y_begin ? tile_y_begin - roi_y_begin : 0;
                        int offset_x = tile_x_begin > roi_x_begin ? tile_x_begin - roi_x_begin : 0;
                        if (has_roi) {
                            tile_y_begin = std::max(roi_y_begin, tile_y_begin);
                            tile_x_begin = std::max(roi_x_begin, tile_x_begin);
                            tile_y_end = std::min(roi_y_end, tile_y_end);
                            tile_x_end = std::min(roi_x_end, tile_x_end);
                        }
                        if (tile_y_begin >= tile_y_end || tile_x_begin >= tile_x_end)
                            continue;

                        XM_CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetDecodeArea(decode_params, tile_x_begin, tile_x_end, tile_y_begin, tile_y_end));

                        nvjpeg2kImage_t output_tile;
                        output_tile.pixel_type = output_image.pixel_type;
                        output_tile.pitch_in_bytes = output_image.pitch_in_bytes;
                        output_tile.num_components = output_image.num_components;
                        output_tile.pixel_data = reinterpret_cast<void**>(tile_decode_output.data());
                        if (native_interleaved) {
                            output_tile.pixel_data[0] = (decode_buffer +
                                                         output_tile.pitch_in_bytes[0] * offset_y +
                                                         static_cast<size_t>(offset_x) * num_components * in_bytes_per_sample);
                        } else if (needs_convert) {
                            for (uint32_t c = 0; c < output_image.num_components; c++) {
                                output_tile.pixel_data[c] = (decode_buffer +
                                                             c * planar_in_plane_nbytes +
                                                             offset_y * planar_in_row_nbytes +
                                                             offset_x * in_bytes_per_sample);
                            }
                        } else {
                            // Decode subset of planes directly to the output
                            uint32_t c = 0;
                            for (; c < image_info.num_planes; c++) {
                                output_tile.pixel_data[c] = (decode_buffer +
                                                             c * output_image.pitch_in_bytes[c] * roi_height +
                                                             offset_y * output_image.pitch_in_bytes[c] +
                                                             offset_x * in_bytes_per_sample);
                            }

                            // Decode remaining planes to a temp buffer
                            assert(output_image.num_components == image_info.num_planes || planar_subset);
                            for (; c < output_image.num_components; c++) {
                                output_tile.pixel_data[c] = (reinterpret_cast<uint8_t*>(t.device_buffer_.data) +
                                                             (c - image_info.num_planes) * planar_in_plane_nbytes +
                                                             offset_y * planar_in_row_nbytes +
                                                             offset_x * in_bytes_per_sample);
                            }
                        }
                        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                                             "nvjpeg2kDecodeTile: y=[" << tile_y_begin << ", " << tile_y_end << "), x=[" << tile_x_begin << ", " << tile_x_end
                                                                       << ")");

                        // Acquire tile resources
                        PerTileResources* tile_res = per_tile_res.Acquire();
                        // sync with previous tile work
                        XM_CHECK_CUDA(cudaEventSynchronize(tile_res->event_));

                        // sync tile stream with user stream
                        XM_CHECK_CUDA(cudaEventRecord(tile_res->event_, image_info.cuda_stream));
                        XM_CHECK_CUDA(cudaStreamWaitEvent(tile_res->stream_, tile_res->event_));
                        {
                            auto tile_idx = tile_y * jpeg2k_info.num_tiles_x + tile_x;
                            nvtx3::scoped_range marker{"nvjpeg2kDecodeTile #" + std::to_string(tile_idx) + " @" + std::to_string((uint64_t)tile_res)};
                            XM_CHECK_NVJPEG2K(nvjpeg2kDecodeTile(handle_, tile_res->state_, t.nvjpeg2k_stream_, decode_params_raii.get(),
                                                                 tile_idx, 0, &output_tile, tile_res->stream_));
                        }
                        // sync thread stream with tile stream
                        XM_CHECK_CUDA(cudaEventRecord(tile_res->event_, tile_res->stream_));
                        XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, tile_res->event_));
                        // Release tile resources
                        per_tile_res.Release(tile_res);
                    }
                }
            }

            if (needs_convert) {
                // TODO: improve perf, right now whole roi is converted, and we could convert only clipped
                nvimgcodecImageInfo_t dec_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
                strcpy(dec_image_info.codec_name, "jpeg2k");
                dec_image_info.color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN; // not used by the converter
                dec_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
                // TODO: for now we ignore alpha channel during conversions
                dec_image_info.sample_format = cs_image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGBA ? NVIMGCODEC_SAMPLEFORMAT_P_RGB : cs_image_info.sample_format; // original (planar) sample format
                dec_image_info.orientation = image_info.orientation;
                dec_image_info.num_planes = num_components;
                for (size_t p = 0; p < dec_image_info.num_planes; p++) {
                    dec_image_info.plane_info[p].struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_PLANE_INFO;
                    dec_image_info.plane_info[p].struct_size = sizeof(nvimgcodecImagePlaneInfo_t);
                    dec_image_info.plane_info[p].struct_next = nullptr;
                    dec_image_info.plane_info[p].width = roi_width;
                    dec_image_info.plane_info[p].height = roi_height;
                    dec_image_info.plane_info[p].row_stride = pitch_in_bytes[p];
                    dec_image_info.plane_info[p].num_channels = 1;
                    dec_image_info.plane_info[p].sample_type = orig_data_type;
                    dec_image_info.plane_info[p].precision = bpp;
                }
                dec_image_info.buffer = decode_buffer;
                dec_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                dec_image_info.cuda_stream = image_info.cuda_stream;

                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "LaunchConvertNormKernel");
                try {
                    nvimgcodec::LaunchConvertNormKernel(image_info, dec_image_info, image_info.cuda_stream);
                } catch (std::runtime_error& e) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not convert image to required format - " << e.what());
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }

            if (has_oob_roi) {
                nvtx3::scoped_range marker{"fill_out_of_bounds_region"};
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "fill_out_of_bounds_region");
                try {
                    nvimgcodec::fill_out_of_bounds_region(
                        image_info,
                        jpeg2k_info.image_width, jpeg2k_info.image_height,
                        codestream_info.code_stream_view->region);
                } catch (std::runtime_error& e) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not fill out of bounds ROI - " << e.what());
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }

            // Record event on user stream to synchronize on next iteration
            XM_CHECK_CUDA(cudaEventRecord(t.event_, image_info.cuda_stream));
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg2k code stream - " << e.info());
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

} // namespace nvjpeg2k
