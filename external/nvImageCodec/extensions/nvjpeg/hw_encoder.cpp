/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "hw_encoder.h"
#include "errors_handling.h"
#include "imgproc/image_info_checks.h"
#include "imgproc/pinned_buffer.h"
#include "log.h"
#include "type_convert.h"
#include <cassert>
#include <cstring>
#include <future>
#include <iostream>
#include <nvimgcodec.h>
#include <nvjpeg.h>
#include <nvtx3/nvtx3.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using nvimgcodec::PinnedBuffer;

namespace nvjpeg {

    struct HwEncoderImpl {
        static constexpr int HW_MAX_DIM = 16384;

        HwEncoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
                      const char* options);
        ~HwEncoderImpl();

        static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder);

        nvimgcodecProcessingStatus_t canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
                                               const nvimgcodecEncodeParams_t* params, int thread_idx);
        static nvimgcodecProcessingStatus_t static_can_encode(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
                                                              const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx) {
            try {
                XM_CHECK_NULL(encoder);
                auto handle = reinterpret_cast<HwEncoderImpl*>(encoder);
                return handle->canEncode(code_stream, image, params, thread_idx);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
        }

        nvimgcodecStatus_t encode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
                                  const nvimgcodecEncodeParams_t* params, int thread_idx);
        static nvimgcodecStatus_t static_encode_sample(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
                                                       const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx) {
            try {
                XM_CHECK_NULL(encoder);
                auto handle = reinterpret_cast<HwEncoderImpl*>(encoder);
                return handle->encode(code_stream, image, params, thread_idx);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        }

        struct PerThreadResources {
            const nvimgcodecFrameworkDesc_t* framework_;
            const char* plugin_id_;
            nvjpegHandle_t handle_;
            const nvimgcodecExecutionParams_t* exec_params_;
            PinnedBuffer pinned_buffer_;

            cudaEvent_t event_;
            nvjpegEncoderState_t state_;
            nvjpegEncoderParams_t encode_params_ = nullptr;
            std::optional<cudaStream_t> stream_ = std::nullopt;

            PerThreadResources(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, nvjpegHandle_t handle,
                               const nvimgcodecExecutionParams_t* exec_params)
                : framework_(framework), plugin_id_(plugin_id), handle_(handle), exec_params_(exec_params), pinned_buffer_(exec_params ? exec_params->pinned_allocator : nullptr), event_(nullptr), state_(nullptr), encode_params_(nullptr) {
                XM_CHECK_CUDA(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
            }

            nvjpegEncoderState_t& state(cudaStream_t cuda_stream) {
                if (stream_ != cuda_stream || !state_) {
                    if (stream_ && stream_ != cuda_stream) { // should not really happen but just in case
                        XM_CHECK_CUDA(cudaStreamSynchronize(stream_.value()));
                        XM_NVJPEG_LOG_DESTROY(nvjpegEncoderStateDestroy(state_));
                    }
                    XM_CHECK_NVJPEG(nvjpegEncoderStateCreateWithBackend(handle_, &state_, NVJPEG_ENC_BACKEND_HARDWARE, cuda_stream));
                    stream_ = cuda_stream;
                }
                return state_;
            }

            nvjpegEncoderParams_t& params(cudaStream_t cuda_stream) {
                if (stream_ != cuda_stream || !encode_params_) {
                    if (stream_ && stream_ != cuda_stream) { // should not really happen but just in case
                        XM_CHECK_CUDA(cudaStreamSynchronize(stream_.value()));
                        XM_NVJPEG_LOG_DESTROY(nvjpegEncoderParamsDestroy(encode_params_));
                    }
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle_, &encode_params_, cuda_stream));
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encode_params_, 0, cuda_stream));
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encode_params_, NVJPEG_CSS_420, cuda_stream));
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encode_params_, NVJPEG_ENCODING_BASELINE_DCT, cuda_stream));
                    stream_ = cuda_stream;
                }
                return encode_params_;
            }

            ~PerThreadResources() {
                if (stream_ && state_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegEncoderStateDestroy(state_));
                }
                if (stream_ && encode_params_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegEncoderParamsDestroy(encode_params_));
                }
                if (event_) {
                    XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
                }
            }
        };

        struct Sample {
            nvimgcodecCodeStreamDesc_t* code_stream_;
            nvimgcodecImageDesc_t* image_;
            const nvimgcodecEncodeParams_t* params;
        };

        const nvimgcodecFrameworkDesc_t* framework_;
        const char* plugin_id_;

        nvjpegHandle_t handle_;
        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;

        std::vector<PerThreadResources> per_thread_;
        std::vector<Sample> samples_;

        const nvimgcodecExecutionParams_t* exec_params_;
        std::string options_;
    };

    NvJpegHwEncoderPlugin::NvJpegHwEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
        : encoder_desc_{
              NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC,
              sizeof(nvimgcodecEncoderDesc_t),
              NULL,
              this,
              plugin_id_,
              "jpeg",
              NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY,
              static_create,
              HwEncoderImpl::static_destroy,
              HwEncoderImpl::static_can_encode,
              HwEncoderImpl::static_encode_sample},
          framework_(framework) {
    }

    bool NvJpegHwEncoderPlugin::isPlatformSupported() {
        nvjpegHandle_t handle = nullptr;
        nvjpegStatus_t can_create_handle = nvjpegCreateSimple(&handle);
        nvjpegEncoderState_t encoder_state = nullptr;
        nvjpegStatus_t can_create_state = nvjpegEncoderStateCreateWithBackend(handle, &encoder_state, NVJPEG_ENC_BACKEND_HARDWARE, nullptr);
        bool supported = can_create_handle == NVJPEG_STATUS_SUCCESS && can_create_state == NVJPEG_STATUS_SUCCESS;
        if (encoder_state)
            XM_CHECK_NVJPEG(nvjpegEncoderStateDestroy(encoder_state));
        if (handle)
            XM_CHECK_NVJPEG(nvjpegDestroy(handle));
        return supported;
    }

    nvimgcodecEncoderDesc_t* NvJpegHwEncoderPlugin::getEncoderDesc() {
        return &encoder_desc_;
    }

    nvimgcodecProcessingStatus_t HwEncoderImpl::canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
                                                          const nvimgcodecEncodeParams_t* params, int thread_idx) {
        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_can_encode");
            XM_CHECK_NULL(code_stream);
            XM_CHECK_NULL(params);

            XM_CHECK_NULL(image);
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
            }

            nvimgcodecJpegImageInfo_t out_jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
            nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &out_jpeg_image_info};
            ret = code_stream->getImageInfo(code_stream->instance, &out_image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
            }

            if (strcmp(out_image_info.codec_name, "jpeg") != 0) {
                return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            }

            if (params->quality_type != NVIMGCODEC_QUALITY_TYPE_DEFAULT && params->quality_type != NVIMGCODEC_QUALITY_TYPE_QUALITY) {
                status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED;
            }
            if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_QUALITY) {
                if (params->quality_value < 1 || params->quality_value > 100) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED;
                }
            }

            // Hardware only support baseline jpeg
            if (out_jpeg_image_info.encoding) {
                if (out_jpeg_image_info.encoding != NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                }
            }

            // Hardware does not support grayscale images
            if (image_info.color_spec != NVIMGCODEC_COLORSPEC_SRGB &&
                image_info.color_spec != NVIMGCODEC_COLORSPEC_SYCC) {
                status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }

            // Hardware requires output subsampling 420
            if (out_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_420) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            // ... and also input subsampling 420 in case input is YUV
            if (image_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC) {
                if (image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_420) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
                }
            }

            if (image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_RGB &&
                image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_RGB &&
                image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_BGR &&
                image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_BGR &&
                image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_YUV) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }

            // nvJPEG hardware API does not support optimized Huffman for now
            auto jpeg_encode_params = static_cast<nvimgcodecJpegEncodeParams_t*>(params->struct_next);
            while (jpeg_encode_params && jpeg_encode_params->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS)
                jpeg_encode_params = static_cast<nvimgcodecJpegEncodeParams_t*>(jpeg_encode_params->struct_next);
            if (jpeg_encode_params && jpeg_encode_params->optimized_huffman) {
                status |= NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
            }

            if (image_info.num_planes == 1) {
                if (image_info.plane_info[0].num_channels != 3) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                }
            } else if (image_info.num_planes == 3) {
                for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                    if (image_info.plane_info[p].num_channels != 1) {
                        status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                    }
                }
            } else {
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
            }

            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                auto sample_type = image_info.plane_info[p].sample_type;
                if (sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                }
            }
            if (!nvimgcodec::check_planes_consistency(framework_, plugin_id_, image_info)) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }

            // Hardware only support images with even dimensions at most 16384 x 16384
            uint32_t w = image_info.plane_info[0].width;
            uint32_t h = image_info.plane_info[0].height;
            if (w > HW_MAX_DIM || h > HW_MAX_DIM || (w % 2) || (h % 2)) {
                status |= NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED;
            }

            // Check for tile geometry - JPEG encoder does not support tiling
            nvimgcodecTileGeometryInfo_t* tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(params->struct_next);
            while (tile_geometry && tile_geometry->struct_type != NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO)
                tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(tile_geometry->struct_next);
            if (tile_geometry) {
                if (tile_geometry->num_tiles_x != 0 || tile_geometry->num_tiles_y != 0 ||
                    tile_geometry->tile_width != 0 || tile_geometry->tile_height != 0) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Tiling is not supported with JPEG encoder.");
                    status |= NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED;
                }
            }
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg can encode - " << e.info());
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
        return status;
    }

    HwEncoderImpl::HwEncoderImpl(
        const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
        : framework_(framework), plugin_id_(plugin_id), device_allocator_{nullptr, nullptr, nullptr}, pinned_allocator_{nullptr, nullptr, nullptr}, exec_params_(exec_params), options_(options) {
        if (exec_params->device_allocator && exec_params->device_allocator->device_malloc && exec_params->device_allocator->device_free) {
            device_allocator_.dev_ctx = exec_params->device_allocator->device_ctx;
            device_allocator_.dev_malloc = exec_params->device_allocator->device_malloc;
            device_allocator_.dev_free = exec_params->device_allocator->device_free;
        }

        if (exec_params->pinned_allocator && exec_params->pinned_allocator->pinned_malloc && exec_params->pinned_allocator->pinned_free) {
            pinned_allocator_.pinned_ctx = exec_params->pinned_allocator->pinned_ctx;
            pinned_allocator_.pinned_malloc = exec_params->pinned_allocator->pinned_malloc;
            pinned_allocator_.pinned_free = exec_params->pinned_allocator->pinned_free;
        }

        if (device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free) {
            XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, 0, &handle_));
        } else {
            XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, 0, &handle_));
        }

        if (exec_params->device_allocator && (exec_params->device_allocator->device_mem_padding != 0)) {
            XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(exec_params->device_allocator->device_mem_padding, handle_));
        }
        if (exec_params->pinned_allocator && (exec_params->pinned_allocator->pinned_mem_padding != 0)) {
            XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(exec_params->pinned_allocator->pinned_mem_padding, handle_));
        }

        auto executor = exec_params_->executor;
        int num_threads = executor->getNumThreads(executor->instance);

        per_thread_.reserve(num_threads);
        while (per_thread_.size() < static_cast<size_t>(num_threads)) {
            per_thread_.emplace_back(framework, plugin_id_, handle_, exec_params_);
        }
    }

    nvimgcodecStatus_t NvJpegHwEncoderPlugin::create(
        nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_create_encoder");
            XM_CHECK_NULL(encoder);
            XM_CHECK_NULL(exec_params);
            if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;

            *encoder = reinterpret_cast<nvimgcodecEncoder_t>(new HwEncoderImpl(plugin_id_, framework_, exec_params, options));
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg encoder - " << e.info());
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t NvJpegHwEncoderPlugin::static_create(
        void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        if (!instance) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }

        NvJpegHwEncoderPlugin* handle = reinterpret_cast<NvJpegHwEncoderPlugin*>(instance);
        return handle->create(encoder, exec_params, options);
    }

    HwEncoderImpl::~HwEncoderImpl() {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_destroy_encoder");
            per_thread_.clear();
            if (handle_)
                XM_NVJPEG_LOG_DESTROY(nvjpegDestroy(handle_));
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg encoder - " << e.info());
        }
    }

    nvimgcodecStatus_t HwEncoderImpl::static_destroy(nvimgcodecEncoder_t encoder) {
        try {
            XM_CHECK_NULL(encoder);
            HwEncoderImpl* impl = reinterpret_cast<HwEncoderImpl*>(encoder);
            delete impl;
        } catch (const NvJpegException& e) {
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t HwEncoderImpl::encode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
                                             const nvimgcodecEncodeParams_t* params, int thread_idx) {
        try {
            XM_CHECK_NULL(image);
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            nvimgcodec::warn_if_custom_precision_unsupported(framework_, plugin_id_, image_info);

            auto& t = per_thread_[thread_idx];

            nvimgcodecJpegImageInfo_t out_jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
            nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &out_jpeg_image_info};
            ret = code_stream->getImageInfo(code_stream->instance, &out_image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

            auto& state = t.state(image_info.cuda_stream);
            auto& encode_params = t.params(image_info.cuda_stream);

            auto nvjpeg_format = nvimgcodec_to_nvjpeg_format(image_info.sample_format);
            if (nvjpeg_format < NVJPEG_OUTPUT_UNCHANGED || nvjpeg_format > NVJPEG_OUTPUT_FORMAT_MAX) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported sample format.");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            nvjpegImage_t input_image;
            unsigned char* ptr = device_buffer;
            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                input_image.channel[p] = ptr;
                input_image.pitch[p] = image_info.plane_info[p].row_stride;
                ptr += input_image.pitch[p] * image_info.plane_info[p].height;
            }

            // only NVIMGCODEC_QUALITY_TYPE_DEFAULT and NVIMGCODEC_QUALITY_TYPE_QUALITY are allowed in canEncode
            int quality = params->quality_value + 0.5f; // round as nvjpeg accept only integer quality
            if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_DEFAULT) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - setting default quality");
                quality = 75;
            }
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - quality: " << quality);
            XM_CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encode_params, quality, image_info.cuda_stream));

            nvjpegInputFormat_t input_format = static_cast<nvjpegInputFormat_t>(nvjpeg_format);
            XM_CHECK_NVJPEG(nvjpegEncode(handle_, state, encode_params, &input_image, NVJPEG_CSS_420, input_format,
                                         image_info.plane_info[0].width, image_info.plane_info[0].height, image_info.cuda_stream));

            XM_CHECK_CUDA(cudaEventRecord(t.event_, image_info.cuda_stream));
            XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

            size_t length;
            XM_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle_, state, NULL, &length, image_info.cuda_stream));

            t.pinned_buffer_.resize(length, image_info.cuda_stream);
            XM_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
                handle_, state, static_cast<uint8_t*>(t.pinned_buffer_.data), &length, image_info.cuda_stream));

            XM_CHECK_CUDA(cudaStreamSynchronize(image_info.cuda_stream));

            nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
            size_t output_size;
            io_stream->reserve(io_stream->instance, length);
            io_stream->seek(io_stream->instance, 0, SEEK_SET);
            io_stream->write(io_stream->instance, &output_size, t.pinned_buffer_.data, t.pinned_buffer_.size);
            io_stream->flush(io_stream->instance);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode jpeg code stream - " << e.info());
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

} // namespace nvjpeg
