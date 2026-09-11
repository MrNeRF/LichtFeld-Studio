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
#define NOMINMAX
#include "cuda_decoder.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <library_types.h>
#include <memory>
#include <numeric>
#include <nvimgcodec.h>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <vector>

#include "errors_handling.h"
#include "log.h"
#include "nvjpeg_utils.h"
#include "type_convert.h"

#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/device_buffer.h"
#include "imgproc/image_info_checks.h"
#include "imgproc/out_of_bound_roi_fill.h"
#include "imgproc/region_orientation.h"
#include "imgproc/type_utils.h"

#define DEFAULT_GPU_HYBRID_HUFFMAN_THRESHOLD 1000u * 1000u

#if WITH_DYNAMIC_NVJPEG_ENABLED
#include "dynlink/dynlink_nvjpeg.h"
#else
#define nvjpegIsSymbolAvailable(T) (true)
#endif

namespace nvjpeg {

    struct Decoder {
        Decoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
                const char* options = nullptr);
        ~Decoder();

        nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const;
        static nvimgcodecStatus_t static_get_metadata(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) {
            try {
                XM_CHECK_NULL(decoder);
                auto handle = reinterpret_cast<Decoder*>(decoder);
                return handle->getMetadata(code_stream, metadata, metadata_count);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
            }
        }

        nvimgcodecProcessingStatus_t canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
                                               const nvimgcodecDecodeParams_t* params, int thread_idx);
        static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
                                                              const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx) {
            try {
                XM_CHECK_NULL(decoder);
                auto handle = reinterpret_cast<Decoder*>(decoder);
                return handle->canDecode(image, code_stream, params, thread_idx);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
        }

        nvimgcodecStatus_t decode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
                                  const nvimgcodecDecodeParams_t* params, int thread_idx);
        static nvimgcodecStatus_t static_decode_sample(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
                                                       const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx) {
            try {
                XM_CHECK_NULL(decoder);
                auto handle = reinterpret_cast<Decoder*>(decoder);
                return handle->decode(image, code_stream, params, thread_idx);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                ;
            }
        }

        static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder) {
            try {
                XM_CHECK_NULL(decoder);
                Decoder* handle = reinterpret_cast<Decoder*>(decoder);
                delete handle;
            } catch (const NvJpegException& e) {
                return e.nvimgcodecStatus();
            }
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        void parseOptions(const char* options);

        const char* plugin_id_;
        nvjpegHandle_t handle_;
        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcodecFrameworkDesc_t* framework_;

        struct ParseState {
            std::optional<uint64_t> parsed_stream_id_ = std::nullopt;
            nvjpegJpegStream_t nvjpeg_stream_ = nullptr;
        };

        struct DecoderData {
            nvjpegJpegDecoder_t decoder = nullptr;
            nvjpegJpegState_t state = nullptr;
        };

        // Set of resources per-thread.
        // Some of them are double-buffered, so that we can simultaneously decode
        // the host part of the next sample, while the GPU part of the previous
        // is still consuming the data from the previous iteration pinned buffer.
        struct PerThreadResources {
            // double-buffered
            struct Page {
                // indexing via nvjpegBackend_t (NVJPEG_BACKEND_GPU_HYBRID and NVJPEG_BACKEND_HYBRID)
                std::array<DecoderData, 3> decoder_data;
                nvjpegBufferPinned_t pinned_buffer_ = nullptr;
                ParseState parse_state_;
            };
            std::array<Page, 2> pages_;
            int current_page_idx = 0;
            cudaEvent_t event_ = nullptr;
            nvjpegBufferDevice_t device_buffer_ = nullptr;
            // helper buffer is currently used for I_YCC/I_YUV decode format.
            nvimgcodec::DeviceBuffer helper_device_buffer_;

            PerThreadResources(const nvimgcodecExecutionParams_t* exec_params)
                : helper_device_buffer_(exec_params ? exec_params->device_allocator : nullptr) {}
        };
        std::vector<PerThreadResources> per_thread_;

        const nvimgcodecExecutionParams_t* exec_params_;
        NvjpegVersion nvjpeg_version_;
        size_t gpu_hybrid_huffman_threshold_ = DEFAULT_GPU_HYBRID_HUFFMAN_THRESHOLD;
        bool preallocate_buffers_ = true;
        // Default true: matches libjpeg's chroma upsampling behaviour.
        bool fancy_upsampling_ = true;
        bool enable_roi_fancy_upsampling_ = true;
        unsigned int nvjpeg_extra_flags_ = 0;
        std::optional<size_t> device_mem_padding_;
        std::optional<size_t> pinned_mem_padding_;
    };

    NvJpegCudaDecoderPlugin::NvJpegCudaDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
        : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg",
                        NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, static_create, Decoder::static_destroy, Decoder::static_get_metadata, Decoder::static_can_decode,
                        Decoder::static_decode_sample, nullptr, nullptr},
          framework_(framework) {
    }

    nvimgcodecDecoderDesc_t* NvJpegCudaDecoderPlugin::getDecoderDesc() {
        return &decoder_desc_;
    }

    void Decoder::parseOptions(const char* options) {
        gpu_hybrid_huffman_threshold_ = DEFAULT_GPU_HYBRID_HUFFMAN_THRESHOLD;
        enable_roi_fancy_upsampling_ = nvjpeg_version_ >= ROI_FANCY_UPSAMPLING_FIX_VERSION;
        std::istringstream iss(options ? options : "");
        std::string token;
        while (std::getline(iss, token, ' ')) {
            std::string::size_type colon = token.find(':');
            std::string::size_type equal = token.find('=');
            if (colon == std::string::npos || equal == std::string::npos || colon > equal)
                continue;
            std::string module = token.substr(0, colon);
            if (module != "" && module != "nvjpeg_cuda_decoder")
                continue;
            std::string option = token.substr(colon + 1, equal - colon - 1);
            std::string value_str = token.substr(equal + 1);

            std::istringstream value(value_str);
            if (option == "hybrid_huffman_threshold") {
                value >> gpu_hybrid_huffman_threshold_;
            } else if (option == "device_memory_padding") {
                size_t padding_value = 0;
                value >> padding_value;
                device_mem_padding_ = padding_value;
            } else if (option == "host_memory_padding") {
                size_t padding_value = 0;
                value >> padding_value;
                pinned_mem_padding_ = padding_value;
            } else if (option == "preallocate_buffers") {
                value >> preallocate_buffers_;
            } else if (option == "fancy_upsampling") {
                value >> fancy_upsampling_;
            } else if (option == "enable_roi_fancy_upsampling") {
                value >> enable_roi_fancy_upsampling_;
            } else if (option == "extra_flags") {
                value >> nvjpeg_extra_flags_;
            }
        }
    }

    Decoder::Decoder(
        const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
        : plugin_id_(plugin_id), device_allocator_{nullptr, nullptr, nullptr}, pinned_allocator_{nullptr, nullptr, nullptr}, framework_(framework), exec_params_(exec_params), nvjpeg_version_(get_nvjpeg_version()) {
        if (!nvjpeg_version_) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to get nvJPEG version");
            throw std::runtime_error("Failed to get nvJPEG version");
        }

        parseOptions(options);

        if (!device_mem_padding_.has_value() && exec_params_->device_allocator && exec_params_->device_allocator->device_mem_padding != 0)
            device_mem_padding_ = exec_params_->device_allocator->device_mem_padding;
        if (!pinned_mem_padding_.has_value() && exec_params_->pinned_allocator && exec_params_->pinned_allocator->pinned_mem_padding != 0)
            pinned_mem_padding_ = exec_params_->pinned_allocator->pinned_mem_padding;

        bool use_nvjpeg_create_ex_v2 = false;
        if (nvjpegIsSymbolAvailable("nvjpegCreateExV2")) {
            if (exec_params_->device_allocator && exec_params_->device_allocator->device_malloc &&
                exec_params_->device_allocator->device_free) {
                device_allocator_.dev_ctx = exec_params_->device_allocator->device_ctx;
                device_allocator_.dev_malloc = exec_params_->device_allocator->device_malloc;
                device_allocator_.dev_free = exec_params_->device_allocator->device_free;
            } else {
                device_mem_padding_ = 0;
            }

            if (exec_params_->pinned_allocator && exec_params_->pinned_allocator->pinned_malloc &&
                exec_params_->pinned_allocator->pinned_free) {
                pinned_allocator_.pinned_ctx = exec_params_->pinned_allocator->pinned_ctx;
                pinned_allocator_.pinned_malloc = exec_params_->pinned_allocator->pinned_malloc;
                pinned_allocator_.pinned_free = exec_params_->pinned_allocator->pinned_free;
            } else {
                pinned_mem_padding_ = 0;
            }
            use_nvjpeg_create_ex_v2 =
                device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free;
        }

        unsigned int nvjpeg_flags = get_nvjpeg_flags(nvjpeg_version_, fancy_upsampling_, nvjpeg_extra_flags_);

        if (fancy_upsampling_) {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "Fancy upsampling enabled; performance may be worse compared to simple upsampling");
        }

        if (use_nvjpeg_create_ex_v2) {
            XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, nvjpeg_flags, &handle_));
        } else {
            XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, nvjpeg_flags, &handle_));
        }

        if (device_mem_padding_.has_value() && device_mem_padding_.value() > 0) {
            XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(device_mem_padding_.value(), handle_));
        }
        if (pinned_mem_padding_.has_value() && pinned_mem_padding_.value() > 0) {
            XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(pinned_mem_padding_.value(), handle_));
        }

        auto executor = exec_params_->executor;
        int num_threads = executor->getNumThreads(executor->instance) + 1; // +1 for the caller thread, which can also run decoding

        per_thread_.reserve(num_threads);
        for (int i = 0; i < num_threads; i++) {
            per_thread_.emplace_back(exec_params_);
            auto& res = per_thread_.back();
            const int npages = res.pages_.size();

            XM_CHECK_CUDA(cudaEventCreateWithFlags(&res.event_, cudaEventDisableTiming));

            for (int page_idx = 0; page_idx < npages; page_idx++) {
                auto& p = res.pages_[page_idx];
                for (auto backend : {NVJPEG_BACKEND_HYBRID, NVJPEG_BACKEND_GPU_HYBRID}) {
                    auto& decoder = p.decoder_data[backend].decoder;
                    auto& state = p.decoder_data[backend].state;
                    XM_CHECK_NVJPEG(nvjpegDecoderCreate(handle_, backend, &decoder));
                    XM_CHECK_NVJPEG(nvjpegDecoderStateCreate(handle_, decoder, &state));
                }
                if (pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free) {
                    XM_CHECK_NVJPEG(nvjpegBufferPinnedCreateV2(handle_, &pinned_allocator_, &p.pinned_buffer_));
#if NVJPEG_BUFFER_RESIZE_API
                    if (preallocate_buffers_ && pinned_mem_padding_ > 0) {
                        if (nvjpegIsSymbolAvailable("nvjpegBufferPinnedResize")) {
                            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                                                 "Preallocating pinned buffer (thread#" << i << " page#" << page_idx
                                                                                        << ") size=" << pinned_mem_padding_.value());
                            XM_CHECK_NVJPEG(nvjpegBufferPinnedResize(p.pinned_buffer_, pinned_mem_padding_.value(), res.stream_));
                        } else {
                            NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "nvjpegBufferPinnedResize not available. Skip preallocation");
                        }
                    }
#endif
                } else {
                    XM_CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle_, nullptr, &p.pinned_buffer_));
                }
                XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &p.parse_state_.nvjpeg_stream_));
            }
            if (device_allocator_.dev_malloc && device_allocator_.dev_free) {
                XM_CHECK_NVJPEG(nvjpegBufferDeviceCreateV2(handle_, &device_allocator_, &res.device_buffer_));
#if NVJPEG_BUFFER_RESIZE_API
                if (preallocate_buffers_ && pinned_mem_padding_ > 0) {
                    if (nvjpegIsSymbolAvailable("nvjpegBufferDeviceResize")) {
                        NVIMGCODEC_LOG_DEBUG(
                            framework_, plugin_id_, "Preallocating device buffer (thread#" << i << ") size=" << device_mem_padding_.value());
                        XM_CHECK_NVJPEG(nvjpegBufferDeviceResize(res.device_buffer_, device_mem_padding_.value(), res.stream_));
                    } else {
                        NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "nvjpegBufferDeviceResize not available. Skip preallocation");
                    }
                }
#endif
            } else {
                XM_CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle_, nullptr, &res.device_buffer_));
            }
        }
    }

    nvimgcodecStatus_t NvJpegCudaDecoderPlugin::create(
        nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_create");
            XM_CHECK_NULL(decoder);
            XM_CHECK_NULL(exec_params);
            if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;

            *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new Decoder(plugin_id_, framework_, exec_params, options));
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg decoder:" << e.info());
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t NvJpegCudaDecoderPlugin::static_create(
        void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        if (!instance) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }

        NvJpegCudaDecoderPlugin* handle = reinterpret_cast<NvJpegCudaDecoderPlugin*>(instance);
        return handle->create(decoder, exec_params, options);
    }

    Decoder::~Decoder() {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_destroy");

            for (auto& res : per_thread_) {
                const int npages = res.pages_.size();
                for (int page_idx = 0; page_idx < npages; page_idx++) {
                    auto& p = res.pages_[page_idx];
                    if (p.parse_state_.nvjpeg_stream_) {
                        XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(p.parse_state_.nvjpeg_stream_));
                    }
                    if (p.pinned_buffer_) {
                        XM_NVJPEG_LOG_DESTROY(nvjpegBufferPinnedDestroy(p.pinned_buffer_));
                    }
                    for (auto& decoder_data : p.decoder_data) {
                        if (decoder_data.state) {
                            XM_NVJPEG_LOG_DESTROY(nvjpegJpegStateDestroy(decoder_data.state));
                        }
                        if (decoder_data.decoder) {
                            XM_NVJPEG_LOG_DESTROY(nvjpegDecoderDestroy(decoder_data.decoder));
                        }
                    }
                }
                if (res.device_buffer_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegBufferDeviceDestroy(res.device_buffer_));
                }
                if (res.event_) {
                    XM_CUDA_LOG_DESTROY(cudaEventDestroy(res.event_));
                }
            }
            per_thread_.clear();

            if (handle_)
                XM_NVJPEG_LOG_DESTROY(nvjpegDestroy(handle_));
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg decoder - " << e.info());
        }
    }
    nvimgcodecStatus_t Decoder::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    nvimgcodecProcessingStatus_t Decoder::canDecode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int tid) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode ");
            XM_CHECK_NULL(image);
            XM_CHECK_NULL(code_stream);
            XM_CHECK_NULL(params);

            nvimgcodecJpegImageInfo_t cs_jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), nullptr};
            nvimgcodecImageInfo_t cs_image_info{
                NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&cs_jpeg_image_info)};

            auto ret = code_stream->getImageInfo(code_stream->instance, &cs_image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

            bool is_jpeg = strcmp(cs_image_info.codec_name, "jpeg") == 0;
            if (!is_jpeg) {
                return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            }

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

            static const std::set<nvimgcodecJpegEncoding_t> encodings_{NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT,
                                                                       NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN};
            bool supported_encoding = encodings_.find(cs_jpeg_image_info.encoding) != encodings_.end();
            if (!supported_encoding) {
                return NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
            }

            nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
            static const std::set<nvimgcodecColorSpec_t> supported_color_space{NVIMGCODEC_COLORSPEC_UNCHANGED, NVIMGCODEC_COLORSPEC_SRGB,
                                                                               NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC, NVIMGCODEC_COLORSPEC_CMYK, NVIMGCODEC_COLORSPEC_YCCK};
            if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
                status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }

            static const std::set<nvimgcodecChromaSubsampling_t> supported_css{NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422,
                                                                               NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY,
                                                                               NVIMGCODEC_SAMPLING_410V};
            if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }

            static const std::set<nvimgcodecSampleFormat_t> supported_sample_format{
                NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED,
                NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
                NVIMGCODEC_SAMPLEFORMAT_P_RGB,
                NVIMGCODEC_SAMPLEFORMAT_I_RGB,
                NVIMGCODEC_SAMPLEFORMAT_P_BGR,
                NVIMGCODEC_SAMPLEFORMAT_I_BGR,
                NVIMGCODEC_SAMPLEFORMAT_P_Y,
                NVIMGCODEC_SAMPLEFORMAT_I_Y,
                NVIMGCODEC_SAMPLEFORMAT_P_YUV,
                NVIMGCODEC_SAMPLEFORMAT_I_YUV,
            };
            if (supported_sample_format.find(image_info.sample_format) == supported_sample_format.end()) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }

            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_YUV || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV) {
                if (cs_image_info.color_spec != NVIMGCODEC_COLORSPEC_SYCC) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "YUV/YCC decoding is only supported if image is encoded into YCC.");
                }
            }

            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV) {
                if (cs_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_444) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "YUV/YCC decoding is only supported if image is encoded without subsampling (444).");
                }
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
                    if (fancy_upsampling_ && !enable_roi_fancy_upsampling_ &&
                        cs_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_444 &&
                        cs_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY) {
                        NVIMGCODEC_LOG_WARNING(framework_, plugin_id_,
                                               "ROI decode with fancy upsampling may produce wrong results on image edge (1 pixel wide) with nvJPEG < 13.2 (available in CTK 13.3)."
                                               " To enable anyway, set enable_roi_fancy_upsampling=1");
                        status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                    }
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

            return status;
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg can decode - " << e.info());
        }
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }

    nvimgcodecStatus_t Decoder::decode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
                                       const nvimgcodecDecodeParams_t* params, int thread_idx) {
        assert(code_stream->io_stream);
        void* encoded_stream_data_raw = nullptr;
        uint8_t* encoded_stream_data = nullptr;
        size_t encoded_stream_data_size = 0;
        if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data_raw, 0, encoded_stream_data_size) !=
            NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        encoded_stream_data = static_cast<uint8_t*>(encoded_stream_data_raw);
        assert(encoded_stream_data != nullptr);
        assert(encoded_stream_data_size > 0);

        PerThreadResources& t = per_thread_[thread_idx];
        t.current_page_idx = (t.current_page_idx + 1) % 2;
        int page_idx = t.current_page_idx;
        auto& p = t.pages_[page_idx];

        try {
            XM_CHECK_NULL(image);
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            nvjpegDecodeParams_t nvjpeg_params_;
            XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &nvjpeg_params_));
            std::unique_ptr<std::remove_pointer<nvjpegDecodeParams_t>::type, decltype(&nvjpegDecodeParamsDestroy)> nvjpeg_params(
                nvjpeg_params_, &nvjpegDecodeParamsDestroy);
            int num_channels = std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
            // For grayscale, use P_Y for nvJPEG decoding (it will work for both P_Y and I_Y requests since they're equivalent for single-channel)
            auto sample_format = num_channels == 1 ? NVIMGCODEC_SAMPLEFORMAT_P_Y : image_info.sample_format;
            nvjpegOutputFormat_t nvjpeg_format = nvimgcodec_to_nvjpeg_format(sample_format);
            XM_CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(nvjpeg_params.get(), nvjpeg_format));
            int allow_cmyk = (image_info.color_spec != NVIMGCODEC_COLORSPEC_UNCHANGED) &&
                             (image_info.color_spec != NVIMGCODEC_COLORSPEC_CMYK) && ((image_info.color_spec != NVIMGCODEC_COLORSPEC_YCCK));
            XM_CHECK_NVJPEG(nvjpegDecodeParamsSetAllowCMYK(nvjpeg_params.get(), allow_cmyk));

            if (params->apply_exif_orientation) {
                nvjpegExifOrientation_t orientation = nvimgcodec_to_nvjpeg_orientation(image_info.orientation);

                // This is a workaround for a known bug in nvjpeg.
                if (nvjpeg_version_ < NvjpegVersion(12, 2, 0)) {
                    if (orientation == NVJPEG_ORIENTATION_ROTATE_90)
                        orientation = NVJPEG_ORIENTATION_ROTATE_270;
                    else if (orientation == NVJPEG_ORIENTATION_ROTATE_270)
                        orientation = NVJPEG_ORIENTATION_ROTATE_90;
                }

                if (orientation == NVJPEG_ORIENTATION_UNKNOWN) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }

                if (orientation != NVJPEG_ORIENTATION_NORMAL) {
                    if (!nvjpegIsSymbolAvailable("nvjpegDecodeParamsSetExifOrientation")) {
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
                        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                    }
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Setting up EXIF orientation " << orientation);
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsSetExifOrientation(nvjpeg_params.get(), orientation));
                }
            }

            nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
            ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream information");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            ret = code_stream->getImageInfo(code_stream->instance, &cs_image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            // dimensions of image that will be decode by nvjpeg (excluding oob roi)
            size_t decoded_image_width = image_info.plane_info[0].width;
            size_t decoded_image_height = image_info.plane_info[0].height;
            bool has_roi = false;
            bool has_oob_roi = false;

            if (codestream_info.code_stream_view) {
                // nvjpeg's ROI primitive operates in the same coordinate space as the decode
                // output: when apply_exif_orientation is on, that is display (post-orientation)
                // coords, so we pass the user-supplied region directly. is_region_out_of_bounds
                // and the clipping bounds therefore use the *effective* image dims that match
                // the region's coord space.
                const auto& region = codestream_info.code_stream_view->region;
                const auto [eff_w, eff_h] = nvimgcodec::oriented_dims(
                    cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height,
                    params->apply_exif_orientation ? image_info.orientation
                                                   : nvimgcodec::kIdentityOrientation);

                if (region.ndim != 0) {
                    assert(region.ndim == 2);
                    has_roi = true;
                    has_oob_roi = nvimgcodec::is_region_out_of_bounds(region, eff_w, eff_h);

                    if (eff_h > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                        eff_w > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Image dimensions exceeds int32, nvjpeg ROI decode is not supported.");
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                    }

                    int roi_y_begin = region.start[0];
                    int roi_x_begin = region.start[1];
                    int roi_y_end = region.end[0];
                    int roi_x_end = region.end[1];

                    NVIMGCODEC_LOG_DEBUG(
                        framework_,
                        plugin_id_,
                        "Input ROI: y=[" << roi_y_begin << ", " << roi_y_end << "); x=[" << roi_x_begin << ", " << roi_x_end << ")");

                    roi_y_begin = std::max(0, roi_y_begin);
                    roi_x_begin = std::max(0, roi_x_begin);
                    roi_y_end = std::min(static_cast<int>(eff_h), roi_y_end);
                    roi_x_end = std::min(static_cast<int>(eff_w), roi_x_end);

                    NVIMGCODEC_LOG_DEBUG(
                        framework_,
                        plugin_id_,
                        "ROI After clipping: y=[" << roi_y_begin << ", " << roi_y_end << "); x=[" << roi_x_begin << ", " << roi_x_end << ")");

                    if (roi_x_end < roi_x_begin || roi_y_end < roi_y_begin) {
                        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "invalid ROI");
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED);
                        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                    }

                    decoded_image_width = roi_x_end - roi_x_begin;
                    decoded_image_height = roi_y_end - roi_y_begin;
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params.get(), roi_x_begin, roi_y_begin, decoded_image_width, decoded_image_height));
                } else {
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params.get(), 0, 0, -1, -1));
                }
            } else {
                XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params.get(), 0, 0, -1, -1));
            }

            {
                nvtx3::scoped_range marker{"nvjpegJpegStreamParse"};
                XM_CHECK_NVJPEG(nvjpegJpegStreamParse(handle_, encoded_stream_data, encoded_stream_data_size,
                                                      false, false, p.parse_state_.nvjpeg_stream_));
            }

            nvjpegJpegEncoding_t jpeg_encoding;
            nvjpegJpegStreamGetJpegEncoding(p.parse_state_.nvjpeg_stream_, &jpeg_encoding);

            int is_gpu_hybrid_supported = -1;                    // zero means is supported
            if (jpeg_encoding == NVJPEG_ENCODING_BASELINE_DCT) { // gpu hybrid is not supported for progressive
                XM_CHECK_NVJPEG(nvjpegDecoderJpegSupported(p.decoder_data[NVJPEG_BACKEND_GPU_HYBRID].decoder, p.parse_state_.nvjpeg_stream_,
                                                           nvjpeg_params.get(), &is_gpu_hybrid_supported));
            }

            bool is_gpu_hybrid = decoded_image_width * decoded_image_height > gpu_hybrid_huffman_threshold_ && is_gpu_hybrid_supported == 0;
            auto& decoder_data = is_gpu_hybrid ? p.decoder_data[NVJPEG_BACKEND_GPU_HYBRID] : p.decoder_data[NVJPEG_BACKEND_HYBRID];
            auto& decoder = decoder_data.decoder;
            auto& state = decoder_data.state;

            XM_CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(state, p.pinned_buffer_));

            {
                nvtx3::scoped_range marker{"nvjpegDecodeJpegHost (is_gpu_hybrid=" + std::to_string(is_gpu_hybrid) + ")"};
                XM_CHECK_NVJPEG(nvjpegDecodeJpegHost(handle_, decoder, state, nvjpeg_params.get(), p.parse_state_.nvjpeg_stream_));
            }

            // Image info that will be used for nvjpeg decoding.
            // nvJPEG can only decode yuv to planar, we need to decode to helper buffer and then convert to interleaved
            // for other cases it will be just equal to image info provided by user
            auto decoded_image_info = image_info;
            if (sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV) {
                assert(nvjpeg_format == NVJPEG_OUTPUT_YUV);

                decoded_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_YUV;
                decoded_image_info.num_planes = image_info.plane_info[0].num_channels;
                for (unsigned int c = 0; c < decoded_image_info.num_planes; ++c) {
                    auto& plane_info = decoded_image_info.plane_info[c];
                    plane_info = image_info.plane_info[0];
                    plane_info.num_channels = 1;
                    plane_info.row_stride = (static_cast<size_t>(plane_info.width) * nvimgcodec::TypeSize(plane_info.sample_type));
                }
                assert(nvimgcodec::GetBufferSize(decoded_image_info) == nvimgcodec::GetImageSize(decoded_image_info));
                t.helper_device_buffer_.resize(nvimgcodec::GetBufferSize(decoded_image_info), image_info.cuda_stream);
                decoded_image_info.buffer = t.helper_device_buffer_.data;
            }

            nvjpegImage_t nvjpeg_image;
            unsigned char* ptr = reinterpret_cast<unsigned char*>(decoded_image_info.buffer);

            for (uint32_t c = 0; c < decoded_image_info.num_planes; ++c) {
                nvjpeg_image.channel[c] = ptr;
                nvjpeg_image.pitch[c] = decoded_image_info.plane_info[c].row_stride;
                ptr += nvjpeg_image.pitch[c] * decoded_image_info.plane_info[c].height;

                if (has_roi) {
                    int roi_y_begin = codestream_info.code_stream_view->region.start[0];
                    int roi_x_begin = codestream_info.code_stream_view->region.start[1];

                    if (roi_y_begin < 0) {
                        nvjpeg_image.channel[c] += (-roi_y_begin) * nvjpeg_image.pitch[c];
                    }

                    if (roi_x_begin < 0) {
                        size_t bytes_per_sample = decoded_image_info.plane_info[c].sample_type >> 11;
                        nvjpeg_image.channel[c] += (-roi_x_begin) * bytes_per_sample * decoded_image_info.plane_info[c].num_channels;
                    }
                }
            }

            // Waits for GPU stage from previous iteration (on this thread)
            XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

            XM_CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(state, t.device_buffer_));

            XM_CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle_, decoder, state, p.parse_state_.nvjpeg_stream_, image_info.cuda_stream));

            {
                nvtx3::scoped_range marker{"nvjpegDecodeJpegDevice"};
                XM_CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle_, decoder, state, &nvjpeg_image, image_info.cuda_stream));
            }

            if (sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV) { // nvJPEG can only decode yuv to planar, we need to convert to interleaved
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "convert from planar to interleaved");

                try {
                    nvimgcodec::LaunchConvertNormKernel(image_info, decoded_image_info, image_info.cuda_stream);
                } catch (std::runtime_error& e) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not convert from planar to interleaved - " << e.what());
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }

            if (has_oob_roi) {
                nvtx3::scoped_range marker{"fill_out_of_bounds_region"};
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "fill_out_of_bounds_region");
                // The output buffer (and the user-given region) is in display coords when
                // apply_exif_orientation is on. Pass the matching dims so OOB pixels are
                // identified relative to the same coordinate system as the region.
                const auto [eff_w, eff_h] = nvimgcodec::oriented_dims(
                    cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height,
                    params->apply_exif_orientation ? image_info.orientation
                                                   : nvimgcodec::kIdentityOrientation);
                try {
                    nvimgcodec::fill_out_of_bounds_region(
                        image_info, eff_w, eff_h,
                        codestream_info.code_stream_view->region);
                } catch (std::runtime_error& e) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not fill out of bounds ROI - " << e.what());
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }
            // this captures the state of image_info.cuda_stream in the cuda event t.event_
            XM_CHECK_CUDA(cudaEventRecord(t.event_, image_info.cuda_stream));

            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg code stream - " << e.info());
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_SUCCESS;
        }
    }

} // namespace nvjpeg
