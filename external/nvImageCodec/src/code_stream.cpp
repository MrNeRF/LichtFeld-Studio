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

#include "code_stream.h"
#include "codec.h"
#include "codec_registry.h"
#include "image_parser.h"
#include "imgproc/exception.h"
#include "log.h"
#include <atomic>
#include <cstring>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <string>

namespace nvimgcodec {

    static std::atomic<uint64_t> s_id(0);

    CodeStream::CodeStream(ICodecRegistry* codec_registry, std::unique_ptr<IIoStreamFactory> io_stream_factory)
        : codec_registry_(codec_registry),
          parser_(nullptr),
          io_stream_factory_(std::move(io_stream_factory)),
          io_stream_(nullptr),
          io_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC, sizeof(nvimgcodecIoStreamDesc_t), nullptr, this,
                          s_id.fetch_add(1, std::memory_order_relaxed), read_static, write_static, putc_static, skip_static, seek_static, tell_static,
                          size_static, reserve_static, flush_static, map_static, unmap_static},
          code_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr, this, &io_stream_desc_,
                            static_get_codestream_info, static_get_image_info} {
    }

    CodeStream::CodeStream(const CodeStream& other, const nvimgcodecCodeStreamView_t* code_stream_view)
        : codec_registry_{other.codec_registry_},
          parser_{other.parser_} // For now we assume there is the same codec so the same parser. It can change in the future.
          ,
          io_stream_factory_{nullptr},
          io_stream_{other.io_stream_},
          io_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC, sizeof(nvimgcodecIoStreamDesc_t), nullptr, this, other.io_stream_desc_.id,
                          read_static, write_static, putc_static, skip_static, seek_static, tell_static, size_static, reserve_static, flush_static,
                          map_static, unmap_static},
          code_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr, this, &io_stream_desc_,
                            static_get_codestream_info, static_get_image_info},
          parse_status_{NVIMGCODEC_STATUS_NOT_INITIALIZED},
          code_stream_view_(*code_stream_view),
          codestream_info_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr, &code_stream_view_, ""},
          tile_geometry_info_{NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr},
          jpeg_info_{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), &tile_geometry_info_},
          image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &jpeg_info_} {
        // code_stream_view was already dereferenced to initialize code_stream_view_;
        // only re-check the other stream's nested-region view.
        if (other.codestream_info_.code_stream_view && other.codestream_info_.code_stream_view->region.ndim > 0 &&
            code_stream_view->region.ndim > 0) {
            throw Exception(INVALID_PARAMETER, "Cannot create a sub code stream with nested regions. This is not supported.", "CodeStream::CodeStream");
        }
    }

    CodeStream::CodeStream(CodeStream&& other)
        : codec_registry_(other.codec_registry_),
          parser_(std::move(other.parser_)),
          io_stream_factory_(std::move(other.io_stream_factory_)),
          io_stream_(std::move(other.io_stream_)),
          io_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC, sizeof(nvimgcodecIoStreamDesc_t), nullptr, this, other.io_stream_desc_.id,
                          read_static, write_static, putc_static, skip_static, seek_static, tell_static, size_static, reserve_static, flush_static,
                          map_static, unmap_static},
          code_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr, this, &io_stream_desc_,
                            static_get_codestream_info, static_get_image_info},
          parse_status_(other.parse_status_),
          code_stream_view_(other.code_stream_view_),
          codestream_info_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr, other.codestream_info_.code_stream_view ? &code_stream_view_ : nullptr, ""},
          tile_geometry_info_(other.tile_geometry_info_),
          jpeg_info_({NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), &tile_geometry_info_}),
          image_info_({NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &jpeg_info_}) {
    }

    CodeStream::~CodeStream() {
    }

    IImageParser* CodeStream::getParser() {
        if (parser_ == nullptr) {
            auto parser = codec_registry_->getParser(&code_stream_desc_);
            if (!parser) {
                throw Exception(UNSUPPORTED_FORMAT_STATUS, "The encoded stream did not match any of the available format parsers",
                                "CodeStream::parse - Encoded stream parsing");
            }

            parser_ = std::move(parser);
        }
        return parser_.get();
    }

    void CodeStream::parseFromFile(const std::string& file_name) {
        io_stream_ = io_stream_factory_->createFileIoStream(file_name, false, true, false);
    }

    void CodeStream::parseFromMem(const unsigned char* data, size_t size) {
        io_stream_ = io_stream_factory_->createMemIoStream(data, size);
    }
    void CodeStream::setOutputToFile(const char* file_name) {
        io_stream_ = io_stream_factory_->createFileIoStream(file_name, false, false, true);
    }

    void CodeStream::setOutputToHostMem(void* ctx, nvimgcodecResizeBufferFunc_t resize_buffer_func) {
        io_stream_ = io_stream_factory_->createMemIoStream(ctx, resize_buffer_func);
    }

    template <typename T>
    void copy(T& dst, const T& src) {
        void* struct_next = dst.struct_next;
        dst = src;
        dst.struct_next = struct_next;
    }

    nvimgcodecStatus_t CodeStream::ensureParsed() {
        if (parse_status_ == NVIMGCODEC_STATUS_NOT_INITIALIZED) {
            IImageParser* parser = getParser();
            if (parser) {
                parse_status_ = parser->getCodeStreamInfo(&code_stream_desc_, &codestream_info_);
                if (parse_status_ == NVIMGCODEC_STATUS_SUCCESS) {
                    parse_status_ = parser->getImageInfo(&code_stream_desc_, &image_info_);
                }
            } else {
                return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
            }
        }
        return parse_status_;
    }

    nvimgcodecStatus_t CodeStream::getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info) {
        assert(codestream_info);

        if (ensureParsed() != NVIMGCODEC_STATUS_SUCCESS) {
            return parse_status_;
        }

        void* struct_next = codestream_info->struct_next;
        *codestream_info = codestream_info_;
        codestream_info->struct_next = struct_next;

        if (codestream_info->struct_next) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t CodeStream::getImageInfo(nvimgcodecImageInfo_t* image_info) {
        assert(image_info);

        if (ensureParsed() != NVIMGCODEC_STATUS_SUCCESS) {
            return parse_status_;
        }

        void* struct_next = image_info->struct_next;
        *image_info = image_info_;
        image_info->struct_next = struct_next;

        while (struct_next) {
            auto* ptr = reinterpret_cast<nvimgcodecImageInfo_t*>(struct_next);
            auto* next_struct_next = ptr->struct_next;
            switch (ptr->struct_type) {
            case NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO:
                copy(*reinterpret_cast<nvimgcodecTileGeometryInfo_t*>(ptr), tile_geometry_info_);
                break;
            case NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO:
                copy(*reinterpret_cast<nvimgcodecJpegImageInfo_t*>(ptr), jpeg_info_);
                break;
            default:
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            struct_next = next_struct_next;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t CodeStream::setImageInfo(const nvimgcodecImageInfo_t* image_info) {
        void* tmp_struct_next = image_info_.struct_next;
        image_info_ = *image_info;
        image_info_.struct_next = tmp_struct_next;

        void* struct_next = image_info->struct_next;
        while (struct_next) {
            auto* ptr = reinterpret_cast<nvimgcodecImageInfo_t*>(struct_next);
            switch (ptr->struct_type) {
            case NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO:
                copy(tile_geometry_info_, *reinterpret_cast<nvimgcodecTileGeometryInfo_t*>(ptr));
                break;
            case NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO:
                copy(jpeg_info_, *reinterpret_cast<nvimgcodecJpegImageInfo_t*>(ptr));
                break;
            default:
                assert(false);
            }
            struct_next = ptr->struct_next;
        }
        parse_status_ = NVIMGCODEC_STATUS_SUCCESS;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    std::string CodeStream::getCodecName() const {
        if (parse_status_ == NVIMGCODEC_STATUS_SUCCESS) {
            return std::string(image_info_.codec_name);
        } else {
            IImageParser* parser = const_cast<CodeStream*>(this)->getParser();
            if (parser) {
                return parser->getCodecName();
            } else {
                return "";
            }
        }
    }

    ICodec* CodeStream::getCodec() const {
        return codec_registry_->getCodecByName(getCodecName().c_str());
    }

    nvimgcodecIoStreamDesc_t* CodeStream::getInputStreamDesc() {
        return &io_stream_desc_;
    }

    nvimgcodecCodeStreamDesc_t* CodeStream::getCodeStreamDesc() {
        return &code_stream_desc_;
    }

    nvimgcodecStatus_t CodeStream::read(size_t* output_size, void* buf, size_t bytes) {
        try {
            assert(io_stream_);
            *output_size = io_stream_->read(buf, bytes);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::write(size_t* output_size, void* buf, size_t bytes) {
        try {
            assert(io_stream_);
            *output_size = io_stream_->write(buf, bytes);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }
    nvimgcodecStatus_t CodeStream::putc(size_t* output_size, unsigned char ch) {
        try {
            assert(io_stream_);
            *output_size = io_stream_->putc(ch);

            return *output_size == 1 ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_BAD_CODESTREAM;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::skip(size_t count) {
        try {
            assert(io_stream_);
            io_stream_->skip(count);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::seek(ptrdiff_t offset, int whence) {
        try {
            assert(io_stream_);
            io_stream_->seek(offset, whence);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::tell(ptrdiff_t* offset) {
        try {
            assert(io_stream_);
            *offset = io_stream_->tell();
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::size(size_t* size) {
        try {
            assert(io_stream_);
            *size = io_stream_->size();
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::reserve(size_t bytes) {
        try {
            assert(io_stream_);
            io_stream_->reserve(bytes);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::flush() {
        try {
            assert(io_stream_);
            io_stream_->flush();
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::map(void** addr, size_t offset, size_t size) {
        try {
            assert(io_stream_);
            *addr = io_stream_->map(offset, size);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::unmap(void* addr, size_t size) {
        try {
            assert(io_stream_);
            io_stream_->unmap(addr, size);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::read_static(void* instance, size_t* output_size, void* buf, size_t bytes) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->read(output_size, buf, bytes);
    }

    nvimgcodecStatus_t CodeStream::write_static(void* instance, size_t* output_size, void* buf, size_t bytes) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->write(output_size, buf, bytes);
    }

    nvimgcodecStatus_t CodeStream::putc_static(void* instance, size_t* output_size, unsigned char ch) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->putc(output_size, ch);
    }

    nvimgcodecStatus_t CodeStream::skip_static(void* instance, size_t count) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->skip(count);
    }

    nvimgcodecStatus_t CodeStream::seek_static(void* instance, ptrdiff_t offset, int whence) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->seek(offset, whence);
    }

    nvimgcodecStatus_t CodeStream::tell_static(void* instance, ptrdiff_t* offset) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->tell(offset);
    }

    nvimgcodecStatus_t CodeStream::size_static(void* instance, size_t* size) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->size(size);
    }

    nvimgcodecStatus_t CodeStream::reserve_static(void* instance, size_t bytes) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->reserve(bytes);
    }

    nvimgcodecStatus_t CodeStream::flush_static(void* instance) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->flush();
    }

    nvimgcodecStatus_t CodeStream::map_static(void* instance, void** addr, size_t offset, size_t size) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->map(addr, offset, size);
    }

    nvimgcodecStatus_t CodeStream::unmap_static(void* instance, void* addr, size_t size) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->unmap(addr, size);
    }

    nvimgcodecStatus_t CodeStream::static_get_codestream_info(void* instance, nvimgcodecCodeStreamInfo_t* codestream_info) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        handle->getCodeStreamInfo(codestream_info);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t CodeStream::static_get_image_info(void* instance, nvimgcodecImageInfo_t* result) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        handle->getImageInfo(result);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

} // namespace nvimgcodec