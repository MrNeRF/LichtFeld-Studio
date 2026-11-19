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

#pragma once

#include <nvimgcodec.h>
#include <string>
#include <memory>
#include "io_stream.h"
#include "iimage_parser.h"
#include "icode_stream.h"
#include "iiostream_factory.h"

namespace nvimgcodec {

class ICodecRegistry;
class ICodec;

class CodeStream : public ICodeStream
{
  public:
    explicit CodeStream(ICodecRegistry* codec_registry, std::unique_ptr<IIoStreamFactory> io_stream_factory);
    CodeStream(const CodeStream& other, const nvimgcodecCodeStreamView_t& code_stream_view); 

    CodeStream(CodeStream&& other);
    CodeStream& operator=(CodeStream&& other) noexcept;

    ~CodeStream();

    void parseFromFile(const std::string& file_name) override;
    void parseFromMem(const unsigned char* data, size_t size) override;
    void setCodeStreamView(const nvimgcodecCodeStreamView_t* view);
    void setOutputToFile(const char* file_name) override;
    void setOutputToHostMem(void* ctx, nvimgcodecResizeBufferFunc_t get_buffer_func) override;
    nvimgcodecStatus_t getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info) override;
    nvimgcodecStatus_t getImageInfo(nvimgcodecImageInfo_t* image_info) override;
    nvimgcodecStatus_t setImageInfo(const nvimgcodecImageInfo_t* image_info) override;
    std::string getCodecName() const override;
    ICodec* getCodec() const override;
    nvimgcodecIoStreamDesc_t* getInputStreamDesc() override;
    nvimgcodecCodeStreamDesc_t* getCodeStreamDesc() override;

  private:
    IImageParser* getParser();
    nvimgcodecStatus_t ensureCodeStreamInfoParsed();
    nvimgcodecStatus_t ensureParsed();
    void moveImpl(CodeStream&& other);

    nvimgcodecStatus_t read(size_t* output_size, void* buf, size_t bytes);
    nvimgcodecStatus_t write(size_t* output_size, void* buf, size_t bytes);
    nvimgcodecStatus_t putc(size_t* output_size, unsigned char ch);
    nvimgcodecStatus_t skip(size_t count);
    nvimgcodecStatus_t seek(ptrdiff_t offset, int whence);
    nvimgcodecStatus_t tell(ptrdiff_t* offset);
    nvimgcodecStatus_t size(size_t* size);
    nvimgcodecStatus_t reserve(size_t bytes);
    nvimgcodecStatus_t flush();
    nvimgcodecStatus_t map(void** addr, size_t offset, size_t size);
    nvimgcodecStatus_t unmap(void* addr, size_t size);

    static nvimgcodecStatus_t read_static(void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcodecStatus_t write_static(
        void* instance, size_t* output_size, void* buf, size_t bytes);
    static nvimgcodecStatus_t putc_static(void* instance, size_t* output_size, unsigned char ch);
    static nvimgcodecStatus_t skip_static(void* instance, size_t count);
    static nvimgcodecStatus_t seek_static(void* instance, ptrdiff_t offset, int whence);
    static nvimgcodecStatus_t tell_static(void* instance, ptrdiff_t* offset);
    static nvimgcodecStatus_t size_static(void* instance, size_t* size);
    static nvimgcodecStatus_t reserve_static(void* instance, size_t bytes);
    static nvimgcodecStatus_t flush_static(void* instance);
    static nvimgcodecStatus_t map_static(void* instance, void** addr, size_t offset, size_t size);
    static nvimgcodecStatus_t unmap_static(void* instance, void* addr, size_t size);

    static nvimgcodecStatus_t static_get_codestream_info(void* instance, nvimgcodecCodeStreamInfo_t* codestream_info);
    static nvimgcodecStatus_t static_get_image_info(void* instance, nvimgcodecImageInfo_t* result);

    ICodecRegistry* codec_registry_ = nullptr;
    std::shared_ptr<IImageParser> parser_;
    std::unique_ptr<IIoStreamFactory> io_stream_factory_;
    std::shared_ptr<IoStream> io_stream_;
    nvimgcodecIoStreamDesc_t io_stream_desc_;
    nvimgcodecCodeStreamDesc_t code_stream_desc_;

    nvimgcodecStatus_t parse_status_ = NVIMGCODEC_STATUS_NOT_INITIALIZED;
    nvimgcodecStatus_t image_info_status_ = NVIMGCODEC_STATUS_NOT_INITIALIZED;
    nvimgcodecCodeStreamView_t code_stream_view_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t), nullptr, 0, {}, 0};
    nvimgcodecCodeStreamInfo_t codestream_info_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr, nullptr, ""};
    nvimgcodecCodeStreamInfoTiffExt_t codestream_info_tiff_ext_{NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr, 0, 0, 0, {}};
    nvimgcodecTileGeometryInfo_t tile_geometry_info_{NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr};
    nvimgcodecJpegImageInfo_t jpeg_info_{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), &tile_geometry_info_};
    nvimgcodecImageInfo_t image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &jpeg_info_};
};
} // namespace nvimgcodec
