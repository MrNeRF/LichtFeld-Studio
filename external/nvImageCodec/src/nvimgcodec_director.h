
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
#include <atomic>
#include <mutex>

#include "code_stream.h"
#include "codec_registry.h"
#include "debug_messenger.h"
#include "default_debug_messenger.h"
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "log.h"
#include "logger.h"
#include "plugin_framework.h"

namespace nvimgcodec {

class IDebugMessenger;

class NvImgCodecDirector
{
  public:
    struct DefaultDebugMessengerManager
    {
        DefaultDebugMessengerManager(ILogger* logger, const nvimgcodecInstanceCreateInfo_t* create_info)
            : logger_(logger)
        {
            if (create_info->create_debug_messenger) {
                if (create_info->debug_messenger_desc) {
                    dbg_messenger_ = std::make_unique<DebugMessenger>(create_info->debug_messenger_desc);
                } else {
                    dbg_messenger_ = std::make_unique<DefaultDebugMessenger>(create_info->message_severity, create_info->message_category);
                }

                logger_->registerDebugMessenger(dbg_messenger_.get());
            }
        };
        ~DefaultDebugMessengerManager()
        {
            if (dbg_messenger_) {
                logger_->unregisterDebugMessenger(dbg_messenger_.get());
            }
        };
        ILogger* logger_;
        std::unique_ptr<IDebugMessenger> dbg_messenger_;
    };

    explicit NvImgCodecDirector(const nvimgcodecInstanceCreateInfo_t* create_info);
    ~NvImgCodecDirector();

    std::unique_ptr<CodeStream> createCodeStream();
    std::unique_ptr<ImageGenericDecoder> createGenericDecoder(const nvimgcodecExecutionParams_t* exec_params, const char* options);
    std::unique_ptr<ImageGenericEncoder> createGenericEncoder(const nvimgcodecExecutionParams_t* exec_params, const char* options);
    void registerDebugMessenger(IDebugMessenger* messenger);
    void unregisterDebugMessenger(IDebugMessenger* messenger);
    void retainRegistryClient();
    void releaseRegistryClient();
    bool hasRegistryClients() const;

    Logger logger_;
    DefaultDebugMessengerManager default_debug_messenger_manager_;
    CodecRegistry codec_registry_;
    std::atomic_size_t registry_client_count_{0};
    std::recursive_mutex registry_mutex_;
    PluginFramework plugin_framework_;
};

} // namespace nvimgcodec
