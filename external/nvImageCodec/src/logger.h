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

#pragma once

#include <algorithm>
#include <nvimgcodec.h>
#include <algorithm>
#include <string>
#include <vector>

#include "default_debug_messenger.h"
#include "idebug_messenger.h"
#include "ilogger.h"

namespace nvimgcodec {

class Logger : public ILogger
{
  public:
    Logger(const std::string& name, IDebugMessenger* messenger = nullptr)
        : name_(name)
    {
        if (messenger != nullptr)
            messengers_.push_back(messenger);
    }

    static ILogger* get_default()
    {
        static DefaultDebugMessenger default_debug_messenger;
        static Logger instance("nvimgcodec", &default_debug_messenger);

        return &instance;
    }

    void log(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const std::string& message) override
    {
        nvimgcodecDebugMessageData_t data{NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA, sizeof(nvimgcodecDebugMessageData_t), nullptr,
            message.c_str(), 0, nullptr, name_.c_str(), 0};

        log(message_severity, message_category, &data);
    }

    void log(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* data) override
    {
        for (auto dbgmsg : messengers_) {
            if ((dbgmsg->getDesc()->message_severity & message_severity) && (dbgmsg->getDesc()->message_category & message_category)) {
                dbgmsg->getDesc()->user_callback(message_severity, message_category, data, dbgmsg->getDesc()->user_data);
            }
        }
    }

    void registerDebugMessenger(IDebugMessenger* messenger) override
    {
        auto it = std::find(messengers_.begin(), messengers_.end(), messenger);
        if (it == messengers_.end()) {
            messengers_.push_back(messenger);
        }
    }

    void unregisterDebugMessenger(IDebugMessenger* messenger) override
    {
        auto it = std::find(messengers_.begin(), messengers_.end(), messenger);
        if (it != messengers_.end()) {
            messengers_.erase(it);
        }
    }

  private:
    std::vector<IDebugMessenger*> messengers_;
    std::string name_;
};

} //namespace nvimgcodec
