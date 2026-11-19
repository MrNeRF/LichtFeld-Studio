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
#include <sstream>
#include <string>

#ifdef NDEBUG
#define NVIMGCODEC_SEVERITY NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO
#else
#define NVIMGCODEC_SEVERITY NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE
#endif

#define NVIMGCODEC_LOG(framework, id, svr, type, msg)                                                                                      \
    do {                                                                                                                                   \
        if (svr >= NVIMGCODEC_SEVERITY) {                                                                                                  \
            std::stringstream ss{};                                                                                                        \
            ss << msg;                                                                                                                     \
            std::string msg_str{ss.str()};                                                                                                 \
            nvimgcodecDebugMessageData_t data{NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA, sizeof(nvimgcodecDebugMessageData_t), nullptr, \
                                              msg_str.c_str(), 0, nullptr, id, NVIMGCODEC_VER};                                            \
            framework->log(framework->instance, svr, type, &data);                                                                         \
        }                                                                                                                                  \
    } while (0)

#define NVIMGCODEC_LOG_TRACE(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_DEBUG(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_INFO(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_WARNING(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_ERROR(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
#define NVIMGCODEC_LOG_FATAL(framework, id, ...) \
    NVIMGCODEC_LOG(framework, id, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)
