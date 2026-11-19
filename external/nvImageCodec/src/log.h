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

#include "logger.h"
#include <iostream>
#include <nvimgcodec.h>
#include <sstream>
#include <string>

namespace nvimgcodec {

#ifdef NDEBUG
#define NVIMGCODEC_SEVERITY NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO
#else
#define NVIMGCODEC_SEVERITY NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE
#endif

#define NVIMGCODEC_LOG(logger, svr, type, msg) \
    do {                                       \
        if (svr >= NVIMGCODEC_SEVERITY) {      \
            std::stringstream ss{};            \
            ss << msg;                         \
            logger->log(svr, type, ss.str());  \
        }                                      \
    } while (0)

#ifdef NDEBUG

#define NVIMGCODEC_LOG_TRACE(...)
#define NVIMGCODEC_LOG_DEBUG(...)

#else

#define NVIMGCODEC_LOG_TRACE(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_TRACE, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

#define NVIMGCODEC_LOG_DEBUG(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEBUG, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

#endif

#define NVIMGCODEC_LOG_INFO(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_INFO, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

#define NVIMGCODEC_LOG_WARNING(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_WARNING, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

#define NVIMGCODEC_LOG_ERROR(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

#define NVIMGCODEC_LOG_FATAL(logger, ...) \
    NVIMGCODEC_LOG(logger, NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_FATAL, NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_GENERAL, __VA_ARGS__)

} // namespace nvimgcodec