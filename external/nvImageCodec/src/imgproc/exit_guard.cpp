/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "imgproc/exit_guard.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace nvimgcodec {

namespace {

ProcessExitFlag& GetProcessExitFlag()
{
    static auto* flag = new ProcessExitFlag();
    return *flag;
}

}  // namespace

bool IsProcessExiting() noexcept
{
    // Must never throw: called from destructors (some implicitly noexcept),
    // so a thrown exception here could call std::terminate() instead of
    // just failing to detect process exit. If registration fails (in
    // practice, essentially never), fall back to always reporting "not
    // exiting" -- the same risk profile callers had before this guard
    // existed.
    static std::once_flag register_once;
    try {
        std::call_once(register_once, []() {
            if (std::atexit([]() { GetProcessExitFlag().Trip(); }) != 0) {
                std::fprintf(stderr, "nvimgcodec: failed to register process-exit guard atexit handler\n");
            }
        });
        return GetProcessExitFlag().IsSet();
    } catch (...) {
        // std::call_once may throw std::system_error, and the first call may
        // fail to allocate the flag. Either way, report "not exiting".
        return false;
    }
}

}  // namespace nvimgcodec
