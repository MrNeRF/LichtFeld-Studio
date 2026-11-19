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

#pragma once

#include <atomic>

namespace nvimgcodec {

// Thread-safe, lock-free one-shot tripwire: false until Trip() is called,
// then permanently true. Exposed mainly so ProcessExitFlag's own semantics
// can be unit-tested independently of std::atexit/IsProcessExiting() below.
class ProcessExitFlag
{
  public:
    bool IsSet() const noexcept { return flag_.load(std::memory_order_relaxed); }
    void Trip() noexcept { flag_.store(true, std::memory_order_relaxed); }

  private:
    std::atomic<bool> flag_{false};
};

// True once std::atexit has fired for the shared library this call is
// compiled into. Extensions (nvJPEG, nvJPEG2000, nvTIFF, ...) live in
// separately dlopen'd .so files that the plugin framework may unload while
// some decoder/encoder objects are still alive; near process exit, the CUDA
// driver or the backing native codec library may already be tearing down.
// Destructors that call into such libraries should check this first and
// skip the teardown calls (intentionally leaking, bounded to one-time per
// object) once it is true, since calling in at that point can segfault.
//
// Registers its own std::atexit handler lazily on first call, so simply
// calling this function anywhere (e.g. an object's constructor) is enough
// to arm the guard.
bool IsProcessExiting() noexcept;

}  // namespace nvimgcodec
