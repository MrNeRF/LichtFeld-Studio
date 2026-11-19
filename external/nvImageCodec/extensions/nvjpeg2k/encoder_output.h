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

#include <cstddef>
#include <nvimgcodec.h>

namespace nvjpeg2k {
namespace detail {

// Clears mapped_data before invoking the callback so a failed unmap is never
// retried during exception cleanup. mapped_capacity is the extent originally
// passed to map(), not the number of bytes ultimately produced by the encoder.
nvimgcodecStatus_t unmap_output(
    nvimgcodecIoStreamDesc_t* io_stream, void*& mapped_data, size_t mapped_capacity) noexcept;

// Commits an already retrieved and synchronized bitstream. If fallback_data is
// null, nvJPEG2K wrote directly to mapped_data; otherwise fallback_data contains
// produced_length bytes that must be copied through the stream interface.
nvimgcodecStatus_t finalize_output(nvimgcodecIoStreamDesc_t* io_stream, void*& mapped_data,
    size_t mapped_capacity, size_t produced_length, const void* fallback_data) noexcept;

} // namespace detail
} // namespace nvjpeg2k
