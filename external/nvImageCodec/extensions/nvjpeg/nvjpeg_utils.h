/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvjpeg.h>
#include "utils/library_version.h"

namespace nvjpeg {

using NvjpegVersion = nvimgcodec::LibraryVersion;

constexpr NvjpegVersion LOSSLESS_JPEG_FIX_VERSION {13, 0, 2};

// Get the current nvJPEG library version
NvjpegVersion get_nvjpeg_version();

constexpr NvjpegVersion ROI_FANCY_UPSAMPLING_FIX_VERSION {13, 2, 0};

unsigned int get_nvjpeg_flags(const NvjpegVersion& version, bool fancy_upsampling = true, unsigned int extra_flags = 0);

}
