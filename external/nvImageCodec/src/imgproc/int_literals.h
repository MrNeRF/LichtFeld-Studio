/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdint>
#include <type_traits>
#include "imgproc/host_dev.h"

namespace nvimgcodec {

using size_t = std::size_t;
using ssize_t = std::make_signed_t<size_t>;

namespace literal {

NVIMGCODEC_HOST_DEV
constexpr ssize_t operator ""_z(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr size_t operator ""_uz(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr size_t operator ""_zu(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr int8_t operator ""_i8(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr uint8_t operator ""_u8(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr int16_t operator ""_i16(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr uint16_t operator ""_u16(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr int32_t operator ""_i32(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr uint32_t operator ""_u32(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr int64_t operator ""_i64(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

NVIMGCODEC_HOST_DEV
constexpr uint64_t operator ""_u64(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

}  // namespace literal
using namespace literal;  // NOLINT
}  // namespace nvimgcodec
