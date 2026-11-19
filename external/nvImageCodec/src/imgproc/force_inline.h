/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace nvimgcodec {

#ifndef NVIMGCODEC_FORCEINLINE
#if defined(_MSC_VER)
#define NVIMGCODEC_FORCEINLINE __forceinline
#elif defined(__GNUC__) && __GNUC__ >= 4
#define NVIMGCODEC_FORCEINLINE __attribute__((always_inline)) inline
#else
#define NVIMGCODEC_FORCEINLINE inline
#endif
#endif

} // namespace nvimgcodec