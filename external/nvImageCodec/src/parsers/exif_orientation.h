
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
#include <vector>

namespace nvimgcodec {

enum class ExifOrientation : uint16_t {
  HORIZONTAL = 1,
  MIRROR_HORIZONTAL = 2,
  ROTATE_180 = 3,
  MIRROR_VERTICAL = 4,
  MIRROR_HORIZONTAL_ROTATE_270_CW = 5,
  ROTATE_90_CW = 6,
  MIRROR_HORIZONTAL_ROTATE_90_CW = 7,
  ROTATE_270_CW = 8
};

inline nvimgcodecOrientation_t FromExifOrientation(ExifOrientation exif_orientation) {
  switch (exif_orientation) {
    case ExifOrientation::HORIZONTAL:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
    case ExifOrientation::MIRROR_HORIZONTAL:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, true, false};
    case ExifOrientation::ROTATE_180:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 180, false, false};
    case ExifOrientation::MIRROR_VERTICAL:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, true};
    case ExifOrientation::MIRROR_HORIZONTAL_ROTATE_270_CW:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 90, false, true};  // 270 CW = 90 CCW
    case ExifOrientation::ROTATE_90_CW:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 270, false, false};  // 90 CW = 270 CCW
    case ExifOrientation::MIRROR_HORIZONTAL_ROTATE_90_CW:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 270, false, true};  // 90 CW = 270 CCW
    case ExifOrientation::ROTATE_270_CW:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 90, false, false};  // 270 CW = 90 CCW
    default:
      return {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
  }
}

}  // namespace nvimgcodec