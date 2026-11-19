/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "imgproc/host_dev.h"
#include "imgproc/geom/box.h"

namespace nvimgcodec {

/**
 * Defines region of interest.
 * 0 dimension is interpreted along x axis (horizontal)
 * 1 dimension is interpreted along y axis (vertical)
 *
 *            image.x ->
 *          +--------------------------------+
 *          |                                |
 *          |   roi.lo    roi.x              |
 *  image.y |         +-----+                |
 *       |  |    roi.y|     |                |
 *       v  |         +-----+ roi.hi         |
 *          |                                |
 *          +--------------------------------+
 *
 * Additionally, by definition, ROI is top-left inclusive and bottom-right exclusive.
 * That means, that `Roi.lo` point is included in actual ROI and `Roi.hi` point isn't.
 */
template <int ndims>
using Roi = Box<ndims, int>;

}  // namespace nvimgcodec
