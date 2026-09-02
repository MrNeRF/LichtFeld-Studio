# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(nvimgcodec_imgproc_static OBJECT 
    convert_kernel_gpu.cu
    pageable_host_buffer.cpp
    pinned_buffer.cpp
    device_buffer.cpp
    exception.cpp
    out_of_bound_roi_fill.cpp
    stream_device.cpp
    color_space_conversion_impl.h
    convert.h
    convert_kernel_gpu.h
    cuda_utils.h
    dev_string.h
    device_buffer.h
    device_guard.h
    exception.h
    force_inline.h
    host_dev.h
    image_info_checks.h
    int_literals.h
    math_util.h
    out_of_bound_roi_fill.h
    pageable_host_buffer.h
    pinned_buffer.h
    region_orientation.h
    roi.h
    sample_format_utils.h
    static_switch.h
    stream_device.h
    surface.h
    tuple_helpers.h
    type_utils.h
    util.h
)
set_property(TARGET nvimgcodec_imgproc_static PROPERTY POSITION_INDEPENDENT_CODE ON)

if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \" /Zc:preprocessor \"  -Xfatbin -compress-all")
endif()
