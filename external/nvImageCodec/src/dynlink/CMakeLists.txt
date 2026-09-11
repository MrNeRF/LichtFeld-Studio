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

# No Python/clang dependencies needed - using pre-generated stub files

if (WITH_DYNAMIC_CUDA)
    # Use pre-generated stub file (no Python/clang dependency required)
    set(CUDA_GENERATED_STUB "${CMAKE_CURRENT_SOURCE_DIR}/dynlink_cuda_gen.cpp")
    add_library(dynlink_cuda STATIC dynlink_cuda.cpp ${CUDA_GENERATED_STUB})

    set_target_properties(dynlink_cuda
        PROPERTIES POSITION_INDEPENDENT_CODE
        ON
        )
endif()

if(WITH_DYNAMIC_NVJPEG)
    # Use pre-generated stub file (no Python/clang dependency required)
    set(NVJPEG_GENERATED_STUB "${CMAKE_CURRENT_SOURCE_DIR}/dynlink_nvjpeg_gen.cpp")
    add_library(dynlink_nvjpeg STATIC dynlink_nvjpeg.cpp ${NVJPEG_GENERATED_STUB})
    set_target_properties(dynlink_nvjpeg PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


if(WITH_DYNAMIC_NVJPEG2K)
    # Use pre-generated stub file (no Python/clang dependency required)
    set(NVJPEG2K_GENERATED_STUB "${CMAKE_CURRENT_SOURCE_DIR}/dynlink_nvjpeg2k_gen.cpp")
    add_library(dynlink_nvjpeg2k STATIC dynlink_nvjpeg2k.cpp ${NVJPEG2K_GENERATED_STUB})
    set_target_properties(dynlink_nvjpeg2k PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(WITH_DYNAMIC_NVTIFF)
    set(NVTIFF_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvtiff_gen.cpp")
    add_custom_command(
        OUTPUT ${NVTIFF_GENERATED_STUB}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Nvtiff --
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvtiff.json" ${NVTIFF_GENERATED_STUB}
        "${NVTIFF_INCLUDE}/nvtiff.h" "-I${NVTIFF_INCLUDE}" "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        # for some reason QNX fails with 'too many errors emitted' is this is not set
        "-ferror-limit=0"
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
        "${NVTIFF_INCLUDE}/nvtiff.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvtiff.json"
        COMMENT "Running nvtiff.h stub generator"
        VERBATIM)

    set_source_files_properties(${NVTIFF_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_nvtiff STATIC dynlink_nvtiff.cpp ${NVTIFF_GENERATED_STUB})
    set_target_properties(dynlink_nvtiff PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
