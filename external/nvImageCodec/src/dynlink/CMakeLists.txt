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

DETERMINE_GCC_SYSTEM_INCLUDE_DIRS("c++" "${CMAKE_CXX_COMPILER}" "${CMAKE_CXX_FLAGS}" INFERED_COMPILER_INCLUDE)

# transform a list of paths into a list of include directives
set(DEFAULT_COMPILER_INCLUDE)
foreach(incl_dir ${INFERED_COMPILER_INCLUDE})
    set(DEFAULT_COMPILER_INCLUDE "${DEFAULT_COMPILER_INCLUDE} -I${incl_dir}")
endforeach(incl_dir)
separate_arguments(DEFAULT_COMPILER_INCLUDE UNIX_COMMAND  "${DEFAULT_COMPILER_INCLUDE}")

find_path(CUDA_MAIN_INCLUDE_DIR NAMES cuda.h HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} NO_DEFAULT_PATH)

if (WITH_DYNAMIC_CUDA)
    set(CUDA_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_cuda_gen.cpp")
    add_custom_command(
        OUTPUT ${CUDA_GENERATED_STUB}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Cuda --
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/cuda.json" ${CUDA_GENERATED_STUB}
        "${CUDA_MAIN_INCLUDE_DIR}/cuda.h" "-I${CUDA_MAIN_INCLUDE_DIR}"

        # for some reason QNX fails with 'too many errors emitted' is this is not set
        "-ferror-limit=0"
        ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
        "${CUDA_MAIN_INCLUDE_DIR}/cuda.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/cuda.json"
        COMMENT "Running cuda.h stub generator"
        VERBATIM)

    set_source_files_properties(${CUDA_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_cuda STATIC dynlink_cuda.cpp ${CUDA_GENERATED_STUB})

    set_target_properties(dynlink_cuda
        PROPERTIES POSITION_INDEPENDENT_CODE
        ON
        )
endif()

if(WITH_DYNAMIC_NVJPEG)
    set(NVJPEG_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvjpeg_gen.cpp")
    add_custom_command(
        OUTPUT ${NVJPEG_GENERATED_STUB}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Nvjpeg --
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvjpeg.json" ${NVJPEG_GENERATED_STUB}
        "${NVJPEG_INCLUDE}/nvjpeg.h" "-I${NVJPEG_INCLUDE}"
        # for some reason QNX fails with 'too many errors emitted' is this is not set
        "-ferror-limit=0"
        ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
        "${NVJPEG_INCLUDE}/nvjpeg.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvjpeg.json"
        COMMENT "Running nvjpeg.h stub generator"
        VERBATIM)

    set_source_files_properties(${NVJPEG_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_nvjpeg STATIC dynlink_nvjpeg.cpp ${NVJPEG_GENERATED_STUB})
    set_target_properties(dynlink_nvjpeg PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


if(WITH_DYNAMIC_NVJPEG2K)
    set(NVJPEG2K_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvjpeg2k_gen.cpp")
    add_custom_command(
        OUTPUT ${NVJPEG2K_GENERATED_STUB}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Nvjpeg2k --
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvjpeg2k.json" ${NVJPEG2K_GENERATED_STUB}
        "${NVJPEG2K_INCLUDE}/nvjpeg2k.h" "-I${NVJPEG2K_INCLUDE}" "-I${CUDA_MAIN_INCLUDE_DIR}"
        # for some reason QNX fails with 'too many errors emitted' is this is not set
        "-ferror-limit=0"
        ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
        "${NVJPEG2K_INCLUDE}/nvjpeg2k.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvjpeg2k.json"
        COMMENT "Running nvjpeg2k.h stub generator"
        VERBATIM)

    set_source_files_properties(${NVJPEG2K_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_nvjpeg2k STATIC dynlink_nvjpeg2k.cpp ${NVJPEG2K_GENERATED_STUB})
    set_target_properties(dynlink_nvjpeg2k PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(WITH_DYNAMIC_NVTIFF)
    set(NVTIFF_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvtiff_gen.cpp")
    add_custom_command(
        OUTPUT ${NVTIFF_GENERATED_STUB}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Nvtiff --
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvtiff.json" ${NVTIFF_GENERATED_STUB}
        "${NVTIFF_INCLUDE}/nvtiff.h" "-I${NVTIFF_INCLUDE}" "-I${CUDA_MAIN_INCLUDE_DIR}"
        # for some reason QNX fails with 'too many errors emitted' is this is not set
        "-ferror-limit=0"
        ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
        "${NVTIFF_INCLUDE}/nvtiff.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvtiff.json"
        COMMENT "Running nvtiff.h stub generator"
        VERBATIM)

    set_source_files_properties(${NVTIFF_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_nvtiff STATIC dynlink_nvtiff.cpp ${NVTIFF_GENERATED_STUB})
    set_target_properties(dynlink_nvtiff PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
