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

set(EXT_LIBRARY_NAME nvjpeg2k_ext)

set(EXT_SRCS
  nvjpeg2k_ext.cpp
  cuda_encoder.cpp
  cuda_decoder.cpp
  exception.cpp
  nvjpeg2k_utils.cpp
)

set(EXT_HEADERS
  cuda_decoder.h
  cuda_encoder.h
  error_handling.h
  exception.h
  log.h
  nvjpeg2k_ext.h
  nvjpeg2k_utils.h
)

add_library(${EXT_LIBRARY_NAME}_obj OBJECT ${EXT_SRCS} ${EXT_HEADERS})
add_library(${EXT_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj> $<TARGET_OBJECTS:nvimgcodec_imgproc_static> ext_module.cpp)
add_library(${EXT_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj> $<TARGET_OBJECTS:nvimgcodec_imgproc_static>)

if (WITH_DYNAMIC_CUDA)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE dynlink_cuda)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE "-Wl,--exclude-libs,$<TARGET_FILE_NAME:dynlink_cuda>")
else()
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE CUDA::cuda_driver)
endif()

if(NOT WITH_DYNAMIC_NVJPEG2K)
  target_link_libraries(${EXT_LIBRARY_NAME}
    PUBLIC ${NVJPEG2K_LIBRARY})
else()
  target_link_libraries(${EXT_LIBRARY_NAME}
    PRIVATE dynlink_nvjpeg2k)
  target_link_libraries(${EXT_LIBRARY_NAME}
    PRIVATE "-Wl,--exclude-libs,$<TARGET_FILE_NAME:dynlink_nvjpeg2k>")
endif()

target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC
  CUDA::cudart_static)

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden -Wl,--exclude-libs,ALL")
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")
  target_link_libraries(${EXT_LIBRARY_NAME}_static PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")

  set_target_properties(${EXT_LIBRARY_NAME} ${EXT_LIBRARY_NAME}_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    NO_SONAME OFF)
else()
  # Windows: set explicit output directories for multi-config generators
  # Must set per-config properties to prevent config subdirectory being added
  set_target_properties(${EXT_LIBRARY_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    OUTPUT_NAME ${EXT_LIBRARY_NAME}_${PROJECT_VERSION_MAJOR}
    ARCHIVE_OUTPUT_NAME ${EXT_LIBRARY_NAME})
  # Set per-config output directories to flatten structure for multi-config generators
  foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(${EXT_LIBRARY_NAME} PROPERTIES
      "RUNTIME_OUTPUT_DIRECTORY_${_config}" "${CMAKE_CURRENT_BINARY_DIR}"
      "LIBRARY_OUTPUT_DIRECTORY_${_config}" "${CMAKE_CURRENT_BINARY_DIR}"
      "ARCHIVE_OUTPUT_DIRECTORY_${_config}" "${CMAKE_CURRENT_BINARY_DIR}")
  endforeach()
endif()

if(UNIX)
  install(TARGETS ${EXT_LIBRARY_NAME} ${EXT_LIBRARY_NAME}_static
    LIBRARY DESTINATION extensions NAMELINK_SKIP COMPONENT lib
    ARCHIVE COMPONENT lib
    PUBLIC_HEADER DESTINATION include COMPONENT lib
  )

else()
  install(TARGETS ${EXT_LIBRARY_NAME}
    RUNTIME DESTINATION extensions COMPONENT lib
    LIBRARY COMPONENT lib
    ARCHIVE COMPONENT lib
    PUBLIC_HEADER DESTINATION include COMPONENT lib
  )
endif()
