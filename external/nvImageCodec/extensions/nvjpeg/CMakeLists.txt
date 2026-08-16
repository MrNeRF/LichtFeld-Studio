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

set(EXT_LIBRARY_NAME nvjpeg_ext)

set(EXT_SRCS
  cuda_decoder.cpp
  hw_decoder.cpp
  cuda_encoder.cpp
  type_convert.cpp
  nvjpeg_ext.cpp
  nvjpeg_utils.cpp
  exception.cpp
)

set(EXT_HEADERS
  cuda_decoder.h
  cuda_encoder.h
  errors_handling.h
  exception.h
  hw_decoder.h
  log.h
  lossless_decoder.h
  nvjpeg_ext.h
  nvjpeg_utils.h
  type_convert.h
)

# Add utils directory for common headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${CUDAToolkit_INCLUDE_DIRS}")
check_cxx_source_compiles(
  "#include <nvjpeg.h>
  int main(){
    return NVJPEG_BACKEND_LOSSLESS_JPEG;
  }"
  NVJPEG_LOSSLESS_SUPPORTED)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${CUDAToolkit_INCLUDE_DIRS}")
check_cxx_source_compiles(
  "#include <nvjpeg.h>
  int main(){
    return NVJPEG_ENC_BACKEND_HARDWARE;
  }"
  NVJPEG_HW_ENCODER_SUPPORTED)

include(CheckCXXSymbolExists)
check_cxx_symbol_exists(nvjpegBufferDeviceResize "nvjpeg.h" NVJPEG_BUFFER_RESIZE_API)
if (NVJPEG_BUFFER_RESIZE_API)
  message("nvJPEG buffer resize API available")
  add_definitions(-DNVJPEG_BUFFER_RESIZE_API=1)
endif()

if(NVJPEG_LOSSLESS_SUPPORTED)
  message("nvJPEG lossless supported")
  add_definitions(-DNVJPEG_LOSSLESS_SUPPORTED=1)
 list(APPEND EXT_SRCS lossless_decoder.cpp)
else()
  message("nvJPEG lossless NOT supported")
  add_definitions(-DNVJPEG_LOSSLESS_SUPPORTED=0)
endif()

if(NVJPEG_HW_ENCODER_SUPPORTED)
  message("nvJPEG hardware encoder supported")
  add_definitions(-DNVJPEG_HW_ENCODER_SUPPORTED=1)
  list(APPEND EXT_SRCS hw_encoder.cpp)
  list(APPEND EXT_HEADERS hw_encoder.h)
else()
  message("nvJPEG hardware encoder NOT supported")
  add_definitions(-DNVJPEG_HW_ENCODER_SUPPORTED=0)
endif()

add_library(${EXT_LIBRARY_NAME}_obj OBJECT ${EXT_SRCS} ${EXT_HEADERS})
add_library(${EXT_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj> $<TARGET_OBJECTS:nvimgcodec_imgproc_static> ext_module.cpp)
add_library(${EXT_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj> $<TARGET_OBJECTS:nvimgcodec_imgproc_static>)

if (WITH_DYNAMIC_CUDA)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE dynlink_cuda)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE "-Wl,--exclude-libs,$<TARGET_FILE_NAME:dynlink_cuda>")
else()
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE CUDA::cuda_driver)
endif()

if(NOT WITH_DYNAMIC_NVJPEG)
  if(UNIX)
    target_link_libraries(${EXT_LIBRARY_NAME}
      PUBLIC ${CUDA_CUDART_LIB} ${CUDA_NVJPEG_LIB} CUDA::culibos)
  else()
    target_link_libraries(${EXT_LIBRARY_NAME}
      PUBLIC ${CUDA_CUDART_LIB} CUDA::nvjpeg)
  endif()
else()
  target_link_libraries(${EXT_LIBRARY_NAME}
    PRIVATE ${CUDA_CUDART_LIB} dynlink_nvjpeg)
  target_link_libraries(${EXT_LIBRARY_NAME}
    PRIVATE "-Wl,--exclude-libs,$<TARGET_FILE_NAME:dynlink_nvjpeg>")
endif()

if(UNIX)
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
    LIBRARY DESTINATION "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR}" NAMELINK_SKIP COMPONENT lib
    ARCHIVE DESTINATION "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${NVIMGCODEC_INSTALL_INCLUDEDIR}" COMPONENT lib
  )

else()
  install(TARGETS ${EXT_LIBRARY_NAME} ${EXT_LIBRARY_NAME}_static
    RUNTIME DESTINATION "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR}" COMPONENT lib
    ARCHIVE DESTINATION "${NVIMGCODEC_INSTALL_EXTENSION_LIBDIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${NVIMGCODEC_INSTALL_INCLUDEDIR}" COMPONENT lib
  )
endif()
