# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: MIT
#
# Generate the GLSL shader headers required by FSR3. The argument lists mirror
# the SDK's CMakeCompileFSR3UpscalerShaders.txt and CMakeShadersFSR3Upscaler.txt.

file(GLOB SHADERS "${FFX_SDK_SOURCE_DIR}/src/backends/vk/shaders/fsr3upscaler/*.glsl")
list(SORT SHADERS)

set(BASE_ARGS
    -reflection -deps=gcc -DFFX_GPU=1
    -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
    -DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2)
set(API_ARGS
    -compiler=glslang -e CS --target-env vulkan1.2 -S comp -DFFX_GLSL=1 -Os)
execute_process(
    COMMAND "${GLSLANG_EXECUTABLE}" -Os --version
    OUTPUT_QUIET
    ERROR_QUIET
    RESULT_VARIABLE GLSLANG_OPTIMIZER_RESULT)
if(NOT GLSLANG_OPTIMIZER_RESULT EQUAL 0)
    message(FATAL_ERROR
        "FSR 3.1 shaders require glslang with the SPIR-V optimiser (vcpkg "
        "feature `opt`, requested by vcpkg.json; re-run the LichtFeld configure "
        "so vcpkg installs it).")
endif()
set(PERM_ARGS
    "-DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH={0,1}"
    "-DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING={0,1}")

file(MAKE_DIRECTORY "${FFX_OUTPUT_DIR}")
foreach(shader IN LISTS SHADERS)
    get_filename_component(pass "${shader}" NAME_WE)
    foreach(variant IN ITEMS base wave64 16bit wave64_16bit)
        if(variant STREQUAL base)
            set(name "${pass}")
            set(half 0)
        elseif(variant STREQUAL wave64)
            set(name "${pass}_wave64")
            set(half 0)
        elseif(variant STREQUAL 16bit)
            set(name "${pass}_16bit")
            set(half 1)
        else()
            set(name "${pass}_wave64_16bit")
            set(half 1)
        endif()

        execute_process(
            COMMAND "${FFX_SC_EXECUTABLE}"
                ${BASE_ARGS} ${API_ARGS} ${PERM_ARGS}
                "-name=${name}" "-DFFX_HALF=${half}"
                "-I${FFX_SDK_SOURCE_DIR}/include/FidelityFX/gpu"
                "-I${FFX_SDK_SOURCE_DIR}/include/FidelityFX/gpu/fsr3upscaler"
                "-output=${FFX_OUTPUT_DIR}"
                "-glslangexe=${GLSLANG_EXECUTABLE}" "${shader}"
            RESULT_VARIABLE result)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "FSR3 GLSL shader generation failed for ${name}")
        endif()
    endforeach()
endforeach()
