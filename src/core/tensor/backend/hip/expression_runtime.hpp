/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "expression_api.h"
#include <hip/hiprtc.h>
struct LfsHipExpression {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
};
namespace {
    void release_expression_native(LfsHipContext* context, LfsHipExpression* expression) {
        check(hipModuleUnload(expression->module));
        context->expressions.erase(expression);
        delete expression;
    }
} // namespace
int lfs_hip_expression_identity_for_device(int32_t ordinal, char* output, size_t capacity, LfsHipError* error) {
    return boundary(error, [&] {
        require(output && capacity >= 128, "Missing HIP compiler identity output");
        validate_device(ordinal);
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, ordinal));
        const auto architecture = lfs_hip_base_architecture(properties.gcnArchName);
        require(lfs_hip_has_architecture(LFS_HIP_COMPILED_ARCH, architecture), "HIP expression target is not in this build");
        int major = 0, minor = 0, runtime = 0, driver = 0;
        if (hiprtcVersion(&major, &minor) != HIPRTC_SUCCESS)
            throw std::runtime_error("Cannot query HIPRTC version");
        check(hipRuntimeGetVersion(&runtime));
        check(hipDriverGetVersion(&driver));
#ifdef _WIN32
        constexpr const char* platform = "windows";
#else
        constexpr const char* platform="linux";
#endif
        std::snprintf(output, capacity, "hip-expression-v2-%s-%.*s-sdk%d-rtc%d.%d-runtime%d-driver%d-fp-contract-off-ieee-denorm", platform, int(architecture.size()), architecture.data(), HIP_VERSION, major, minor, runtime, driver);
    });
}
int lfs_hip_expression_compile_for_device(int32_t ordinal, const char* source, char* artifact, size_t capacity, size_t* bytes,
                                          char* diagnostics, size_t diagnostic_capacity, LfsHipError* error) {
    return boundary(error, [&] {
        require(source && bytes && (!capacity || artifact) && (!diagnostic_capacity || diagnostics), "Invalid HIP expression compile descriptor");
        validate_device(ordinal);
        DeviceScope selected(ordinal);
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, ordinal));
        const auto architecture = lfs_hip_base_architecture(properties.gcnArchName);
        require(lfs_hip_has_architecture(LFS_HIP_COMPILED_ARCH, architecture), "HIP expression target is not in this build");
        *bytes = 0;
        if (diagnostic_capacity)
            diagnostics[0] = 0;
        struct Program {
            hiprtcProgram program = nullptr;
            ~Program() {
                if (program)
                    hiprtcDestroyProgram(&program);
            }
        } program;
        const auto rtc_check = [](hiprtcResult result) {
            if (result != HIPRTC_SUCCESS)
                throw std::runtime_error(std::string("HIPRTC expression: ") + hiprtcGetErrorString(result));
        };
        rtc_check(hiprtcCreateProgram(&program.program, source, "lfs_expression.hip", 0, nullptr, nullptr));
        const std::string target = "--gpu-architecture=" + std::string(architecture);
        const char* options[]{"--std=c++20", target.c_str(), "-O3", "-ffp-contract=off", "-fno-fast-math", "-fdenormal-fp-math=ieee", "-Xclang", "-fdenormal-fp-math-f32=ieee"};
        const auto result = hiprtcCompileProgram(program.program, 8, options);
        if (result != HIPRTC_SUCCESS) {
            size_t size = 0;
            if (hiprtcGetProgramLogSize(program.program, &size) == HIPRTC_SUCCESS && size) {
                std::vector<char> log(size);
                if (hiprtcGetProgramLog(program.program, log.data()) == HIPRTC_SUCCESS && diagnostic_capacity) {
                    const auto copied = std::min(size_t(diagnostic_capacity - 1), size - 1);
                    std::memcpy(diagnostics, log.data(), copied);
                    diagnostics[copied] = 0;
                }
            }
            rtc_check(result);
        }
        rtc_check(hiprtcGetCodeSize(program.program, bytes));
        require(*bytes && *bytes <= 16 * 1024 * 1024, "HIP expression artifact exceeds budget");
        if (!artifact)
            return;
        require(capacity >= *bytes, "HIP expression artifact output is too small");
        rtc_check(hiprtcGetCode(program.program, artifact));
    });
}
int lfs_hip_expression_identity(char* output, size_t capacity, LfsHipError* error) {
    int ordinal = 0;
    const int status = boundary(error, [&] { check(hipGetDevice(&ordinal)); });
    return status ? status : lfs_hip_expression_identity_for_device(ordinal, output, capacity, error);
}
int lfs_hip_expression_compile(const char* source, char* artifact, size_t capacity, size_t* bytes,
                               char* diagnostics, size_t diagnostic_capacity, LfsHipError* error) {
    int ordinal = 0;
    const int status = boundary(error, [&] { check(hipGetDevice(&ordinal)); });
    return status ? status : lfs_hip_expression_compile_for_device(ordinal, source, artifact, capacity, bytes, diagnostics, diagnostic_capacity, error);
}
int lfs_hip_expression_load(LfsHipContext* context, const void* artifact, size_t bytes, LfsHipExpression** output, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && artifact && bytes && bytes <= 16 * 1024 * 1024 && output, "Invalid HIP expression artifact");
        *output = nullptr;
        DeviceScope device(context->device);
        auto expression = std::make_unique<LfsHipExpression>();
        check(hipModuleLoadData(&expression->module, artifact));
        try {
            check(hipModuleGetFunction(&expression->function, expression->module, "lfs_expression"));
            context->expressions.insert(expression.get());
        } catch (...) {
            (void)hipModuleUnload(expression->module);
            throw;
        }
        *output = expression.release();
    });
}
int lfs_hip_expression_launch(LfsHipContext* context, LfsHipExpression* expression, const uint32_t* words,
                              size_t count, size_t work, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && expression && context->expressions.contains(expression) && words && count && count <= 256,
                "Invalid HIP expression launch");
        if (!work)
            return;
        require(work <= UINT32_MAX, "HIP expression work exceeds uint32");
        struct Arguments {
            uint32_t words[256]{};
        } arguments;
        std::copy_n(words, count, arguments.words);
        void* parameters[]{&arguments};
        DeviceScope device(context->device);
        check(hipModuleLaunchKernel(expression->function, unsigned(std::min<size_t>((work + 255) / 256, 4096)), 1, 1, 256, 1, 1, 0,
                                    context->stream, parameters, nullptr));
    });
}
int lfs_hip_expression_destroy(LfsHipContext* context, LfsHipExpression* expression, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && expression && context->expressions.contains(expression), "Unknown HIP expression module");
        DeviceScope device(context->device);
        check(hipStreamSynchronize(context->stream));
        release_expression_native(context, expression);
    });
}
