/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Standalone HIP bring-up. This does not register an application tensor backend.
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    void check(hipError_t result, const char* operation) {
        if (result != hipSuccess)
            throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
    }

    struct Buffer {
        void* pointer = nullptr;
        explicit Buffer(size_t bytes) { check(hipMalloc(&pointer, bytes), "hipMalloc"); }
        ~Buffer() {
            if (pointer)
                (void)hipFree(pointer);
        }
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        template <class T>
        T* as() const { return static_cast<T*>(pointer); }
    };

    __global__ void affine(const float* input, float* output, size_t count) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count)
            output[i] = input[i] * 1.25f + 0.125f;
    }

    __global__ void gather(const float* input, const int* indices, float* output, size_t count) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count)
            output[i] = input[indices[i]];
    }

    __global__ void adam(float* parameter, const float* gradient, float* first, float* second,
                         size_t count, float correction1, float correction2) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count)
            return;
        const float g = gradient[i];
        const float m = 0.9f * first[i] + 0.1f * g;
        const float v = 0.999f * second[i] + 0.001f * g * g;
        parameter[i] -= 0.001f * (m / correction1) / (sqrtf(v / correction2) + 1.0e-8f);
        first[i] = m;
        second[i] = v;
    }

    void check_close(const std::vector<float>& actual, const std::vector<float>& expected,
                     float tolerance, const char* label) {
        float maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float error = std::abs(actual[i] - expected[i]);
            if (!std::isfinite(actual[i]) || error > tolerance)
                throw std::runtime_error(std::string(label) + " mismatch at " + std::to_string(i));
            maximum = std::max(maximum, error);
        }
        std::printf("PASS %s: %zu values, maximum absolute error %.9g\n", label, actual.size(), maximum);
    }
} // namespace

int main() try {
    int runtime_version = 0;
    check(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion");
    std::printf("HIP runtime version: %d\n", runtime_version);
#ifdef _WIN32
    char module_path[32768]{};
    const auto module = GetModuleHandleA("amdhip64_7.dll");
    if (!module || !GetModuleFileNameA(module, module_path, sizeof(module_path)))
        throw std::runtime_error("Cannot identify the loaded HIP runtime");
    std::printf("HIP runtime module: %s\n", module_path);
#endif
    int count = 0;
    check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    if (count != 1)
        throw std::runtime_error("Bring-up expects exactly one HIP adapter");
    check(hipSetDevice(0), "hipSetDevice");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    std::printf("HIP device: %s, target %s, warp size %d\n", properties.name, properties.gcnArchName, properties.warpSize);
    if (std::string(properties.gcnArchName).find("gfx1150") != 0)
        throw std::runtime_error("This bring-up binary must run on the intended gfx1150 adapter");
    constexpr size_t n = (1u << 20) + 19; // Non-multiple workgroup tail.
    constexpr size_t offset = 13;
    constexpr float guard = -9876.5f;
    std::vector<float> input(n), expected(n), actual(n), padded(n + 2 * offset, guard);
    std::vector<int> indices(n);
    for (size_t i = 0; i < n; ++i) {
        input[i] = float(int(i % 127) - 63) / 128.0f;
        indices[i] = int((i * 17 + 23) % n);
        expected[i] = input[i] * 1.25f + 0.125f;
    }
    Buffer source(n * sizeof(float)), target(padded.size() * sizeof(float)), index(n * sizeof(int));
    check(hipMemcpy(source.pointer, input.data(), n * sizeof(float), hipMemcpyHostToDevice), "input upload");
    check(hipMemcpy(target.pointer, padded.data(), padded.size() * sizeof(float), hipMemcpyHostToDevice), "guards upload");
    check(hipMemcpy(index.pointer, indices.data(), n * sizeof(int), hipMemcpyHostToDevice), "index upload");
    const dim3 block(256), grid(static_cast<unsigned>((n + 255) / 256));
    hipLaunchKernelGGL(affine, grid, block, 0, 0, source.as<float>(), target.as<float>() + offset, n);
    check(hipGetLastError(), "affine launch");
    check(hipMemcpy(actual.data(), target.as<float>() + offset, n * sizeof(float), hipMemcpyDeviceToHost), "affine readback");
    check_close(actual, expected, 1.0e-7f, "affine/offset/tail");
    hipLaunchKernelGGL(gather, grid, block, 0, 0, source.as<float>(), index.as<int>(), target.as<float>() + offset, n);
    check(hipGetLastError(), "gather launch");
    check(hipMemcpy(actual.data(), target.as<float>() + offset, n * sizeof(float), hipMemcpyDeviceToHost), "gather readback");
    for (size_t i = 0; i < n; ++i)
        expected[i] = input[indices[i]];
    check_close(actual, expected, 0, "gather/offset/tail");
    check(hipMemcpy(padded.data(), target.pointer, padded.size() * sizeof(float), hipMemcpyDeviceToHost), "guard readback");
    for (size_t i = 0; i < offset; ++i)
        if (padded[i] != guard || padded[n + offset + i] != guard)
            throw std::runtime_error("A kernel overwrote an offset/tail guard");
    std::puts("PASS unchanged prefix/suffix guards");

    Buffer parameter(n * sizeof(float)), first(n * sizeof(float)), second(n * sizeof(float));
    std::vector<float> cpu_parameter = input, cpu_first(n, 0), cpu_second(n, 0);
    check(hipMemcpy(parameter.pointer, input.data(), n * sizeof(float), hipMemcpyHostToDevice), "parameter upload");
    check(hipMemset(first.pointer, 0, n * sizeof(float)), "first clear");
    check(hipMemset(second.pointer, 0, n * sizeof(float)), "second clear");
    for (int step = 1; step <= 32; ++step) {
        const float c1 = float(1.0 - std::pow(0.9, step));
        const float c2 = float(1.0 - std::pow(0.999, step));
        hipLaunchKernelGGL(adam, grid, block, 0, 0, parameter.as<float>(), source.as<float>(), first.as<float>(), second.as<float>(), n, c1, c2);
        check(hipGetLastError(), "Adam launch");
        for (size_t i = 0; i < n; ++i) {
            const float g = input[i];
            cpu_first[i] = 0.9f * cpu_first[i] + 0.1f * g;
            cpu_second[i] = 0.999f * cpu_second[i] + 0.001f * g * g;
            cpu_parameter[i] -= 0.001f * (cpu_first[i] / c1) / (std::sqrt(cpu_second[i] / c2) + 1.0e-8f);
        }
    }
    check(hipMemcpy(actual.data(), parameter.pointer, n * sizeof(float), hipMemcpyDeviceToHost), "Adam parameter readback");
    check_close(actual, cpu_parameter, 2.0e-6f, "Adam parameters/32 steps");
    check(hipMemcpy(actual.data(), first.pointer, n * sizeof(float), hipMemcpyDeviceToHost), "Adam first readback");
    check_close(actual, cpu_first, 2.0e-6f, "Adam first moments");
    check(hipMemcpy(actual.data(), second.pointer, n * sizeof(float), hipMemcpyDeviceToHost), "Adam second readback");
    check_close(actual, cpu_second, 2.0e-6f, "Adam second moments");
    std::puts("HIP COMPILED KERNEL SMOKE PASSED; this is not an application training benchmark");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP SMOKE FAILED: %s\n", error.what());
    return 1;
}
