/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// MSVC host with CUDA; HIP compiler/runtime details remain inside provider.hip.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "probe_api.h"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <vector>
#include <windows.h>

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        std::fprintf(stderr, "No CUDA device for the coexistence test\n");
        return 1;
    }
    cudaDeviceProp property{};
    if (cudaGetDeviceProperties(&property, 0) != cudaSuccess)
        return 1;
    std::printf("CUDA device stays active: %s\n", property.name);
    constexpr size_t bytes = 1024 * 1024;
    void* cuda_pointer = nullptr;
    if (cudaMalloc(&cuda_pointer, bytes) != cudaSuccess)
        return 1;
    if (cudaMemset(cuda_pointer, 0x3C, bytes) != cudaSuccess)
        return 1;
    const auto provider = LoadLibraryExA(argv[1], nullptr,
                                         LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!provider) {
        std::fprintf(stderr, "Cannot load the HIP probe DLL: %lu\n", GetLastError());
        (void)cudaFree(cuda_pointer);
        return 1;
    }
    const auto affine = reinterpret_cast<LfsHipProbeAffine>(GetProcAddress(provider, "lfs_hip_probe_affine"));
    if (!affine)
        return 1;
    std::vector<float> input((1u << 20) + 19, 0.25f), output(input.size());
    char error[512]{};
    const int status = affine(input.data(), output.data(), input.size(), error, sizeof(error));
    if (status || !std::all_of(output.begin(), output.end(), [](auto value) { return value == 0.4375f; })) {
        std::fprintf(stderr, "HIP provider failed: %s\n", error);
        return 1;
    }
    std::vector<unsigned char> original(bytes);
    if (cudaMemcpy(original.data(), cuda_pointer, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 1;
    if (!std::all_of(original.begin(), original.end(), [](auto value) { return value == 0x3C; }))
        return 1;
    if (cudaMemset(cuda_pointer, 0x7B, bytes) != cudaSuccess)
        return 1;
    if (cudaMemcpy(original.data(), cuda_pointer, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 1;
    if (!std::all_of(original.begin(), original.end(), [](auto value) { return value == 0x7B; }))
        return 1;
    if (cudaFree(cuda_pointer) != cudaSuccess)
        return 1;
    FreeLibrary(provider);
    std::puts("PASS: MSVC CUDA allocation survives Clang HIP kernel execution in the same process");
    return 0;
}
