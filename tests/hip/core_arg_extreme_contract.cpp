/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/detail/gpu_backend_ops.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace lfs::core;
using namespace lfs::core::internal;
namespace {
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class F> void reject(F&& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Invalid HIP indexed-extrema contract was accepted");
}
StorageRef host(void* data) {
    return {GpuBackend::HIP, data, 0, DataType::UInt8, nullptr, STORAGE_REF_HOST_MEMORY};
}
struct Buffer {
    GpuBackendOps& backend;
    StorageRef base;
    Buffer(GpuBackendOps& gpu, size_t bytes) : backend(gpu), base(gpu.allocate(bytes, 256, {})) {}
    ~Buffer() { backend.deallocate(base, {}); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
bool same_float(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }

void test_shape(GpuBackendOps& gpu, size_t outer, size_t reduced, size_t inner) {
    const size_t count = outer * reduced * inner, outputs = outer * inner;
    std::vector<float> input(count + 8, -77.f), initial_values(outputs + 4, -77.f), values(initial_values.size());
    std::vector<int64_t> initial_indices(outputs + 4, -777), indices(initial_indices.size());
    for (size_t i = 0; i < count; ++i)
        input[i + 4] = float(int(i % 257) - 128) * 0.125f;
    for (size_t row = 0; row < outputs; ++row) {
        const size_t base = 4 + (row / inner) * reduced * inner + row % inner;
        if (row % 7 == 0) {
            for (size_t k = 0; k < reduced; ++k)
                input[base + k * inner] = k ? 0.f : -0.f;
        } else if (row % 7 == 1) {
            input[base] = std::bit_cast<float>(uint32_t(0x7fc12345));
            input[base + (reduced - 1) * inner] = std::bit_cast<float>(uint32_t(0x7fc12345));
        } else if (row % 7 == 2) {
            input[base + (reduced / 2) * inner] = std::bit_cast<float>(uint32_t(0x7fc12345));
            input[base + (reduced - 1) * inner] = std::bit_cast<float>(uint32_t(0x7fc54321));
        } else if (row % 7 == 3) {
            input[base] = -std::numeric_limits<float>::infinity();
            input[base + (reduced - 1) * inner] = std::numeric_limits<float>::infinity();
        } else if (row % 7 == 4) {
            input[base + std::min<size_t>(4095, reduced - 1) * inner] = 4096.f;
            input[base + std::min<size_t>(4096, reduced - 1) * inner] = 4096.f;
        }
    }
    Buffer source(gpu, input.size() * 4), val_buffer(gpu, values.size() * 4), idx_buffer(gpu, indices.size() * 8);
    auto in = source.base, val = val_buffer.base, idx = idx_buffer.base;
    in.dtype = val.dtype = DataType::Float32;
    idx.dtype = DataType::Int64;
    in.byte_offset = idx.byte_offset = 16;
    val.byte_offset = 8;
    gpu.copy_host_to_device({host(input.data()), source.base, input.size() * 4});
    for (bool maximum : {false, true}) {
        gpu.copy_host_to_device({host(initial_values.data()), val_buffer.base, values.size() * 4});
        gpu.copy_host_to_device({host(initial_indices.data()), idx_buffer.base, indices.size() * 8});
        require(gpu.arg_extreme(in, val, idx, {.outer = outer, .reduce = reduced, .inner = inner, .maximum = maximum}, {}),
                "HIP indexed extrema did not use the native path");
        gpu.copy_device_to_host({val_buffer.base, host(values.data()), values.size() * 4});
        gpu.copy_device_to_host({idx_buffer.base, host(indices.data()), indices.size() * 8});
        require(values[0] == -77.f && values[1] == -77.f && values[outputs + 2] == -77.f && values[outputs + 3] == -77.f &&
                    indices[0] == -777 && indices[1] == -777 && indices[outputs + 2] == -777 && indices[outputs + 3] == -777,
                "HIP indexed extrema changed allocation guards");
        for (size_t row = 0; row < outputs; ++row) {
            const size_t base = 4 + (row / inner) * reduced * inner + row % inner;
            float expected = input[base];
            int64_t position = 0;
            for (size_t k = 1; k < reduced && !std::isnan(expected); ++k) {
                const float candidate = input[base + k * inner];
                if (std::isnan(candidate) || (maximum ? candidate > expected : candidate < expected)) {
                    expected = candidate;
                    position = int64_t(k);
                }
            }
            require(same_float(values[row + 2], expected) && indices[row + 2] == position,
                    "HIP indexed extrema disagree with independent CPU loop (value bits or first index)");
        }
    }
    auto copied = input;
    gpu.copy_device_to_host({source.base, host(copied.data()), copied.size() * 4});
    require(std::memcmp(copied.data(), input.data(), input.size() * 4) == 0, "HIP indexed extrema changed source storage");
}

void test_invalid(GpuBackendOps& gpu) {
    Buffer source(gpu, 64), val_buffer(gpu, 64), idx_buffer(gpu, 64);
    auto in = source.base, val = val_buffer.base, idx = idx_buffer.base;
    in.dtype = val.dtype = DataType::Float32;
    idx.dtype = DataType::Int64;
    std::array<float, 16> initial{};
    initial.fill(-77.f);
    gpu.copy_host_to_device({host(initial.data()), val_buffer.base, 64});
    ArgExtremeProgram program{.outer = 2, .reduce = 3, .inner = 2};
    auto run = [&] { gpu.arg_extreme(in, val, idx, program, {}); };
    program.reduce = 0;
    reject(run);
    program.reduce = SIZE_MAX;
    reject(run);
    program.reduce = 3;
    program.outer = SIZE_MAX;
    reject(run);
    program.outer = 2;
    program.reduce = 100;
    reject(run);
    program.reduce = 3;
    reject([&] { gpu.arg_extreme(in, in, idx, program, {}); });
    auto alias = val;
    alias.dtype = DataType::Int64;
    reject([&] { gpu.arg_extreme(in, val, alias, program, {}); });
    auto wrong = idx;
    wrong.dtype = DataType::Int32;
    reject([&] { gpu.arg_extreme(in, val, wrong, program, {}); });
    auto unaligned = idx;
    unaligned.byte_offset = 4;
    reject([&] { gpu.arg_extreme(in, val, unaligned, program, {}); });
    program.outer = 0;
    run();
    std::array<float, 16> observed{};
    gpu.copy_device_to_host({val_buffer.base, host(observed.data()), 64});
    require(observed == initial, "Invalid or empty HIP indexed extrema partially changed values");
    gpu.copy_host_to_device({host(initial.data()), in, 64});
    program.outer = 2;
    run(); // Recovery after rejected descriptors.
}

void test_public() {
    const float nan = std::bit_cast<float>(uint32_t(0x7fc12345));
    auto cpu = Tensor::from_vector(std::vector<float>{-0.f, 0.f, -0.f, 5.f, nan, nan, 3.f, 7.f, 7.f}, {3, 3}, Device::CPU);
    auto input = cpu.to(Device::GPU);
    for (bool transpose : {false, true}) {
        auto a = transpose ? input.transpose(0, 1) : input;
        auto b = transpose ? cpu.transpose(0, 1) : cpu;
        for (bool maximum : {false, true})
            for (bool keep : {false, true}) {
                auto actual = maximum ? a.max_with_indices(-1, keep) : a.min_with_indices(-1, keep);
                auto expected = maximum ? b.max_with_indices(-1, keep) : b.min_with_indices(-1, keep);
                auto values = actual.first.cpu(), indices = actual.second.cpu();
                require(values.shape() == expected.first.shape() && indices.shape() == expected.second.shape(), "HIP indexed-extrema public shape changed");
                require(std::memcmp(values.ptr<float>(), expected.first.ptr<float>(), values.numel() * 4) == 0 &&
                            std::equal(indices.ptr<int64_t>(), indices.ptr<int64_t>() + indices.numel(), expected.second.ptr<int64_t>()),
                        "HIP indexed-extrema public API differs from CPU");
            }
    }
    require(input.argmax().cpu().ptr<int64_t>()[0] == 4 && input.argmin().cpu().ptr<int64_t>()[0] == 4,
            "HIP flat argmax/argmin lost the first NaN index");
}

void test_workspace_reuse() {
    std::vector<float> data(8193);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = float(data.size() - 1 - i);
    auto input = Tensor::from_vector(data, {data.size()}, Device::GPU);
    auto [sorted, order] = input.sort();
    auto maximum = sorted.max_with_indices(0);
    require(maximum.first.cpu().ptr<float>()[0] == 8192.f && maximum.second.cpu().ptr<int64_t>()[0] == 8192,
            "HIP indexed extrema corrupted a preceding sort workspace");
    auto [descending, reverse_order] = sorted.sort(0, true);
    auto minimum = descending.min_with_indices(0);
    require(minimum.first.cpu().ptr<float>()[0] == 0.f && minimum.second.cpu().ptr<int64_t>()[0] == 8192 &&
                input.argmax().cpu().ptr<int64_t>()[0] == 0,
            "HIP sort/indexed-extrema workspace reuse changed results");
}
} // namespace

int main() try {
    GpuBackendScope scope(GpuBackend::HIP);
    auto& gpu = backend_ops(GpuBackend::HIP);
    for (auto shape : {std::array<size_t, 3>{1, 1, 1}, {7, 2, 3}, {3, 31, 5}, {2, 32, 1}, {3, 33, 1},
                       {5, 263, 7}, {2, 8193, 1}, {3, 8193, 7}, {2, 1000003, 1}, {3, 8193, 263}, {8193, 17, 1}})
        test_shape(gpu, shape[0], shape[1], shape[2]);
    test_invalid(gpu);
    test_public();
    require(gpu.stats().allocated_bytes > 0 && gpu.stats().allocated_bytes <= 32 * 1024,
            "HIP indexed extrema retained live buffers or exceeded the bounded workspace");
    gpu.trim();
    require(gpu.stats().allocated_bytes == 0, "HIP indexed-extrema workspace was not released by trim");
    test_workspace_reuse();
    gpu.trim();
    require(gpu.stats().allocated_bytes == 0, "HIP shared sort/extrema workspace was not released by trim");
    std::puts("PASS integrated HIP indexed extrema: contiguous/strided axes, multi-pass million-value reductions, exact values/first-index ties/NaNs/signed zeros, guards, invalid descriptors, empty work and public argmin/argmax");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP INDEXED EXTREMA FAILED: %s\n", error.what());
    return 1;
}
