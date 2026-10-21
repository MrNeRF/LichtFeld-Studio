/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/detail/gpu_backend_ops.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>
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
    require(rejected, "Invalid HIP flat index operation was accepted");
}
StorageRef host(void* data) {
    return {GpuBackend::HIP, data, 0, DataType::UInt8, nullptr, STORAGE_REF_HOST_MEMORY};
}
struct Buffer {
    GpuBackendOps& backend;
    StorageRef base;
    Buffer(GpuBackendOps& backend, size_t bytes) : backend(backend), base(backend.allocate(bytes, 256, {})) {}
    ~Buffer() { backend.deallocate(base, {}); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
template <class Index> void test_index_put(GpuBackendOps& gpu) {
    std::array<float, 9> initial{-77, -77, 1, 2, 3, 4, 5, -77, -77};
    std::array<float, 4> values{8, 9, 10, 8};
    std::array<Index, 4> indices{4, 0, 2, 4};
    Buffer destination(gpu, sizeof(initial)), source(gpu, sizeof(values)), index(gpu, sizeof(indices));
    auto output = destination.base, src = source.base, idx = index.base;
    output.dtype = src.dtype = DataType::Float32;
    output.byte_offset = 2 * sizeof(float);
    idx.dtype = std::is_same_v<Index, int32_t> ? DataType::Int32 : DataType::Int64;
    gpu.copy_host_to_device({host(values.data()), src, sizeof(values)});
    auto upload_indices = [&] { gpu.copy_host_to_device({host(indices.data()), idx, sizeof(indices)}); };
    auto reset = [&] { gpu.copy_host_to_device({host(initial.data()), destination.base, sizeof(initial)}); };
    std::array<float, 9> result{};
    auto read = [&] { gpu.copy_device_to_host({destination.base, host(result.data()), sizeof(result)}); };
    IndexProgram program{};
    program.input_size = 5;
    program.index_size = indices.size();
    auto run = [&] { gpu.index_put(output, idx, src, program, {}); };
    upload_indices();
    reset();
    run();
    read();
    require(result == std::array<float, 9>{-77, -77, 9, 2, 10, 4, 8, -77, -77},
            "HIP index_put changed values or allocation guards");
    indices = {Index(-1), Index(-5), Index(2), Index(9)};
    upload_indices();
    reset();
    run();
    read();
    require(result == std::array<float, 9>{-77, -77, 9, 2, 10, 4, 8, -77, -77},
            "HIP index_put differs from the flat negative-index/clamp contract");
    program.input_size = 0;
    reset();
    reject(run);
    read();
    require(result == initial, "Invalid HIP index_put extent partially changed output");
    program.input_size = 5;
    run(); // Recovery after rejected descriptor.
    reject([&] { gpu.index_put(output, idx, output, program, {}); });
    auto alias = output;
    alias.dtype = idx.dtype;
    reject([&] { gpu.index_put(output, alias, src, program, {}); });
    auto wrong = src;
    wrong.dtype = DataType::Int32;
    reject([&] { gpu.index_put(output, idx, wrong, program, {}); });
    program.index_size = 0;
    reset();
    run();
    read();
    require(result == initial, "Empty HIP index_put changed output");
}

template <class Index> void test_gather_unary(GpuBackendOps& gpu) {
    const float infinity = std::numeric_limits<float>::infinity();
    std::array<float, 8> input{-infinity, -4.f, -0.f, 0.f, 0.25f, 4.f, infinity, std::numeric_limits<float>::quiet_NaN()};
    constexpr size_t count = 263;
    std::vector<Index> indices(count);
    const std::array<Index, 15> cases{std::numeric_limits<Index>::min(), Index(-12), Index(-8), Index(-7), Index(-1),
                                     Index(0), Index(1), Index(2), Index(3), Index(4), Index(5), Index(6), Index(7), Index(8),
                                     std::numeric_limits<Index>::max()};
    for (size_t i = 0; i < count; ++i)
        indices[i] = cases[i % cases.size()];
    std::vector<float> initial(count + 4, -77.f), result(initial.size());
    Buffer source(gpu, sizeof(input) + 16), index(gpu, indices.size() * sizeof(Index) + 16), destination(gpu, initial.size() * sizeof(float));
    auto src = source.base, idx = index.base, output = destination.base;
    src.dtype = output.dtype = DataType::Float32;
    src.byte_offset = idx.byte_offset = output.byte_offset = 8;
    idx.dtype = std::is_same_v<Index, int32_t> ? DataType::Int32 : DataType::Int64;
    gpu.copy_host_to_device({host(input.data()), src, sizeof(input)});
    gpu.copy_host_to_device({host(indices.data()), idx, indices.size() * sizeof(Index)});
    IndexProgram program{};
    program.input_size = input.size();
    program.index_size = count;
    for (auto op : {PointwiseOp::Abs, PointwiseOp::Sqrt, PointwiseOp::Neg}) {
        gpu.copy_host_to_device({host(initial.data()), destination.base, initial.size() * sizeof(float)});
        gpu.gather_fused_unary(src, idx, output, op, program, {});
        gpu.copy_device_to_host({destination.base, host(result.data()), result.size() * sizeof(float)});
        require(result[0] == -77.f && result[1] == -77.f && result[count + 2] == -77.f && result[count + 3] == -77.f,
                "HIP fused gather changed allocation guards");
        for (size_t i = 0; i < count; ++i) {
            int64_t position = indices[i];
            if (position < 0)
                position += int64_t(input.size());
            position = std::clamp(position, int64_t(0), int64_t(input.size() - 1));
            const float value = input[size_t(position)];
            const float expected = op == PointwiseOp::Abs ? std::abs(value) : op == PointwiseOp::Sqrt ? std::sqrt(value) : -value;
            if (std::isnan(expected))
                require(std::isnan(result[i + 2]), "HIP fused gather lost NaN propagation");
            else
                require(std::bit_cast<uint32_t>(result[i + 2]) == std::bit_cast<uint32_t>(expected),
                        "HIP fused gather differs from CPU oracle, including signed zero/infinity");
        }
    }
    auto run = [&] { gpu.gather_fused_unary(src, idx, output, PointwiseOp::Abs, program, {}); };
    gpu.copy_host_to_device({host(initial.data()), destination.base, initial.size() * sizeof(float)});
    reject([&] { gpu.gather_fused_unary(src, idx, output, PointwiseOp::Exp, program, {}); });
    reject([&] { gpu.gather_fused_unary(src, idx, src, PointwiseOp::Abs, program, {}); });
    program.input_size = 0;
    reject(run);
    program.input_size = input.size() + 100;
    reject(run);
    program.input_size = input.size();
    auto unaligned = src;
    ++unaligned.byte_offset;
    reject([&] { gpu.gather_fused_unary(unaligned, idx, output, PointwiseOp::Abs, program, {}); });
    gpu.copy_device_to_host({destination.base, host(result.data()), result.size() * sizeof(float)});
    require(result == initial, "Rejected HIP fused gather partially changed output");
    program.index_size = 0;
    run();
    gpu.copy_device_to_host({destination.base, host(result.data()), result.size() * sizeof(float)});
    require(result == initial, "Empty HIP fused gather changed output");
    program.index_size = count;
    run(); // Recovery after rejected descriptors.
}
} // namespace

int main() try {
    GpuBackendScope scope(GpuBackend::HIP);
    auto& gpu = backend_ops(GpuBackend::HIP);
    test_index_put<int32_t>(gpu);
    test_index_put<int64_t>(gpu);
    test_gather_unary<int32_t>(gpu);
    test_gather_unary<int64_t>(gpu);
    auto input = Tensor::from_vector(std::vector<float>{-4.f, -0.f, 0.25f, 4.f}, {4}, Device::GPU);
    auto indices = Tensor::from_vector(std::vector<int>{-4, -1, 1, 2}, {4}, Device::GPU);
    require(input.gather_lazy(indices).map(ops::abs_op{}).eval().cpu().to_vector() == std::vector<float>{4.f, 4.f, 0.f, 0.25f},
            "HIP public fused gather abs failed");
    require(input.gather_lazy(indices).map(ops::neg_op{}).eval().cpu().to_vector() == std::vector<float>{4.f, -4.f, 0.f, -0.25f},
            "HIP public fused gather negation failed");
    auto roots = input.gather_lazy(indices).map(ops::sqrt_op{}).eval().cpu().to_vector();
    require(std::isnan(roots[0]) && roots[1] == 2.f && roots[2] == 0.f && std::signbit(roots[2]) && roots[3] == 0.5f,
            "HIP public fused gather sqrt failed");
    std::puts("PASS integrated HIP flat indexing: Int32/Int64, wrapped/clamped indices, guards, duplicate writes, fused abs/sqrt/neg with IEEE values, invalid descriptors without partial writes, recovery and empty operations");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP INDEX CONTRACT FAILED: %s\n", error.what());
    return 1;
}
