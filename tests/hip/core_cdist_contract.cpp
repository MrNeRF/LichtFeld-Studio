/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/detail/gpu_backend_ops.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace lfs::core;
using namespace lfs::core::internal;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template <class F> void reject(F&& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Invalid HIP cdist contract was accepted");
}
StorageRef host(void* data) { return {GpuBackend::HIP, data, 0, DataType::UInt8, nullptr, STORAGE_REF_HOST_MEMORY}; }
struct Buffer {
    GpuBackendOps& backend;
    StorageRef base;
    Buffer(GpuBackendOps& gpu, size_t bytes) : backend(gpu), base(gpu.allocate(bytes, 256, {})) {}
    ~Buffer() { backend.deallocate(base, {}); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

void compare(const Tensor& lhs, const Tensor& rhs) {
    auto a = lhs.to(Device::GPU), b = rhs.to(Device::GPU);
    for (float p : {0.f, 0.5f, 1.f, 1.5f, 2.f, std::numeric_limits<float>::infinity()}) {
        auto actual = a.cdist(b, p);
        require(gpu_backend_of(actual) == GpuBackend::HIP, "HIP cdist returned a different backend");
        auto observed = actual.cpu(), expected = lhs.cdist(rhs, p);
        require(observed.shape() == expected.shape(), "HIP cdist output shape differs from CPU");
        for (size_t i = 0; i < expected.numel(); ++i) {
            const float x = observed.ptr<float>()[i], y = expected.ptr<float>()[i];
            if (std::isnan(y)) require(std::isnan(x), "HIP cdist lost NaN propagation");
            else if (std::isinf(y)) require(x == y, "HIP cdist changed infinity");
            else if (p == 0.f || p == 1.f || std::isinf(p)) require(x == y, "HIP cdist exact norm differs from CPU");
            else require(std::isfinite(x) && std::abs(x - y) <= 2e-5f * std::max(1.f, std::abs(y)),
                         "HIP cdist differs from CPU beyond Float32 tolerance");
        }
    }
}

void test_shape(size_t n, size_t m, size_t d) {
    std::vector<float> a(n * d), b(m * d);
    for (size_t i = 0; i < a.size(); ++i) a[i] = float(int(i % 61) - 30) * 0.125f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = float(int((i * 13) % 71) - 35) * 0.25f;
    compare(Tensor::from_vector(a, {n, d}, Device::CPU), Tensor::from_vector(b, {m, d}, Device::CPU));
    // Exercise materialization of both noncontiguous public operands.
    auto lhs = Tensor::from_vector(a, {d, n}, Device::CPU).transpose(0, 1);
    auto rhs = Tensor::from_vector(b, {d, m}, Device::CPU).transpose(0, 1);
    compare(lhs, rhs);
    auto gpu_lhs = Tensor::from_vector(a, {d, n}, Device::GPU).transpose(0, 1);
    auto gpu_rhs = Tensor::from_vector(b, {d, m}, Device::GPU).transpose(0, 1);
    auto actual = gpu_lhs.cdist(gpu_rhs).cpu(), expected = lhs.cdist(rhs);
    for (size_t i = 0; i < expected.numel(); ++i)
        require(std::abs(actual.ptr<float>()[i] - expected.ptr<float>()[i]) <= 2e-5f * std::max(1.f, std::abs(expected.ptr<float>()[i])),
                "HIP cdist noncontiguous GPU materialization differs from CPU");
    if (n && m && d) {
        auto& gpu = backend_ops(GpuBackend::HIP);
        Buffer a_buffer(gpu, a.size() * 4), b_buffer(gpu, b.size() * 4), out_buffer(gpu, (n * m + 4) * 4);
        auto av = a_buffer.base, bv = b_buffer.base, out = out_buffer.base;
        av.dtype = bv.dtype = out.dtype = DataType::Float32;
        out.byte_offset = 8;
        std::vector<float> guarded(n * m + 4, -77.f);
        gpu.copy_host_to_device({host(a.data()), av, a.size() * 4});
        gpu.copy_host_to_device({host(b.data()), bv, b.size() * 4});
        gpu.copy_host_to_device({host(guarded.data()), out_buffer.base, guarded.size() * 4});
        gpu.cdist(av, bv, out, n, m, d, 2.f, {});
        gpu.copy_device_to_host({out_buffer.base, host(guarded.data()), guarded.size() * 4});
        require(guarded[0] == -77.f && guarded[1] == -77.f && guarded[n * m + 2] == -77.f && guarded[n * m + 3] == -77.f,
                "HIP direct/tiled cdist changed output guards");
    }
}

void test_invalid(GpuBackendOps& gpu) {
    Buffer a(gpu, 48), b(gpu, 48), destination(gpu, 64);
    auto lhs = a.base, rhs = b.base, output = destination.base;
    lhs.dtype = rhs.dtype = output.dtype = DataType::Float32;
    output.byte_offset = 8;
    std::array<float, 16> initial{}, result{};
    initial.fill(-77.f);
    gpu.copy_host_to_device({host(initial.data()), destination.base, 64});
    auto run = [&](size_t n, size_t m, size_t d, float p) { gpu.cdist(lhs, rhs, output, n, m, d, p, {}); };
    reject([&] { run(3, 3, 4, -1.f); });
    reject([&] { run(3, 3, 4, std::numeric_limits<float>::quiet_NaN()); });
    reject([&] { run(SIZE_MAX, 3, 4, 2.f); });
    reject([&] { run(4, 3, 4, 2.f); });
    reject([&] { run(3, 3, SIZE_MAX, 2.f); });
    reject([&] { gpu.cdist(lhs, rhs, lhs, 3, 3, 4, 2.f, {}); });
    auto wrong = rhs;
    wrong.dtype = DataType::Float16;
    reject([&] { gpu.cdist(lhs, wrong, output, 3, 3, 4, 2.f, {}); });
    auto unaligned = output;
    ++unaligned.byte_offset;
    reject([&] { gpu.cdist(lhs, rhs, unaligned, 3, 3, 4, 2.f, {}); });
    run(0, 3, 4, 2.f);
    gpu.copy_device_to_host({destination.base, host(result.data()), 64});
    require(result == initial, "Rejected/empty HIP cdist partially changed output");
    std::array<float, 12> values{};
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(i);
    gpu.copy_host_to_device({host(values.data()), lhs, 48});
    gpu.copy_host_to_device({host(values.data()), rhs, 48});
    run(3, 3, 4, 2.f);
    gpu.copy_device_to_host({destination.base, host(result.data()), 64});
    require(result[2] == 0.f && result[6] == 0.f && result[10] == 0.f && result[0] == -77.f && result[11] == -77.f,
            "HIP cdist recovery or output guards failed");
}
} // namespace

int main() try {
    GpuBackendScope scope(GpuBackend::HIP);
    for (auto shape : {std::array<size_t, 3>{0, 19, 3}, {17, 0, 3}, {3, 5, 0}, {1, 1, 3}, {17, 19, 3},
                       {17, 19, 17}, {7, 35, 33}, {263, 17, 65}, {3, 5, 263}})
        test_shape(shape[0], shape[1], shape[2]);
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    compare(Tensor::from_vector(std::vector<float>{0.f, -0.f, 1.f, inf, 1.f, 2.f, nan, 0.f, 0.f}, {3, 3}, Device::CPU),
            Tensor::from_vector(std::vector<float>{0.f, 0.f, 1.f, -inf, 2.f, 3.f, inf, 1.f, 2.f, nan, 0.f, 0.f}, {4, 3}, Device::CPU));
    test_invalid(backend_ops(GpuBackend::HIP));
    std::puts("PASS integrated HIP cdist: p=0/0.5/1/1.5/2/infinity, short/tiled/tail/empty layouts, noncontiguous operands, IEEE values, guards, invalid descriptors and recovery");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP CDIST FAILED: %s\n", error.what());
    return 1;
}
