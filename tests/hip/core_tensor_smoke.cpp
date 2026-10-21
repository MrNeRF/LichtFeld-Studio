/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_readback.hpp>
#include <core/tensor_completion.hpp>
#include <core/tensor_upload.hpp>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace lfs::core;
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
int main() try {
    // HIP is deliberately absent from automatic/default candidate selection.
    const auto before = configured_gpu_backend();
#if LFS_HAS_CUDA
    // Exercise the existing Tensor path on either side of the HIP scope.
    auto check_cuda = [] {
        if (!gpu_backend_available(GpuBackend::CUDA)) {
            std::puts("CUDA Tensor control skipped: no visible CUDA device");
            return;
        }
        GpuBackendScope cuda(GpuBackend::CUDA);
        auto input = Tensor::from_vector(std::vector<float>{-1.f, 0.f, 1.f}, {3}, Device::GPU);
        require(gpu_backend_of(input) == GpuBackend::CUDA, "CUDA control metadata changed");
        const auto result = input.exp().mul(0.5f).add(1.f).cpu().to_vector();
        for (size_t i = 0; i < result.size(); ++i)
            require(std::abs(result[i] - (std::exp(float(i) - 1) * 0.5f + 1)) < 2e-6f, "CUDA control expression changed");
        require(input.clone().cpu().to_vector() == std::vector<float>({-1.f, 0.f, 1.f}), "CUDA control clone changed");
        auto matrix = Tensor::from_vector(std::vector<float>{0, 1, 2, 3, 4, 5}, {2, 3}, Device::GPU);
        auto indices = Tensor::from_vector(std::vector<int>{1, 0}, {2}, Device::GPU).to(DataType::Int64);
        require(matrix.index_select(0, indices).cpu().to_vector() == std::vector<float>({3, 4, 5, 0, 1, 2}), "CUDA index_select control changed");
        auto gather = Tensor::from_vector(std::vector<int>{2, 0, 0, 1}, {2, 2}, Device::GPU).to(DataType::Int64);
        require(matrix.gather(1, gather).cpu().to_vector() == std::vector<float>({2, 0, 3, 4}), "CUDA gather control changed");
        auto values = Tensor::from_vector(std::vector<float>{6, 7, 8, 9, 10, 11}, {2, 3}, Device::GPU);
        matrix.index_copy_(0, indices, values);
        require(matrix.cpu().to_vector() == std::vector<float>({9, 10, 11, 6, 7, 8}), "CUDA index_copy control changed");
        matrix.index_add_(0, indices, values);
        require(matrix.cpu().to_vector() == std::vector<float>({18, 20, 22, 12, 14, 16}), "CUDA index_add control changed");
        Tensor::manual_seed(9);
        const auto cuda_normal = Tensor::randn({263}, Device::GPU).cpu().to_vector();
        Tensor::manual_seed(9);
        require(Tensor::randn({263}, Device::GPU).cpu().to_vector() == cuda_normal, "CUDA odd normal seed control changed");
        Tensor::manual_seed(42);
    };
    check_cuda();
#endif

    {
        GpuBackendScope scope(GpuBackend::HIP);
        std::vector<float> input(263);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = (float(i) - 131) * 0.01f;
        auto a = Tensor::from_vector(input, {input.size()}, Device::GPU);
        {
            auto source = Tensor::from_vector(std::vector<float>{1,2,3,4,5,6}, {2,3}, Device::GPU);
            {
                TensorUpload upload;
                auto destination = Tensor::empty({2,3}, Device::GPU);
                auto host_source = Tensor::from_vector(std::vector<float>{7,8,9,10,11,12}, {2,3}, Device::CPU);
                upload.enqueue(destination, host_source);
                host_source.ptr<float>()[0] = -100;
                bool busy_rejected = false;
                try { upload.enqueue(destination, host_source); } catch (const std::logic_error&) { busy_rejected = true; }
                require(busy_rejected, "HIP TensorUpload accepted a busy slot");
                upload.wait();
                require(!upload.pending() && upload.poll(), "HIP TensorUpload did not finish");
                require(destination.cpu().to_vector() == std::vector<float>({7,8,9,10,11,12}), "HIP TensorUpload retained borrowed CPU pointer");
                std::vector<float> byte_source(1031);
                std::iota(byte_source.begin(), byte_source.end(), 0.f);
                auto grown = Tensor::empty({byte_source.size()}, Device::GPU);
                upload.enqueue(grown, std::as_bytes(std::span(byte_source)), TensorExecutionTarget::default_queue(GpuBackend::HIP));
                std::fill(byte_source.begin(), byte_source.end(), -5.f);
                upload.wait();
                const auto copied = grown.cpu().to_vector();
                for (size_t i = 0; i < copied.size(); ++i)
                    require(copied[i] == float(i), "HIP TensorUpload byte snapshot or staging growth failed");
                std::puts("PASS actual HIP TensorUpload: immutable CPU/span snapshots, busy slot, reuse/growth and no CUDA queue");
            }
            TensorReadback slot;
            slot.enqueue(source.transpose(0,1));
            source = {};
            std::vector<float> snapshot(6);
            slot.wait(std::as_writable_bytes(std::span(snapshot)));
            require(snapshot == std::vector<float>({1,4,2,5,3,6}) && !slot.pending(), "HIP actual TensorReadback strided snapshot failed");
            slot.enqueue_range(a, 4, 7*4);
            std::vector<float> subset(7);
            slot.wait(std::as_writable_bytes(std::span(subset)));
            require(std::equal(subset.begin(), subset.end(), input.begin()+1), "HIP actual TensorReadback range failed");
            const Tensor* captured[] = {&a};
            TensorCompletion completion(captured);
            completion.wait();
            require(completion.ready() && completion.ready(), "HIP TensorCompletion did not retain completion state");
            TensorCompletion obligation;
            obligation.include(GpuBackend::HIP);
            require(!obligation.ready(), "HIP backend-wide completion falsely ready before wait");
            obligation.wait();
            require(obligation.ready(), "HIP backend-wide completion did not settle");
            {
                TensorWorkQueue work(GpuBackend::HIP);
                Tensor payload;
                auto completion = work.execute([&] { payload = Tensor::full({263}, 3.f, Device::GPU); });
                TensorFence fence(GpuBackend::HIP);
                require(fence.ready(), "New HIP fence is not signaled");
                work.record(fence);
                work.wait_for(fence);
                work.record(fence); // re-record without a host wait
                std::atomic<bool> callback{false};
                work.enqueue_host_callback([](void* pointer) { static_cast<std::atomic<bool>*>(pointer)->store(true, std::memory_order_release); }, &callback);
                work.wait();
                completion.wait();
                require(callback.load(std::memory_order_acquire) && work.ready() && fence.ready() && fence.ready(), "HIP queue fence/callback completion failed");
                require(std::abs(payload.sum_scalar()-789.f) < 1e-4f, "HIP work queue used wrong backend");
                TensorReadback prepared;
                auto host_destination = Tensor::empty({263}, Device::CPU);
                prepared.prepare(payload, host_destination);
                prepared.enqueue(work);
                prepared.wait();
                require(host_destination.to_vector() == std::vector<float>(263,3.f), "HIP prepared readback failed");
                TensorReadbackRing ring(GpuBackend::HIP, 2, 64, work);
                ring.enqueue(payload, 4, 5*4, 0, 0);
                ring.enqueue(payload, 0, 2*4, 0, 32, true);
                ring.seal(0);
                ring.wait(0);
                require(ring.poll(0), "HIP packed checkpoint readback not ready");
                auto bytes = ring.slot_bytes(0);
                std::vector<float> packed(16);
                std::memcpy(packed.data(), bytes.data(), bytes.size());
                for (size_t i=0; i<packed.size(); ++i)
                    require(packed[i] == ((i<5 || i==8 || i==9) ? 3.f : 0.f), "HIP packed readback offset/padding failed");
                ring.release(0);
                auto second = Tensor::full({263}, -2.f, Device::GPU);
                ring.enqueue(second, 0, 64, 0, 0);
                ring.seal(0);
                ring.wait(0);
                std::memcpy(packed.data(), ring.slot_bytes(0).data(), 64);
                require(packed == std::vector<float>(16,-2.f), "HIP checkpoint readback slot reuse failed");
                ring.release(0);
                ring.seal(1);
                ring.wait(1);
                ring.release(1);
                std::puts("PASS actual HIP work queue, native callbacks, reusable fences, prepared readback and packed checkpoint ring with optional GPU staging");
            }
            auto queue = TensorExecutionTarget::default_queue(GpuBackend::HIP);
            queue.wait_for(queue);
            std::puts("PASS actual HIP TensorReadback and TensorCompletion: strided/range snapshots, ownership, repeated readiness and default-queue bridge");
        }

        require(Tensor::eye(2, 3, Device::GPU).cpu().to_vector() == std::vector<float>({1,0,0,0,1,0}), "HIP rectangular eye failed");
        require(Tensor::eye(3, 2, Device::GPU).cpu().to_vector() == std::vector<float>({1,0,0,1,0,0}), "HIP tall eye failed");
        const auto diagonal = Tensor::from_vector(std::vector<float>{-2,0,3}, {3}, Device::GPU);
        require(Tensor::diag(diagonal).cpu().to_vector() == std::vector<float>({-2,0,0,0,0,0,0,0,3}), "HIP diag failed");
        const auto dot_rhs = Tensor::from_vector(std::vector<float>{4,5,6}, {3}, Device::GPU);
        require(std::abs(diagonal.dot(dot_rhs).cpu().to_vector().at(0) - 10.f) < 1e-5f, "HIP dot failed");

        {
            std::vector<float> values(3 * 513);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = float(int(i % 9) - 4) * 0.25f;
            auto matrix = Tensor::from_vector(values, {3, 513}, Device::GPU);
            const auto unchanged = matrix.cpu().to_vector();
            for (int axis = 0; axis < 2; ++axis) {
                const auto actual = matrix.cumsum(axis).cpu().to_vector();
                auto expected = values;
                for (size_t r = 0; r < 3; ++r)
                    for (size_t c = 0; c < 513; ++c) {
                        if (axis == 0 && r)
                            expected[r * 513 + c] += expected[(r - 1) * 513 + c];
                        if (axis == 1 && c)
                            expected[r * 513 + c] += expected[r * 513 + c - 1];
                    }
                require(actual == expected, "HIP axis cumsum differs from exact CPU reference");
                require(matrix.transpose(0, 1).cumsum(1 - axis).transpose(0, 1).contiguous().cpu().to_vector() == expected,
                        "HIP noncontiguous cumsum differs from reference");
            }
            require(matrix.cpu().to_vector() == unchanged, "HIP cumsum mutated its input");
            std::vector<int32_t> integers(1000003);
            for (size_t i = 0; i < integers.size(); ++i)
                integers[i] = i % 2 ? -7 : 9;
            integers[0] += 10000000;
            auto wide = Tensor::from_blob(integers.data(), {integers.size()}, Device::CPU, DataType::Int32).to(Device::GPU);
            const auto scanned = wide.cumsum(0).cpu();
            const auto* result = static_cast<const int32_t*>(scanned.data_ptr());
            int32_t sum = 0;
            for (size_t i = 0; i < integers.size(); ++i) {
                sum += integers[i];
                require(result[i] == sum, "HIP large Int32 scan lost a prefix");
            }
            std::puts("PASS HIP inclusive scans: axes, noncontiguous input, tile tails, 1,000,003 Int32 values");
        }
        {
            const size_t dimensions[] = {3, 17, 5};
            std::vector<float> values(3 * 17 * 5);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = float(int((i * 13) % 23) - 11) * 0.25f;
            auto tensor = Tensor::from_vector(values, {3, 17, 5}, Device::GPU);
            for (int axis = 0; axis < 3; ++axis)
                for (bool descending : {false, true}) {
                    const auto [sorted, order] = tensor.sort(axis, descending);
                    const auto actual = sorted.cpu().to_vector();
                    const auto cpu_order = order.cpu();
                    const auto* indices = static_cast<const int64_t*>(cpu_order.data_ptr());
                    size_t outer = 1, inner = 1;
                    for (int d = 0; d < axis; ++d)
                        outer *= dimensions[d];
                    for (int d = axis + 1; d < 3; ++d)
                        inner *= dimensions[d];
                    for (size_t o = 0; o < outer; ++o)
                        for (size_t in = 0; in < inner; ++in) {
                            std::vector<size_t> expected(dimensions[axis]);
                            std::iota(expected.begin(), expected.end(), 0);
                            auto at = [&](size_t c) { return (o * dimensions[axis] + c) * inner + in; };
                            std::stable_sort(expected.begin(), expected.end(), [&](size_t l, size_t r) {
                                return descending ? values[at(l)] > values[at(r)] : values[at(l)] < values[at(r)];
                            });
                            for (size_t c = 0; c < expected.size(); ++c)
                                require(actual[at(c)] == values[at(expected[c])] && indices[at(c)] == int64_t(expected[c]),
                                        "HIP segmented sort values, stability or relative indices differ");
                        }
                }
            require(tensor.cpu().to_vector() == values, "HIP sort modified its source");
            std::vector<float> large(1000003);
            for (size_t i = 0; i < large.size(); ++i)
                large[i] = float(large.size() - i);
            auto [sorted, order] = Tensor::from_vector(large, {large.size()}, Device::GPU).sort();
            const auto actual = sorted.cpu().to_vector();
            const auto cpu_order = order.cpu();
            const auto* indices = static_cast<const int64_t*>(cpu_order.data_ptr());
            for (size_t i = 0; i < actual.size(); ++i)
                require(actual[i] == float(i + 1) && indices[i] == int64_t(actual.size() - i - 1), "HIP million-value sort lost data or index");
            std::puts("PASS HIP stable sort: all axes, both directions, relative indices, million-value vector");
        }
        {
            constexpr size_t length = 513;
            for (DataType dtype : {DataType::Float32, DataType::Float16, DataType::Int32, DataType::Int64, DataType::UInt8, DataType::Bool, DataType::UInt32}) {
                const size_t width = dtype_size(dtype);
                std::vector<uint8_t> raw(length * width);
                for (size_t i = 0; i < raw.size(); ++i)
                    raw[i] = uint8_t((i * 43 + 17) & 255);
                auto values = Tensor::from_blob(raw.data(), {length}, Device::CPU, dtype).to(Device::GPU);
                for (unsigned pattern = 0; pattern < 3; ++pattern) {
                    std::vector<uint8_t> mask(length), expected;
                    for (size_t i = 0; i < length; ++i) {
                        mask[i] = pattern == 0 ? 0 : pattern == 1 ? 1
                                                                  : i % 3 == 0;
                        if (mask[i])
                            expected.insert(expected.end(), raw.begin() + i * width, raw.begin() + (i + 1) * width);
                    }
                    auto flags = Tensor::from_blob(mask.data(), {length}, Device::CPU, DataType::Bool).to(Device::GPU);
                    const auto selected = values.masked_select(flags).cpu();
                    require(selected.dtype() == dtype && selected.bytes() == expected.size() &&
                                (expected.empty() || std::memcmp(selected.data_ptr(), expected.data(), expected.size()) == 0),
                            "HIP masked selection changed order, dtype or value bits");
                }
            }
            std::vector<float> values{0, -0.0f, 1, std::bit_cast<float>(0x7fc01234u), -INFINITY, 0, 2,
                                      std::bit_cast<float>(1u), std::bit_cast<float>(0x80000001u)};
            auto input_values = Tensor::from_vector(values, {values.size()}, Device::GPU);
            auto positions = input_values.nonzero().cpu();
            const std::vector<int64_t> expected{2, 3, 4, 6, 7, 8};
            require(positions.numel() == expected.size() && std::memcmp(positions.data_ptr(), expected.data(), expected.size() * 8) == 0,
                    "HIP nonzero mishandled signed zero, NaN or infinity");
            std::vector<float> large(1000003);
            std::vector<uint8_t> flags(large.size());
            std::vector<float> expected_values;
            std::vector<int64_t> expected_positions;
            for (size_t i = 0; i < large.size(); ++i) {
                large[i] = float(i % 127);
                flags[i] = i % 17 == 0;
                if (flags[i]) {
                    expected_values.push_back(large[i]);
                    expected_positions.push_back(i);
                }
            }
            auto large_input = Tensor::from_vector(large, {large.size()}, Device::GPU);
            auto large_flags = Tensor::from_blob(flags.data(), {flags.size()}, Device::CPU, DataType::Bool).to(Device::GPU);
            require(large_input.masked_select(large_flags).cpu().to_vector() == expected_values, "HIP million-value masked selection failed");
            positions = large_flags.nonzero().cpu();
            require(positions.numel() == expected_positions.size() &&
                        std::memcmp(positions.data_ptr(), expected_positions.data(), expected_positions.size() * 8) == 0,
                    "HIP million-value boolean nonzero failed");
            std::puts("PASS HIP stable selection: seven value dtypes, empty/full/sparse masks, IEEE predicates and million-value inputs");
        }
        {
            auto equal_bytes = [&](const Tensor& actual, const Tensor& expected) {
                const auto cpu = actual.cpu();
                require(cpu.dtype() == expected.dtype() && cpu.shape() == expected.shape() && cpu.bytes() == expected.bytes() &&
                            (!cpu.bytes() || std::memcmp(cpu.data_ptr(), expected.data_ptr(), cpu.bytes()) == 0),
                        "HIP concatenation/stack byte reference mismatch");
            };
            for (DataType dtype : {DataType::Float32, DataType::Float16, DataType::Int32, DataType::Int64, DataType::UInt8, DataType::Bool, DataType::UInt32}) {
                std::vector<uint8_t> raw(3 * 17 * 5 * dtype_size(dtype));
                for (size_t i = 0; i < raw.size(); ++i)
                    raw[i] = uint8_t((i * 17 + 37) & 255);
                const auto original = Tensor::from_blob(raw.data(), {3, 17, 5}, Device::CPU, dtype);
                for (bool transpose : {false, true}) {
                    const auto cpu = transpose ? original.transpose(0, 2) : original;
                    auto gpu = original.to(Device::GPU);
                    if (transpose)
                        gpu = gpu.transpose(0, 2);
                    for (int axis = 0; axis < 3; ++axis) {
                        const auto tail_cpu = cpu.slice(axis, 0, 1);
                        const auto tail_gpu = gpu.slice(axis, 0, 1);
                        equal_bytes(Tensor::cat({gpu, tail_gpu}, axis), Tensor::cat({cpu, tail_cpu}, axis));
                        auto empty_shape = cpu.shape().dims();
                        empty_shape[axis] = 0;
                        const auto empty_gpu = Tensor::empty(TensorShape(empty_shape), Device::GPU, dtype);
                        equal_bytes(Tensor::cat({empty_gpu, gpu}, axis), cpu.contiguous());
                    }
                    for (int axis : {0, 1, 3})
                        equal_bytes(Tensor::stack({gpu, gpu}, axis), Tensor::stack({cpu, cpu}, axis));
                }
            }
            std::vector<float> values(3 * 4 * 5);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = i % 3 ? float(i) : 0;
            values[7] = std::bit_cast<float>(0x7fc01234u);
            const auto cpu = Tensor::from_vector(values, {3, 4, 5}, Device::CPU);
            const auto gpu = cpu.to(Device::GPU);
            equal_bytes(gpu.nonzero(), cpu.nonzero());
            equal_bytes(gpu.transpose(0, 2).nonzero(), cpu.transpose(0, 2).nonzero());
            equal_bytes((gpu > 1).nonzero(), (cpu > 1).nonzero());
            std::puts("PASS HIP cat/stack: all axes, seven dtypes, strided/empty inputs; multidimensional nonzero coordinates exact");
        }
        require(gpu_backend_of(a) == GpuBackend::HIP, "Factory did not create HIP Tensor metadata");
        auto result = a.exp().mul(0.5f).add(1.f);
        const auto output = result.cpu().to_vector();
        require(output.size() == input.size(), "HIP expression size changed");
        for (size_t i = 0; i < input.size(); ++i)
            require(std::abs(output[i] - (std::exp(input[i]) * 0.5f + 1)) < 2e-6f, "HIP Tensor expression differs from CPU");
        std::vector<float> matrix_values(264);
        for (size_t i = 0; i < matrix_values.size(); ++i)
            matrix_values[i] = float(i);
        auto matrix = Tensor::from_vector(matrix_values, {24, 11}, Device::GPU);
        const auto transposed = matrix.transpose(0, 1).contiguous().cpu().to_vector();
        for (size_t row = 0; row < 11; ++row)
            for (size_t column = 0; column < 24; ++column)
                require(transposed[row * 24 + column] == matrix_values[column * 11 + row], "HIP Tensor transpose/contiguous failed");
        auto bias = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {11}, Device::GPU);
        const auto broadcast = matrix.add(bias).cpu().to_vector();
        for (size_t i = 0; i < matrix_values.size(); ++i)
            require(broadcast[i] == matrix_values[i] + float(i % 11 + 1), "HIP Tensor matrix/vector broadcast failed");
        auto inplace = a.clone();
        inplace.mul_(2.f).div_(2.f);
        require(inplace.cpu().to_vector() == input, "HIP in-place scalar Tensor functor dispatch failed");
        const auto columns = matrix.sum(0).cpu().to_vector();
        const auto rows = matrix.mean(1, true).cpu().to_vector();
        require(columns.size() == 11 && rows.size() == 24, "HIP Tensor axis reduction shape changed");
        for (size_t col = 0; col < 11; ++col)
            require(columns[col] == 3036.f + 24.f * float(col), "HIP Tensor column sum failed");
        for (size_t row = 0; row < 24; ++row)
            require(std::abs(rows[row] - (11.f * float(row) + 5.f)) < 3e-5f, "HIP Tensor row mean failed");
        const auto half_expected = Tensor::from_vector(input, {input.size()}, Device::CPU).to(DataType::Float16).to(DataType::Float32).to_vector();
        require(a.to(DataType::Float16).to(DataType::Float32).cpu().to_vector() == half_expected, "HIP Tensor half roundtrip differs from CPU");
        const auto byte_expected = Tensor::from_vector(input, {input.size()}, Device::CPU).to(DataType::UInt8).to(DataType::Float32).to_vector();
        require(a.to(DataType::UInt8).to(DataType::Float32).cpu().to_vector() == byte_expected, "HIP Tensor byte conversion differs from CPU");
        auto picks = Tensor::from_vector(std::vector<int>{23, 0, 12, 12}, {4}, Device::GPU).to(DataType::Int64);
        const auto selected = matrix.index_select(0, picks).cpu().to_vector();
        for (size_t row = 0; row < 4; ++row)
            for (size_t col = 0; col < 11; ++col) {
                const size_t source_rows[] = {23, 0, 12, 12};
                require(selected[row * 11 + col] == matrix_values[source_rows[row] * 11 + col], "HIP Tensor index_select failed");
            }
        auto gather_indices = Tensor::from_vector(std::vector<int>{10, 0, 5, 1, 2, 3}, {2, 3}, Device::GPU).to(DataType::Int64);
        const auto gathered = matrix.gather(1, gather_indices).cpu().to_vector();
        require(gathered == std::vector<float>({10, 0, 5, 12, 13, 14}), "HIP Tensor multidimensional gather failed");
        const auto transposed_gather = matrix.gather(1, gather_indices.transpose(0, 1)).cpu().to_vector();
        require(transposed_gather == std::vector<float>({10, 1, 11, 13, 27, 25}), "HIP Tensor gather with noncontiguous Int64 indices failed");
        int64_t wide_values[] = {INT64_MIN, INT64_MAX};
        auto wide_cpu = Tensor::from_blob(wide_values, {2}, Device::CPU, DataType::Int64);
        auto wide = wide_cpu.to(Device::GPU);
        const auto clamped = matrix.index_select(0, wide, BoundaryMode::Clamp).cpu().to_vector();
        const auto wrapped = matrix.index_select(0, wide, BoundaryMode::Wrap).cpu().to_vector();
        for (size_t row = 0; row < 2; ++row)
            for (size_t col = 0; col < 11; ++col) {
                require(clamped[row * 11 + col] == matrix_values[(row ? 23 : 0) * 11 + col], "HIP Tensor clamp narrowed an Int64 index");
                auto index = wide_values[row] % 24;
                if (index < 0)
                    index += 24;
                require(wrapped[row * 11 + col] == matrix_values[size_t(index) * 11 + col], "HIP Tensor wrap narrowed an Int64 index");
            }
        size_t nonzero = 0;
        for (float value : input)
            nonzero += value != 0;
        require(a.count_nonzero() == nonzero, "HIP Tensor float count_nonzero failed");
        require(a.ne(0.f).count_nonzero() == nonzero, "HIP Tensor predicate mask count_nonzero failed");
        require(matrix.transpose(0, 1).count_nonzero() == 263, "HIP Tensor noncontiguous count_nonzero failed");
        auto update = Tensor::from_vector(std::vector<float>{0, 1, 2, 3, 4, 5}, {2, 3}, Device::GPU);
        auto swap = Tensor::from_vector(std::vector<int>{1, 0}, {2}, Device::GPU).to(DataType::Int64);
        update.index_copy_(0, swap, update);
        require(update.cpu().to_vector() == std::vector<float>({3, 4, 5, 0, 1, 2}), "HIP aliased index_copy did not snapshot source");
        update.index_add_(0, swap, update);
        require(update.cpu().to_vector() == std::vector<float>({3, 5, 7, 3, 5, 7}), "HIP aliased index_add did not snapshot source");
        auto scatter_indices = Tensor::from_vector(std::vector<int>{2, 0}, {2}, Device::GPU).to(DataType::Int64);
        auto scatter_values = Tensor::from_vector(std::vector<float>{1, 2, 3, 4}, {2, 2}, Device::GPU);
        auto scattered = Tensor::full({2, 3}, 0.f, Device::GPU);
        scattered.scatter_(1, scatter_indices, scatter_values);
        require(scattered.cpu().to_vector() == std::vector<float>({2, 0, 1, 4, 0, 3}), "HIP Tensor scatter assignment failed");
        scattered.index_fill_(1, scatter_indices, 7.f);
        require(scattered.cpu().to_vector() == std::vector<float>({7, 0, 7, 7, 0, 7}), "HIP Tensor index_fill failed");
        auto repeated = Tensor::from_vector(std::vector<int>{0, 0, 1}, {3}, Device::GPU).to(DataType::Int64);
        auto contributions = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3}, Device::GPU);
        auto accumulate = Tensor::full({2, 3}, 0.f, Device::GPU);
        accumulate.index_add_(0, repeated, contributions);
        require(accumulate.cpu().to_vector() == std::vector<float>({5, 7, 9, 7, 8, 9}), "HIP repeated index_add lost contributions");
        auto index_owner = Tensor::from_vector(std::vector<int>{1, 0}, {2}, Device::GPU);
        auto replacement = Tensor::from_vector(std::vector<int>{9, 8}, {2}, Device::GPU);
        index_owner.index_copy_(0, index_owner, replacement);
        require(index_owner.cpu().to(DataType::Float32).to_vector() == std::vector<float>({8, 9}), "HIP index_copy did not snapshot aliased indices");
        const auto stable_values = accumulate.cpu().to_vector();
        auto invalid_updates = Tensor::from_vector(std::vector<int>{0, 1, -1}, {3}, Device::GPU).to(DataType::Int64);
        for (bool add : {false, true}) {
            bool rejected = false;
            try {
                if (add)
                    accumulate.index_add_(0, invalid_updates, contributions);
                else
                    accumulate.index_copy_(0, invalid_updates, contributions);
            } catch (const std::exception&) { rejected = true; }
            require(rejected && accumulate.cpu().to_vector() == stable_values, "Invalid Tensor indexed update partially changed destination");
        }
        auto rank_one = Tensor::from_vector(std::vector<float>{1, 2}, {2}, Device::GPU);
        rank_one.index_add_(0, swap, rank_one);
        require(rank_one.cpu().to_vector() == std::vector<float>({3, 3}), "HIP rank-one index_add alias failed");
        auto backing = Tensor::from_vector(std::vector<float>{0, 1, 2, 3, 4, 5}, {2, 3}, Device::GPU);
        auto view = backing.transpose(0, 1);
        view.index_copy_(1, swap, view);
        require(backing.cpu().to_vector() == std::vector<float>({3, 4, 5, 0, 1, 2}), "HIP indexed copy failed to write back a transposed view");
        view.index_add_(1, swap, view);
        require(backing.cpu().to_vector() == std::vector<float>({3, 5, 7, 3, 5, 7}), "HIP indexed add failed to write back a transposed view");
        Tensor::manual_seed(123);
        const auto normal_first = Tensor::randn({263}, Device::GPU).cpu().to_vector();
        const auto normal_next = Tensor::randn({263}, Device::GPU).cpu().to_vector();
        require(normal_first != normal_next, "HIP normal did not advance global random stream");
        Tensor::manual_seed(123);
        require(Tensor::randn({263}, Device::GPU).cpu().to_vector() == normal_first, "HIP normal manual_seed is not reproducible");
        const auto explicit_first = Tensor::uniform({263}, 0, 1, Device::GPU, DataType::Float32, 17).cpu().to_vector();
        const auto explicit_repeat = Tensor::uniform({263}, 0, 1, Device::GPU, DataType::Float32, 17).cpu().to_vector();
        require(explicit_first == explicit_repeat, "HIP explicit uniform seed changed sequence");
        Tensor::manual_seed(71);
        const auto global_expected = Tensor::rand({263}, Device::GPU).cpu().to_vector();
        Tensor::manual_seed(71);
        (void)Tensor::uniform({263}, 0, 1, Device::GPU, DataType::Float32, 99).cpu();
        require(Tensor::rand({263}, Device::GPU).cpu().to_vector() == global_expected, "HIP explicit seed changed global RNG state");
        auto random_backing = Tensor::full({2, 3}, 0.f, Device::GPU);
        auto random_view = random_backing.transpose(0, 1);
        Tensor::manual_seed(33);
        random_view.normal_(0, 1);
        const auto strided_normal = random_view.contiguous().cpu().to_vector();
        Tensor::manual_seed(33);
        require(Tensor::randn({3, 2}, Device::GPU).cpu().to_vector() == strided_normal, "HIP random view writeback changed draw order");
        const auto bernoulli = Tensor::bernoulli({263}, 0.37f, Device::GPU).cpu().to_vector();
        for (float value : bernoulli)
            require(value == 0 || value == 1, "HIP Tensor Bernoulli factory returned nonbinary value");
        const auto integers = Tensor::randint({263}, -7, 13, Device::GPU).to(DataType::Float32).cpu().to_vector();
        for (float value : integers)
            require(value >= -7 && value < 13 && std::floor(value) == value, "HIP Tensor randint factory returned invalid value");
        Tensor::manual_seed(42);
        const auto copy = a.clone().cpu().to_vector();
        require(copy == input, "HIP Tensor clone changed values");
        double sum = 0;
        for (float value : input)
            sum += value;
        require(std::abs(a.sum_scalar() - sum) < 1e-5, "HIP Tensor sum failed");
        auto filled = Tensor::full({263}, 0.25f, Device::GPU);
        require(std::abs(filled.sum_scalar() - 65.75f) < 1e-5, "HIP Tensor factory fill failed");
    }
    require(configured_gpu_backend() == before, "HIP scope changed process backend default");
    require(gpu_backend_live(GpuBackend::HIP), "Native HIP context was not reported live");
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)), "HIP Tensor shutdown failed after releasing tensors");
    require(!gpu_backend_live(GpuBackend::HIP), "HIP context remained live after shutdown");
#if LFS_HAS_CUDA
    check_cuda();
#endif
    std::puts("PASS actual lfs_core Tensor: HIP factory metadata, lazy expression, CPU download, clone, sum and full; process default unchanged");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP TENSOR SMOKE FAILED: %s\n", error.what());
    return 1;
}
