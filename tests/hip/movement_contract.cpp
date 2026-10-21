/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor/internal/tensor_impl.hpp"
#include <core/tensor_backend.hpp>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace lfs::core;
using namespace lfs::core::internal;
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

static void upload_contract(GpuBackendOps& ops) {
    for (const auto dtype : {DataType::Float32, DataType::Float16, DataType::Int32,
                            DataType::Int64, DataType::UInt32, DataType::UInt8, DataType::Bool}) {
        const size_t width = dtype_size(dtype);
        std::vector<unsigned char> host(40 * width), expected(24 * width);
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = dtype == DataType::Bool ? static_cast<unsigned char>(i % 2) : static_cast<unsigned char>((37 * i + 11) % 251);
        size_t position = 0;
        for (size_t a = 0; a < 2; ++a)
            for (size_t b = 0; b < 3; ++b)
                for (size_t c = 0; c < 4; ++c) {
                    std::memcpy(expected.data() + position * width, host.data() + (3 + a * 20 + b * 4 + c) * width, width);
                    ++position;
                }
        StridedLayout layout{.rank = 3, .dims = {2, 3, 4}, .strides = {20, 4, 1}, .element_count = 24};
        auto input = raw_storage_ref(host.data(), dtype);
        input.byte_offset = 3 * width;
        auto output = Tensor::empty({2, 3, 4}, Device::GPU, dtype);
        ops.strided_upload(input, storage_ref(output), layout, {});
        std::fill(host.begin(), host.end(), 0); // No borrowed host pointer survives return.
        const auto actual = output.cpu();
        require(std::memcmp(actual.data_ptr(), expected.data(), expected.size()) == 0,
                "Strided HIP upload changed raw bits, layout, offset or host lifetime");
    }
    std::vector<int64_t> large{9007199254740993LL, -9007199254740995LL, 17};
    auto output = Tensor::empty({2, 3}, Device::GPU, DataType::Int64);
    const StridedLayout broadcast{.rank = 2, .dims = {2, 3}, .strides = {0, 1}, .element_count = 6};
    ops.strided_upload(raw_storage_ref(large.data(), DataType::Int64), storage_ref(output), broadcast, {});
    const auto actual = output.cpu();
    for (size_t i = 0; i < 6; ++i)
        require(actual.ptr<int64_t>()[i] == large[i % 3], "Broadcast strided HIP upload lost Int64 precision");

    auto empty = Tensor::empty({0, 3});
    ops.strided_upload(raw_storage_ref(nullptr, DataType::Float32), storage_ref(empty),
                       StridedLayout{.rank = 2, .dims = {0, 3}, .strides = {3, 1}, .element_count = 0}, {});
    auto invalid = broadcast;
    invalid.element_count = 7;
    bool rejected = false;
    try { ops.strided_upload(raw_storage_ref(large.data(), DataType::Int64), storage_ref(output), invalid, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Strided HIP upload accepted inconsistent element count");
    invalid = broadcast;
    invalid.dims[0] = SIZE_MAX;
    rejected = false;
    try { ops.strided_upload(raw_storage_ref(large.data(), DataType::Int64), storage_ref(output), invalid, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Strided HIP upload accepted overflowing shape");
}

static void padding_contract(GpuBackendOps& ops) {
    const std::vector<float> values{1, 2, 3, 4, 5, 6};
    auto source = Tensor::from_vector(values, {2, 3}, Device::GPU);
    for (const bool transposed : {false, true}) {
        const size_t h = transposed ? 3 : 2, w = transposed ? 2 : 3;
        StridedLayout in{.rank = 2, .dims = {h, w}, .strides = {transposed ? 1u : 3u, transposed ? 3u : 1u}, .element_count = 6};
        StridedLayout out{.rank = 2, .dims = {5, 7}, .strides = {7, 1}, .element_count = 35};
        auto destination = Tensor::full({5, 7}, -99.f);
        std::array<size_t, MAX_TENSOR_RANK> before{};
        before[0] = 1; before[1] = 2;
        ops.pad(storage_ref(source), storage_ref(destination), in, out, before, {});
        auto expected = std::vector<float>(35, -99.f);
        for (size_t row = 0; row < h; ++row)
            for (size_t column = 0; column < w; ++column)
                expected[(row + 1) * 7 + column + 2] = values[transposed ? column * 3 + row : row * 3 + column];
        require(destination.cpu().to_vector() == expected, "HIP padding changed the interior or border values");
        before[0] = 5;
        bool rejected = false;
        try { ops.pad(storage_ref(source), storage_ref(destination), in, out, before, {}); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && destination.cpu().to_vector() == expected, "Rejected HIP padding mutated the destination");
    }
    auto aliased = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, {8}, Device::GPU);
    const StridedLayout in{.rank = 1, .dims = {3}, .strides = {1}, .element_count = 3};
    const StridedLayout out{.rank = 1, .dims = {8}, .strides = {1}, .element_count = 8};
    std::array<size_t, MAX_TENSOR_RANK> before{}; before[0] = 2;
    ops.pad(storage_ref(aliased), storage_ref(aliased), in, out, before, {});
    require(aliased.cpu().to_vector() == std::vector<float>({1, 2, 1, 2, 3, 6, 7, 8}), "Aliased HIP padding overwrote its source");
    std::vector<int64_t> integer_values{9007199254740993LL, -9007199254740995LL, 17};
    auto integers = Tensor::from_blob(integer_values.data(), {3}, Device::CPU, DataType::Int64).gpu();
    auto padded = Tensor::zeros({8}, Device::GPU, DataType::Int64);
    ops.pad(storage_ref(integers), storage_ref(padded), in, out, before, {});
    const auto exact = padded.cpu();
    const std::vector<int64_t> expected{0, 0, 9007199254740993LL, -9007199254740995LL, 17, 0, 0, 0};
    require(std::memcmp(exact.data_ptr(), expected.data(), expected.size() * sizeof(int64_t)) == 0,
            "HIP padding converted Int64 values through floating point");
}

int main() try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)), "Cannot select HIP");
    {
        auto& ops = backend_ops(GpuBackend::HIP);
        upload_contract(ops);
        padding_contract(ops);
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)), "HIP movement retained storage");
    std::puts("PASS HIP movement: strided upload, offsets, seven dtypes, exact Int64, broadcast/empty layouts, padding, alias safety and rejected writes");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL HIP movement: %s\n", error.what());
    return 1;
}
