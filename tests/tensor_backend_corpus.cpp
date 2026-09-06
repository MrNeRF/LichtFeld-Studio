/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"
#include "core/tensor/backend/facade_trace.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using lfs::core::BoundaryMode;
using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::GpuBackend;
using lfs::core::MovementArgs;
using lfs::core::MovementOp;
using lfs::core::ScatterMode;
using lfs::core::Tensor;
using lfs::core::TensorShape;
namespace internal = lfs::core::internal;
namespace tensor_ops = lfs::core::tensor_ops;

namespace {

    constexpr uint64_t kSeed = 0x4c46535f56334b30ULL;

    // Frozen host-launcher map. This is intentionally a source table, not inferred
    // from symbols, so each lane B row remains reviewable when dispatch is refactored.
    // | launcher | public Tensor call |
    // | launch_unary_op_generic | Tensor::exp; Tensor::isfinite for UInt8 |
    // | launch_ieee_round_float | Tensor::round |
    // | launch_binary_op_generic | Tensor::add(Tensor) |
    // | launch_ieee_maximum_float | Tensor::maximum(Tensor) |
    // | launch_ieee_minimum_float | Tensor::minimum(Tensor) |
    // | launch_scalar_op_generic | Tensor::pow(float) |
    // | launch_broadcast_binary | Tensor::add(broadcast Tensor) |
    // | launch_ieee_maximum_float_broadcast | Tensor::maximum(broadcast Tensor) |
    // | launch_ieee_minimum_float_broadcast | Tensor::minimum(broadcast Tensor) |
    // | direct_sum_scalar | Tensor::sum_scalar |
    // | direct_mean_scalar | Tensor::mean_scalar |
    // | direct_max_scalar | Tensor::max_scalar |
    // | direct_min_scalar | Tensor::min_scalar |
    // | launch_reduce_op | Tensor::sum |
    // | launch_column_reduce | Tensor::sum(0) on rank 2 |
    // | launch_strided_reduce_fast | Tensor::sum(1) with inner size at least 256 |
    // | launch_fused_transform_reduce | Tensor::add(float).sum |
    // | launch_fused_segmented_transform_reduce | Tensor::add(float).sum(-1) |
    // | launch_count_nonzero_bool | Tensor::count_nonzero on Bool |
    // | launch_count_nonzero_float | Tensor::count_nonzero on Float32 |
    // | launch_cumsum | Tensor::cumsum |
    // | launch_sort_1d | Tensor::sort on rank 1 |
    // | launch_sort_2d | Tensor::sort on rank 2 |
    // | launch_gather | Tensor::gather |
    // | launch_gather_fused_unary | Tensor::gather_lazy(...).map(...).eval |
    // | launch_take | Tensor::take |
    // | launch_index_select | Tensor::index_select |
    // | launch_scatter | Tensor::scatter_ |
    // | launch_index_copy | Tensor::index_copy_ |
    // | launch_index_add | Tensor::index_add_ |
    // | launch_strided_scatter | Tensor::copy_from into rank 5 transposed view |
    // | launch_strided_scatter_immediate | Tensor::copy_from into rank 2 transposed view |
    // | launch_strided_scatter_int32_to_float32 | Tensor::copy_from Int32 into Float32 view |
    // | launch_masked_fill | Tensor::masked_fill_ |
    // | launch_masked_select | Tensor::masked_select |
    // | launch_masked_scatter | Tensor::operator[](mask) assignment |
    // | launch_and_live | Tensor::and_live_ |
    // | launch_where | Tensor::where |
    // | launch_nonzero | Tensor::nonzero on Float32 |
    // | launch_nonzero_bool | Tensor::nonzero on Bool |
    // | launch_sgemm | Tensor::mm |
    // | launch_sgemm_tn | Tensor::linear |
    // | launch_sgemm_batched | Tensor::bmm |
    // | launch_sgemm_bias_relu | Tensor::conv1x1_bias_relu_out with output_size >= 500000 |
    // | launch_dot_product | Tensor::dot |
    // | launch_diag | Tensor::diag |
    // | launch_max_pool2d | Tensor::max_pool2d |
    // | launch_adaptive_avg_pool2d | Tensor::adaptive_avg_pool2d |
    // | launch_bias_add | Tensor::linear(weight, bias) |
    // | launch_bias_relu | Tensor::linear_bias_relu_out fallback |
    // | launch_relu | Tensor::relu_out |
    // | launch_uniform | Tensor::uniform |
    // | launch_bernoulli | Tensor::bernoulli |
    // | launch_randint | Tensor::randint |
    // | launch_multinomial | Tensor::multinomial |
    // | launch_strided_copy | Tensor::contiguous on rank 5 view |
    // | launch_strided_copy_immediate | Tensor::contiguous on rank 2 view |
    // | launch_strided_upload | non-contiguous CPU Tensor::to(Device::GPU) |
    // | launch_convert_type | Tensor::to(DataType) |
    // | launch_cat_last_dim | Tensor::cat(..., last dim) |
    // | launch_cat_middle_dim | Tensor::cat(..., middle dim) |
    // | launch_pad | Tensor::movement(MovementOp::Pad) |
    // | launch_fill_strided | Tensor::fill_ on a view |
    // | launch_load_op | Tensor::full |
    // | launch_clamp_scalar | Tensor::clamp_ on Float32 |
    // | launch_clamp_fused | Tensor::clamp on Float32 |
    // | launch_clamp_scalar_int | Tensor::clamp_ on Int32 |
    // | launch_fused_pointwise_chain | chained Tensor pointwise expression materialization |
    // | launch_cdist | Tensor::cdist |
    // | launch_eye | Tensor::eye |
    // | has_nan_gpu | Tensor::has_nan |
    // | has_inf_gpu | Tensor::has_inf |

    // Scan rows hold prefix sums, so their bound scales with the largest
    // magnitude of the row instead of the element, which passes through zero.
    // Reduce rows sum many inputs of similar scale, so an output that cancels
    // toward zero still carries the rounding error of its partials; their bound
    // scales with the largest magnitude of the row.
    enum class Rule { Digest,
                      Tolerance,
                      Scan,
                      Reduce,
                      Permutation,
                      Stat };

    struct Profile {
        std::string_view name;
        size_t rows;
        size_t cols;
    };

    constexpr std::array kProfiles{
        Profile{"7", 1, 7},
        Profile{"33x127", 33, 127},
        Profile{"1024x1031", 1024, 1031},
    };

    struct Entry {
        std::string_view launcher;
        std::string_view call;
        std::span<const DataType> dtypes;
        Rule rule;
    };

    constexpr std::array kF32{DataType::Float32};
    constexpr std::array kI32{DataType::Int32};
    constexpr std::array kBool{DataType::Bool};
    constexpr std::array kF32I32{DataType::Float32, DataType::Int32};
    constexpr std::array kF32I32Bool{DataType::Float32, DataType::Int32, DataType::Bool};
    constexpr std::array kUnary{DataType::Float32, DataType::Int32, DataType::UInt8};
    constexpr std::array kBinary{DataType::Float32, DataType::Float16, DataType::Int32,
                                 DataType::Int64, DataType::UInt8};
    constexpr std::array kReduce{DataType::Float32, DataType::Float16, DataType::Int32, DataType::Bool};
    constexpr std::array kIndex{DataType::Float32, DataType::Int64, DataType::Int32, DataType::UInt8};
    constexpr std::array kScatter{DataType::Float32, DataType::Int32, DataType::UInt8};
    constexpr std::array kMask{DataType::Float32, DataType::Float16, DataType::Int32,
                               DataType::Int64, DataType::UInt8};
    constexpr std::array kSeven{DataType::Float32, DataType::Float16, DataType::Int32,
                                DataType::Int64, DataType::UInt8, DataType::Bool, DataType::UInt32};

#define ENTRY(name, call, types, rule) \
    Entry { #name, call, types, Rule::rule }
    const std::array<Entry, 72> kEntries{{
        ENTRY(launch_unary_op_generic, "Tensor::exp|Tensor::isfinite", kUnary, Digest),
        ENTRY(launch_ieee_round_float, "Tensor::round", kF32, Digest),
        ENTRY(launch_binary_op_generic, "Tensor::add(Tensor)", kBinary, Digest),
        ENTRY(launch_ieee_maximum_float, "Tensor::maximum(Tensor)", kF32, Digest),
        ENTRY(launch_ieee_minimum_float, "Tensor::minimum(Tensor)", kF32, Digest),
        ENTRY(launch_scalar_op_generic, "Tensor::pow(float)", kF32I32, Digest),
        ENTRY(launch_broadcast_binary, "Tensor::add(broadcast)", kBinary, Digest),
        ENTRY(launch_ieee_maximum_float_broadcast, "Tensor::maximum(broadcast)", kF32, Digest),
        ENTRY(launch_ieee_minimum_float_broadcast, "Tensor::minimum(broadcast)", kF32, Digest),
        ENTRY(direct_sum_scalar, "Tensor::sum_scalar", kF32, Reduce),
        ENTRY(direct_mean_scalar, "Tensor::mean_scalar", kF32, Reduce),
        ENTRY(direct_max_scalar, "Tensor::max_scalar", kF32, Digest),
        ENTRY(direct_min_scalar, "Tensor::min_scalar", kF32, Digest),
        ENTRY(launch_reduce_op, "Tensor::sum", kReduce, Reduce),
        ENTRY(launch_column_reduce, "Tensor::sum(0)", kF32, Reduce),
        ENTRY(launch_strided_reduce_fast, "Tensor::sum(1)", kF32, Reduce),
        ENTRY(launch_fused_transform_reduce, "Tensor::add(float).sum", kF32, Reduce),
        ENTRY(launch_fused_segmented_transform_reduce, "Tensor::add(float).sum(-1)", kF32, Reduce),
        ENTRY(launch_count_nonzero_bool, "Tensor::count_nonzero(Bool)", kBool, Digest),
        ENTRY(launch_count_nonzero_float, "Tensor::count_nonzero(Float32)", kF32, Digest),
        ENTRY(launch_cumsum, "Tensor::cumsum", kF32I32, Scan),
        ENTRY(launch_sort_1d, "Tensor::sort(rank1)", kF32, Permutation),
        ENTRY(launch_sort_2d, "Tensor::sort(rank2)", kF32, Permutation),
        ENTRY(launch_gather, "Tensor::gather", kF32, Digest),
        ENTRY(launch_gather_fused_unary, "Tensor::gather_lazy.map.eval", kF32, Digest),
        ENTRY(launch_take, "Tensor::take", kF32, Digest),
        ENTRY(launch_index_select, "Tensor::index_select", kIndex, Digest),
        ENTRY(launch_scatter, "Tensor::scatter_", kScatter, Digest),
        ENTRY(launch_index_copy, "Tensor::index_copy_", kScatter, Digest),
        ENTRY(launch_index_add, "Tensor::index_add_", kF32I32, Tolerance),
        ENTRY(launch_strided_scatter, "Tensor::copy_from(rank5 view)", kSeven, Digest),
        ENTRY(launch_strided_scatter_immediate, "Tensor::copy_from(rank2 view)", kSeven, Digest),
        ENTRY(launch_strided_scatter_int32_to_float32, "Tensor::copy_from(Int32,Float32 view)", kF32, Digest),
        ENTRY(launch_masked_fill, "Tensor::masked_fill_", kMask, Digest),
        ENTRY(launch_masked_select, "Tensor::masked_select", kMask, Digest),
        ENTRY(launch_masked_scatter, "Tensor::operator[](mask)=Tensor", kMask, Digest),
        ENTRY(launch_and_live, "Tensor::and_live_", kBool, Digest),
        ENTRY(launch_where, "Tensor::where", kF32, Digest),
        ENTRY(launch_nonzero, "Tensor::nonzero(Float32)", kF32, Digest),
        ENTRY(launch_nonzero_bool, "Tensor::nonzero(Bool)", kBool, Digest),
        ENTRY(launch_sgemm, "Tensor::mm", kF32, Tolerance),
        ENTRY(launch_sgemm_tn, "Tensor::linear", kF32, Tolerance),
        ENTRY(launch_sgemm_batched, "Tensor::bmm", kF32, Tolerance),
        ENTRY(launch_sgemm_bias_relu, "Tensor::conv1x1_bias_relu_out", kF32, Tolerance),
        ENTRY(launch_dot_product, "Tensor::dot", kF32, Tolerance),
        ENTRY(launch_diag, "Tensor::diag", kF32, Digest),
        ENTRY(launch_max_pool2d, "Tensor::max_pool2d", kF32, Digest),
        ENTRY(launch_adaptive_avg_pool2d, "Tensor::adaptive_avg_pool2d", kF32, Tolerance),
        ENTRY(launch_bias_add, "Tensor::linear(weight,bias)", kF32, Tolerance),
        ENTRY(launch_bias_relu, "Tensor::linear_bias_relu_out(fallback)", kF32, Tolerance),
        ENTRY(launch_relu, "Tensor::relu_out", kF32, Digest),
        ENTRY(launch_uniform, "Tensor::uniform", kF32, Stat),
        ENTRY(launch_bernoulli, "Tensor::bernoulli", kF32, Stat),
        ENTRY(launch_randint, "Tensor::randint", kI32, Stat),
        ENTRY(launch_multinomial, "Tensor::multinomial", kF32, Stat),
        ENTRY(launch_strided_copy, "Tensor::contiguous(rank5 view)", kSeven, Digest),
        ENTRY(launch_strided_copy_immediate, "Tensor::contiguous(rank2 view)", kSeven, Digest),
        ENTRY(launch_strided_upload, "Tensor::to(CUDA,CPU-view)", kSeven, Digest),
        ENTRY(launch_convert_type, "Tensor::to(DataType)", kSeven, Digest),
        ENTRY(launch_cat_last_dim, "Tensor::cat(last dim)", kF32, Digest),
        ENTRY(launch_cat_middle_dim, "Tensor::cat(middle dim)", kF32, Digest),
        ENTRY(launch_pad, "Tensor::movement(Pad)", kF32, Digest),
        ENTRY(launch_fill_strided, "Tensor::fill_(view)", kF32I32Bool, Digest),
        ENTRY(launch_load_op, "Tensor::full", kF32, Digest),
        ENTRY(launch_clamp_scalar, "Tensor::clamp_(Float32)", kF32, Digest),
        ENTRY(launch_clamp_fused, "Tensor::clamp", kF32, Digest),
        ENTRY(launch_clamp_scalar_int, "Tensor::clamp_(Int32)", kI32, Digest),
        ENTRY(launch_fused_pointwise_chain, "Tensor::add.mul.sub", kF32, Digest),
        ENTRY(launch_cdist, "Tensor::cdist", kF32, Tolerance),
        ENTRY(launch_eye, "Tensor::eye", kF32, Digest),
        ENTRY(has_nan_gpu, "Tensor::has_nan", kF32, Digest),
        ENTRY(has_inf_gpu, "Tensor::has_inf", kF32, Digest),
    }};
#undef ENTRY

    static_assert(kEntries.size() == 72);

    std::string compact_dtype_name(DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return "f32";
        case DataType::Float16: return "f16";
        case DataType::Int32: return "i32";
        case DataType::Int64: return "i64";
        case DataType::UInt8: return "u8";
        case DataType::Bool: return "bool";
        case DataType::UInt32: return "u32";
        }
        throw std::runtime_error("unknown dtype");
    }

    void cuda_check(cudaError_t status, std::string_view operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
        }
    }

    size_t elements(const Profile& profile) { return profile.rows * profile.cols; }

    Tensor make_tensor(const TensorShape& shape, DataType dtype, uint64_t seed, bool positive = false) {
        std::mt19937_64 generator(seed);
        std::uniform_real_distribution<float> real_dist(positive ? 0.05f : -2.0f, 2.0f);
        std::uniform_int_distribution<int> int_dist(positive ? 1 : -7, 7);
        const size_t count = shape.elements();
        if (dtype == DataType::Float32 || dtype == DataType::Float16 || dtype == DataType::UInt32) {
            std::vector<float> values(count);
            std::generate(values.begin(), values.end(), [&] { return real_dist(generator); });
            if (dtype == DataType::UInt32) {
                for (float& value : values)
                    value = std::abs(value * 1024.0f);
            }
            // Convert on the host so both backends receive identical bytes; a backend
            // conversion defect must fail the convert row, not every other row.
            Tensor cpu = Tensor::from_vector(values, shape, Device::CPU);
            if (dtype != DataType::Float32)
                cpu = cpu.to(dtype);
            return cpu.to(Device::GPU);
        }
        if (dtype == DataType::Bool) {
            std::vector<bool> values(count);
            for (size_t i = 0; i < count; ++i)
                values[i] = (generator() & 1U) != 0;
            return Tensor::from_vector(values, shape, Device::CPU).to(Device::GPU);
        }
        std::vector<int> values(count);
        std::generate(values.begin(), values.end(), [&] { return int_dist(generator); });
        Tensor cpu = Tensor::from_vector(values, shape, Device::CPU);
        if (dtype != DataType::Int32)
            cpu = cpu.to(dtype);
        return cpu.to(Device::GPU);
    }

    Tensor make_profile_tensor(const Profile& profile, DataType dtype, uint64_t seed,
                               bool positive = false, bool noncontiguous = false) {
        if (noncontiguous) {
            return make_tensor(TensorShape({profile.cols, profile.rows}), dtype, seed, positive)
                .transpose(0, 1);
        }
        return make_tensor(TensorShape({profile.rows, profile.cols}), dtype, seed, positive);
    }

    Tensor indices(size_t count, size_t bound, uint64_t seed, bool duplicates = false) {
        (void)seed;
        std::vector<int> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<int>(duplicates ? (i % std::min<size_t>(bound, 17)) : (i % bound));
        }
        return Tensor::from_vector(values, {count}, Device::CPU).to(Device::GPU);
    }

    struct PreparedInputs {
        Profile profile;
        DataType dtype;
        Tensor a;
        Tensor b;
        Tensor input;
        Tensor rhs;
        Tensor indices;
        Tensor source;
        Tensor mask;
        Tensor weight;
        Tensor bias;
        MovementArgs movement_args;
        std::vector<Tensor> destinations;
        std::vector<Tensor> destination_views;
        size_t next_destination = 0;
    };

    Tensor clone_preserving_layout(const Tensor& tensor, bool noncontiguous) {
        if (!noncontiguous)
            return tensor.clone();
        return tensor.transpose(0, 1).clone().transpose(0, 1);
    }

    void prepare_destinations(PreparedInputs& inputs, const Tensor& prototype, size_t count,
                              bool noncontiguous = false) {
        inputs.destinations.reserve(count);
        for (size_t i = 0; i < count; ++i)
            inputs.destinations.push_back(clone_preserving_layout(prototype, noncontiguous));
    }

    void prepare_empty_destinations(PreparedInputs& inputs, const TensorShape& shape, DataType dtype,
                                    size_t count) {
        inputs.destinations.reserve(count);
        for (size_t i = 0; i < count; ++i)
            inputs.destinations.push_back(Tensor::empty(shape, Device::GPU, dtype));
    }

    PreparedInputs prepare(size_t entry_index, const Profile& profile, DataType dtype,
                           bool noncontiguous = false, size_t execution_count = 1) {
        const auto& entry = kEntries[entry_index];
        const uint64_t seed = kSeed + entry_index * 0x10001ULL + elements(profile);
        const TensorShape shape({profile.rows, profile.cols});
        PreparedInputs inputs{.profile = profile,
                              .dtype = dtype,
                              .a = make_profile_tensor(profile, dtype, seed, false, noncontiguous),
                              .b = make_profile_tensor(profile, dtype, seed + 1, true, noncontiguous)};
        const std::string_view name = entry.launcher;

        if (name == "launch_broadcast_binary" || name == "launch_ieee_maximum_float_broadcast" ||
            name == "launch_ieee_minimum_float_broadcast") {
            inputs.rhs = make_tensor({profile.rows, 1}, dtype, seed + 2, true);
        }
        if (name == "launch_strided_reduce_fast") {
            inputs.input = make_tensor({2, 3, std::max<size_t>(profile.cols, 257)}, dtype, seed);
        }
        if (name == "has_nan_gpu" || name == "has_inf_gpu") {
            // The dirty copy is prepared here so the timed window holds only the two checks.
            std::vector<float> host = inputs.a.cpu().to_vector();
            host[(inputs.a.numel() * 7 + 3) % inputs.a.numel()] =
                name == "has_nan_gpu" ? std::numeric_limits<float>::quiet_NaN()
                                      : std::numeric_limits<float>::infinity();
            inputs.rhs = Tensor::from_vector(host, inputs.a.shape(), Device::CPU).to(Device::GPU);
        }
        if (name == "launch_sort_1d" || name == "launch_sort_2d") {
            inputs.input = name == "launch_sort_1d" ? make_tensor({elements(profile)}, dtype, seed)
                                                    : inputs.a;
        }
        if (name == "launch_gather" || name == "launch_take" ||
            name == "launch_gather_fused_unary") {
            inputs.input = inputs.a.flatten();
            inputs.indices = indices(inputs.input.numel(), inputs.input.numel(), seed + 3);
        }
        if (name == "launch_index_select") {
            inputs.indices = indices(profile.rows, profile.rows, seed + 3);
        }
        if (name == "launch_scatter" || name == "launch_index_copy" || name == "launch_index_add") {
            const size_t count = elements(profile);
            inputs.input = make_tensor({count}, dtype, seed);
            inputs.indices = indices(count, count, seed + 3, name == "launch_index_add");
            inputs.source = make_tensor({count}, dtype, seed + 4);
            prepare_destinations(inputs, inputs.input, execution_count);
        }
        if (name == "launch_strided_scatter" || name == "launch_strided_copy") {
            const size_t n = std::max<size_t>(7, std::min<size_t>(elements(profile), 1048581));
            inputs.input = make_tensor({1, 1, 1, n, 2}, dtype, seed);
            inputs.rhs = inputs.input.transpose(3, 4);
            if (name == "launch_strided_scatter") {
                inputs.source = make_tensor(inputs.rhs.shape(), dtype, seed + 4);
                inputs.destinations.reserve(execution_count);
                inputs.destination_views.reserve(execution_count);
                for (size_t i = 0; i < execution_count; ++i) {
                    inputs.destinations.push_back(inputs.input.clone());
                    inputs.destination_views.push_back(inputs.destinations.back().transpose(3, 4));
                }
            }
        }
        if (name == "launch_strided_scatter_immediate" ||
            name == "launch_strided_scatter_int32_to_float32") {
            inputs.input = make_tensor({profile.cols, profile.rows}, dtype, seed);
            inputs.rhs = inputs.input.transpose(0, 1);
            const auto source_dtype = name == "launch_strided_scatter_int32_to_float32"
                                          ? DataType::Int32
                                          : dtype;
            inputs.source = make_tensor(inputs.rhs.shape(), source_dtype, seed + 4);
            inputs.destinations.reserve(execution_count);
            inputs.destination_views.reserve(execution_count);
            for (size_t i = 0; i < execution_count; ++i) {
                inputs.destinations.push_back(inputs.input.clone());
                inputs.destination_views.push_back(inputs.destinations.back().transpose(0, 1));
            }
        }
        if (name == "launch_masked_fill" || name == "launch_masked_select" ||
            name == "launch_masked_scatter") {
            inputs.mask = inputs.a.to(DataType::Float32).gt(0.0f).contiguous();
            if (name != "launch_masked_select")
                prepare_destinations(inputs, inputs.a, execution_count, noncontiguous);
            if (name == "launch_masked_scatter") {
                const size_t selected = inputs.mask.count_nonzero();
                inputs.source = make_tensor({selected}, dtype, seed + 4, true);
            }
        }
        if (name == "launch_and_live") {
            inputs.rhs = make_profile_tensor(profile, DataType::Bool, seed + 4);
            prepare_destinations(inputs, inputs.a, execution_count, noncontiguous);
        }
        if (name == "launch_where") {
            inputs.mask = inputs.a.gt(0.0f).contiguous();
        }
        if (name == "launch_sgemm") {
            const size_t k = std::max<size_t>(3, std::min<size_t>(profile.cols, 33));
            inputs.input = make_tensor({profile.rows, k}, dtype, seed);
            inputs.rhs = make_tensor({k, profile.cols}, dtype, seed + 1);
        }
        if (name == "launch_sgemm_bias_relu") {
            const size_t spatial = elements(profile);
            const size_t out_channels = (500000 + spatial - 1) / spatial;
            inputs.input = make_tensor({1, 1, profile.rows, profile.cols}, dtype, seed);
            inputs.weight = make_tensor({out_channels, 1}, dtype, seed + 1);
            inputs.bias = make_tensor({out_channels}, dtype, seed + 2);
            prepare_empty_destinations(inputs, {1, out_channels, profile.rows, profile.cols},
                                       DataType::Float32, execution_count);
        }
        if (name == "launch_sgemm_tn" || name == "launch_bias_add" ||
            name == "launch_bias_relu") {
            const size_t in = std::max<size_t>(3, std::min<size_t>(profile.cols, 33));
            const size_t out = std::max<size_t>(5, std::min<size_t>(profile.cols, 65));
            inputs.input = make_tensor({profile.rows, in}, dtype, seed);
            inputs.weight = make_tensor({out, in}, dtype, seed + 1);
            inputs.bias = make_tensor({out}, dtype, seed + 2);
            if (name == "launch_bias_relu")
                prepare_empty_destinations(inputs, {profile.rows, out}, DataType::Float32,
                                           execution_count);
        }
        if (name == "launch_sgemm_batched") {
            const size_t batch = std::max<size_t>(1, std::min<size_t>(profile.rows, 16));
            inputs.input = make_tensor({batch, 3, 5}, dtype, seed);
            inputs.rhs = make_tensor({batch, 5, 7}, dtype, seed + 1);
        }
        if (name == "launch_dot_product") {
            inputs.input = make_tensor({elements(profile)}, dtype, seed);
            inputs.rhs = make_tensor({elements(profile)}, dtype, seed + 1);
        }
        if (name == "launch_diag") {
            const size_t diagonal_size =
                profile.rows == 1 ? 7 : std::min<size_t>(profile.cols, 1031);
            inputs.input = make_tensor({diagonal_size}, dtype, seed);
        }
        if (name == "launch_max_pool2d" || name == "launch_adaptive_avg_pool2d") {
            const size_t h = std::max<size_t>(3, profile.rows);
            const size_t w = std::max<size_t>(5, profile.cols);
            inputs.input = make_tensor({1, 1, h, w}, dtype, seed);
        }
        if (name == "launch_relu") {
            prepare_empty_destinations(inputs, shape, dtype, execution_count);
        }
        if (name == "launch_multinomial") {
            const size_t categories = std::max<size_t>(7, profile.cols);
            std::vector<float> host_weights(categories, 1.0f);
            inputs.input =
                Tensor::from_vector(host_weights, {categories}, Device::CPU).to(Device::GPU);
        }
        if (name == "launch_strided_copy_immediate") {
            inputs.input = make_tensor({profile.cols, profile.rows}, dtype, seed).transpose(0, 1);
        }
        if (name == "launch_strided_upload") {
            auto host_base = make_tensor({profile.cols, profile.rows}, dtype, seed).cpu();
            inputs.input = host_base.transpose(0, 1);
        }
        if (name == "launch_cat_middle_dim") {
            inputs.input = inputs.a.unsqueeze(0);
            inputs.rhs = inputs.b.unsqueeze(0);
        }
        if (name == "launch_pad") {
            inputs.movement_args.args = std::vector<std::pair<int, int>>{{1, 2}, {2, 1}};
        }
        if (name == "launch_fill_strided") {
            inputs.input = make_tensor({profile.cols, profile.rows}, dtype, seed);
            inputs.destinations.reserve(execution_count);
            inputs.destination_views.reserve(execution_count);
            for (size_t i = 0; i < execution_count; ++i) {
                inputs.destinations.push_back(inputs.input.clone());
                inputs.destination_views.push_back(inputs.destinations.back().transpose(0, 1));
            }
        }
        if (name == "launch_clamp_scalar" || name == "launch_clamp_scalar_int") {
            prepare_destinations(inputs, inputs.a, execution_count, noncontiguous);
        }
        if (name == "launch_cdist") {
            const size_t features = std::max<size_t>(3, std::min<size_t>(profile.cols, 16));
            inputs.input = make_tensor({profile.rows, features}, dtype, seed);
            inputs.rhs =
                make_tensor({std::max<size_t>(3, profile.rows / 2), features}, dtype, seed + 1);
        }
        return inputs;
    }

    Tensor& next_destination(PreparedInputs& inputs) {
        if (inputs.next_destination >= inputs.destinations.size())
            throw std::runtime_error("prepared destination count exhausted");
        return inputs.destinations[inputs.next_destination++];
    }

    std::vector<Tensor> execute(size_t entry_index, PreparedInputs& inputs) {
        const std::string_view name = kEntries[entry_index].launcher;
        const TensorShape shape({inputs.profile.rows, inputs.profile.cols});
        auto& a = inputs.a;
        auto& b = inputs.b;

        if (name == "launch_unary_op_generic")
            return {inputs.dtype == DataType::UInt8 ? a.isfinite() : a.exp()};
        if (name == "launch_ieee_round_float")
            return {a.round()};
        if (name == "launch_binary_op_generic")
            return {a.add(b)};
        if (name == "launch_ieee_maximum_float")
            return {a.maximum(b)};
        if (name == "launch_ieee_minimum_float")
            return {a.minimum(b)};
        if (name == "launch_scalar_op_generic")
            return {a.pow(2)};
        if (name == "launch_broadcast_binary")
            return {a.add(inputs.rhs)};
        if (name == "launch_ieee_maximum_float_broadcast")
            return {a.maximum(inputs.rhs)};
        if (name == "launch_ieee_minimum_float_broadcast")
            return {a.minimum(inputs.rhs)};
        if (name == "direct_sum_scalar")
            return {Tensor::from_vector({a.sum_scalar()}, {1}, Device::CPU)};
        if (name == "direct_mean_scalar")
            return {Tensor::from_vector({a.mean_scalar()}, {1}, Device::CPU)};
        if (name == "direct_max_scalar")
            return {Tensor::from_vector({a.max_scalar()}, {1}, Device::CPU)};
        if (name == "direct_min_scalar")
            return {Tensor::from_vector({a.min_scalar()}, {1}, Device::CPU)};
        if (name == "launch_reduce_op")
            return {a.sum()};
        if (name == "launch_column_reduce")
            return {a.sum(0)};
        if (name == "launch_strided_reduce_fast")
            return {inputs.input.sum(1)};
        if (name == "launch_fused_transform_reduce")
            return {a.add(0.25f).sum()};
        if (name == "launch_fused_segmented_transform_reduce")
            return {a.add(0.25f).sum(-1)};
        if (name == "launch_count_nonzero_bool" || name == "launch_count_nonzero_float")
            return {Tensor::from_vector({static_cast<int>(a.count_nonzero())}, {1}, Device::CPU)};
        if (name == "launch_cumsum")
            return {a.cumsum(-1)};
        if (name == "launch_sort_1d" || name == "launch_sort_2d") {
            auto [values, order] = inputs.input.sort(-1, false);
            return {values, order, inputs.input.contiguous()};
        }
        if (name == "launch_gather")
            return {inputs.input.gather(0, inputs.indices)};
        if (name == "launch_take")
            return {inputs.input.take(inputs.indices)};
        if (name == "launch_gather_fused_unary")
            return {inputs.input.gather_lazy(inputs.indices).map(lfs::core::ops::abs_op{}).eval()};
        if (name == "launch_index_select")
            return {a.index_select(0, inputs.indices)};
        if (name == "launch_scatter" || name == "launch_index_copy" || name == "launch_index_add") {
            auto& destination = next_destination(inputs);
            if (name == "launch_scatter")
                destination.scatter_(0, inputs.indices, inputs.source, ScatterMode::None);
            if (name == "launch_index_copy")
                destination.index_copy_(0, inputs.indices, inputs.source);
            if (name == "launch_index_add")
                destination.index_add_(0, inputs.indices, inputs.source);
            return {destination};
        }
        if (name == "launch_strided_copy")
            return {inputs.rhs.contiguous()};
        if (name == "launch_strided_scatter" || name == "launch_strided_scatter_immediate" ||
            name == "launch_strided_scatter_int32_to_float32") {
            const size_t destination_index = inputs.next_destination;
            auto& destination = next_destination(inputs);
            inputs.destination_views[destination_index].copy_from(inputs.source);
            return {destination};
        }
        if (name == "launch_masked_fill" || name == "launch_masked_scatter") {
            auto& destination = next_destination(inputs);
            if (name == "launch_masked_fill")
                destination.masked_fill_(inputs.mask, 1.0f);
            else
                destination[inputs.mask] = inputs.source;
            return {destination};
        }
        if (name == "launch_masked_select")
            return {a.masked_select(inputs.mask)};
        if (name == "launch_and_live") {
            auto& destination = next_destination(inputs);
            destination.and_live_(inputs.rhs);
            return {destination};
        }
        if (name == "launch_where")
            return {Tensor::where(inputs.mask, a, b)};
        if (name == "launch_nonzero" || name == "launch_nonzero_bool")
            return {a.nonzero()};
        if (name == "launch_sgemm")
            return {inputs.input.mm(inputs.rhs)};
        if (name == "launch_sgemm_bias_relu") {
            auto& output = next_destination(inputs);
            inputs.input.conv1x1_bias_relu_out(inputs.weight, inputs.bias, output);
            return {output};
        }
        if (name == "launch_sgemm_tn")
            return {inputs.input.linear(inputs.weight)};
        if (name == "launch_bias_add")
            return {inputs.input.linear(inputs.weight, inputs.bias)};
        if (name == "launch_bias_relu") {
            auto& output = next_destination(inputs);
            inputs.input.linear_bias_relu_out(inputs.weight, inputs.bias, output);
            return {output};
        }
        if (name == "launch_sgemm_batched")
            return {inputs.input.bmm(inputs.rhs)};
        if (name == "launch_dot_product")
            return {inputs.input.dot(inputs.rhs)};
        if (name == "launch_diag")
            return {Tensor::diag(inputs.input)};
        if (name == "launch_max_pool2d")
            return {inputs.input.max_pool2d(2, 2, 0)};
        if (name == "launch_adaptive_avg_pool2d")
            return {inputs.input.adaptive_avg_pool2d(3, 5)};
        if (name == "launch_relu") {
            auto& output = next_destination(inputs);
            a.relu_out(output);
            return {output};
        }
        if (name == "launch_uniform")
            return {Tensor::uniform(shape, -2.0f, 2.0f)};
        if (name == "launch_bernoulli")
            return {Tensor::bernoulli(shape, 0.35f)};
        if (name == "launch_randint")
            return {Tensor::randint(shape, -7, 8)};
        if (name == "launch_multinomial")
            return {Tensor::multinomial(inputs.input, elements(inputs.profile), true)};
        if (name == "launch_strided_copy_immediate")
            return {inputs.input.contiguous()};
        if (name == "launch_strided_upload")
            return {inputs.input.to(Device::GPU)};
        if (name == "launch_convert_type") {
            const DataType target =
                inputs.dtype == DataType::Float32 ? DataType::Int32 : DataType::Float32;
            return {a.to(target)};
        }
        if (name == "launch_cat_last_dim")
            return {Tensor::cat({a, b}, 1)};
        if (name == "launch_cat_middle_dim")
            return {Tensor::cat({inputs.input, inputs.rhs}, 1)};
        if (name == "launch_pad")
            return {a.movement(MovementOp::Pad, inputs.movement_args)};
        if (name == "launch_fill_strided") {
            const size_t destination_index = inputs.next_destination;
            auto& destination = next_destination(inputs);
            inputs.destination_views[destination_index].fill_(1.0f);
            return {destination};
        }
        if (name == "launch_load_op")
            return {Tensor::full(shape, 1.25f, Device::GPU, inputs.dtype)};
        if (name == "launch_clamp_scalar" || name == "launch_clamp_scalar_int") {
            auto& destination = next_destination(inputs);
            destination.clamp_(-1.0f, 1.0f);
            return {destination};
        }
        if (name == "launch_clamp_fused")
            return {a.clamp(-1.0f, 1.0f)};
        if (name == "launch_fused_pointwise_chain") {
            tensor_ops::FusedPointwiseOpChain chain{};
            chain.num_ops = 3;
            chain.ops[0].kind = static_cast<uint8_t>(internal::PointwiseOp::AddScalar);
            chain.ops[0].scalar = 0.25f;
            chain.ops[1].kind = static_cast<uint8_t>(internal::PointwiseOp::MulScalar);
            chain.ops[1].scalar = 1.5f;
            chain.ops[2].kind = static_cast<uint8_t>(internal::PointwiseOp::SubScalar);
            chain.ops[2].scalar = 0.5f;
            Tensor output = internal::allocate_like(a, a.shape(), DataType::Float32);
            internal::backend_ops_for(a).fused_pointwise_chain(
                internal::storage_ref(a), internal::storage_ref(output),
                output.numel(), chain, std::span<const internal::StorageRef>{},
                internal::ExecContext{output.stream()});
            return {std::move(output)};
        }
        if (name == "launch_cdist")
            return {inputs.input.cdist(inputs.rhs)};
        if (name == "launch_eye")
            return {Tensor::eye(inputs.profile.rows, inputs.profile.cols)};
        if (name == "has_nan_gpu" || name == "has_inf_gpu") {
            // The clean input answers false; the copy with one injected special value
            // (prepared outside the timed window) answers true, so a constant-false
            // entry fails the row.
            const Tensor& dirty = inputs.rhs;
            const bool clean_result = name == "has_nan_gpu" ? a.has_nan() : a.has_inf();
            const bool dirty_result = name == "has_nan_gpu" ? dirty.has_nan() : dirty.has_inf();
            return {Tensor::from_vector(std::vector<float>{clean_result ? 1.0f : 0.0f,
                                                           dirty_result ? 1.0f : 0.0f},
                                        {2}, Device::CPU)};
        }
        throw std::runtime_error("no runner for " + std::string(name));
    }

    std::vector<Tensor> run_entry(size_t entry_index, const Profile& profile, DataType dtype,
                                  bool noncontiguous = false) {
        auto inputs = prepare(entry_index, profile, dtype, noncontiguous);
        return execute(entry_index, inputs);
    }

    class Sha256 {
    public:
        void update(std::span<const std::byte> bytes) {
            bit_length_ += static_cast<uint64_t>(bytes.size()) * 8;
            for (std::byte byte : bytes) {
                block_[block_size_++] = std::to_integer<uint8_t>(byte);
                if (block_size_ == 64)
                    transform();
            }
        }

        std::string finish() {
            block_[block_size_++] = 0x80;
            if (block_size_ > 56) {
                while (block_size_ < 64)
                    block_[block_size_++] = 0;
                transform();
            }
            while (block_size_ < 56)
                block_[block_size_++] = 0;
            for (int shift = 56; shift >= 0; shift -= 8)
                block_[block_size_++] = static_cast<uint8_t>(bit_length_ >> shift);
            transform();
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (uint32_t word : state_)
                out << std::setw(8) << word;
            return out.str();
        }

    private:
        static constexpr std::array<uint32_t, 64> k{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::array<uint32_t, 8> state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::array<uint8_t, 64> block_{};
        size_t block_size_ = 0;
        uint64_t bit_length_ = 0;

        void transform() {
            std::array<uint32_t, 64> words{};
            for (size_t i = 0; i < 16; ++i)
                words[i] = (uint32_t{block_[4 * i]} << 24) | (uint32_t{block_[4 * i + 1]} << 16) |
                           (uint32_t{block_[4 * i + 2]} << 8) | uint32_t{block_[4 * i + 3]};
            for (size_t i = 16; i < 64; ++i) {
                const uint32_t s0 = std::rotr(words[i - 15], 7) ^ std::rotr(words[i - 15], 18) ^
                                    (words[i - 15] >> 3);
                const uint32_t s1 = std::rotr(words[i - 2], 17) ^ std::rotr(words[i - 2], 19) ^
                                    (words[i - 2] >> 10);
                words[i] = words[i - 16] + s0 + words[i - 7] + s1;
            }
            auto [a, b, c, d, e, f, g, h] = state_;
            for (size_t i = 0; i < 64; ++i) {
                const uint32_t s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
                const uint32_t choice = (e & f) ^ (~e & g);
                const uint32_t temp1 = h + s1 + choice + k[i] + words[i];
                const uint32_t s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
                const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
                const uint32_t temp2 = s0 + majority;
                h = g;
                g = f;
                f = e;
                e = d + temp1;
                d = c;
                c = b;
                b = a;
                a = temp1 + temp2;
            }
            state_[0] += a;
            state_[1] += b;
            state_[2] += c;
            state_[3] += d;
            state_[4] += e;
            state_[5] += f;
            state_[6] += g;
            state_[7] += h;
            block_size_ = 0;
        }
    };

    std::vector<std::byte> download(const std::vector<Tensor>& outputs) {
        for (const auto& output : outputs)
            (void)output.data_ptr();
        const bool has_cuda_output = std::ranges::any_of(outputs, [](const Tensor& output) {
            return output.device() == Device::GPU &&
                   lfs::core::gpu_backend_of(output) == GpuBackend::CUDA;
        });
        if (has_cuda_output)
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize before download");
        std::vector<std::byte> bytes;
        for (const auto& output : outputs) {
            const Tensor host = output.device() == Device::CPU ? output.contiguous()
                                                               : output.cpu().contiguous();
            const auto* begin = static_cast<const std::byte*>(host.data_ptr());
            bytes.insert(bytes.end(), begin, begin + host.bytes());
        }
        return bytes;
    }

    std::string digest(std::span<const std::byte> bytes) {
        Sha256 hash;
        hash.update(bytes);
        return hash.finish();
    }

    bool integral_dtype(DataType dtype) {
        return dtype == DataType::Int32 || dtype == DataType::Int64 || dtype == DataType::UInt8 ||
               dtype == DataType::UInt32 || dtype == DataType::Bool;
    }

    // Wrap-around integer arithmetic is associative, so integral inputs on a
    // tolerance entry are bitwise-reproducible and compared by digest.
    Rule effective_rule(const Entry& entry, DataType dtype) {
        const bool tolerance = entry.rule == Rule::Tolerance || entry.rule == Rule::Scan ||
                               entry.rule == Rule::Reduce;
        return tolerance && integral_dtype(dtype) ? Rule::Digest : entry.rule;
    }

    // Scan rows carry their line length so the comparator can bound each
    // prefix sum by the largest magnitude of its own line.
    std::string rule_name(Rule rule, const Profile& profile) {
        switch (rule) {
        case Rule::Digest: return "digest";
        case Rule::Tolerance: return "tolerance:1e-5:1e-6";
        case Rule::Scan: return "tolerance-scan:1e-5:1e-6:" + std::to_string(profile.cols);
        case Rule::Reduce: return "tolerance-reduce:1e-5:1e-6";
        case Rule::Permutation: return "permutation";
        case Rule::Stat: return "stat";
        }
        throw std::runtime_error("unknown rule");
    }

    std::string statistics(const std::vector<Tensor>& outputs, std::string_view launcher,
                           const Profile& profile) {
        auto values = outputs.front().to(DataType::Float32).cpu().to_vector();
        if (values.empty())
            return "mean=0,var=0,min=0,max=0,ks=0";
        const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0.0;
        for (float value : values)
            variance += (value - mean) * (value - mean);
        variance /= values.size();
        std::sort(values.begin(), values.end());
        const float minimum = values.front();
        const float maximum = values.back();
        const auto theoretical_cdf = [&](const float value) {
            if (launcher == "launch_uniform")
                return std::clamp((value + 2.0) / 4.0, 0.0, 1.0);
            if (launcher == "launch_bernoulli")
                return value < 1.0f ? 0.65 : 1.0;
            if (launcher == "launch_randint")
                return std::clamp((value + 8.0) / 15.0, 0.0, 1.0);
            if (launcher == "launch_multinomial")
                return std::clamp((value + 1.0) /
                                      static_cast<double>(std::max<size_t>(7, profile.cols)),
                                  0.0, 1.0);
            return 0.0;
        };
        // Evaluate the empirical CDF once per distinct value, on both sides of the
        // step, so ties in discrete outputs do not inflate the statistic. The left
        // side compares against the theoretical CDF just below the value, which for
        // a discrete distribution is the CDF of the previous atom.
        double ks = 0.0;
        for (size_t i = 0; i < values.size();) {
            size_t j = i;
            while (j < values.size() && values[j] == values[i])
                ++j;
            const double cdf = theoretical_cdf(values[i]);
            const double cdf_below = i == 0 ? 0.0 : theoretical_cdf(values[i - 1]);
            ks = std::max(ks, std::abs(static_cast<double>(j) / values.size() - cdf));
            ks = std::max(ks, std::abs(static_cast<double>(i) / values.size() - cdf_below));
            i = j;
        }
        auto min_it = &minimum;
        auto max_it = &maximum;
        std::ostringstream out;
        out << std::setprecision(6) << "mean=" << mean << ",var=" << variance << ",min=" << *min_it
            << ",max=" << *max_it << ",ks=" << ks;
        return out.str();
    }

    struct Options {
        std::filesystem::path output;
        GpuBackend backend = GpuBackend::CUDA;
        std::string only;
        bool dump = false;
        bool time = false;
        std::filesystem::path reference;
    };

    Options parse_options(int argc, char** argv) {
        Options options;
        std::string backend;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--backend" && i + 1 < argc)
                backend = argv[++i];
            else if (arg == "--out" && i + 1 < argc)
                options.output = argv[++i];
            else if (arg == "--only" && i + 1 < argc)
                options.only = argv[++i];
            else if (arg == "--dump")
                options.dump = true;
            else if (arg == "--time")
                options.time = true;
            else if (arg == "--reference" && i + 1 < argc)
                options.reference = argv[++i];
            else
                throw std::runtime_error(
                    "usage: tensor_backend_corpus --backend cuda|vulkan --out DIR [--dump] "
                    "[--time] [--only LAUNCHER_SUBSTRING] [--reference DIR]");
        }
        if (backend == "vulkan")
            options.backend = GpuBackend::Vulkan;
        else if (backend != "cuda")
            throw std::runtime_error("--backend cuda|vulkan is required");
        if (options.output.empty())
            throw std::runtime_error("--out DIR is required");
        if (!options.reference.empty())
            options.dump = true;
        return options;
    }

    bool is_vulkan_not_implemented(const std::exception& error) {
        const std::string_view message(error.what());
        return message.contains("Vulkan backend:") &&
               message.contains("is not implemented yet");
    }

    std::string not_implemented_line(const Entry& entry, const Profile& profile,
                                     const DataType dtype, const std::exception& error) {
        std::string message(error.what());
        std::ranges::replace(message, '\n', ' ');
        std::ostringstream line;
        line << entry.launcher << ' ' << entry.call << ' ' << profile.name << ' '
             << compact_dtype_name(dtype) << " notimpl " << message;
        return line.str();
    }

    // CUDA rows are timed on the device between two events; Vulkan rows are
    // timed on the host from the call to the completion of every submitted
    // batch, so a Vulkan timing includes recording and submission and is only
    // comparable with another Vulkan timing.
    std::array<double, 3> time_case_vulkan(size_t entry_index, PreparedInputs& inputs,
                                           size_t warmup_count, size_t sample_count) {
        auto& ops = internal::backend_ops(GpuBackend::Vulkan);
        for (size_t i = 0; i < warmup_count; ++i) {
            const auto outputs = execute(entry_index, inputs);
            for (const auto& output : outputs)
                (void)output.data_ptr();
        }
        ops.synchronize_stream(internal::ExecContext{});
        std::vector<double> samples;
        samples.reserve(sample_count);
        for (size_t i = 0; i < sample_count; ++i) {
            if (kEntries[entry_index].rule == Rule::Stat)
                Tensor::manual_seed(kSeed + entry_index);
            const auto begin = std::chrono::steady_clock::now();
            const auto outputs = execute(entry_index, inputs);
            for (const auto& output : outputs)
                (void)output.data_ptr();
            ops.synchronize_stream(internal::ExecContext{});
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        }
        std::sort(samples.begin(), samples.end());
        return {samples[50], samples[25], samples[75]};
    }

    std::array<double, 3> time_case(size_t entry_index, const Profile& profile, DataType dtype,
                                    bool noncontiguous, GpuBackend backend) {
        constexpr size_t warmup_count = 5;
        constexpr size_t sample_count = 100;
        auto inputs = prepare(entry_index, profile, dtype, noncontiguous,
                              warmup_count + sample_count);
        if (backend == GpuBackend::Vulkan)
            return time_case_vulkan(entry_index, inputs, warmup_count, sample_count);
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
        cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
        for (size_t i = 0; i < warmup_count; ++i) {
            const auto outputs = execute(entry_index, inputs);
            for (const auto& output : outputs)
                (void)output.data_ptr();
        }
        std::vector<double> samples;
        samples.reserve(sample_count);
        for (size_t i = 0; i < sample_count; ++i) {
            if (kEntries[entry_index].rule == Rule::Stat)
                Tensor::manual_seed(kSeed + entry_index);
            cuda_check(cudaEventRecord(start), "cudaEventRecord start");
            const auto outputs = execute(entry_index, inputs);
            for (const auto& output : outputs)
                (void)output.data_ptr();
            cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
            cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
            float milliseconds = 0.0f;
            cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
            samples.push_back(milliseconds * 1000.0);
        }
        cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
        cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");
        std::sort(samples.begin(), samples.end());
        return {samples[50], samples[25], samples[75]};
    }

    bool accepts_noncontiguous(std::string_view launcher) {
        constexpr std::array names{
            "launch_unary_op_generic", "launch_ieee_round_float", "launch_binary_op_generic",
            "launch_ieee_maximum_float", "launch_ieee_minimum_float", "launch_scalar_op_generic",
            "launch_broadcast_binary", "launch_ieee_maximum_float_broadcast",
            "launch_ieee_minimum_float_broadcast", "launch_reduce_op", "launch_count_nonzero_bool",
            "launch_count_nonzero_float", "launch_cumsum", "launch_sort_2d", "launch_index_select",
            "launch_masked_fill", "launch_masked_select", "launch_masked_scatter", "launch_where",
            "launch_nonzero", "launch_nonzero_bool", "launch_relu", "launch_convert_type",
            "launch_cat_last_dim", "launch_cat_middle_dim", "launch_pad", "launch_clamp_scalar",
            "launch_clamp_fused", "launch_clamp_scalar_int", "has_nan_gpu", "has_inf_gpu"};
        return std::ranges::find(names, launcher) != names.end();
    }

    using FacadeEntry = internal::FacadeEntry;

    // Facade entries a launcher row may legitimately reach. Rows whose public call
    // the lazy executor routes through the fused chain above the lazy size
    // threshold, or eagerly below it, list every entry that is correct for them;
    // the recorded entry is written next to the digest so the op matrix reports
    // the kernel that ran, not the one the row is named after.
    const std::map<std::string_view, std::vector<FacadeEntry>> kExpectedEntries{
        {"launch_unary_op_generic", {FacadeEntry::unary, FacadeEntry::fused_pointwise_chain}},
        {"launch_ieee_round_float", {FacadeEntry::unary, FacadeEntry::fused_pointwise_chain}},
        {"launch_binary_op_generic", {FacadeEntry::binary, FacadeEntry::fused_pointwise_chain}},
        {"launch_ieee_maximum_float", {FacadeEntry::binary}},
        {"launch_ieee_minimum_float", {FacadeEntry::binary}},
        {"launch_scalar_op_generic", {FacadeEntry::scalar, FacadeEntry::unary, FacadeEntry::fused_pointwise_chain}},
        {"launch_broadcast_binary", {FacadeEntry::broadcast_binary}},
        {"launch_ieee_maximum_float_broadcast", {FacadeEntry::broadcast_binary}},
        {"launch_ieee_minimum_float_broadcast", {FacadeEntry::broadcast_binary}},
        {"direct_sum_scalar", {FacadeEntry::sum_scalar}},
        {"direct_mean_scalar", {FacadeEntry::mean_scalar}},
        {"direct_max_scalar", {FacadeEntry::max_scalar}},
        {"direct_min_scalar", {FacadeEntry::min_scalar}},
        {"launch_reduce_op", {FacadeEntry::reduce}},
        {"launch_column_reduce", {FacadeEntry::column_reduce}},
        {"launch_strided_reduce_fast", {FacadeEntry::strided_reduce}},
        {"launch_fused_transform_reduce", {FacadeEntry::fused_transform_reduce, FacadeEntry::reduce, FacadeEntry::sum_scalar}},
        {"launch_fused_segmented_transform_reduce", {FacadeEntry::fused_segmented_transform_reduce, FacadeEntry::reduce}},
        {"launch_count_nonzero_bool", {FacadeEntry::count_nonzero_bool}},
        {"launch_count_nonzero_float", {FacadeEntry::count_nonzero_float}},
        {"launch_cumsum", {FacadeEntry::cumsum}},
        {"launch_sort_1d", {FacadeEntry::sort_1d}},
        {"launch_sort_2d", {FacadeEntry::sort_2d}},
        {"launch_gather", {FacadeEntry::gather, FacadeEntry::index_select}},
        {"launch_gather_fused_unary", {FacadeEntry::gather_fused_unary}},
        {"launch_take", {FacadeEntry::take}},
        {"launch_index_select", {FacadeEntry::index_select}},
        {"launch_scatter", {FacadeEntry::scatter}},
        {"launch_index_copy", {FacadeEntry::index_copy}},
        {"launch_index_add", {FacadeEntry::index_add}},
        {"launch_strided_scatter", {FacadeEntry::strided_scatter}},
        {"launch_strided_scatter_immediate", {FacadeEntry::strided_scatter_immediate}},
        {"launch_strided_scatter_int32_to_float32", {FacadeEntry::strided_scatter_int32_to_float32}},
        {"launch_masked_fill", {FacadeEntry::masked_fill}},
        {"launch_masked_select", {FacadeEntry::masked_select}},
        {"launch_masked_scatter", {FacadeEntry::masked_scatter}},
        {"launch_and_live", {FacadeEntry::and_live}},
        {"launch_where", {FacadeEntry::where}},
        {"launch_nonzero", {FacadeEntry::nonzero, FacadeEntry::count_nonzero_float}},
        {"launch_nonzero_bool", {FacadeEntry::nonzero_bool, FacadeEntry::count_nonzero_bool}},
        {"launch_sgemm", {FacadeEntry::sgemm}},
        {"launch_sgemm_tn", {FacadeEntry::sgemm_tn}},
        {"launch_sgemm_batched", {FacadeEntry::sgemm_batched}},
        {"launch_sgemm_bias_relu", {FacadeEntry::sgemm_bias_relu}},
        {"launch_dot_product", {FacadeEntry::dot_product}},
        {"launch_diag", {FacadeEntry::diag}},
        {"launch_max_pool2d", {FacadeEntry::max_pool2d}},
        {"launch_adaptive_avg_pool2d", {FacadeEntry::adaptive_avg_pool2d}},
        {"launch_bias_add", {FacadeEntry::bias_add}},
        {"launch_bias_relu", {FacadeEntry::bias_relu}},
        {"launch_relu", {FacadeEntry::relu}},
        {"launch_uniform", {FacadeEntry::uniform}},
        {"launch_bernoulli", {FacadeEntry::bernoulli}},
        {"launch_randint", {FacadeEntry::randint}},
        {"launch_multinomial", {FacadeEntry::multinomial}},
        {"launch_strided_copy", {FacadeEntry::strided_copy}},
        {"launch_strided_copy_immediate", {FacadeEntry::strided_copy_immediate}},
        {"launch_strided_upload", {FacadeEntry::strided_upload}},
        {"launch_convert_type", {FacadeEntry::convert_type}},
        {"launch_cat_last_dim", {FacadeEntry::cat_last_dim}},
        {"launch_cat_middle_dim", {FacadeEntry::cat_middle_dim}},
        {"launch_pad", {FacadeEntry::pad}},
        {"launch_fill_strided", {FacadeEntry::fill_strided}},
        {"launch_load_op", {FacadeEntry::load_fill}},
        {"launch_clamp_scalar", {FacadeEntry::clamp_scalar}},
        {"launch_clamp_fused", {FacadeEntry::clamp_fused}},
        {"launch_clamp_scalar_int", {FacadeEntry::clamp_scalar_int}},
        {"launch_fused_pointwise_chain", {FacadeEntry::fused_pointwise_chain, FacadeEntry::scalar}},
        {"launch_cdist", {FacadeEntry::cdist}},
        {"launch_eye", {FacadeEntry::eye}},
        {"has_nan_gpu", {FacadeEntry::has_nan}},
        {"has_inf_gpu", {FacadeEntry::has_inf}},
    };

    std::string executed_entries(const Entry& entry,
                                 const std::array<uint64_t, internal::kFacadeEntryCount>& before,
                                 const std::array<uint64_t, internal::kFacadeEntryCount>& after) {
        const auto expected = kExpectedEntries.find(entry.launcher);
        if (expected == kExpectedEntries.end())
            throw std::runtime_error(std::string("no facade entry expectation for ") +
                                     std::string(entry.launcher));
        std::string reached;
        bool matched = false;
        for (size_t index = 0; index < internal::kFacadeEntryCount; ++index) {
            if (after[index] == before[index])
                continue;
            const auto facade_entry = static_cast<FacadeEntry>(index);
            if (std::ranges::find(expected->second, facade_entry) != expected->second.end()) {
                matched = true;
                if (!reached.empty())
                    reached += ',';
                reached += std::string(internal::facade_entry_name(facade_entry)) + '=' +
                           std::to_string(after[index] - before[index]);
            }
        }
        if (!matched) {
            std::string advanced;
            for (size_t index = 0; index < internal::kFacadeEntryCount; ++index) {
                if (after[index] != before[index])
                    advanced += (advanced.empty() ? "" : ",") +
                                std::string(internal::facade_entry_name(static_cast<FacadeEntry>(index)));
            }
            // LFS_CORPUS_TRACE_SURVEY=1 records every mismatch in entries.txt instead of
            // stopping at the first, so the expectation table can be corrected in one run.
            if (std::getenv("LFS_CORPUS_TRACE_SURVEY") != nullptr)
                return "UNEXPECTED:" + (advanced.empty() ? std::string("none") : advanced);
            throw std::runtime_error(std::string(entry.launcher) +
                                     " reached none of its expected facade entries; advanced: " +
                                     (advanced.empty() ? "none" : advanced));
        }
        return reached;
    }

    std::vector<std::string> corpus_pass(const Options& options, bool write_files,
                                         double* aggregate_median_ms = nullptr) {
        std::vector<std::string> lines;
        std::ofstream timing;
        std::ofstream entries_file;
        if (write_files && options.time)
            timing.open(options.output / "timing.txt");
        if (write_files)
            entries_file.open(options.output / "entries.txt");
        internal::facade_trace_enable_for_testing(true);
        double median_sum_us = 0.0;
        size_t case_index = 0;
        for (size_t entry_index = 0; entry_index < kEntries.size(); ++entry_index) {
            const auto& entry = kEntries[entry_index];
            if (!options.only.empty() && entry.launcher.find(options.only) == std::string_view::npos)
                continue;
            std::vector<std::pair<Profile, bool>> variants;
            for (const auto& profile : kProfiles)
                variants.emplace_back(profile, false);
            if (accepts_noncontiguous(entry.launcher))
                variants.emplace_back(Profile{"33x127T", 33, 127}, true);
            for (const auto& [profile, noncontiguous] : variants) {
                for (DataType dtype : entry.dtypes) {
                    try {
                        const Rule rule = effective_rule(entry, dtype);
                        if (rule == Rule::Stat)
                            Tensor::manual_seed(kSeed + entry_index);
                        const auto trace_before = internal::facade_trace_snapshot_for_testing();
                        auto outputs = run_entry(entry_index, profile, dtype, noncontiguous);
                        auto bytes = download(outputs);
                        const auto trace_after = internal::facade_trace_snapshot_for_testing();
                        const std::string reached = executed_entries(entry, trace_before, trace_after);
                        if (rule == Rule::Permutation) {
                            const auto order = outputs.at(1).cpu().to_vector_int64();
                            const size_t width = entry.launcher == "launch_sort_1d" ? order.size()
                                                                                    : profile.cols;
                            for (size_t offset = 0; offset < order.size(); offset += width) {
                                std::vector<int64_t> row(
                                    order.begin() + static_cast<ptrdiff_t>(offset),
                                    order.begin() + static_cast<ptrdiff_t>(offset + width));
                                std::sort(row.begin(), row.end());
                                for (size_t i = 0; i < row.size(); ++i) {
                                    if (row[i] != static_cast<int64_t>(i))
                                        throw std::runtime_error("sort indices failed permutation check");
                                }
                            }
                            // The permutation must reproduce the sorted values: gather the
                            // unsorted input through the indices and compare bitwise.
                            const auto sorted_values = outputs.front().cpu().to_vector();
                            const auto unsorted_values = outputs.at(2).cpu().to_vector();
                            for (size_t offset = 0; offset < order.size(); offset += width) {
                                for (size_t i = 0; i < width; ++i) {
                                    const auto source = static_cast<size_t>(order[offset + i]);
                                    const float expected = unsorted_values[offset + source];
                                    const float actual = sorted_values[offset + i];
                                    if (std::bit_cast<uint32_t>(expected) != std::bit_cast<uint32_t>(actual))
                                        throw std::runtime_error("sort indices do not reproduce the sorted values");
                                }
                            }
                            bytes = download({outputs.front()});
                        }
                        const std::string result =
                            rule == Rule::Stat ? statistics(outputs, entry.launcher, profile)
                                               : digest(bytes);
                        std::ostringstream line;
                        line << entry.launcher << ' ' << entry.call << ' ' << profile.name << ' '
                             << compact_dtype_name(dtype) << ' ' << rule_name(rule, profile) << ' ' << result;
                        lines.push_back(line.str());
                        if (write_files)
                            entries_file << case_index << ' ' << entry.launcher << ' ' << profile.name
                                         << ' ' << compact_dtype_name(dtype) << ' ' << reached << '\n';
                        if (write_files && options.dump) {
                            std::ofstream dump(options.output / (std::to_string(case_index) + ".bin"),
                                               std::ios::binary);
                            dump.write(reinterpret_cast<const char*>(bytes.data()),
                                       static_cast<std::streamsize>(bytes.size()));
                        }
                        if (write_files && options.time) {
                            const auto quartiles =
                                time_case(entry_index, profile, dtype, noncontiguous,
                                          options.backend);
                            median_sum_us += quartiles[0];
                            timing << entry.launcher << ' ' << profile.name << ' ' << compact_dtype_name(dtype) << ' '
                                   << std::fixed << std::setprecision(3) << quartiles[0] << ' ' << quartiles[1]
                                   << ' ' << quartiles[2] << '\n';
                        }
                    } catch (const std::exception& error) {
                        if (options.backend != GpuBackend::Vulkan ||
                            !is_vulkan_not_implemented(error)) {
                            throw;
                        }
                        lines.push_back(not_implemented_line(entry, profile, dtype, error));
                        if (write_files)
                            std::cout << lines.back() << '\n';
                    }
                    ++case_index;
                }
            }
        }
        if (write_files && options.time) {
            *aggregate_median_ms = median_sum_us / 1000.0;
            timing << "aggregate_median_ms " << std::fixed << std::setprecision(3)
                   << *aggregate_median_ms << '\n';
        }
        return lines;
    }

} // namespace

namespace {

    float half_to_float(const uint16_t bits) {
        const uint32_t sign = (bits & 0x8000u) << 16;
        const uint32_t exponent = (bits >> 10) & 0x1fu;
        const uint32_t mantissa = bits & 0x3ffu;
        uint32_t word;
        if (exponent == 0) {
            if (mantissa == 0) {
                word = sign;
            } else {
                uint32_t m = mantissa;
                int e = -1;
                do {
                    ++e;
                    m <<= 1;
                } while ((m & 0x400u) == 0);
                word = sign | ((127 - 15 - e) << 23) | ((m & 0x3ffu) << 13);
            }
        } else if (exponent == 31) {
            word = sign | 0x7f800000u | (mantissa << 13);
        } else {
            word = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
        return std::bit_cast<float>(word);
    }

    std::vector<float> as_floats(const std::vector<std::byte>& bytes, const std::string& dtype) {
        std::vector<float> values;
        if (dtype == "f16") {
            values.resize(bytes.size() / 2);
            for (size_t i = 0; i < values.size(); ++i) {
                uint16_t half;
                std::memcpy(&half, bytes.data() + i * 2, 2);
                values[i] = half_to_float(half);
            }
        } else {
            values.resize(bytes.size() / 4);
            std::memcpy(values.data(), bytes.data(), values.size() * 4);
        }
        return values;
    }

    // Compares this run's rows and dumped bytes against a --dump reference run:
    // digest and permutation rows must match textually, tolerance rows are compared
    // element-wise within rtol scaled by log2(n), stat rows compare mean and
    // variance. Returns the number of failing rows.
    size_t compare_against_reference(const Options& options, const std::vector<std::string>& lines) {
        std::ifstream reference_manifest(options.reference / "manifest.txt");
        if (!reference_manifest)
            throw std::runtime_error("reference manifest not found under " + options.reference.string());
        std::vector<std::string> reference_lines;
        for (std::string line; std::getline(reference_manifest, line);)
            reference_lines.push_back(line);
        if (reference_lines.size() != lines.size())
            throw std::runtime_error("reference manifest row count differs");
        size_t failures = 0;
        size_t identical = 0;
        size_t within = 0;
        double worst_error = 0.0;
        std::string worst_row;
        const auto read_bytes = [](const std::filesystem::path& file) {
            std::ifstream stream(file, std::ios::binary);
            const std::vector<char> chars((std::istreambuf_iterator<char>(stream)),
                                          std::istreambuf_iterator<char>());
            std::vector<std::byte> bytes(chars.size());
            std::memcpy(bytes.data(), chars.data(), chars.size());
            return bytes;
        };
        for (size_t i = 0; i < lines.size(); ++i) {
            if (lines[i] == reference_lines[i]) {
                ++identical;
                continue;
            }
            std::istringstream fields(lines[i]);
            std::string launcher, call, profile, dtype, rule;
            fields >> launcher >> call >> profile >> dtype >> rule;
            {
                std::istringstream reference_fields(reference_lines[i]);
                std::string reference_key[4];
                reference_fields >> reference_key[0] >> reference_key[1] >> reference_key[2] >> reference_key[3];
                if (reference_key[0] != launcher || reference_key[1] != call ||
                    reference_key[2] != profile || reference_key[3] != dtype) {
                    std::cerr << "row " << i << " key differs from the reference: " << lines[i]
                              << " vs " << reference_lines[i] << '\n';
                    ++failures;
                    continue;
                }
            }
            const auto reference_file = options.reference / (std::to_string(i) + ".bin");
            const auto candidate_file = options.output / (std::to_string(i) + ".bin");
            const bool scan = rule.starts_with("tolerance-scan:");
            const bool reduce = rule.starts_with("tolerance-reduce:");
            if (scan || reduce || rule.starts_with("tolerance:")) {
                if (!std::filesystem::is_regular_file(reference_file) ||
                    !std::filesystem::is_regular_file(candidate_file)) {
                    std::cerr << "row " << i << " has no dumped bytes to compare: " << lines[i] << '\n';
                    ++failures;
                    continue;
                }
                const auto reference = as_floats(read_bytes(reference_file), dtype);
                const auto candidate = as_floats(read_bytes(candidate_file), dtype);
                if (reference.size() != candidate.size()) {
                    std::cerr << "row " << i << " element count differs\n";
                    ++failures;
                    continue;
                }
                const double rtol = 1e-5 * std::max(1.0, std::log2(static_cast<double>(std::max<size_t>(2, reference.size()))));
                const double atol = 1e-6;
                // Scan rows: the floor is the largest magnitude of the element's own
                // scan line (line length is the last rule field). Reduce rows: the
                // largest magnitude of the row.
                size_t line_length = reference.size();
                if (scan) {
                    const auto last_colon = rule.rfind(':');
                    line_length = std::max<size_t>(1, std::stoul(rule.substr(last_colon + 1)));
                }
                std::vector<double> floors(reference.size(), 0.0);
                if (scan || reduce) {
                    const size_t span = scan ? line_length : reference.size();
                    for (size_t start = 0; start < reference.size(); start += span) {
                        const size_t stop = std::min(reference.size(), start + span);
                        double floor = 0.0;
                        for (size_t k = start; k < stop; ++k)
                            if (!std::isnan(reference[k]) && !std::isinf(reference[k]))
                                floor = std::max(floor, std::abs(static_cast<double>(reference[k])));
                        std::fill(floors.begin() + start, floors.begin() + stop, floor);
                    }
                }
                double row_worst = 0.0;
                bool row_ok = true;
                for (size_t k = 0; k < reference.size(); ++k) {
                    const double a = reference[k];
                    const double b = candidate[k];
                    if (std::isnan(a) || std::isnan(b)) {
                        if (std::isnan(a) != std::isnan(b))
                            row_ok = false;
                        continue;
                    }
                    const double error = std::abs(a - b);
                    const double bound = atol + rtol * std::max(std::abs(a), floors[k]);
                    row_worst = std::max(row_worst, std::abs(a) > 0 ? error / std::abs(a) : error);
                    if (error > bound)
                        row_ok = false;
                }
                if (row_worst > worst_error) {
                    worst_error = row_worst;
                    worst_row = lines[i];
                }
                if (row_ok) {
                    ++within;
                } else {
                    std::cerr << "row " << i << " exceeds tolerance (max relative error "
                              << row_worst << "): " << lines[i] << '\n';
                    ++failures;
                }
            } else if (rule == "stat") {
                const auto parse = [](const std::string& line, const char* key) {
                    const auto position = line.find(key);
                    return position == std::string::npos ? 0.0 : std::stod(line.substr(position + std::strlen(key)));
                };
                // Two generators only agree in distribution: the candidate must pass
                // the Kolmogorov test against the theoretical distribution at its
                // sample size (1% level), and the moments must match the reference
                // once the sample is large enough for them to be stable.
                size_t samples = 1;
                for (const Profile& candidate_profile : kProfiles) {
                    if (profile.starts_with(candidate_profile.name))
                        samples = candidate_profile.rows * candidate_profile.cols;
                }
                const double ks_b = parse(lines[i], "ks=");
                const double ks_critical = 1.63 / std::sqrt(static_cast<double>(samples));
                const double mean_a = parse(reference_lines[i], "mean=");
                const double mean_b = parse(lines[i], "mean=");
                const double var_a = parse(reference_lines[i], "var=");
                const double var_b = parse(lines[i], "var=");
                // Five standard errors of the sample mean and of the sample variance.
                const double count = static_cast<double>(samples);
                const double mean_bound = 5.0 * std::sqrt(std::max(var_a, 1e-12) / count) + 1e-9;
                const double variance_bound = 5.0 * var_a * std::sqrt(2.0 / count) + 1e-9;
                const bool moments_ok =
                    samples < 1000 ||
                    (std::abs(mean_a - mean_b) <= mean_bound && std::abs(var_a - var_b) <= variance_bound);
                if (ks_b <= ks_critical && moments_ok) {
                    ++within;
                } else {
                    std::cerr << "row " << i << " statistics differ (ks " << ks_b << " critical "
                              << ks_critical << "): " << lines[i] << " vs " << reference_lines[i] << '\n';
                    ++failures;
                }
            } else {
                std::cerr << "row " << i << " differs from the reference: " << lines[i] << '\n';
                ++failures;
            }
        }
        std::cout << "reference compare: " << identical << " identical, " << within
                  << " within rule, " << failures << " failing";
        if (!worst_row.empty())
            std::cout << "; max relative error " << worst_error << " at " << worst_row;
        std::cout << '\n';
        return failures;
    }

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto backend_status = lfs::core::set_default_gpu_backend(options.backend);
        if (!backend_status)
            throw std::runtime_error(
                std::string(backend_status.error().user_message()));
        if (options.backend == GpuBackend::CUDA) {
            int device = -1;
            cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        }
        std::filesystem::create_directories(options.output);
        double aggregate_median_ms = 0.0;
        const auto first = corpus_pass(options, true, &aggregate_median_ms);
        const auto second = corpus_pass(options, false);
        size_t tolerated_differences = 0;
        if (first.size() != second.size())
            throw std::runtime_error("in-process manifest case count changed");
        for (size_t i = 0; i < first.size(); ++i) {
            if (first[i] == second[i])
                continue;
            if (first[i].find(" tolerance") != std::string::npos ||
                first[i].find(" permutation ") != std::string::npos) {
                ++tolerated_differences;
                std::cerr << "nondeterministic tolerance-rule case " << i << "\nfirst: "
                          << first[i] << "\nsecond: " << second[i] << '\n';
                continue;
            }
            throw std::runtime_error("in-process manifest self-check failed at case " +
                                     std::to_string(i));
        }
        std::ofstream manifest(options.output / "manifest.txt");
        for (const auto& line : first)
            manifest << line << '\n';
        std::cout << "tensor_backend_corpus: " << first.size()
                  << " cases, self-check passed with " << tolerated_differences
                  << " tolerance-rule differences\n";
        if (options.time)
            std::cout << "aggregate_median_ms " << std::fixed << std::setprecision(3)
                      << aggregate_median_ms << '\n';
        if (!options.reference.empty() && compare_against_reference(options, first) != 0)
            return 2;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tensor_backend_corpus: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "tensor_backend_corpus: unknown failure\n";
        return 1;
    }
}
