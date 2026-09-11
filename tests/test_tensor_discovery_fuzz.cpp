/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace lfs::core;

namespace {

    enum class ViewKind {
        Transpose,
        Permute,
        Slice,
        Unsqueeze,
        Squeeze,
        Expand,
    };

    struct ViewStep {
        ViewKind kind{};
        int first = 0;
        int second = 0;
        int64_t start = 0;
        int64_t end = 0;
        std::vector<int64_t> dimensions;
    };

    enum class PointwiseKind {
        AddScalar,
        MulScalar,
        Abs,
        Neg,
        Clamp,
        Square,
    };

    struct PointwiseStep {
        PointwiseKind kind{};
        float scalar = 0.0f;
    };

    enum class TerminalOp {
        Identity,
        Cast,
        AddScalar,
        MulScalar,
        Abs,
        Neg,
        Clamp,
        CompareScalar,
        SumAll,
        SumDim,
        SumAxes,
        MeanAll,
        MeanDim,
        StdDim,
        VarDim,
        Cumsum,
        Nonzero,
        Sort,
        InplaceAdd,
        BroadcastAdd,
        MaskedSelect,
        IndexSelect,
        Cat,
        Stack,
        Matmul,
        MinWithIndices,
        MaxWithIndices,
        SerializeRoundTrip,
    };

    struct FuzzProgram {
        uint64_t seed = 0;
        size_t iteration = 0;
        DataType dtype = DataType::Float32;
        Device device = Device::CPU;
        std::vector<int64_t> base_shape;
        std::vector<float> values;
        std::vector<ViewStep> views;
        std::vector<PointwiseStep> pointwise;
        TerminalOp terminal = TerminalOp::Identity;
        DataType cast_dtype = DataType::Float32;
        int dim = 0;
        std::vector<int> axes;
        float scalar = 1.0f;
        bool keepdim = false;
        bool unbiased = false;
        bool descending = false;
    };

    struct LfsRun {
        std::vector<Tensor> outputs;
        std::string exception;
    };

    struct TorchRun {
        std::vector<torch::Tensor> outputs;
        std::string exception;
    };

    constexpr std::array<DataType, 6> kDtypes = {
        DataType::Float32,
        DataType::Float16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::Bool,
    };

    std::string_view terminal_name(const TerminalOp op) {
        switch (op) {
        case TerminalOp::Identity: return "identity";
        case TerminalOp::Cast: return "cast";
        case TerminalOp::AddScalar: return "add_scalar";
        case TerminalOp::MulScalar: return "mul_scalar";
        case TerminalOp::Abs: return "abs";
        case TerminalOp::Neg: return "neg";
        case TerminalOp::Clamp: return "clamp";
        case TerminalOp::CompareScalar: return "compare_scalar";
        case TerminalOp::SumAll: return "sum_all";
        case TerminalOp::SumDim: return "sum_dim";
        case TerminalOp::SumAxes: return "sum_axes";
        case TerminalOp::MeanAll: return "mean_all";
        case TerminalOp::MeanDim: return "mean_dim";
        case TerminalOp::StdDim: return "std_dim";
        case TerminalOp::VarDim: return "var_dim";
        case TerminalOp::Cumsum: return "cumsum";
        case TerminalOp::Nonzero: return "nonzero";
        case TerminalOp::Sort: return "sort";
        case TerminalOp::InplaceAdd: return "inplace_add";
        case TerminalOp::BroadcastAdd: return "broadcast_add";
        case TerminalOp::MaskedSelect: return "masked_select";
        case TerminalOp::IndexSelect: return "index_select";
        case TerminalOp::Cat: return "cat";
        case TerminalOp::Stack: return "stack";
        case TerminalOp::Matmul: return "matmul";
        case TerminalOp::MinWithIndices: return "min_with_indices";
        case TerminalOp::MaxWithIndices: return "max_with_indices";
        case TerminalOp::SerializeRoundTrip: return "serialize_round_trip";
        }
        return "unknown";
    }

    std::string_view view_name(const ViewKind kind) {
        switch (kind) {
        case ViewKind::Transpose: return "transpose";
        case ViewKind::Permute: return "permute";
        case ViewKind::Slice: return "slice/narrow";
        case ViewKind::Unsqueeze: return "unsqueeze";
        case ViewKind::Squeeze: return "squeeze";
        case ViewKind::Expand: return "expand";
        }
        return "unknown";
    }

    std::string_view pointwise_name(const PointwiseKind kind) {
        switch (kind) {
        case PointwiseKind::AddScalar: return "add";
        case PointwiseKind::MulScalar: return "mul";
        case PointwiseKind::Abs: return "abs";
        case PointwiseKind::Neg: return "neg";
        case PointwiseKind::Clamp: return "clamp";
        case PointwiseKind::Square: return "square";
        }
        return "unknown";
    }

    torch::ScalarType torch_dtype(const DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return torch::kFloat32;
        case DataType::Float16: return torch::kFloat16;
        case DataType::Int32: return torch::kInt32;
        case DataType::Int64: return torch::kInt64;
        case DataType::UInt8: return torch::kUInt8;
        case DataType::Bool: return torch::kBool;
        }
        throw std::runtime_error("unsupported fuzz dtype");
    }

    DataType lfs_dtype(const torch::ScalarType dtype) {
        switch (dtype) {
        case torch::kFloat32: return DataType::Float32;
        case torch::kFloat16: return DataType::Float16;
        case torch::kInt32: return DataType::Int32;
        case torch::kInt64: return DataType::Int64;
        case torch::kUInt8: return DataType::UInt8;
        case torch::kBool: return DataType::Bool;
        default: throw std::runtime_error("unsupported Torch result dtype");
        }
    }

    size_t shape_numel(const std::vector<int64_t>& shape) {
        size_t result = 1;
        for (const int64_t extent : shape) {
            if (extent == 0) {
                return 0;
            }
            result *= static_cast<size_t>(extent);
        }
        return result;
    }

    std::vector<size_t> lfs_shape(const std::vector<int64_t>& shape) {
        std::vector<size_t> result;
        result.reserve(shape.size());
        for (const int64_t extent : shape) {
            result.push_back(static_cast<size_t>(extent));
        }
        return result;
    }

    torch::Tensor make_torch_base_cpu(const FuzzProgram& program) {
        auto source = torch::tensor(program.values, torch::kFloat32).reshape(program.base_shape);
        return source.to(torch_dtype(program.dtype));
    }

    torch::Tensor make_torch_base(const FuzzProgram& program) {
        auto result = make_torch_base_cpu(program);
        return program.device == Device::CUDA ? result.cuda() : result;
    }

    Tensor make_lfs_base(const FuzzProgram& program) {
        auto result = Tensor::from_vector(
            program.values, TensorShape(lfs_shape(program.base_shape)), Device::CPU);
        if (program.dtype != DataType::Float32) {
            result = result.to(program.dtype);
        }
        return program.device == Device::CUDA ? result.cuda() : result;
    }

    torch::Tensor apply_view(torch::Tensor tensor, const ViewStep& step) {
        switch (step.kind) {
        case ViewKind::Transpose:
            return tensor.transpose(step.first, step.second);
        case ViewKind::Permute:
            return tensor.permute(step.dimensions);
        case ViewKind::Slice:
            return tensor.slice(step.first, step.start, step.end);
        case ViewKind::Unsqueeze:
            return tensor.unsqueeze(step.first);
        case ViewKind::Squeeze:
            return tensor.squeeze(step.first);
        case ViewKind::Expand:
            return tensor.expand(step.dimensions);
        }
        throw std::runtime_error("unknown Torch view step");
    }

    Tensor apply_view(Tensor tensor, const ViewStep& step) {
        switch (step.kind) {
        case ViewKind::Transpose:
            return tensor.transpose(step.first, step.second);
        case ViewKind::Permute: {
            std::vector<int> dimensions(step.dimensions.begin(), step.dimensions.end());
            return tensor.permute(dimensions);
        }
        case ViewKind::Slice:
            return tensor.slice(static_cast<size_t>(step.first),
                                static_cast<size_t>(step.start),
                                static_cast<size_t>(step.end));
        case ViewKind::Unsqueeze:
            return tensor.unsqueeze(step.first);
        case ViewKind::Squeeze:
            return tensor.squeeze(step.first);
        case ViewKind::Expand: {
            std::vector<int> dimensions(step.dimensions.begin(), step.dimensions.end());
            return tensor.expand(dimensions);
        }
        }
        throw std::runtime_error("unknown LFS view step");
    }

    torch::Tensor apply_views(torch::Tensor tensor, const FuzzProgram& program) {
        for (const auto& step : program.views) {
            tensor = apply_view(std::move(tensor), step);
        }
        return tensor;
    }

    Tensor apply_views(Tensor tensor, const FuzzProgram& program) {
        std::optional<Tensor> current;
        current.emplace(std::move(tensor));
        for (const auto& step : program.views) {
            Tensor next = apply_view(std::move(*current), step);
            current.reset();
            current.emplace(std::move(next));
        }
        return std::move(*current);
    }

    torch::Tensor apply_pointwise(torch::Tensor tensor, const FuzzProgram& program) {
        for (const auto& step : program.pointwise) {
            switch (step.kind) {
            case PointwiseKind::AddScalar: tensor = tensor.add(step.scalar); break;
            case PointwiseKind::MulScalar: tensor = tensor.mul(step.scalar); break;
            case PointwiseKind::Abs: tensor = tensor.abs(); break;
            case PointwiseKind::Neg: tensor = tensor.neg(); break;
            case PointwiseKind::Clamp: tensor = tensor.clamp(-2.0f, 2.0f); break;
            case PointwiseKind::Square: tensor = tensor.square(); break;
            }
        }
        return tensor;
    }

    Tensor apply_pointwise(Tensor tensor, const FuzzProgram& program) {
        std::optional<Tensor> current;
        current.emplace(std::move(tensor));
        for (const auto& step : program.pointwise) {
            Tensor next = [&]() -> Tensor {
                switch (step.kind) {
                case PointwiseKind::AddScalar: return current->add(step.scalar);
                case PointwiseKind::MulScalar: return current->mul(step.scalar);
                case PointwiseKind::Abs: return current->abs();
                case PointwiseKind::Neg: return current->neg();
                case PointwiseKind::Clamp: return current->clamp(-2.0f, 2.0f);
                case PointwiseKind::Square: return current->square();
                }
                throw std::runtime_error("unknown LFS pointwise step");
            }();
            current.reset();
            current.emplace(std::move(next));
        }
        return std::move(*current);
    }

    Tensor make_lfs_rhs(const Tensor& input, const size_t rows, const size_t columns) {
        std::vector<float> data(rows * columns);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
        }
        auto result = Tensor::from_vector(data, {rows, columns}, Device::CPU);
        return input.device() == Device::CUDA ? result.cuda() : result;
    }

    torch::Tensor make_torch_rhs(const torch::Tensor& input,
                                 const int64_t rows,
                                 const int64_t columns) {
        auto result = torch::arange(rows * columns, torch::kFloat32)
                          .remainder(7)
                          .sub(3)
                          .reshape({rows, columns});
        return input.is_cuda() ? result.cuda() : result;
    }

    LfsRun run_lfs(const FuzzProgram& program) {
        try {
            Tensor input = apply_pointwise(
                apply_views(make_lfs_base(program), program), program);
            switch (program.terminal) {
            case TerminalOp::Identity:
                return {{input}, {}};
            case TerminalOp::Cast:
                return {{input.to(program.cast_dtype)}, {}};
            case TerminalOp::AddScalar:
                return program.dtype == DataType::Int32
                           ? LfsRun{{input.add(static_cast<int>(program.scalar))}, {}}
                           : LfsRun{{input.add(program.scalar)}, {}};
            case TerminalOp::MulScalar:
                return program.dtype == DataType::Int32
                           ? LfsRun{{input.mul(static_cast<int>(program.scalar))}, {}}
                           : LfsRun{{input.mul(program.scalar)}, {}};
            case TerminalOp::Abs:
                return {{input.abs()}, {}};
            case TerminalOp::Neg:
                return {{input.neg()}, {}};
            case TerminalOp::Clamp:
                return {{input.clamp(-1, 2)}, {}};
            case TerminalOp::CompareScalar:
                return program.dtype == DataType::Float32
                           ? LfsRun{{input.gt(program.scalar)}, {}}
                           : LfsRun{{input.gt(static_cast<int>(program.scalar))}, {}};
            case TerminalOp::SumAll:
                return {{input.sum()}, {}};
            case TerminalOp::SumDim:
                return {{input.sum(program.dim, program.keepdim)}, {}};
            case TerminalOp::SumAxes:
                return {{input.sum(program.axes, program.keepdim)}, {}};
            case TerminalOp::MeanAll:
                return {{input.mean()}, {}};
            case TerminalOp::MeanDim:
                return {{input.mean(program.dim, program.keepdim)}, {}};
            case TerminalOp::StdDim:
                return {{input.std(program.dim, program.keepdim, program.unbiased)}, {}};
            case TerminalOp::VarDim:
                return {{input.var(program.dim, program.keepdim, program.unbiased)}, {}};
            case TerminalOp::Cumsum:
                return {{input.cumsum(program.dim)}, {}};
            case TerminalOp::Nonzero:
                return {{input.nonzero()}, {}};
            case TerminalOp::Sort: {
                auto [values, indices] = input.sort(program.dim, program.descending);
                return {{std::move(values), std::move(indices)}, {}};
            }
            case TerminalOp::InplaceAdd:
                if (program.dtype == DataType::Int32) {
                    input.add_(static_cast<int>(program.scalar));
                } else {
                    input.add_(program.scalar);
                }
                return {{input}, {}};
            case TerminalOp::BroadcastAdd: {
                std::vector<size_t> shape(input.ndim(), 1);
                const auto rhs = Tensor::full(TensorShape(shape), program.scalar,
                                              input.device(), input.dtype());
                return {{input.add(rhs)}, {}};
            }
            case TerminalOp::MaskedSelect: {
                const auto mask = input.gt(program.scalar);
                return {{input.masked_select(mask)}, {}};
            }
            case TerminalOp::IndexSelect: {
                const int resolved_dim = program.dim < 0
                                             ? program.dim + static_cast<int>(input.ndim())
                                             : program.dim;
                const size_t extent = input.size(static_cast<size_t>(resolved_dim));
                std::vector<int> indices(extent);
                for (size_t i = 0; i < extent; ++i) {
                    indices[i] = static_cast<int>(extent - i - 1);
                }
                auto index = Tensor::from_vector(indices, {indices.size()}, Device::CPU)
                                 .to(DataType::Int64);
                if (input.device() == Device::CUDA) {
                    index = index.cuda();
                }
                return {{input.index_select(program.dim, index)}, {}};
            }
            case TerminalOp::Cat:
                return {{Tensor::cat({input, input}, program.dim)}, {}};
            case TerminalOp::Stack:
                return {{Tensor::stack({input, input}, program.dim)}, {}};
            case TerminalOp::Matmul: {
                const size_t rows = input.size(1);
                const auto rhs = make_lfs_rhs(input, rows, 2);
                return {{input.matmul(rhs)}, {}};
            }
            case TerminalOp::MinWithIndices: {
                auto [values, indices] = input.min_with_indices(program.dim, program.keepdim);
                return {{std::move(values), std::move(indices)}, {}};
            }
            case TerminalOp::MaxWithIndices: {
                auto [values, indices] = input.max_with_indices(program.dim, program.keepdim);
                return {{std::move(values), std::move(indices)}, {}};
            }
            case TerminalOp::SerializeRoundTrip: {
                std::stringstream stream;
                stream << input;
                Tensor loaded;
                stream >> loaded;
                return {{std::move(loaded)}, {}};
            }
            }
        } catch (const std::exception& error) {
            return {{}, error.what()};
        } catch (...) {
            return {{}, "non-standard exception"};
        }
        return {{}, "unreachable terminal"};
    }

    TorchRun run_torch(const FuzzProgram& program) {
        try {
            auto input = apply_pointwise(
                apply_views(make_torch_base(program), program), program);
            switch (program.terminal) {
            case TerminalOp::Identity:
                return {{input}, {}};
            case TerminalOp::Cast:
                return {{input.to(torch_dtype(program.cast_dtype))}, {}};
            case TerminalOp::AddScalar:
                return program.dtype == DataType::Int32
                           ? TorchRun{{input.add(static_cast<int64_t>(program.scalar))}, {}}
                           : TorchRun{{input.add(program.scalar)}, {}};
            case TerminalOp::MulScalar:
                return program.dtype == DataType::Int32
                           ? TorchRun{{input.mul(static_cast<int64_t>(program.scalar))}, {}}
                           : TorchRun{{input.mul(program.scalar)}, {}};
            case TerminalOp::Abs:
                return {{input.abs()}, {}};
            case TerminalOp::Neg:
                return {{input.neg()}, {}};
            case TerminalOp::Clamp:
                return {{input.clamp(-1, 2)}, {}};
            case TerminalOp::CompareScalar:
                return program.dtype == DataType::Float32
                           ? TorchRun{{input.gt(program.scalar)}, {}}
                           : TorchRun{{input.gt(static_cast<int64_t>(program.scalar))}, {}};
            case TerminalOp::SumAll:
                return {{input.sum()}, {}};
            case TerminalOp::SumDim:
                return {{input.sum(program.dim, program.keepdim)}, {}};
            case TerminalOp::SumAxes: {
                std::vector<int64_t> axes(program.axes.begin(), program.axes.end());
                return {{input.sum(axes, program.keepdim)}, {}};
            }
            case TerminalOp::MeanAll:
                return {{input.mean()}, {}};
            case TerminalOp::MeanDim:
                return {{input.mean(program.dim, program.keepdim)}, {}};
            case TerminalOp::StdDim:
                return {{input.std(c10::IntArrayRef{program.dim},
                                   program.unbiased, program.keepdim)},
                        {}};
            case TerminalOp::VarDim:
                return {{input.var(c10::IntArrayRef{program.dim},
                                   program.unbiased, program.keepdim)},
                        {}};
            case TerminalOp::Cumsum:
                return {{input.cumsum(program.dim)}, {}};
            case TerminalOp::Nonzero:
                return {{input.nonzero()}, {}};
            case TerminalOp::Sort: {
                auto [values, indices] = input.sort(program.dim, program.descending);
                return {{std::move(values), std::move(indices)}, {}};
            }
            case TerminalOp::InplaceAdd:
                if (program.dtype == DataType::Int32) {
                    input.add_(static_cast<int64_t>(program.scalar));
                } else {
                    input.add_(program.scalar);
                }
                return {{input}, {}};
            case TerminalOp::BroadcastAdd: {
                std::vector<int64_t> shape(static_cast<size_t>(input.dim()), 1);
                const auto rhs = torch::full(shape, program.scalar, input.options());
                return {{input.add(rhs)}, {}};
            }
            case TerminalOp::MaskedSelect: {
                const auto mask = input.gt(program.scalar);
                return {{input.masked_select(mask)}, {}};
            }
            case TerminalOp::IndexSelect: {
                const int64_t extent = input.size(program.dim);
                auto index = torch::arange(extent - 1, -1, -1,
                                           torch::TensorOptions().dtype(torch::kInt64));
                if (input.is_cuda()) {
                    index = index.cuda();
                }
                return {{input.index_select(program.dim, index)}, {}};
            }
            case TerminalOp::Cat:
                return {{torch::cat({input, input}, program.dim)}, {}};
            case TerminalOp::Stack:
                return {{torch::stack({input, input}, program.dim)}, {}};
            case TerminalOp::Matmul: {
                const auto rhs = make_torch_rhs(input, input.size(1), 2);
                return {{input.matmul(rhs)}, {}};
            }
            case TerminalOp::MinWithIndices: {
                auto [values, indices] = input.min(program.dim, program.keepdim);
                return {{std::move(values), std::move(indices)}, {}};
            }
            case TerminalOp::MaxWithIndices: {
                auto [values, indices] = input.max(program.dim, program.keepdim);
                return {{std::move(values), std::move(indices)}, {}};
            }
            case TerminalOp::SerializeRoundTrip:
                return {{input.cpu().contiguous()}, {}};
            }
        } catch (const std::exception& error) {
            return {{}, error.what()};
        } catch (...) {
            return {{}, "non-standard exception"};
        }
        return {{}, "unreachable terminal"};
    }

    std::optional<std::string> compare_output(const Tensor& actual,
                                              const torch::Tensor& expected,
                                              const size_t output_index) {
        if (actual.ndim() != static_cast<size_t>(expected.dim())) {
            return "shape: output " + std::to_string(output_index) + " rank " +
                   std::to_string(actual.ndim()) + " versus " +
                   std::to_string(expected.dim());
        }
        for (size_t dim = 0; dim < actual.ndim(); ++dim) {
            if (actual.size(dim) != static_cast<size_t>(expected.size(dim))) {
                return "shape: output " + std::to_string(output_index) + " dimension " +
                       std::to_string(dim) + " is " + std::to_string(actual.size(dim)) +
                       " versus " + std::to_string(expected.size(dim));
            }
        }
        const DataType expected_dtype = lfs_dtype(expected.scalar_type());
        if (actual.dtype() != expected_dtype) {
            return "dtype: output " + std::to_string(output_index) + " is " +
                   std::string(dtype_name(actual.dtype())) + " versus " +
                   std::string(dtype_name(expected_dtype));
        }
        if (actual.numel() != static_cast<size_t>(expected.numel())) {
            return "shape: output element count differs";
        }

        const bool exact = expected_dtype == DataType::Int32 ||
                           expected_dtype == DataType::Int64 ||
                           expected_dtype == DataType::UInt8 ||
                           expected_dtype == DataType::Bool;
        if (exact) {
            const auto actual_values = actual.cpu()
                                           .contiguous()
                                           .to(DataType::Int64)
                                           .to_vector_int64();
            const auto expected_cpu = expected.detach()
                                          .to(torch::kCPU)
                                          .contiguous()
                                          .to(torch::kInt64);
            const auto* expected_data = expected_cpu.data_ptr<int64_t>();
            for (size_t index = 0; index < actual_values.size(); ++index) {
                if (actual_values[index] != expected_data[index]) {
                    return "value: exact output " + std::to_string(output_index) +
                           " index " + std::to_string(index) + " is " +
                           std::to_string(actual_values[index]) + " versus " +
                           std::to_string(expected_data[index]);
                }
            }
            return std::nullopt;
        }

        const auto actual_values = actual.cpu()
                                       .contiguous()
                                       .to(DataType::Float32)
                                       .to_vector();
        const auto expected_cpu = expected.detach()
                                      .to(torch::kCPU)
                                      .contiguous()
                                      .to(torch::kFloat32);
        const auto* expected_data = expected_cpu.data_ptr<float>();
        const float rtol = expected_dtype == DataType::Float16 ? 3e-3f : 2e-4f;
        const float atol = expected_dtype == DataType::Float16 ? 3e-3f : 2e-5f;
        for (size_t index = 0; index < actual_values.size(); ++index) {
            const float a = actual_values[index];
            const float e = expected_data[index];
            if (std::isnan(e)) {
                if (!std::isnan(a)) {
                    return "value: output " + std::to_string(output_index) +
                           " index " + std::to_string(index) + " expected NaN, got " +
                           std::to_string(a);
                }
            } else if (std::isinf(e)) {
                if (a != e) {
                    return "value: output " + std::to_string(output_index) +
                           " index " + std::to_string(index) + " expected infinity";
                }
            } else if (!std::isfinite(a) ||
                       std::abs(a - e) > atol + rtol * std::abs(e)) {
                return "value: output " + std::to_string(output_index) +
                       " index " + std::to_string(index) + " is " +
                       std::to_string(a) + " versus " + std::to_string(e);
            }
        }
        return std::nullopt;
    }

    std::optional<std::string> mismatch_for(const FuzzProgram& program) {
        auto actual = run_lfs(program);
        if (program.device == Device::CUDA) {
            const cudaError_t status = cudaDeviceSynchronize();
            const cudaError_t launch_status = cudaGetLastError();
            const cudaError_t failure = status != cudaSuccess ? status : launch_status;
            if (failure != cudaSuccess && actual.exception.empty()) {
                actual.exception = std::string("CUDA synchronization failed: ") +
                                   cudaGetErrorString(failure);
            }
        }
        const auto expected = run_torch(program);
        if (!actual.exception.empty() || !expected.exception.empty()) {
            if (actual.exception.empty() != expected.exception.empty()) {
                return "exception-asymmetry: LFS=" +
                       (actual.exception.empty() ? std::string("none") : actual.exception) +
                       "; Torch=" +
                       (expected.exception.empty() ? std::string("none") : expected.exception);
            }
            return std::nullopt;
        }
        if (actual.outputs.size() != expected.outputs.size()) {
            return "shape: output count is " + std::to_string(actual.outputs.size()) +
                   " versus " + std::to_string(expected.outputs.size());
        }
        for (size_t index = 0; index < actual.outputs.size(); ++index) {
            if (auto mismatch = compare_output(actual.outputs[index], expected.outputs[index], index)) {
                return mismatch;
            }
        }
        return std::nullopt;
    }

    std::string describe_program(const FuzzProgram& program) {
        std::ostringstream description;
        description << "seed=0x" << std::hex << program.seed << std::dec
                    << " iteration=" << program.iteration
                    << " device=" << (program.device == Device::CPU ? "CPU" : "CUDA")
                    << " dtype=" << dtype_name(program.dtype)
                    << " shape=[";
        for (size_t i = 0; i < program.base_shape.size(); ++i) {
            if (i != 0)
                description << ',';
            description << program.base_shape[i];
        }
        description << "] values=[";
        for (size_t i = 0; i < std::min<size_t>(program.values.size(), 12); ++i) {
            if (i != 0)
                description << ',';
            if (std::isnan(program.values[i]))
                description << "nan";
            else if (std::isinf(program.values[i]))
                description << (program.values[i] > 0 ? "inf" : "-inf");
            else
                description << program.values[i];
        }
        if (program.values.size() > 12)
            description << ",...";
        description << "] views=";
        if (program.views.empty())
            description << "none";
        for (size_t i = 0; i < program.views.size(); ++i) {
            if (i != 0)
                description << "->";
            const auto& view = program.views[i];
            description << view_name(view.kind) << '(';
            switch (view.kind) {
            case ViewKind::Transpose:
                description << view.first << ',' << view.second;
                break;
            case ViewKind::Permute:
            case ViewKind::Expand:
                for (size_t dim = 0; dim < view.dimensions.size(); ++dim) {
                    if (dim != 0)
                        description << ',';
                    description << view.dimensions[dim];
                }
                break;
            case ViewKind::Slice:
                description << view.first << ',' << view.start << ',' << view.end;
                break;
            case ViewKind::Unsqueeze:
            case ViewKind::Squeeze:
                description << view.first;
                break;
            }
            description << ')';
        }
        description << " terminal=" << terminal_name(program.terminal)
                    << " dim=" << program.dim
                    << " keepdim=" << program.keepdim
                    << " unbiased=" << program.unbiased
                    << " scalar=" << program.scalar;
        description << " chain=";
        if (program.pointwise.empty())
            description << "none";
        for (size_t i = 0; i < program.pointwise.size(); ++i) {
            if (i != 0)
                description << "->";
            description << pointwise_name(program.pointwise[i].kind);
            if (program.pointwise[i].kind == PointwiseKind::AddScalar ||
                program.pointwise[i].kind == PointwiseKind::MulScalar) {
                description << '(' << program.pointwise[i].scalar << ')';
            }
        }
        return description.str();
    }

    std::string mismatch_class(const TerminalOp terminal, const std::string& mismatch) {
        const size_t separator = mismatch.find(':');
        std::string category =
            mismatch.substr(0, separator == std::string::npos ? mismatch.size() : separator);
        if (mismatch.find("slice range") != std::string::npos)
            category += "-slice-contract";
        if (mismatch.find("broadcast_to does not support") != std::string::npos)
            category += "-expand-dtype";
        if (mismatch.find("squeeze dimension") != std::string::npos)
            category += "-squeeze";
        if (mismatch.find("broadcast-compatible") != std::string::npos ||
            mismatch.find("Cannot broadcast") != std::string::npos)
            category += "-broadcast";
        return std::string(terminal_name(terminal)) + '|' + category;
    }

    FuzzProgram minimize(FuzzProgram program) {
        const auto initial_mismatch = mismatch_for(program);
        if (!initial_mismatch)
            return program;
        const std::string target_class = mismatch_class(program.terminal, *initial_mismatch);
        size_t attempts = 0;
        constexpr size_t kMaxAttempts = 96;
        auto still_fails = [&](const FuzzProgram& candidate) {
            if (++attempts > kMaxAttempts)
                return false;
            const auto mismatch = mismatch_for(candidate);
            return mismatch && mismatch_class(candidate.terminal, *mismatch) == target_class;
        };

        for (size_t index = 0; index < program.views.size() && attempts < kMaxAttempts;) {
            auto candidate = program;
            candidate.views.erase(candidate.views.begin() + static_cast<std::ptrdiff_t>(index));
            if (still_fails(candidate)) {
                program = std::move(candidate);
            } else {
                ++index;
            }
        }

        for (size_t index = 0; index < program.pointwise.size() && attempts < kMaxAttempts;) {
            auto candidate = program;
            candidate.pointwise.erase(
                candidate.pointwise.begin() + static_cast<std::ptrdiff_t>(index));
            if (still_fails(candidate)) {
                program = std::move(candidate);
            } else {
                ++index;
            }
        }

        for (size_t index = 0; index < program.values.size() && attempts < kMaxAttempts; ++index) {
            if (program.values[index] == 0.0f)
                continue;
            auto candidate = program;
            candidate.values[index] = 0.0f;
            if (still_fails(candidate)) {
                program = std::move(candidate);
            }
        }

        for (size_t dim = 0; dim < program.base_shape.size() && attempts < kMaxAttempts; ++dim) {
            if (program.base_shape[dim] <= 1)
                continue;
            auto candidate = program;
            candidate.base_shape[dim] = 1;
            candidate.values.resize(shape_numel(candidate.base_shape), 0.0f);
            if (still_fails(candidate)) {
                program = std::move(candidate);
            }
        }

        if (program.device == Device::CUDA && attempts < kMaxAttempts) {
            auto candidate = program;
            candidate.device = Device::CPU;
            if (still_fails(candidate)) {
                program = std::move(candidate);
            }
        }
        return program;
    }

    template <typename Generator>
    size_t random_index(Generator& generator, const size_t size) {
        return std::uniform_int_distribution<size_t>(0, size - 1)(generator);
    }

    template <typename Generator>
    bool random_bool(Generator& generator) {
        return std::uniform_int_distribution<int>(0, 1)(generator) != 0;
    }

    template <typename Generator>
    FuzzProgram generate_program(Generator& generator,
                                 const uint64_t seed,
                                 const size_t iteration,
                                 const std::string_view device_mode) {
        FuzzProgram program;
        program.seed = seed;
        program.iteration = iteration;
        program.dtype = kDtypes[random_index(generator, kDtypes.size())];
        if (device_mode == "cpu") {
            program.device = Device::CPU;
        } else if (device_mode == "cuda") {
            program.device = Device::CUDA;
        } else {
            program.device = random_bool(generator) ? Device::CPU : Device::CUDA;
        }

        const int shape_class = std::uniform_int_distribution<int>(0, 9)(generator);
        if (shape_class == 0) {
            program.base_shape = {};
        } else if (shape_class <= 2) {
            const int rank = std::uniform_int_distribution<int>(1, 3)(generator);
            program.base_shape.assign(static_cast<size_t>(rank), 1);
            program.base_shape[random_index(generator, program.base_shape.size())] = 0;
            for (int dim = 0; dim < rank; ++dim) {
                if (program.base_shape[static_cast<size_t>(dim)] != 0) {
                    program.base_shape[static_cast<size_t>(dim)] =
                        std::uniform_int_distribution<int>(1, 3)(generator);
                }
            }
        } else {
            const int rank = std::uniform_int_distribution<int>(1, 3)(generator);
            for (int dim = 0; dim < rank; ++dim) {
                const bool singleton = std::uniform_int_distribution<int>(0, 2)(generator) == 0;
                program.base_shape.push_back(singleton
                                                 ? 1
                                                 : std::uniform_int_distribution<int>(2, 4)(generator));
            }
        }

        const size_t elements = shape_numel(program.base_shape);
        program.values.resize(elements);
        for (size_t index = 0; index < elements; ++index) {
            if (program.dtype == DataType::Bool) {
                program.values[index] = random_bool(generator) ? 1.0f : 0.0f;
            } else if (program.dtype == DataType::UInt8) {
                program.values[index] = static_cast<float>(
                    std::uniform_int_distribution<int>(0, 5)(generator));
            } else {
                program.values[index] = static_cast<float>(
                    std::uniform_int_distribution<int>(-3, 5)(generator));
                if (program.dtype == DataType::Float32) {
                    const int special = std::uniform_int_distribution<int>(0, 199)(generator);
                    if (special == 0)
                        program.values[index] = std::numeric_limits<float>::quiet_NaN();
                    if (special == 1)
                        program.values[index] = std::numeric_limits<float>::infinity();
                    if (special == 2)
                        program.values[index] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        torch::Tensor planning = make_torch_base_cpu(program);
        const int wanted_views = std::uniform_int_distribution<int>(0, 4)(generator);
        for (int view_index = 0, tries = 0;
             view_index < wanted_views && tries < wanted_views * 6 + 6;
             ++tries) {
            ViewStep step;
            step.kind = static_cast<ViewKind>(std::uniform_int_distribution<int>(0, 5)(generator));
            const int rank = static_cast<int>(planning.dim());
            bool valid = true;
            switch (step.kind) {
            case ViewKind::Transpose:
                if (rank < 2) {
                    valid = false;
                    break;
                }
                step.first = std::uniform_int_distribution<int>(0, rank - 1)(generator);
                do {
                    step.second = std::uniform_int_distribution<int>(0, rank - 1)(generator);
                } while (step.second == step.first);
                break;
            case ViewKind::Permute:
                if (rank < 2) {
                    valid = false;
                    break;
                }
                step.dimensions.resize(static_cast<size_t>(rank));
                for (int dim = 0; dim < rank; ++dim)
                    step.dimensions[static_cast<size_t>(dim)] = dim;
                std::shuffle(step.dimensions.begin(), step.dimensions.end(), generator);
                break;
            case ViewKind::Slice: {
                if (rank == 0) {
                    valid = false;
                    break;
                }
                step.first = std::uniform_int_distribution<int>(0, rank - 1)(generator);
                const int64_t extent = planning.size(step.first);
                if (extent == 0) {
                    valid = false;
                    break;
                }
                step.start = std::uniform_int_distribution<int64_t>(0, extent - 1)(generator);
                step.end = std::uniform_int_distribution<int64_t>(step.start + 1, extent)(generator);
                break;
            }
            case ViewKind::Unsqueeze:
                if (rank >= 5) {
                    valid = false;
                    break;
                }
                step.first = std::uniform_int_distribution<int>(0, rank)(generator);
                break;
            case ViewKind::Squeeze:
                if (rank == 0) {
                    valid = false;
                    break;
                }
                step.first = std::uniform_int_distribution<int>(0, rank - 1)(generator);
                if (planning.size(step.first) != 1 || rank == 1)
                    valid = false;
                break;
            case ViewKind::Expand: {
                std::vector<int> singleton_dims;
                for (int dim = 0; dim < rank; ++dim) {
                    if (planning.size(dim) == 1)
                        singleton_dims.push_back(dim);
                }
                if (program.dtype != DataType::Float32 ||
                    singleton_dims.empty() || planning.numel() > 32) {
                    valid = false;
                    break;
                }
                step.dimensions = planning.sizes().vec();
                const int dim = singleton_dims[random_index(generator, singleton_dims.size())];
                step.dimensions[static_cast<size_t>(dim)] =
                    std::uniform_int_distribution<int>(2, 3)(generator);
                break;
            }
            }
            if (!valid)
                continue;
            try {
                planning = apply_view(std::move(planning), step);
                program.views.push_back(std::move(step));
                ++view_index;
            } catch (...) {
                // Generation only keeps view programs that Torch accepts.
            }
        }

        std::vector<TerminalOp> terminals = {
            TerminalOp::Identity,
            TerminalOp::Cast,
            TerminalOp::SerializeRoundTrip,
        };
        const bool float32 = program.dtype == DataType::Float32;
        const bool int32 = program.dtype == DataType::Int32;
        const int rank = static_cast<int>(planning.dim());
        const bool nonempty = planning.numel() > 0;

        if (float32) {
            const int chain_length = std::uniform_int_distribution<int>(0, 3)(generator);
            for (int i = 0; i < chain_length; ++i) {
                PointwiseStep step;
                step.kind = static_cast<PointwiseKind>(
                    std::uniform_int_distribution<int>(0, 5)(generator));
                step.scalar = static_cast<float>(
                    std::uniform_int_distribution<int>(-2, 3)(generator));
                if (step.kind == PointwiseKind::MulScalar && step.scalar == 0.0f) {
                    step.scalar = 2.0f;
                }
                program.pointwise.push_back(step);
            }
        }

        if (float32 || int32) {
            terminals.insert(terminals.end(), {
                                                  TerminalOp::AddScalar,
                                                  TerminalOp::MulScalar,
                                                  TerminalOp::Abs,
                                                  TerminalOp::Neg,
                                                  TerminalOp::Clamp,
                                                  TerminalOp::CompareScalar,
                                                  TerminalOp::BroadcastAdd,
                                                  TerminalOp::MaskedSelect,
                                              });
            if (float32) {
                terminals.push_back(TerminalOp::SumAll);
                terminals.push_back(TerminalOp::InplaceAdd);
            }
        } else if (program.dtype == DataType::Bool) {
            terminals.push_back(TerminalOp::CompareScalar);
            terminals.push_back(TerminalOp::SumAll);
        } else if (program.dtype == DataType::UInt8) {
            terminals.push_back(TerminalOp::CompareScalar);
        }
        if (float32) {
            terminals.push_back(TerminalOp::MeanAll);
            terminals.push_back(TerminalOp::Cat);
            terminals.push_back(TerminalOp::Stack);
        }
        if (float32 || int32 || program.dtype == DataType::Bool) {
            terminals.push_back(TerminalOp::Nonzero);
        }
        if (rank > 0) {
            terminals.push_back(TerminalOp::IndexSelect);
            if (float32) {
                terminals.insert(terminals.end(), {
                                                      TerminalOp::SumDim,
                                                      TerminalOp::SumAxes,
                                                      TerminalOp::MeanDim,
                                                      TerminalOp::StdDim,
                                                      TerminalOp::VarDim,
                                                      TerminalOp::Cumsum,
                                                  });
                if (nonempty) {
                    terminals.push_back(TerminalOp::Sort);
                    if (rank > 1) {
                        terminals.push_back(TerminalOp::MinWithIndices);
                        terminals.push_back(TerminalOp::MaxWithIndices);
                    }
                }
            }
        }
        if (float32 && rank == 2 && nonempty && planning.size(1) > 0) {
            terminals.push_back(TerminalOp::Matmul);
        }

        program.terminal = terminals[random_index(generator, terminals.size())];
        program.scalar = static_cast<float>(std::uniform_int_distribution<int>(-2, 3)(generator));
        if (program.dtype == DataType::Bool || program.dtype == DataType::UInt8) {
            program.scalar = static_cast<float>(std::uniform_int_distribution<int>(0, 1)(generator));
        }
        if (program.scalar == 0.0f && program.terminal == TerminalOp::MulScalar) {
            program.scalar = 2.0f;
        }
        program.keepdim = random_bool(generator);
        program.unbiased = random_bool(generator);
        program.descending = random_bool(generator);
        if (rank > 0) {
            program.dim = std::uniform_int_distribution<int>(0, rank - 1)(generator);
            if (random_bool(generator))
                program.dim -= rank;
            const int positive_dim = program.dim < 0 ? program.dim + rank : program.dim;
            program.axes = {positive_dim};
            if (rank > 1 && random_bool(generator)) {
                int second = std::uniform_int_distribution<int>(0, rank - 1)(generator);
                if (second == positive_dim)
                    second = (second + 1) % rank;
                program.axes.push_back(second);
                if (random_bool(generator))
                    std::reverse(program.axes.begin(), program.axes.end());
            }
        }
        program.cast_dtype = kDtypes[random_index(generator, kDtypes.size())];
        if (program.cast_dtype == DataType::UInt8) {
            program.cast_dtype = DataType::Int64;
        }
        if (program.terminal == TerminalOp::Cat) {
            program.dim = rank == 0 ? 0 : std::uniform_int_distribution<int>(0, rank - 1)(generator);
        } else if (program.terminal == TerminalOp::Stack) {
            program.dim = std::uniform_int_distribution<int>(0, rank)(generator);
        }
        return program;
    }

    uint64_t env_uint64(const char* name, const uint64_t fallback) {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0')
            return fallback;
        char* end = nullptr;
        const uint64_t parsed = std::strtoull(value, &end, 0);
        return end != value && *end == '\0' ? parsed : fallback;
    }

    std::string env_string(const char* name, std::string fallback) {
        const char* value = std::getenv(name);
        return value == nullptr || *value == '\0' ? std::move(fallback) : std::string(value);
    }

} // namespace

TEST(DiscoveryFuzz, RandomizedProgramsMatchTorch) {
    ASSERT_TRUE(torch::cuda::is_available()) << "Discovery fuzzer requires CUDA";

    const uint64_t seed = env_uint64("LFS_DISCOVERY_SEED", 0x5eedc0deULL);
    const size_t start_iteration = static_cast<size_t>(
        env_uint64("LFS_DISCOVERY_START", 0));
    const size_t iterations = static_cast<size_t>(
        env_uint64("LFS_DISCOVERY_ITERS", 512));
    const std::string device_mode = env_string("LFS_DISCOVERY_DEVICE", "mixed");
    const bool allow_mismatches = env_uint64("LFS_DISCOVERY_ALLOW_MISMATCHES", 0) != 0;
    const bool emit_mismatches = env_uint64("LFS_DISCOVERY_EMIT_MISMATCHES", 1) != 0;
    ASSERT_TRUE(device_mode == "mixed" || device_mode == "cpu" || device_mode == "cuda");

    std::mt19937_64 generator(seed);
    std::map<std::string, size_t> dtype_counts;
    std::map<std::string, size_t> device_counts;
    std::map<std::string, size_t> terminal_counts;
    std::map<std::string, size_t> view_counts;
    std::map<std::string, size_t> pointwise_counts;
    std::map<std::string, size_t> mismatch_counts;
    std::map<std::string, std::string> first_mismatches;
    size_t mismatch_total = 0;

    for (size_t iteration = 0; iteration < start_iteration + iterations; ++iteration) {
        const auto program = generate_program(generator, seed, iteration, device_mode);
        if (iteration < start_iteration)
            continue;
        ++dtype_counts[std::string(dtype_name(program.dtype))];
        ++device_counts[program.device == Device::CPU ? "CPU" : "CUDA"];
        ++terminal_counts[std::string(terminal_name(program.terminal))];
        if (program.views.empty())
            ++view_counts["none"];
        for (const auto& view : program.views) {
            ++view_counts[std::string(view_name(view.kind))];
        }
        for (const auto& step : program.pointwise) {
            ++pointwise_counts[std::string(pointwise_name(step.kind))];
        }

        const auto mismatch = mismatch_for(program);
        if (!mismatch)
            continue;
        ++mismatch_total;
        const std::string signature = mismatch_class(program.terminal, *mismatch);
        ++mismatch_counts[signature];
        if (emit_mismatches && !first_mismatches.contains(signature)) {
            const auto minimized = minimize(program);
            const auto minimized_mismatch = mismatch_for(minimized).value_or(*mismatch);
            std::ostringstream detail;
            detail << *mismatch << "\n  original: " << describe_program(program)
                   << "\n  minimized: " << describe_program(minimized)
                   << "\n  minimized mismatch: " << minimized_mismatch;
            first_mismatches.emplace(signature, detail.str());
            std::cout << "DISCOVERY_FUZZ_MISMATCH signature=" << signature << '\n'
                      << detail.str() << '\n';
        }
    }

    auto print_counts = [](const std::string_view label,
                           const std::map<std::string, size_t>& counts) {
        std::cout << label << '=';
        bool first = true;
        for (const auto& [name, count] : counts) {
            if (!first)
                std::cout << ',';
            first = false;
            std::cout << name << ':' << count;
        }
        std::cout << '\n';
    };

    std::cout << "DISCOVERY_FUZZ_SUMMARY seed=0x" << std::hex << seed << std::dec
              << " start=" << start_iteration
              << " iterations=" << iterations
              << " mismatches=" << mismatch_total
              << " signatures=" << mismatch_counts.size() << '\n';
    print_counts("DISCOVERY_FUZZ_DTYPES", dtype_counts);
    print_counts("DISCOVERY_FUZZ_DEVICES", device_counts);
    print_counts("DISCOVERY_FUZZ_TERMINALS", terminal_counts);
    print_counts("DISCOVERY_FUZZ_VIEWS", view_counts);
    print_counts("DISCOVERY_FUZZ_POINTWISE", pointwise_counts);
    print_counts("DISCOVERY_FUZZ_SIGNATURES", mismatch_counts);

    if (!allow_mismatches && mismatch_total != 0) {
        std::ostringstream failures;
        failures << mismatch_total << " mismatches in " << mismatch_counts.size()
                 << " signatures";
        for (const auto& [signature, detail] : first_mismatches) {
            failures << "\n[" << signature << "] " << detail;
        }
        FAIL() << failures.str();
    }
}
