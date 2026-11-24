// SPDX-License-Identifier: GPL-3.0-or-later
#include "sharpness_tensor.hpp"
#include "media/sharpness_edge.hpp"
#include "sharpness_program.hpp"
#include <algorithm>
#include <array>
#include <numeric>
namespace lfs::io {
    std::optional<double> TensorSharpnessScorer::score(const core::Tensor& rgb, int width, int height, SharpnessAlgorithm algorithm) {
        if (width < 3 || height < 3)
            return 0.0;
        const auto backend = core::gpu_backend_of(rgb);
        if (!backend || rgb.dtype() != core::DataType::UInt8 || !rgb.is_contiguous() ||
            uint64_t(width) * uint64_t(height) * 3 > rgb.numel())
            return std::nullopt;
        core::TensorExecutionTarget::Scope scope(rgb.execution_target());
        if (!module_ || module_->backend() != *backend) {
            auto loaded = core::GpuKernelModule::load(sharpness_program_entries(), *backend);
            if (!loaded)
                return std::nullopt;
            module_ = std::move(*loaded);
        }
        struct Parameters {
            uint64_t rgb, counts;
            uint32_t width, height, algorithm, groups;
        };
        const auto groups = uint32_t(std::min<int64_t>(1024, (int64_t(width - 2) * (height - 2) + 255) / 256));
        if (!partials_.is_valid() || partials_.numel() != groups || core::gpu_backend_of(partials_) != core::gpu_backend_of(rgb))
            partials_ = core::Tensor::empty({size_t(groups)}, core::Device::GPU, core::DataType::Int32);
        Parameters p{0, 0, uint32_t(width), uint32_t(height), uint32_t(algorithm), groups};
        using Module = core::GpuKernelModule;
        const std::array bindings{Module::Binding{0, &rgb}, Module::Binding{8, &partials_, Module::Access::ReadWrite}};
        auto status = module_->dispatch({.function = "countEdges", .arguments = {std::as_bytes(std::span(&p, 1)), bindings}, .groups = {groups, 1, 1}, .group = {256, 1, 1}});
        if (!status)
            throw lfs::Exception(status.error());
        const auto counts = partials_.to_vector_int();
        return detail::edgeRatio(std::accumulate(counts.begin(), counts.end(), 0LL), width, height);
    }
} // namespace lfs::io
