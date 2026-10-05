/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/gpu_kernel_module.hpp"

#include "../internal/tensor_impl.hpp"
#include "core/guarded_task.hpp"
#include "core/tensor_backend.hpp"
#include "gpu_program.hpp"
#include "metal/metal_module.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfs::core {

    struct GpuKernelModule::Impl {
        GpuBackend backend;
        std::unique_ptr<internal::MetalModule> metal;
        std::unique_ptr<internal::GpuProgram> program;
        struct Signature {
            uint32_t parameter_bytes;
            std::vector<uint32_t> tensor_offsets;
            std::array<uint32_t, 3> thread_group;
        };
        std::map<std::pair<std::string, Stage>, Signature> entries;
    };

    namespace {
        Error program_error(const ErrorCode code, std::string detail) {
            return make_error({.code = code, .domain = ErrorDomain::Tensor, .user_message = "GPU program operation failed", .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        template <class T, class F>
        Result<T> program_boundary(F&& body) {
            try {
                return body();
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): normalize native failures at the typed tensor-program boundary.
                return detail::task_failure_from_current_exception<T>(
                    {.name = "tensor.program", .domain = ErrorDomain::Tensor, .site = LFS_SOURCE_SITE_CURRENT()});
            }
        }

        Result<internal::ProgramArguments> bind_arguments(internal::GpuProgram& program,
                                                          const GpuBackend backend,
                                                          const GpuKernelModule::Arguments& arguments,
                                                          const uint32_t parameter_bytes,
                                                          const std::span<const uint32_t> tensor_offsets) {
            if (arguments.parameters.size() != parameter_bytes || arguments.tensors.size() != tensor_offsets.size())
                return program_error(ErrorCode::InvalidArgument,
                                     std::format("Program expects {} parameter bytes / {} tensor bindings, observed {} / {}",
                                                 parameter_bytes, tensor_offsets.size(), arguments.parameters.size(), arguments.tensors.size()));
            internal::ProgramArguments result;
            result.parameters.assign(arguments.parameters.begin(), arguments.parameters.end());
            std::set<uint32_t> offsets;
            for (const auto& binding : arguments.tensors) {
                const auto* tensor = binding.tensor;
                const size_t offset = binding.parameter_offset;
                if (std::find(tensor_offsets.begin(), tensor_offsets.end(), offset) == tensor_offsets.end())
                    return program_error(ErrorCode::InvalidArgument, std::format("Offset {} is not a reflected Slang tensor pointer", offset));
                if (offset % 8 || offset > result.parameters.size() || result.parameters.size() - offset < 8 ||
                    !offsets.insert(binding.parameter_offset).second)
                    return program_error(ErrorCode::InvalidArgument,
                                         std::format("Invalid/repeated tensor binding offset {} in {} parameter bytes", offset, result.parameters.size()));
                uint64_t pointer = 0;
                std::memcpy(&pointer, result.parameters.data() + offset, sizeof(pointer));
                if (pointer != 0)
                    return program_error(ErrorCode::InvalidArgument, std::format("Pointer field at {} must be zero, observed {}", offset, pointer));
                if (!tensor)
                    continue;
                if (!tensor->is_valid() || tensor->numel() == 0 || !tensor->is_contiguous() ||
                    tensor->device() != Device::GPU || gpu_backend_of(*tensor) != backend)
                    return program_error(ErrorCode::InvalidArgument,
                                         std::format("Binding at {} requires a nonempty contiguous {} tensor; present={}, valid={}",
                                                     offset, gpu_backend_name(backend), true, tensor->is_valid()));
                pointer = program.address(*tensor);
                std::memcpy(result.parameters.data() + offset, &pointer, sizeof(pointer));
                (binding.access == GpuKernelModule::Access::Read ? result.reads : result.writes).push_back(tensor);
            }
            return result;
        }
    } // namespace

    GpuKernelModule::GpuKernelModule() : impl_(std::make_unique<Impl>()) {}

    Result<std::unique_ptr<GpuKernelModule>> GpuKernelModule::load(const std::span<const Entry> entries,
                                                                   const GpuBackend backend) {
        return program_boundary<std::unique_ptr<GpuKernelModule>>([&]() -> Result<std::unique_ptr<GpuKernelModule>> {
            auto module = std::unique_ptr<GpuKernelModule>(new GpuKernelModule);
            module->impl_->backend = backend;
            for (const auto& entry : entries) {
                if (entry.backend == backend && !entry.code.empty() && !entry.name.empty())
                    module->impl_->entries.emplace(std::pair{std::string(entry.name), entry.stage},
                                                   Impl::Signature{entry.parameter_bytes,
                                                                   {entry.tensor_offsets.begin(), entry.tensor_offsets.end()},
                                                                   entry.thread_group});
            }
            if (module->impl_->entries.empty())
                return program_error(ErrorCode::Unsupported, std::format("Program has no {} artifacts", gpu_backend_name(backend)));
            switch (backend) {
#ifdef LFS_TENSOR_VULKAN
            case GpuBackend::Vulkan: module->impl_->program = internal::make_vulkan_program(entries); break;
#endif
#ifdef LFS_TENSOR_METAL
            case GpuBackend::Metal: module->impl_->program = internal::make_metal_program(entries); break;
#endif
#if LFS_HAS_CUDA
            case GpuBackend::CUDA: module->impl_->program = internal::make_cuda_program(entries); break;
#endif
            default: return program_error(ErrorCode::Unsupported, std::format("{} programs were not built", gpu_backend_name(backend)));
            }
            return module;
        });
    }

    bool GpuKernelModule::supports_raster() const {
        return impl_->program && impl_->program->supports_raster();
    }

    Result<void> GpuKernelModule::dispatch(const Dispatch& dispatch) {
        return program_boundary<void>([&]() -> Result<void> {
            if (!impl_->program || !impl_->entries.contains({std::string(dispatch.function), Stage::Compute}))
                return Result<void>::failure(program_error(ErrorCode::NotFound, std::format("Compute entry '{}' not found", dispatch.function)));
            const auto& signature = impl_->entries.at({std::string(dispatch.function), Stage::Compute});
            for (size_t i = 0; i < 3; ++i) {
                if (dispatch.group[i] != signature.thread_group[i])
                    return Result<void>::failure(program_error(ErrorCode::InvalidArgument,
                                                               std::format("Threadgroup dimension {}: shader requires {}, observed {}", i, signature.thread_group[i], dispatch.group[i])));
                if (dispatch.group[i] == 0)
                    return Result<void>::failure(program_error(ErrorCode::InvalidArgument, std::format("Threadgroup dimension {} is zero", i)));
                if (!dispatch.indirect && dispatch.groups[i] == 0)
                    return {};
            }
            if (const auto* indirect = dispatch.indirect) {
                if (impl_->backend == GpuBackend::CUDA)
                    return Result<void>::failure(program_error(ErrorCode::Unsupported, "Indirect dispatch is unavailable on CUDA programs"));
                if (!indirect->is_valid() || !indirect->is_contiguous() || indirect->device() != Device::GPU ||
                    gpu_backend_of(*indirect) != impl_->backend ||
                    (indirect->dtype() != DataType::Int32 && indirect->dtype() != DataType::UInt32) ||
                    indirect->numel() < dispatch.indirect_offset + 3)
                    return Result<void>::failure(program_error(
                        ErrorCode::InvalidArgument,
                        std::format("Indirect arguments need three contiguous {} Int32/UInt32 values at element {}; valid={}, numel={}",
                                    gpu_backend_name(impl_->backend), dispatch.indirect_offset, indirect->is_valid(),
                                    indirect->is_valid() ? indirect->numel() : 0)));
            }
            auto args = bind_arguments(*impl_->program, impl_->backend, dispatch.arguments, signature.parameter_bytes, signature.tensor_offsets);
            if (!args)
                return Result<void>::failure(std::move(args).error());
            if (dispatch.indirect)
                args->reads.push_back(dispatch.indirect);
            impl_->program->dispatch(dispatch, *args);
            return {};
        });
    }

    Result<void> GpuKernelModule::draw(const Draw& draw) {
        return draw_batch(std::span(&draw, 1));
    }

    Result<void> GpuKernelModule::draw_batch(const std::span<const Draw> draws) {
        return program_boundary<void>([&]() -> Result<void> {
            if (!supports_raster())
                return Result<void>::failure(program_error(ErrorCode::Unsupported, std::format("Raster is unsupported by {}", gpu_backend_name(impl_->backend))));
            if (draws.empty())
                return {};
            const auto* color = draws.front().color;
            const auto* depth = draws.front().depth;
            const auto valid_storage = [&](const Tensor* t) {
                return t && t->is_valid() && t->is_contiguous() && t->device() == Device::GPU && gpu_backend_of(*t) == impl_->backend;
            };
            if (!valid_storage(color) || color->ndim() != 3 || color->size(2) != 4 || color->numel() == 0 ||
                (color->dtype() != DataType::Float32 && color->dtype() != DataType::UInt8))
                return Result<void>::failure(program_error(ErrorCode::InvalidArgument, "Raster color must be a nonempty contiguous GPU Float32/UInt8 [H,W,4] tensor"));
            if (depth && (!valid_storage(depth) || depth->dtype() != DataType::Float32 ||
                          depth->ndim() != 2 || depth->size(0) != color->size(0) || depth->size(1) != color->size(1)))
                return Result<void>::failure(program_error(ErrorCode::InvalidArgument, std::format("Depth must be Float32 [{},{}] on the color backend", color->size(0), color->size(1))));
            const auto width = static_cast<uint64_t>(color->size(1));
            const auto height = static_cast<uint64_t>(color->size(0));
            std::vector<internal::ProgramArguments> arguments;
            std::vector<Draw> recorded;
            arguments.reserve(draws.size());
            recorded.reserve(draws.size());
            for (size_t i = 0; i < draws.size(); ++i) {
                const auto& draw = draws[i];
                if (draw.color != color || draw.depth != depth)
                    return Result<void>::failure(program_error(ErrorCode::InvalidArgument, std::format("Batched draw {} names different attachments than draw 0", i)));
                if (i > 0 && (draw.clear_color || draw.clear_depth))
                    return Result<void>::failure(program_error(ErrorCode::InvalidArgument, std::format("Batched draw {} clears; only draw 0 may clear", i)));
                if (!impl_->entries.contains({std::string(draw.vertex), Stage::Vertex}) ||
                    !impl_->entries.contains({std::string(draw.fragment), Stage::Fragment}))
                    return Result<void>::failure(program_error(ErrorCode::NotFound, std::format("Raster entries '{}'/'{}' not found", draw.vertex, draw.fragment)));
                if (draw.scissor) {
                    const auto& rect = *draw.scissor;
                    if (rect.x > width || rect.y > height ||
                        static_cast<uint64_t>(rect.x) + rect.width > width ||
                        static_cast<uint64_t>(rect.y) + rect.height > height)
                        return Result<void>::failure(program_error(
                            ErrorCode::InvalidArgument,
                            std::format("Raster scissor [{},{},{},{}] exceeds attachment [{},{}]",
                                        rect.x, rect.y, rect.width, rect.height, width, height)));
                }
                if (draw.viewport && !(draw.viewport->width > 0 && draw.viewport->height > 0 &&
                                       std::isfinite(draw.viewport->x) && std::isfinite(draw.viewport->y) &&
                                       std::isfinite(draw.viewport->width) && std::isfinite(draw.viewport->height)))
                    return Result<void>::failure(program_error(ErrorCode::InvalidArgument,
                                                               std::format("Raster viewport [{},{},{},{}] must be finite with a positive extent",
                                                                           draw.viewport->x, draw.viewport->y, draw.viewport->width, draw.viewport->height)));
                if (draw.vertex_count % 3)
                    return Result<void>::failure(program_error(ErrorCode::InvalidArgument, std::format("Triangle-list vertex count {} is not divisible by three", draw.vertex_count)));
                const auto& vertex = impl_->entries.at({std::string(draw.vertex), Stage::Vertex});
                const auto& fragment = impl_->entries.at({std::string(draw.fragment), Stage::Fragment});
                if (vertex.parameter_bytes != fragment.parameter_bytes || vertex.tensor_offsets != fragment.tensor_offsets)
                    return Result<void>::failure(program_error(ErrorCode::InvalidArgument, "Vertex and fragment parameter blocks must have the same reflected layout"));
                // An empty scissor or no vertices rasterizes nothing; a first
                // draw that clears still records so its clear takes effect.
                const bool empty = draw.vertex_count == 0 || draw.instance_count == 0 ||
                                   (draw.scissor && (draw.scissor->width == 0 || draw.scissor->height == 0));
                if (empty && !(i == 0 && (draw.clear_color || draw.clear_depth)))
                    continue;
                auto args = bind_arguments(*impl_->program, impl_->backend, draw.arguments, vertex.parameter_bytes, vertex.tensor_offsets);
                if (!args)
                    return Result<void>::failure(std::move(args).error());
                arguments.push_back(std::move(*args));
                recorded.push_back(draw);
                if (empty)
                    recorded.back().vertex_count = 0;
            }
            if (!recorded.empty())
                impl_->program->draw(recorded, arguments);
            return {};
        });
    }

    GpuKernelModule::GpuKernelModule(const GpuBackend backend, std::string source, const bool fast_math)
        : impl_(std::make_unique<Impl>()) {
        impl_->backend = backend;
        if (backend != GpuBackend::Metal)
            throw std::invalid_argument(
                std::format("GPU kernel modules are unavailable on the {} backend", gpu_backend_name(backend)));
        impl_->metal = internal::make_metal_module(std::move(source), fast_math);
    }

    GpuKernelModule::~GpuKernelModule() = default;

    uint64_t GpuKernelModule::address(const Tensor& tensor) const {
        if (!tensor.is_valid() || tensor.numel() == 0)
            return 0;
        if (tensor.device() != Device::GPU || gpu_backend_of(tensor) != impl_->backend)
            throw std::invalid_argument(std::format("GPU kernel operand must live on the {} backend, got {} storage",
                                                    gpu_backend_name(impl_->backend),
                                                    tensor.device() == Device::GPU ? "another GPU backend's" : "CPU"));
        // Kernels index operands densely from this address.
        if (!tensor.is_contiguous()) {
            const auto strides = tensor.strides();
            std::string text;
            for (size_t i = 0; i < strides.rank; ++i)
                text += std::format("{}{}", i ? ", " : "", strides.values[i]);
            throw std::invalid_argument(std::format("GPU kernel operand must be contiguous, got shape {} strides [{}]",
                                                    tensor.shape().str(), text));
        }
        return impl_->metal->address(internal::storage_ref(tensor));
    }

    void GpuKernelModule::launch(const Launch& launch) {
        std::vector<internal::StorageRef> uses;
        uses.reserve(launch.uses.size());
        for (const Tensor* const tensor : launch.uses) {
            if (tensor != nullptr && tensor->is_valid() && tensor->numel() != 0)
                uses.push_back(internal::storage_ref(*tensor));
        }
        impl_->metal->launch(launch.function, launch.constants, uses, launch.params, launch.groups, launch.group);
    }

    uint32_t GpuKernelModule::groups_for(const size_t count, const uint32_t width) {
        const size_t groups = (count + width - 1) / width;
        if (groups > UINT32_MAX)
            throw std::overflow_error(std::format("{} items exceed the dispatch limit at {} per group", count, width));
        return static_cast<uint32_t>(groups);
    }

} // namespace lfs::core
