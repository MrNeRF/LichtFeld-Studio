/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "tensor_scene_temporal.hpp"

#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "tensor_scene_temporal_program.hpp"

#include <cstring>
#include <format>

namespace lfs::rendering {
    namespace {
        using Module = core::GpuKernelModule;

        lfs::Result<void> invalid(std::string detail) {
            return lfs::Result<void>::failure(make_error({
                .code = ErrorCode::InvalidArgument,
                .domain = ErrorDomain::Rendering,
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }

        struct alignas(16) DispatchParameters {
            std::uint64_t pointers[10]{};
            TensorSceneMotionParameters motion;
            TensorSceneResolveParameters resolve;
            std::uint32_t count = 0, alignment[3]{}, padding[4]{};
        };
        static_assert(offsetof(DispatchParameters, motion) == 80);
        static_assert(offsetof(DispatchParameters, resolve) == 240);
        static_assert(offsetof(DispatchParameters, count) == 352);
        static_assert(sizeof(DispatchParameters) == 384);
    } // namespace

    struct TensorSceneTemporalKernels::Impl {
        explicit Impl(const core::GpuBackend value) : backend(value) {}
        core::GpuBackend backend;
        std::unique_ptr<Module> module;

        lfs::Result<void> ensureModule() {
            if (module)
                return {};
            auto loaded = Module::load(tensor_scene_temporal_program_entries(), backend);
            if (!loaded)
                return lfs::Result<void>::failure(std::move(loaded).error());
            module = std::move(*loaded);
            return {};
        }
        std::size_t parameterBytes(const std::string_view function) const {
            for (const auto& entry : tensor_scene_temporal_program_entries())
                if (entry.backend == backend && entry.name == function)
                    return entry.parameter_bytes;
            return 0;
        }
    };

    TensorSceneTemporalKernels::TensorSceneTemporalKernels(const core::GpuBackend backend)
        : impl_(std::make_unique<Impl>(backend)) {}
    TensorSceneTemporalKernels::~TensorSceneTemporalKernels() = default;

    lfs::Result<void> TensorSceneTemporalKernels::motion(
        const core::Tensor& depth, const TensorSceneMotionParameters& parameters,
        core::Tensor& output) {
        const auto width = parameters.render_info[0], height = parameters.render_info[1];
        if (!depth.is_valid() || depth.device() != core::Device::GPU ||
            depth.dtype() != core::DataType::Float32 || depth.ndim() != 2 ||
            depth.size(1) < width || depth.size(0) < height || !depth.is_contiguous() ||
            width == 0 || height == 0) {
            return invalid("Tensor scene motion requires a contiguous Float32 GPU depth image");
        }
        const core::GpuBackendScope scope(impl_->backend);
        if (!output.is_valid() || output.dtype() != core::DataType::Float32 ||
            output.ndim() != 3 || output.size(0) != height || output.size(1) != width ||
            output.size(2) != 2) {
            output = core::Tensor::empty({height, width, 2}, core::Device::GPU,
                                         core::DataType::Float32);
        }
        if (auto loaded = impl_->ensureModule(); !loaded)
            return loaded;
        DispatchParameters dispatch{.motion = parameters};
        const std::array bindings{
            Module::Binding{0, &depth}, Module::Binding{8, &output, Module::Access::ReadWrite},
            Module::Binding{16, nullptr}, Module::Binding{24, nullptr},
            Module::Binding{32, nullptr}, Module::Binding{40, nullptr},
            Module::Binding{48, nullptr}, Module::Binding{56, nullptr},
            Module::Binding{64, nullptr}, Module::Binding{72, nullptr},
        };
        const auto arguments = std::as_bytes(std::span(&dispatch, 1));
        return impl_->module->dispatch({
            .function = "scene_motion",
            .arguments = {arguments.first(impl_->parameterBytes("scene_motion")), bindings},
            .groups = {Module::groups_for(width, 8), Module::groups_for(height, 8), 1},
            .group = {8, 8, 1},
        });
    }

    lfs::Result<void> TensorSceneTemporalKernels::resolve(
        const core::Tensor& current, const core::Tensor* history,
        const core::Tensor& motion_tensor, const core::Tensor* current_depth,
        const core::Tensor* history_depth,
        const TensorSceneResolveParameters& parameters, core::Tensor& output) {
        const auto render_width = parameters.extents[0], render_height = parameters.extents[1];
        const auto output_width = parameters.extents[2], output_height = parameters.extents[3];
        const bool current_type = current.dtype() == core::DataType::UInt8 ||
                                  current.dtype() == core::DataType::Float32;
        if (!current.is_valid() || current.device() != core::Device::GPU || !current_type ||
            current.ndim() != 3 || current.size(2) < 4 || !current.is_contiguous() ||
            !motion_tensor.is_valid() || motion_tensor.device() != core::Device::GPU ||
            motion_tensor.dtype() != core::DataType::Float32 || motion_tensor.ndim() != 3 ||
            motion_tensor.size(0) < render_height || motion_tensor.size(1) < render_width ||
            motion_tensor.size(2) < 2 || !motion_tensor.is_contiguous() ||
            render_width == 0 || render_height == 0 || output_width == 0 || output_height == 0) {
            return invalid("Tensor scene resolve inputs do not match the declared extents");
        }
        const auto valid_depth = [render_width, render_height](const core::Tensor* depth) {
            return depth && depth->is_valid() && depth->device() == core::Device::GPU &&
                   depth->dtype() == core::DataType::Float32 && depth->ndim() == 2 &&
                   depth->size(0) >= render_height && depth->size(1) >= render_width &&
                   depth->is_contiguous();
        };
        const bool wants_history = parameters.control[0] >= 0.5f;
        if (wants_history &&
            (!history || !history->is_valid() || history->device() != core::Device::GPU ||
             history->dtype() != core::DataType::Float32 || history->ndim() != 3 ||
             history->size(0) != output_height || history->size(1) != output_width ||
             history->size(2) != 4 || !history->is_contiguous())) {
            return invalid("Tensor scene resolve history does not match the output extent");
        }
        if (parameters.depth_control[0] >= 0.5f &&
            (!valid_depth(current_depth) || !valid_depth(history_depth))) {
            return invalid("Tensor scene resolve depth history is incomplete");
        }
        const core::GpuBackendScope scope(impl_->backend);
        output = core::Tensor::empty({output_height, output_width, 4}, core::Device::GPU,
                                     core::DataType::Float32);
        if (auto loaded = impl_->ensureModule(); !loaded)
            return loaded;
        DispatchParameters dispatch{.resolve = parameters};
        const std::array bindings{
            Module::Binding{0, nullptr}, Module::Binding{8, nullptr},
            Module::Binding{16, &current}, Module::Binding{24, history},
            Module::Binding{32, &motion_tensor}, Module::Binding{40, current_depth},
            Module::Binding{48, history_depth},
            Module::Binding{56, &output, Module::Access::ReadWrite},
            Module::Binding{64, nullptr}, Module::Binding{72, nullptr},
        };
        const auto arguments = std::as_bytes(std::span(&dispatch, 1));
        return impl_->module->dispatch({
            .function = "scene_temporal_resolve",
            .arguments = {arguments.first(impl_->parameterBytes("scene_temporal_resolve")), bindings},
            .groups = {Module::groups_for(output_width, 8), Module::groups_for(output_height, 8), 1},
            .group = {8, 8, 1},
        });
    }

    lfs::Result<void> TensorSceneTemporalKernels::spatial(
        const core::Tensor& current, const TensorSceneResolveParameters& parameters,
        core::Tensor& output) {
        const auto render_width = parameters.extents[0], render_height = parameters.extents[1];
        const auto output_width = parameters.extents[2], output_height = parameters.extents[3];
        const bool current_type = current.dtype() == core::DataType::UInt8 ||
                                  current.dtype() == core::DataType::Float32;
        if (!current.is_valid() || current.device() != core::Device::GPU || !current_type ||
            current.ndim() != 3 || current.size(0) < render_height ||
            current.size(1) < render_width || current.size(2) < 4 || !current.is_contiguous() ||
            render_width == 0 || render_height == 0 || output_width == 0 || output_height == 0) {
            return invalid("Tensor scene spatial input does not match the declared extents");
        }
        const core::GpuBackendScope scope(impl_->backend);
        output = core::Tensor::empty({output_height, output_width, 4}, core::Device::GPU,
                                     core::DataType::Float32);
        if (auto loaded = impl_->ensureModule(); !loaded)
            return loaded;
        DispatchParameters dispatch{.resolve = parameters};
        const std::array bindings{
            Module::Binding{0, nullptr}, Module::Binding{8, nullptr},
            Module::Binding{16, &current}, Module::Binding{24, nullptr},
            Module::Binding{32, nullptr}, Module::Binding{40, nullptr},
            Module::Binding{48, nullptr},
            Module::Binding{56, &output, Module::Access::ReadWrite},
            Module::Binding{64, nullptr}, Module::Binding{72, nullptr},
        };
        const auto arguments = std::as_bytes(std::span(&dispatch, 1));
        return impl_->module->dispatch({
            .function = "scene_spatial",
            .arguments = {arguments.first(impl_->parameterBytes("scene_spatial")), bindings},
            .groups = {Module::groups_for(output_width, 8), Module::groups_for(output_height, 8), 1},
            .group = {8, 8, 1},
        });
    }

    lfs::Result<void> TensorSceneTemporalKernels::resolveSamples(
        const std::span<const TensorSceneResolveSample> samples, core::Tensor& output) {
        if (samples.empty())
            return invalid("Tensor scene resolve sample input is empty");
        const core::GpuBackendScope scope(impl_->backend);
        auto input = core::Tensor::from_blob(const_cast<TensorSceneResolveSample*>(samples.data()),
                                             {samples.size(), sizeof(TensorSceneResolveSample)},
                                             core::Device::CPU, core::DataType::UInt8)
                         .to(core::Device::GPU);
        output = core::Tensor::empty({samples.size(), sizeof(TensorSceneResolveResult)},
                                     core::Device::GPU, core::DataType::UInt8);
        if (auto loaded = impl_->ensureModule(); !loaded)
            return loaded;
        const DispatchParameters dispatch{.count = static_cast<std::uint32_t>(samples.size())};
        const std::array bindings{
            Module::Binding{0, nullptr}, Module::Binding{8, nullptr},
            Module::Binding{16, nullptr}, Module::Binding{24, nullptr},
            Module::Binding{32, nullptr}, Module::Binding{40, nullptr},
            Module::Binding{48, nullptr}, Module::Binding{56, nullptr},
            Module::Binding{64, &input},
            Module::Binding{72, &output, Module::Access::ReadWrite},
        };
        const auto arguments = std::as_bytes(std::span(&dispatch, 1));
        return impl_->module->dispatch({
            .function = "scene_temporal_resolve_samples",
            .arguments = {arguments.first(impl_->parameterBytes("scene_temporal_resolve_samples")), bindings},
            .groups = {Module::groups_for(samples.size(), 64), 1, 1},
            .group = {64, 1, 1},
        });
    }
} // namespace lfs::rendering
