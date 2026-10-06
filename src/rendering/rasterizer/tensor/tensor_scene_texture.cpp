/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "tensor_scene_texture.hpp"
#include "tensor_scene_texture_program.hpp"

namespace lfs::rendering {
    lfs::Result<void> TensorSceneTextureKernels::dispatch(
        bool unpack, const SceneTextureParameters& parameters,
        const std::array<core::Tensor*, 8>& tensors) {
        using Module = core::GpuKernelModule;
        if (!module_) {
            auto loaded = Module::load(tensor_scene_texture_program_entries(), backend_);
            if (!loaded)
                return lfs::Result<void>::failure(std::move(loaded).error());
            module_ = std::move(*loaded);
        }
        struct alignas(16) Arguments {
            uint64_t pointers[8]{};
            SceneTextureParameters parameters;
        } args{.parameters = parameters};
        static_assert(offsetof(Arguments, parameters) == 64);
        static_assert(sizeof(Arguments) == 144);

        std::array<Module::Binding, 8> bindings;
        for (size_t i = 0; i < bindings.size(); ++i) {
            const bool writable = unpack ? i == 6 : i >= 3 && i <= 5;
            bindings[i] = {uint32_t(i * 8), tensors[i],
                           writable ? Module::Access::ReadWrite : Module::Access::Read};
        }
        const auto name = unpack ? "scene_texture_unpack" : "scene_texture_pack";
        uint32_t bytes = 0;
        for (const auto& entry : tensor_scene_texture_program_entries())
            if (entry.backend == backend_ && entry.name == name)
                bytes = entry.parameter_bytes;
        const uint32_t width = parameters.extents[unpack ? 2 : 0];
        const uint32_t height = parameters.extents[unpack ? 3 : 1];
        return module_->dispatch({
            .function = name,
            .arguments = {std::as_bytes(std::span(&args, 1)).first(bytes), bindings},
            .groups = {Module::groups_for(width, 8), Module::groups_for(height, 8), 1},
            .group = {8, 8, 1},
        });
    }
} // namespace lfs::rendering
