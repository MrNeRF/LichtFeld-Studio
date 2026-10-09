/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lens_views.hpp"
#include "core/crash_handler.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "lens_views_program.hpp"
#include <array>
#include <span>
namespace lfs::preprocessing {
    namespace {
        using namespace core;
        struct Parameters {
            uint64_t camera = 0, view = 0, radial = 0, tangent = 0, prism = 0, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
            float fx = 0, fy = 0, cx = 0, cy = 0;
            uint32_t width = 0, height = 0, model = 0, faces = 0, pixels = 0;
            LensFace face;
        };
        static_assert(sizeof(Parameters) == 168);

        void run(std::string_view entry, const LensCamera& camera, const LensFace& face,
                 std::array<const Tensor*, 6> tensors, size_t count, int faces = 0) {
            if (!count)
                return;
            const auto backend = gpu_backend_of(*tensors[0]).value_or(default_gpu_backend());
            const GpuBackendScope scope(backend);
            auto loaded = GpuKernelModule::load(lens_views_program_entries(), backend);
            if (!loaded)
                throw Exception(std::move(loaded).error());
            auto& program = **loaded;
            auto cam = Tensor::empty({64}, Device::GPU);
            auto view = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f}, {16}, Device::GPU);
            auto radial = Tensor::from_vector(std::vector<float>(camera.projection.radial, camera.projection.radial + 6), {6}, Device::GPU);
            auto tangent = Tensor::from_vector(std::vector<float>(camera.projection.tangential, camera.projection.tangential + 2), {2}, Device::GPU);
            auto prism = Tensor::from_vector(std::vector<float>(camera.projection.thin_prism, camera.projection.thin_prism + 4), {4}, Device::GPU);
            const bool panorama = camera.projection.model == 3;
            Parameters p{.fx = panorama ? float(camera.width) : camera.fx, .fy = panorama ? float(camera.height) : camera.fy, .cx = panorama ? 0.f : camera.cx, .cy = panorama ? 0.f : camera.cy, .width = uint32_t(camera.width), .height = uint32_t(camera.height), .model = uint32_t(camera.projection.model), .faces = uint32_t(faces), .pixels = uint32_t(count), .face = face};
            std::array<GpuKernelModule::Binding, 11> bindings{{{0, &cam, GpuKernelModule::Access::ReadWrite}, {8, &view}, {16, &radial}, {24, &tangent}, {32, &prism}}};
            for (size_t i = 0; i < 6; ++i)
                bindings[i + 5] = {uint32_t(40 + 8 * i), tensors[i], GpuKernelModule::Access::ReadWrite};
            auto dispatch = [&](std::string_view name, uint32_t groups) {
                auto result = program.dispatch({.function = name, .arguments = {std::as_bytes(std::span(&p, 1)), bindings}, .groups = {groups, 1, 1}, .group = {256, 1, 1}});
                if (!result)
                    throw Exception(std::move(result).error());
            };
            if (entry != "blend")
                dispatch("setup", 1);
            dispatch(entry, uint32_t((count + 255) / 256));
        }
    } // namespace
    void lens_rays(const LensCamera& camera, core::Tensor& rays) {
        run("rays", camera, {}, {&rays}, size_t(camera.width) * camera.height);
    }
    void sample_lens_face(const core::Tensor& lens_rgb, const LensCamera& camera, const LensFace& face, core::Tensor& face_rgb) {
        run("sampleFace", camera, face, {&lens_rgb, &face_rgb}, size_t(face.size) * face.size);
    }
    void project_face_to_lens(const LensCamera& camera, const LensFace& face, const core::Tensor& points, const core::Tensor& normals, const core::Tensor& mask,
                              core::Tensor& distance, core::Tensor& normal, core::Tensor& weight) {
        run("projectFace", camera, face, {&points, &normals, &mask, &distance, &normal, &weight}, size_t(camera.width) * camera.height);
    }
    void blend_lens_faces(const core::Tensor& distance, const core::Tensor& normal, const core::Tensor& weight, const core::Tensor& scales, int faces, int pixels,
                          core::Tensor& out_distance, core::Tensor& out_normal) {
        run("blend", {}, {}, {&distance, &normal, &weight, &scales, &out_distance, &out_normal}, pixels, faces);
    }
} // namespace lfs::preprocessing
