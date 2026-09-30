/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Gsplat uses the tensor recorder and the shared stable pair sorter.

#include "lfs/training/ops/gsplat_vulkan.hpp"
#include "core/memory_pressure.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor_readback.hpp"
#include "lfs/training/ops/pair_sort_vulkan.hpp"
#include "vulkan/dispatch.hpp"
#include "vulkan/gsplat_state.hpp"
#include <climits>

#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "lfs/training/vram_ledger.hpp"

#include <bit>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace lfs::training {
    namespace {
        namespace ops = lfs::gpu_ops;
        namespace mk = vulkan;
        template <class P>
        void launch(uint32_t stage, const P& p, std::initializer_list<const core::Tensor*> tensors, uint32_t groups, const core::Tensor* control = nullptr, size_t word = 0) {
            std::vector<mk::StorageRef> refs;
            for (auto t : tensors)
                if (t->is_valid() && t->numel())
                    refs.push_back(mk::ref(*t));
            if (control)
                refs.push_back(mk::ref(*control));
            mk::dispatch("gsplat_forward", p, refs, refs, groups, stage, control ? mk::ref(*control) : mk::StorageRef{}, word * 4);
        }
        template <class P>
        void launch_items(uint32_t stage, const P& p, std::initializer_list<const core::Tensor*> tensors, size_t count) {
            if (count)
                launch(stage, p, tensors, mk::groups(count));
        }
        using core::DataType;
        using core::Device;
        using ops::Tensor;

        constexpr uint32_t kTile = 16;
        constexpr size_t kGradStride = 14; // gsplat.metal kGsplatGradStride
        constexpr size_t kCameraFloats = 64;

        struct SetupParams {
            uint64_t view, radial, tangential, prism, camera;
            float fx, fy, cx, cy;
            uint32_t width, height, model, radial_count, tangential_count, prism_count;
        };

        struct ProjectParams {
            uint64_t camera, means, scales, quats, opacities, sh0, sh_rest, sh_codes, sh_bounds;
            uint64_t radii, means2d, colors, depth_keys, tile_counts;
            uint32_t count, width, height, tiles_x, tiles_y, degree, layout_rest, sh_format;
        };

        struct BinParams {
            uint64_t counts, ends, radii, means2d, depths, keys, ids, offsets;
            uint32_t count, tiles_x, tiles_y;
            uint64_t control;
        };
        struct ScanParams {
            uint64_t input, output, totals;
            uint32_t count;
        };
        struct RasterParams {
            uint64_t camera, means, scales, quats, opacities, colors, bg_color, bg_image, tile_offsets, gaussian_ids;
            uint64_t image, alpha, last_ids, depth, depths;
            uint32_t count, width, height, tiles_x, tiles_y, mode;
            uint64_t control;
        };
        static_assert(sizeof(SetupParams) == 80 && offsetof(SetupParams, fx) == 40);
        static_assert(sizeof(ProjectParams) == 144 && offsetof(ProjectParams, count) == 112);
        static_assert(sizeof(BinParams) == 88 && offsetof(BinParams, count) == 64);
        static_assert(sizeof(RasterParams) == 152 && offsetof(RasterParams, count) == 120);
        static_assert(sizeof(ScanParams) == 32 && offsetof(ScanParams, count) == 24);
        struct BackParams {
            uint64_t camera, means, scales, quats, opacities, colors, bg_color, bg_image, tile_offsets, gaussian_ids;
            uint64_t alpha, last_ids, v_image, v_alpha, grads, densification, error_map, edge_map, edge_scores;
            uint64_t v_depth, depths, rendered_depth;
            uint32_t count, width, height, tiles_x, tiles_y, edge_factor, mode;
        };
        static_assert(sizeof(BackParams) == 208 && offsetof(BackParams, count) == 176);
        template <class P>
        void launch_backward(uint32_t stage, const P& p, std::initializer_list<const Tensor*> tensors, uint32_t groups) {
            std::vector<mk::StorageRef> refs;
            for (auto t : tensors)
                if (t->is_valid() && t->numel())
                    refs.push_back(mk::ref(*t));
            const auto context = core::internal::acquire_vulkan_context();
            const auto shader = context->caps().shader_atomic_float ? "gsplat_backward_atomic" : "gsplat_backward";
            mk::dispatch(shader, p, refs, refs, groups, stage);
        }
        struct AccumulateParams {
            uint64_t camera, means, scales, opacities, grads, radii, means2d;
            uint64_t means_grad, scaling_grad, rotation_grad, opacity_grad, sh0_grad, sh_rest_grad, grad_norms, shares;
            uint32_t count, degree, layout_rest, width, height;
        };

        static_assert(sizeof(AccumulateParams) == 144 && offsetof(AccumulateParams, count) == 120);
        static_assert(alignof(SetupParams) == 8 && alignof(ProjectParams) == 8 && alignof(BinParams) == 8 &&
                      alignof(RasterParams) == 8 && alignof(BackParams) == 8 && alignof(AccumulateParams) == 8);

        using Frame = vulkan::GsplatFrame;
        using vulkan::VulkanGsplatState;

        VulkanGsplatState& state_of(ops::GsplatSaved& saved) {
            if (!saved.backend)
                throw std::logic_error("gsplat raster op called without a created state");
            return static_cast<VulkanGsplatState&>(*saved.backend);
        }

        // A contiguous Float32 GPU operand with `numel` elements.
        Tensor float_input(const Tensor& tensor, const size_t numel, const std::string_view name) {
            if (!tensor.is_valid() || tensor.device() != Device::GPU || tensor.dtype() != DataType::Float32 ||
                tensor.numel() != numel)
                throw std::invalid_argument(std::format(
                    "gsplat {} must be a Float32 GPU tensor of {} elements, got {} {} {} of {} elements", name, numel,
                    tensor.is_valid() ? core::device_name(tensor.device()) : "invalid",
                    tensor.is_valid() ? core::dtype_name(tensor.dtype()) : "", tensor.is_valid() ? tensor.shape().str() : "",
                    tensor.is_valid() ? tensor.numel() : 0));
            return tensor.is_contiguous() ? tensor : tensor.contiguous();
        }

        // Gradient accumulators are updated in place, so they must already be contiguous.
        Tensor& gradient_output(Tensor& tensor, const size_t numel, const std::string_view name) {
            if (!tensor.is_valid() || tensor.device() != Device::GPU || tensor.dtype() != DataType::Float32 ||
                !tensor.is_contiguous() || tensor.numel() < numel)
                throw std::invalid_argument(std::format(
                    "gsplat {} gradient must be a contiguous Float32 GPU tensor of at least {} elements, got {} of {}",
                    name, numel, tensor.is_valid() ? tensor.shape().str() : "invalid",
                    tensor.is_valid() ? tensor.numel() : 0));
            return tensor;
        }

        // The first `count` coefficients of a distortion tensor on the device.
        Tensor distortion(const Tensor& source, const size_t count, Tensor& cache) {
            if (source.dtype() != DataType::Float32)
                throw std::invalid_argument(std::format("gsplat distortion coefficients must be Float32, got {}",
                                                        core::dtype_name(source.dtype())));
            const auto flat = source.contiguous().reshape({static_cast<int>(source.numel())}).slice(0, 0, count);
            if (flat.device() == Device::GPU)
                return flat.contiguous();
            cache = flat.contiguous().to(Device::GPU);
            return cache;
        }

        Tensor reuse(Tensor& cache, const core::TensorShape& shape, const DataType dtype) {
            if (!cache.is_valid() || cache.shape() != shape || cache.dtype() != dtype)
                cache = Tensor::empty(shape, Device::GPU, dtype);
            return cache;
        }

        Tensor reuse_capacity(Tensor& cache, size_t count, DataType dtype) {
            if (!cache.is_valid() || cache.numel() < count || cache.dtype() != dtype)
                cache = Tensor::empty({count}, Device::GPU, dtype);
            return cache;
        }

        Tensor inclusive_counts(const Tensor& input, Tensor* cache = nullptr) {
            auto output = cache ? reuse(*cache, input.shape(), DataType::Int64)
                                : Tensor::empty(input.shape(), Device::GPU, DataType::Int64);
            auto totals = Tensor::empty({(input.numel() + 255) / 256}, Device::GPU, DataType::Int64);
            ScanParams p{mk::address(input), mk::address(output), mk::address(totals), uint32_t(input.numel())};
            launch_items(4, p, {&input, &output, &totals}, input.numel());
            if (totals.numel() > 1) {
                auto prefix = inclusive_counts(totals);
                p.totals = mk::address(prefix);
                launch_items(5, p, {&output, &prefix}, input.numel());
            }
            return output;
        }

        ops::State create() { return std::make_unique<VulkanGsplatState>(); }

        void release(ops::GsplatSaved& saved) noexcept {
            if (!saved.backend)
                return;
            auto& state = static_cast<VulkanGsplatState&>(*saved.backend);
            state.live = false;
            state.frame = {};
            if (state.intersection_readback.pending())
                state.intersection_readback = {};
        }

        ops::RasterResult forward(ops::GsplatSaved& saved, const ops::SplatInputs& splats, const Tensor& view,
                                  const Tensor& radial, const Tensor& tangential, const Tensor& bg_color,
                                  const Tensor& bg_image, const ops::GsplatParams& params,
                                  const ops::RenderOutputs& output) {
            auto& state = state_of(saved);
            release(saved);
            const auto model = params.camera_model;
            if (model == core::CameraModelType::ORTHO)
                throw std::invalid_argument("gsplat has no orthographic camera model");
            const uint32_t bases = params.sh.active_bases;
            const uint32_t layout_bases = params.sh.layout_bases;
            const auto degree = static_cast<uint32_t>(std::lround(std::sqrt(static_cast<double>(bases)))) - 1u;
            if ((degree + 1) * (degree + 1) != bases || bases > 16 || layout_bases < bases || layout_bases > 16)
                throw std::invalid_argument(std::format(
                    "gsplat needs square SH basis counts with active <= layout <= 16, got active {} layout {}", bases,
                    layout_bases));

            Frame f;
            f.mode = params.render_mode;
            const auto count = splats.means.is_valid() && splats.means.ndim() > 0 ? splats.means.shape()[0] : 0;
            LFS_ASSERT_MSG(count <= INT_MAX, "Gsplat primitive count exceeds signed indexing");
            f.count = static_cast<uint32_t>(count);
            f.means = float_input(splats.means, count * 3, "means");
            f.scales = float_input(splats.raw_scales, count * 3, "scales");
            f.quats = float_input(splats.raw_rotations, count * 4, "rotations");
            f.opacities = float_input(splats.raw_opacities, count, "opacities");
            const Tensor sh0 = float_input(splats.sh0, count * 3, "sh0");
            f.degree = degree;
            f.layout_rest = layout_bases - 1;

            Tensor sh_rest, sh_codes, sh_bounds;
            if (degree > 0 && splats.shN.is_valid() && splats.shN.numel() > 0) {
                if (params.sh.storage == ops::ShStorage::Q16 && splats.sh_value_bounds.is_valid()) {
                    const auto cells = core::sh_value_quant::sh_value_u16_count(count, f.layout_rest);
                    if (splats.shN.numel() < cells || dtype_size(splats.shN.dtype()) != 2)
                        throw std::invalid_argument(std::format(
                            "gsplat Q16 SH-rest needs {} 16-bit cells, got {} {}", cells, splats.shN.numel(),
                            core::dtype_name(splats.shN.dtype())));
                    sh_codes = splats.shN.contiguous();
                    sh_bounds = float_input(splats.sh_value_bounds, 2 * core::sh_value_quant::n_bounds_for_prims(count),
                                            "SH value bounds");
                } else {
                    const auto floats = core::sh_swizzled_float_count(count, f.layout_rest);
                    if ((splats.shN.dtype() != DataType::Float32 && splats.shN.dtype() != DataType::Float16) || splats.shN.numel() < floats)
                        throw std::invalid_argument(std::format(
                            "gsplat SH-rest must be Float32 or Q16 codes with bounds and hold {} floats, got {} of {}",
                            floats, core::dtype_name(splats.shN.dtype()), splats.shN.numel()));
                    sh_rest = splats.shN.contiguous();
                }
            }

            const auto full_w = static_cast<uint32_t>(params.full_image.w);
            const auto full_h = static_cast<uint32_t>(params.full_image.h);
            f.width = params.tile_w > 0 ? static_cast<uint32_t>(params.tile_w) : full_w;
            f.height = params.tile_h > 0 ? static_cast<uint32_t>(params.tile_h) : full_h;
            if (f.width == 0 || f.height == 0)
                throw std::invalid_argument(std::format("gsplat needs a nonempty image, got {}x{}", f.width, f.height));
            const uint32_t tiles_x = (f.width + kTile - 1) / kTile;
            const uint32_t tiles_y = (f.height + kTile - 1) / kTile;
            const uint32_t tiles = tiles_x * tiles_y;

            // Equirectangular K carries the full image size and the tile offset.
            const bool equirect = model == core::CameraModelType::EQUIRECTANGULAR;
            const auto [fx, fy, cx, cy] = params.intrinsics;
            SetupParams setup{
                .fx = equirect ? static_cast<float>(full_w) : fx,
                .fy = equirect ? static_cast<float>(full_h) : fy,
                .cx = equirect ? static_cast<float>(params.tile_x) : cx - static_cast<float>(params.tile_x),
                .cy = equirect ? static_cast<float>(params.tile_y) : cy - static_cast<float>(params.tile_y),
                .width = f.width,
                .height = f.height,
                .model = static_cast<uint32_t>(model),
            };
            const Tensor view_matrix = float_input(view, 16, "view");
            Tensor radial_d, tangential_d, prism_d;
            const auto has = [](const Tensor& t, const size_t n) { return t.is_valid() && t.numel() >= n; };
            switch (model) {
            case core::CameraModelType::PINHOLE:
                if (has(radial, 1))
                    radial_d = distortion(radial, std::min<size_t>(radial.numel(), 6), state.radial);
                if (has(tangential, 2))
                    tangential_d = distortion(tangential, 2, state.tangential);
                break;
            case core::CameraModelType::FISHEYE:
                if (has(radial, 4))
                    radial_d = distortion(radial, 4, state.radial);
                break;
            case core::CameraModelType::THIN_PRISM_FISHEYE:
                if (radial.is_valid() && radial.numel() == 4)
                    radial_d = distortion(radial, 4, state.radial);
                if (tangential.is_valid() && tangential.numel() == 4)
                    prism_d = distortion(tangential, 4, state.prism);
                break;
            default: break;
            }
            setup.radial_count = static_cast<uint32_t>(radial_d.is_valid() ? radial_d.numel() : 0);
            setup.tangential_count = static_cast<uint32_t>(tangential_d.is_valid() ? tangential_d.numel() : 0);
            setup.prism_count = static_cast<uint32_t>(prism_d.is_valid() ? prism_d.numel() : 0);
            f.camera = reuse(state.camera, {kCameraFloats}, DataType::Float32);
            setup.view = mk::address(view_matrix);
            setup.radial = mk::address(radial_d);
            setup.tangential = mk::address(tangential_d);
            setup.prism = mk::address(prism_d);
            setup.camera = mk::address(f.camera);
            launch(0, setup, {&view_matrix, &radial_d, &tangential_d, &prism_d, &f.camera}, 1);

            f.radii = reuse(state.radii, {count, 2}, DataType::Int32);
            f.means2d = reuse(state.means2d, {count, 2}, DataType::Float32);
            f.colors = reuse(state.colors, {count, 3}, DataType::Float32);
            Tensor depth_keys = reuse(state.depth_keys, {count}, DataType::Int32);
            const Tensor tile_counts = reuse(state.tile_counts, {count}, DataType::Int64);
            f.depths = depth_keys;
            if (count > 0) {
                const ProjectParams project{
                    mk::address(f.camera), mk::address(f.means), mk::address(f.scales), mk::address(f.quats),
                    mk::address(f.opacities), mk::address(sh0), mk::address(sh_codes.is_valid() ? sh_codes : sh_rest), mk::address(sh_codes),
                    mk::address(sh_bounds), mk::address(f.radii), mk::address(f.means2d), mk::address(f.colors),
                    mk::address(depth_keys), mk::address(tile_counts), f.count, f.width, f.height, tiles_x, tiles_y,
                    degree, f.layout_rest, params.sh.storage == ops::ShStorage::Q16 ? 3u : params.sh.storage == ops::ShStorage::IeeeFloat16 ? 2u
                                                                                                                                            : 1u};
                launch_items(1, project,
                             {&f.camera, &f.means, &f.scales, &f.quats, &f.opacities, &sh0, &sh_rest, &sh_codes,
                              &sh_bounds, &f.radii, &f.means2d, &f.colors, &depth_keys, &tile_counts},
                             count);
            }

            const uint64_t tiles_wide = uint64_t(tiles_x) * tiles_y;
            LFS_ASSERT_MSG(tiles_wide < INT_MAX && uint64_t(f.width) * f.height <= INT_MAX, "Gsplat image exceeds signed indexing");
            int64_t intersections = 0;
            Tensor ends;
            if (count)
                ends = inclusive_counts(tile_counts, &state.ends);
            if (count && !state.indirect) {
                intersections = ends.slice(0, count - 1, count).item<int64_t>();
            }
            if (intersections < 0 || intersections > INT_MAX)
                return {ops::RasterResult::Code::InstanceOverflow, false, "Gsplat tile intersections exceed signed indexing"};
            const auto plane = [&](const size_t channels) {
                return core::TensorShape({channels, static_cast<size_t>(f.height), static_cast<size_t>(f.width)});
            };
            const bool rgb = params.render_mode == ops::GsplatRenderMode::RGB || params.render_mode == ops::GsplatRenderMode::RGB_D || params.render_mode == ops::GsplatRenderMode::RGB_ED;
            const Tensor image = rgb ? reuse(state.image, plane(3), DataType::Float32) : Tensor{};
            const Tensor depth = params.render_mode != ops::GsplatRenderMode::RGB ? reuse(state.depth, plane(1), DataType::Float32) : Tensor{};
            f.alpha = reuse(state.alpha, plane(1), DataType::Float32);
            f.last_ids = reuse(state.last_ids, plane(1), DataType::Int32);
            if (bg_image.is_valid() && !bg_image.is_empty())
                f.bg_image = float_input(bg_image, 3 * size_t{f.width} * f.height, "background image");
            else if (bg_color.is_valid() && bg_color.numel() > 0)
                f.bg_color = float_input(bg_color, 3, "background colour");
            const auto record_forward = [&](uint32_t capacity, const Tensor* control) {
                f.tile_offsets = reuse(state.tile_offsets, {size_t(tiles) + 1}, DataType::Int32);
                f.tile_offsets.zero_();
                if (capacity) {
                    auto keys_a = reuse_capacity(state.keys_a, size_t(capacity), DataType::Int64);
                    auto keys_b = reuse_capacity(state.keys_b, size_t(capacity), DataType::Int64);
                    auto ids_a = reuse_capacity(state.ids_a, size_t(capacity), DataType::UInt32);
                    auto ids_b = reuse_capacity(state.ids_b, size_t(capacity), DataType::UInt32);
                    BinParams bin{mk::address(tile_counts), mk::address(ends), mk::address(f.radii), mk::address(f.means2d),
                                  mk::address(depth_keys), mk::address(keys_a), mk::address(ids_a), mk::address(f.tile_offsets),
                                  f.count, tiles_x, tiles_y, control ? mk::address(*control) : 0};
                    launch(2, bin, {&tile_counts, &ends, &f.radii, &f.means2d, &depth_keys, &keys_a, &ids_a}, mk::groups(count), control, 4);
                    const bool in_a = vulkan_pair_sort({&keys_a, &keys_b, &ids_a, &ids_b}, capacity, 0,
                                                       32 + std::bit_width(tiles - 1), true, nullptr, control);
                    f.gaussian_ids = in_a ? ids_a : ids_b;
                    const auto& keys = in_a ? keys_a : keys_b;
                    bin.keys = mk::address(keys);
                    bin.count = capacity;
                    launch(3, bin, {&keys, &f.tile_offsets}, mk::groups(capacity), control, 12);
                }

                if (!control && capacity == 0) {
                    // As CUDA: an empty intersection list clears the outputs, background included.
                    if (image.is_valid())
                        state.image.zero_();
                    if (depth.is_valid())
                        state.depth.zero_();
                    f.alpha.zero_();
                    f.last_ids.zero_();
                } else {
                    const RasterParams raster{
                        .camera = mk::address(f.camera),
                        .means = mk::address(f.means),
                        .scales = mk::address(f.scales),
                        .quats = mk::address(f.quats),
                        .opacities = mk::address(f.opacities),
                        .colors = mk::address(f.colors),
                        .bg_color = mk::address(f.bg_color),
                        .bg_image = mk::address(f.bg_image),
                        .tile_offsets = mk::address(f.tile_offsets),
                        .gaussian_ids = mk::address(f.gaussian_ids),
                        .image = mk::address(image),
                        .alpha = mk::address(f.alpha),
                        .last_ids = mk::address(f.last_ids),
                        .depth = mk::address(depth),
                        .depths = mk::address(f.depths),
                        .count = f.count,
                        .width = f.width,
                        .height = f.height,
                        .tiles_x = tiles_x,
                        .tiles_y = tiles_y,
                        .mode = static_cast<uint32_t>(params.render_mode),
                        .control = control ? mk::address(*control) : 0,
                    };
                    launch(6, raster,
                           {&f.camera, &f.means, &f.scales, &f.quats, &f.opacities, &f.colors, &f.bg_color,
                            &f.bg_image, &f.tile_offsets, &f.gaussian_ids, &image, &depth, &f.depths, &f.alpha, &f.last_ids},
                           std::min(tiles, core::internal::acquire_vulkan_context()->caps().max_workgroup_count[0]), control, 20);
                }
            };
            if (!state.indirect) {
                f.intersections = uint32_t(intersections);
                record_forward(f.intersections, nullptr);
            } else {
                if (!state.control.is_valid())
                    state.control = Tensor::empty({32}, Device::GPU, DataType::UInt32);
                size_t capacity = state.keys_a.is_valid() ? state.keys_a.numel() : std::min(size_t{1} << 20, count * 8);
                capacity = std::max(size_t{256}, capacity);
                for (;;) {
                    struct ControlParams {
                        uint64_t ends, control;
                        uint32_t count, capacity, tiles, max_groups;
                    };
                    const ControlParams control{mk::address(ends), mk::address(state.control), f.count,
                                                uint32_t(capacity), tiles, core::internal::acquire_vulkan_context()->caps().max_workgroup_count[0]};
                    launch(7, control, {&ends, &state.control}, 1);
                    state.intersection_readback.enqueue_range(state.control, sizeof(uint32_t), 2 * sizeof(uint32_t));
                    record_forward(uint32_t(capacity), &state.control);
                    // Submit the dependent work before checking capacity. Overflow
                    // dispatches are empty, so retry cannot expose partial output.
                    core::internal::acquire_vulkan_context()->recorders().flush_current();
                    std::array<uint32_t, 2> status{};
                    state.intersection_readback.wait(std::as_writable_bytes(std::span(status)));
                    if (status[0] == 0) {
                        f.intersections = status[1];
                        break;
                    }
                    if (status[0] == 2)
                        return {ops::RasterResult::Code::InstanceOverflow, false, "Gsplat tile intersections exceed signed indexing"};
                    // One eighth headroom avoids repeated growth without reserving
                    // another full pair buffer at late training sizes.
                    capacity = std::min(size_t(INT_MAX), (size_t(status[1]) + status[1] / 8 + 255) & ~size_t{255});
                }
            }
            output.image = image;
            output.alpha = f.alpha;
            output.depth = depth;
            output.normal = {};
            state.frame = std::move(f);
            state.live = true;
            return {ops::RasterResult::Code::Success, true, {}};
        }

        void backward(ops::GsplatSaved& saved, const Tensor& grad_image, const Tensor& grad_alpha,
                      const ops::GsplatGradients& gradients, Tensor& densification, const Tensor& error_map,
                      const Tensor& edge_map, Tensor& edge_scores, Tensor& max_screen_share) {
            auto& state = state_of(saved);
            if (!state.live)
                throw std::logic_error("gsplat backward called without a live forward frame");
            // The frame ends with this backward, as on CUDA.
            struct EndFrame {
                ops::GsplatSaved& saved;
                ~EndFrame() { release(saved); }
            } const end_frame{saved};
            const Frame& f = state.frame;
            const size_t count = f.count;
            const size_t pixels = size_t{f.width} * f.height;
            const auto plane = [&](const size_t channels) {
                return core::TensorShape({channels, static_cast<size_t>(f.height), static_cast<size_t>(f.width)});
            };
            Tensor v_image = Tensor::zeros(plane(3), Device::GPU, DataType::Float32), v_depth;
            if (grad_image.is_valid() && grad_image.numel()) {
                if (grad_image.numel() == 3 * pixels)
                    v_image = float_input(grad_image, 3 * pixels, "image gradient");
                else if (grad_image.numel() == 4 * pixels) {
                    auto channels = float_input(grad_image, 4 * pixels, "image gradient").reshape({4, int(f.height), int(f.width)});
                    v_image = channels.slice(0, 0, 3);
                    v_depth = channels.slice(0, 3, 4);
                } else {
                    LFS_ASSERT_MSG(f.mode == ops::GsplatRenderMode::D || f.mode == ops::GsplatRenderMode::ED, "Gsplat image gradient has the wrong channel count");
                    v_depth = float_input(grad_image, pixels, "depth gradient");
                }
            }
            const Tensor v_alpha = grad_alpha.is_valid() && grad_alpha.numel() > 0
                                       ? float_input(grad_alpha, pixels, "alpha gradient")
                                       : Tensor::zeros(plane(1), Device::GPU, DataType::Float32);

            const bool update_densification =
                densification.is_valid() && densification.ndim() == 2 && densification.shape()[1] >= count;
            if (update_densification)
                gradient_output(densification, 2 * count, "densification");
            Tensor errors;
            if (update_densification && error_map.is_valid() && error_map.numel() > 0) {
                const bool plane_shape = (error_map.ndim() == 2 || (error_map.ndim() == 3 && error_map.shape()[0] == 1)) &&
                                         error_map.shape()[error_map.ndim() - 2] == f.height &&
                                         error_map.shape()[error_map.ndim() - 1] == f.width;
                if (!plane_shape)
                    throw std::invalid_argument(std::format("gsplat pixel error map must be [{}, {}] or [1, {}, {}], got {}",
                                                            f.height, f.width, f.height, f.width, error_map.shape().str()));
                errors = float_input(error_map.device() == Device::GPU ? error_map : error_map.to(Device::GPU), pixels,
                                     "pixel error map");
            }
            const bool edge_scoring =
                edge_map.is_valid() && edge_scores.is_valid() && edge_map.device() == Device::GPU &&
                edge_scores.device() == Device::GPU && edge_map.dtype() == DataType::Float32 &&
                edge_scores.dtype() == DataType::Float32 && edge_map.ndim() == 2 && edge_scores.ndim() == 1 &&
                edge_map.shape()[0] == f.height && edge_map.shape()[1] == f.width && edge_scores.numel() == count &&
                edge_scores.is_contiguous();
            const Tensor edges = edge_scoring ? edge_map.contiguous() : Tensor();
            const Tensor grads = reuse(state.grads, {count, kGradStride}, DataType::Float32);
            state.grads.zero_();

            if (f.intersections > 0) {
                const uint32_t tiles_x = (f.width + kTile - 1) / kTile;
                const uint32_t tiles_y = (f.height + kTile - 1) / kTile;
                const BackParams raster{
                    .camera = mk::address(f.camera),
                    .means = mk::address(f.means),
                    .scales = mk::address(f.scales),
                    .quats = mk::address(f.quats),
                    .opacities = mk::address(f.opacities),
                    .colors = mk::address(f.colors),
                    .bg_color = mk::address(f.bg_color),
                    .bg_image = mk::address(f.bg_image),
                    .tile_offsets = mk::address(f.tile_offsets),
                    .gaussian_ids = mk::address(f.gaussian_ids),
                    .alpha = mk::address(f.alpha),
                    .last_ids = mk::address(f.last_ids),
                    .v_image = mk::address(v_image),
                    .v_alpha = mk::address(v_alpha),
                    .grads = mk::address(grads),
                    .densification = update_densification && errors.is_valid() ? mk::address(densification) : 0,
                    .error_map = mk::address(errors),
                    .edge_map = mk::address(edges),
                    .edge_scores = edge_scoring ? mk::address(edge_scores) : 0,
                    .v_depth = mk::address(v_depth),
                    .depths = mk::address(f.depths),
                    .rendered_depth = mk::address(state.depth),
                    .count = f.count,
                    .width = f.width,
                    .height = f.height,
                    .tiles_x = tiles_x,
                    .tiles_y = tiles_y,
                    .edge_factor = f.mode == ops::GsplatRenderMode::RGB ? 32u : 1u,
                    .mode = static_cast<uint32_t>(f.mode),
                };
                // Two pixels per thread.
                launch_backward(0, raster,
                                {&f.camera, &f.means, &f.scales, &f.quats, &f.opacities, &f.colors, &f.bg_color,
                                 &f.bg_image, &f.tile_offsets, &f.gaussian_ids, &f.alpha, &f.last_ids, &v_image,
                                 &v_alpha, &grads, &densification, &errors, &edges, &edge_scores, &v_depth, &f.depths, &state.depth},
                                std::min(tiles_x * tiles_y, core::internal::acquire_vulkan_context()->caps().max_workgroup_count[0]));
            }

            auto slot = [&](const ops::AdamSlot which) -> Tensor& { return gradients.get(gradients.owner, which); };
            Tensor& means_grad = gradient_output(slot(ops::AdamSlot::Means), count * 3, "means");
            Tensor& scaling_grad = gradient_output(slot(ops::AdamSlot::Scaling), count * 3, "scaling");
            Tensor& rotation_grad = gradient_output(slot(ops::AdamSlot::Rotation), count * 4, "rotation");
            Tensor& opacity_grad = gradient_output(slot(ops::AdamSlot::Opacity), count, "opacity");
            // SH-rest gradient storage is resolved only once a rest band is active.
            Tensor sh_rest_grad;
            if (f.degree > 0) {
                Tensor& rest = slot(ops::AdamSlot::ShN);
                if (rest.is_valid() && rest.numel() > 0)
                    sh_rest_grad = gradient_output(rest, core::sh_swizzled_float_count(count, f.layout_rest), "SH-rest");
            }
            Tensor& sh0_grad = gradient_output(slot(ops::AdamSlot::Sh0), count * 3, "sh0");
            const bool shares = max_screen_share.is_valid() && max_screen_share.numel() == count && count > 0;
            if (shares)
                gradient_output(max_screen_share, count, "max screen share");
            if (count == 0)
                return;
            const AccumulateParams accumulate{
                .camera = mk::address(f.camera),
                .means = mk::address(f.means),
                .scales = mk::address(f.scales),
                .opacities = mk::address(f.opacities),
                .grads = mk::address(grads),
                .radii = mk::address(f.radii),
                .means2d = mk::address(f.means2d),
                .means_grad = mk::address(means_grad),
                .scaling_grad = mk::address(scaling_grad),
                .rotation_grad = mk::address(rotation_grad),
                .opacity_grad = mk::address(opacity_grad),
                .sh0_grad = mk::address(sh0_grad),
                .sh_rest_grad = mk::address(sh_rest_grad),
                .grad_norms = update_densification && !errors.is_valid() ? mk::address(densification) : 0,
                .shares = shares ? mk::address(max_screen_share) : 0,
                .count = f.count,
                .degree = f.degree,
                .layout_rest = f.layout_rest,
                .width = f.width,
                .height = f.height,
            };
            launch_backward(1, accumulate,
                            {&f.camera, &f.means, &f.scales, &f.opacities, &grads, &f.radii, &f.means2d, &means_grad,
                             &scaling_grad, &rotation_grad, &opacity_grad, &sh0_grad, &sh_rest_grad, &densification,
                             &max_screen_share},
                            std::min(uint32_t((count + 127) / 128), core::internal::acquire_vulkan_context()->caps().max_workgroup_count[0]));
        }

        void record_vram(const ops::GsplatSaved& saved, const Tensor& image, const Tensor& alpha,
                         const Tensor& gt_tile, const Tensor& bg_tile, const Tensor& error_map) {
            if (!saved.backend)
                return;
            const auto& state = static_cast<const VulkanGsplatState&>(*saved.backend);
            constexpr std::string_view scope = "rasterizer.gsplat";
            record_vram_tensor(scope, "forward.gaussian_ids", state.frame.gaussian_ids);
            record_vram_tensor(scope, "forward.tile_offsets", state.frame.tile_offsets);
            record_vram_tensor(scope, "forward.radii", state.frame.radii);
            record_vram_tensor(scope, "forward.means2d", state.frame.means2d);
            record_vram_tensor(scope, "forward.colors", state.frame.colors);
            record_vram_tensor(scope, "forward.last_ids", state.last_ids);
            record_vram_tensor(scope, "output.image", image);
            record_vram_tensor(scope, "output.alpha", alpha);
            record_vram_tensor(scope, "output.depth", state.depth);
            record_vram_tensor(scope, "cache.radii", state.radii);
            record_vram_tensor(scope, "cache.means2d", state.means2d);
            record_vram_tensor(scope, "cache.colors", state.colors);
            record_vram_tensor(scope, "cache.depth_keys", state.depth_keys);
            record_vram_tensor(scope, "cache.tile_counts", state.tile_counts);
            record_vram_tensor(scope, "cache.tile_offsets", state.tile_offsets);
            record_vram_tensor(scope, "cache.ends", state.ends);
            record_vram_tensor(scope, "cache.keys_a", state.keys_a);
            record_vram_tensor(scope, "cache.keys_b", state.keys_b);
            record_vram_tensor(scope, "cache.ids_a", state.ids_a);
            record_vram_tensor(scope, "cache.ids_b", state.ids_b);
            record_vram_tensor(scope, "cache.grads", state.grads);
            record_vram_tensor(scope, "cache.control", state.control);
            record_vram_tensor(scope, "camera.block", state.camera);
            record_vram_tensor(scope, "camera.radial", state.radial);
            record_vram_tensor(scope, "camera.tangential", state.tangential);
            record_vram_tensor(scope, "camera.thin_prism", state.prism);
            record_vram_tensor("train.inputs", "gt_tile", gt_tile);
            record_vram_tensor("train.inputs", "background_tile", bg_tile);
            record_vram_tensor("train.losses", "densification_error_map.live", error_map);
        }

        bool release_caches(ops::GsplatSaved& saved) noexcept {
            if (!saved.backend)
                return true;
            auto& state = static_cast<VulkanGsplatState&>(*saved.backend);
            state.camera = {};
            state.radial = {};
            state.tangential = {};
            state.prism = {};
            state.image = {};
            state.alpha = {};
            state.last_ids = {};
            state.depth = {};
            state.radii = {};
            state.means2d = {};
            state.colors = {};
            state.depth_keys = {};
            state.tile_counts = {};
            state.tile_offsets = {};
            state.ends = {};
            state.keys_a = {};
            state.keys_b = {};
            state.ids_a = {};
            state.ids_b = {};
            state.grads = {};
            state.control = {};
            return true;
        }
    } // namespace

    namespace vulkan {
        gpu_ops::State gsplat_create() { return create(); }
        gpu_ops::RasterResult gsplat_forward(gpu_ops::GsplatSaved& saved, const gpu_ops::SplatInputs& splats,
                                             gpu_ops::In view, gpu_ops::In radial, gpu_ops::In tangential, gpu_ops::In bg, gpu_ops::In bg_image,
                                             const gpu_ops::GsplatParams& params, const gpu_ops::RenderOutputs& output) {
            try {
                return forward(saved, splats, view, radial, tangential, bg, bg_image, params, output);
            } catch (const core::MemoryAllocationError& error) {
                auto& state = state_of(saved);
                state.message = error.what();
                release(saved);
                return {gpu_ops::RasterResult::Code::ResourceExhausted, false, state.message};
            } catch (const std::bad_alloc& error) {
                auto& state = state_of(saved);
                state.message = error.what();
                release(saved);
                return {gpu_ops::RasterResult::Code::ResourceExhausted, false, state.message};
            }
        }
        void gsplat_release(gpu_ops::GsplatSaved& saved) noexcept { release(saved); }
    } // namespace vulkan
    const gpu_ops::GsplatRasterOps& vulkan_gsplat_ops() {
        static const gpu_ops::GsplatRasterOps ops{
            .create = create,
            .forward = vulkan::gsplat_forward,
            .backward = backward,
            .release = release,
            .record_vram = record_vram,
            .release_caches = release_caches};
        return ops;
    }
} // namespace lfs::training
