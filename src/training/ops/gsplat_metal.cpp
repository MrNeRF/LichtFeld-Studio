/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal GsplatRasterOps; see ops/gsplat_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

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
        namespace mk = metal;
        using core::DataType;
        using core::Device;
        using ops::Tensor;

        constexpr uint32_t kTile = 16;
        constexpr size_t kGradStride = 14; // gsplat.metal kGsplatGradStride
        constexpr uint32_t kRadixBlock = mk::kGroupWidth * 8;
        constexpr size_t kCameraFloats = 64;

        struct SetupParams {
            uint64_t view, radial, tangential, prism, camera;
            float fx, fy, cx, cy;
            uint32_t width, height, model, radial_count, tangential_count, prism_count;
        };

        struct ProjectParams {
            uint64_t camera, means, scales, quats, opacities, sh0, sh_rest, sh_codes, sh_bounds;
            uint64_t radii, means2d, colors, depth_keys, tile_counts;
            uint32_t count, width, height, tiles_x, tiles_y, degree, layout_rest;
        };

        struct BinParams {
            uint64_t order, tile_counts, ranked_counts, ranked_ends, radii, means2d, tile_keys, gaussian_ids,
                tile_offsets;
            uint32_t count, tiles_x, tiles_y;
        };

        struct RadixParams {
            uint64_t keys_in, values_in, keys_out, values_out, counts, ends;
            uint32_t count, shift, blocks;
        };

        struct RasterParams {
            uint64_t camera, means, scales, quats, opacities, colors, bg_color, bg_image, tile_offsets, gaussian_ids;
            uint64_t image, alpha, last_ids, v_image, v_alpha, grads, densification, error_map, edge_map, edge_scores;
            uint32_t count, width, height, tiles_x;
        };

        struct AccumulateParams {
            uint64_t camera, means, scales, opacities, grads, radii, means2d;
            uint64_t means_grad, scaling_grad, rotation_grad, opacity_grad, sh0_grad, sh_rest_grad, grad_norms, shares;
            uint32_t count, degree, layout_rest, width, height;
        };

        // What backward needs from the forward it follows.
        struct Frame {
            Tensor means, scales, quats, opacities, camera, bg_color, bg_image;
            Tensor radii, means2d, colors, tile_offsets, gaussian_ids, alpha, last_ids;
            uint32_t count = 0, width = 0, height = 0, degree = 0, layout_rest = 0, intersections = 0;
        };

        struct MetalGsplatState : ops::BackendState {
            Frame frame;
            bool live = false;
            Tensor camera, radial, tangential, prism, image, alpha, last_ids;
        };

        MetalGsplatState& state_of(ops::GsplatSaved& saved) {
            if (!saved.backend)
                throw std::logic_error("gsplat raster op called without a created state");
            return static_cast<MetalGsplatState&>(*saved.backend);
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

        // Stable LSD radix sort of `keys` (uint bits in Int32 storage) over
        // their low `bits`, carrying `values`; empty `values` start as indices.
        void radix_sort(Tensor& keys, Tensor& values, const uint32_t count, const uint32_t bits) {
            if (count == 0 || bits == 0)
                return;
            const uint32_t blocks = (count + kRadixBlock - 1) / kRadixBlock;
            const auto counts = Tensor::empty({size_t{256} * blocks}, Device::GPU, DataType::Int32);
            auto keys_out = Tensor::empty({count}, Device::GPU, DataType::Int32);
            auto values_out = Tensor::empty({count}, Device::GPU, DataType::Int32);
            for (uint32_t shift = 0; shift < bits; shift += 8) {
                RadixParams params{mk::address(keys), mk::address(values), mk::address(keys_out),
                                   mk::address(values_out), mk::address(counts), 0, count, shift, blocks};
                mk::launch("gsplat_radix_histogram", params, {&keys, &counts}, blocks);
                const Tensor ends = counts.cumsum(0);
                params.ends = mk::address(ends);
                mk::launch("gsplat_radix_scatter", params, {&keys, &values, &keys_out, &values_out, &counts, &ends},
                           blocks);
                std::swap(keys, keys_out);
                std::swap(values, values_out);
                if (!values_out.is_valid())
                    values_out = Tensor::empty({count}, Device::GPU, DataType::Int32);
            }
        }

        ops::State create() { return std::make_unique<MetalGsplatState>(); }

        void release(ops::GsplatSaved& saved) noexcept {
            if (!saved.backend)
                return;
            auto& state = static_cast<MetalGsplatState&>(*saved.backend);
            state.live = false;
            state.frame = {};
        }

        ops::RasterResult forward(ops::GsplatSaved& saved, const ops::SplatInputs& splats, const Tensor& view,
                                  const Tensor& radial, const Tensor& tangential, const Tensor& bg_color,
                                  const Tensor& bg_image, const ops::GsplatParams& params,
                                  const ops::RenderOutputs& output) {
            auto& state = state_of(saved);
            release(saved);
            // CUDA depth modes read colour slots the SH pass never writes.
            if (params.render_mode != ops::GsplatRenderMode::RGB)
                throw std::invalid_argument(std::format("Metal gsplat renders RGB only, got render mode {}",
                                                        static_cast<int>(params.render_mode)));
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
            const auto count = splats.means.is_valid() && splats.means.ndim() > 0 ? splats.means.shape()[0] : 0;
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
                    if (splats.shN.dtype() != DataType::Float32 || splats.shN.numel() < floats)
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
            mk::launch("gsplat_camera_setup", setup, {&view_matrix, &radial_d, &tangential_d, &prism_d, &f.camera}, 1, 1);

            f.radii = Tensor::empty({count, 2}, Device::GPU, DataType::Int32);
            f.means2d = Tensor::empty({count, 2}, Device::GPU, DataType::Float32);
            f.colors = Tensor::empty({count, 3}, Device::GPU, DataType::Float32);
            Tensor depth_keys = Tensor::empty({count}, Device::GPU, DataType::Int32);
            const Tensor tile_counts = Tensor::empty({count}, Device::GPU, DataType::Int32);
            if (count > 0) {
                const ProjectParams project{
                    mk::address(f.camera), mk::address(f.means), mk::address(f.scales), mk::address(f.quats),
                    mk::address(f.opacities), mk::address(sh0), mk::address(sh_rest), mk::address(sh_codes),
                    mk::address(sh_bounds), mk::address(f.radii), mk::address(f.means2d), mk::address(f.colors),
                    mk::address(depth_keys), mk::address(tile_counts), f.count, f.width, f.height, tiles_x, tiles_y,
                    degree, f.layout_rest};
                mk::launch_items("gsplat_project", project,
                                 {&f.camera, &f.means, &f.scales, &f.quats, &f.opacities, &sh0, &sh_rest, &sh_codes,
                                  &sh_bounds, &f.radii, &f.means2d, &f.colors, &depth_keys, &tile_counts},
                                 count);
            }

            // Gaussians by depth bits, their tiles in that order, then a stable
            // sort of the tile ids: CUDA's (tile, depth, gaussian) order.
            Tensor order;
            radix_sort(depth_keys, order, f.count, 32);
            BinParams bin{.tiles_x = tiles_x, .tiles_y = tiles_y};
            int64_t intersections = 0;
            Tensor ranked, ranked_ends;
            if (count > 0) {
                ranked = Tensor::empty({count}, Device::GPU, DataType::Int32);
                bin.order = mk::address(order);
                bin.tile_counts = mk::address(tile_counts);
                bin.ranked_counts = mk::address(ranked);
                bin.count = f.count;
                mk::launch_items("gsplat_rank_counts", bin, {&order, &tile_counts, &ranked}, count);
                ranked_ends = ranked.cumsum(0);
                // The frame's one host read: the intersection count sizes the tile sort.
                intersections = ranked_ends.slice(0, count - 1, count).item<int32_t>();
                // Per-gaussian counts are at most the tile count; a wrapped
                // Int32 scan shows up as a negative total.
                if (intersections < 0)
                    return {ops::RasterResult::Code::ResourceExhausted, false,
                            "gsplat tile intersections exceed the Int32 range"};
            }
            f.intersections = static_cast<uint32_t>(intersections);
            auto tile_keys = Tensor::empty({static_cast<size_t>(intersections)}, Device::GPU, DataType::Int32);
            f.gaussian_ids = Tensor::empty({static_cast<size_t>(intersections)}, Device::GPU, DataType::Int32);
            if (intersections > 0) {
                bin.ranked_ends = mk::address(ranked_ends);
                bin.radii = mk::address(f.radii);
                bin.means2d = mk::address(f.means2d);
                bin.tile_keys = mk::address(tile_keys);
                bin.gaussian_ids = mk::address(f.gaussian_ids);
                mk::launch_items("gsplat_fill_intersections", bin,
                                 {&order, &ranked, &ranked_ends, &f.radii, &f.means2d, &tile_keys, &f.gaussian_ids},
                                 count);
                radix_sort(tile_keys, f.gaussian_ids, f.intersections, std::bit_width(tiles - 1));
                f.tile_offsets = Tensor::empty({size_t{tiles} + 1}, Device::GPU, DataType::Int32);
                bin.tile_keys = mk::address(tile_keys);
                bin.tile_offsets = mk::address(f.tile_offsets);
                bin.count = f.intersections;
                mk::launch_items("gsplat_tile_offsets", bin, {&tile_keys, &f.tile_offsets}, f.intersections);
            } else {
                f.tile_offsets = Tensor::zeros({size_t{tiles} + 1}, Device::GPU, DataType::Int32);
            }

            const auto plane = [&](const size_t channels) {
                return core::TensorShape({channels, static_cast<size_t>(f.height), static_cast<size_t>(f.width)});
            };
            const Tensor image = reuse(state.image, plane(3), DataType::Float32);
            f.alpha = reuse(state.alpha, plane(1), DataType::Float32);
            f.last_ids = reuse(state.last_ids, plane(1), DataType::Int32);
            if (bg_image.is_valid() && !bg_image.is_empty())
                f.bg_image = float_input(bg_image, 3 * size_t{f.width} * f.height, "background image");
            else if (bg_color.is_valid() && bg_color.numel() > 0)
                f.bg_color = float_input(bg_color, 3, "background colour");
            if (intersections == 0) {
                // As CUDA: an empty intersection list clears the outputs, background included.
                state.image.zero_();
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
                    .count = f.count,
                    .width = f.width,
                    .height = f.height,
                    .tiles_x = tiles_x,
                };
                mk::launch_2d("gsplat_rasterize_forward", raster,
                              {&f.camera, &f.means, &f.scales, &f.quats, &f.opacities, &f.colors, &f.bg_color,
                               &f.bg_image, &f.tile_offsets, &f.gaussian_ids, &image, &f.alpha, &f.last_ids},
                              tiles_x, tiles_y, kTile, kTile);
            }
            output.image = image;
            output.alpha = f.alpha;
            output.depth = Tensor();
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
            const Tensor v_image = grad_image.is_valid() && grad_image.numel() > 0
                                       ? float_input(grad_image, 3 * pixels, "image gradient")
                                       : Tensor::zeros(plane(3), Device::GPU, DataType::Float32);
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
            const Tensor grads = Tensor::zeros({count, kGradStride}, Device::GPU, DataType::Float32);

            if (f.intersections > 0) {
                const uint32_t tiles_x = (f.width + kTile - 1) / kTile;
                const uint32_t tiles_y = (f.height + kTile - 1) / kTile;
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
                    .alpha = mk::address(f.alpha),
                    .last_ids = mk::address(f.last_ids),
                    .v_image = mk::address(v_image),
                    .v_alpha = mk::address(v_alpha),
                    .grads = mk::address(grads),
                    .densification = update_densification && errors.is_valid() ? mk::address(densification) : 0,
                    .error_map = mk::address(errors),
                    .edge_map = mk::address(edges),
                    .edge_scores = edge_scoring ? mk::address(edge_scores) : 0,
                    .count = f.count,
                    .width = f.width,
                    .height = f.height,
                    .tiles_x = tiles_x,
                };
                // Two pixels per thread.
                mk::launch_2d("gsplat_rasterize_backward", raster,
                              {&f.camera, &f.means, &f.scales, &f.quats, &f.opacities, &f.colors, &f.bg_color,
                               &f.bg_image, &f.tile_offsets, &f.gaussian_ids, &f.alpha, &f.last_ids, &v_image,
                               &v_alpha, &grads, &densification, &errors, &edges, &edge_scores},
                              tiles_x, tiles_y, kTile / 2, kTile);
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
            mk::launch_items("gsplat_accumulate", accumulate,
                             {&f.camera, &f.means, &f.scales, &f.opacities, &grads, &f.radii, &f.means2d, &means_grad,
                              &scaling_grad, &rotation_grad, &opacity_grad, &sh0_grad, &sh_rest_grad, &densification,
                              &max_screen_share},
                             count);
        }

        void record_vram(const ops::GsplatSaved& saved, const Tensor& image, const Tensor& alpha,
                         const Tensor& gt_tile, const Tensor& bg_tile, const Tensor& error_map) {
            if (!saved.backend)
                return;
            const auto& state = static_cast<const MetalGsplatState&>(*saved.backend);
            constexpr std::string_view scope = "rasterizer.gsplat";
            record_vram_tensor(scope, "forward.gaussian_ids", state.frame.gaussian_ids);
            record_vram_tensor(scope, "forward.tile_offsets", state.frame.tile_offsets);
            record_vram_tensor(scope, "forward.radii", state.frame.radii);
            record_vram_tensor(scope, "forward.means2d", state.frame.means2d);
            record_vram_tensor(scope, "forward.colors", state.frame.colors);
            record_vram_tensor(scope, "forward.last_ids", state.last_ids);
            record_vram_tensor(scope, "output.image", image);
            record_vram_tensor(scope, "output.alpha", alpha);
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
            auto& state = static_cast<MetalGsplatState&>(*saved.backend);
            state.camera = {};
            state.radial = {};
            state.tangential = {};
            state.prism = {};
            state.image = {};
            state.alpha = {};
            state.last_ids = {};
            return true;
        }
    } // namespace

    const lfs::gpu_ops::GsplatRasterOps& metal_gsplat_ops() {
        static const lfs::gpu_ops::GsplatRasterOps ops{
            .create = create,
            .forward = forward,
            .backward = backward,
            .release = release,
            .record_vram = record_vram,
            .release_caches = release_caches,
        };
        return ops;
    }
} // namespace lfs::training
