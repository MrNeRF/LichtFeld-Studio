/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/splat_tile_source.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data_transform.hpp"
#include <algorithm>

namespace lfs::io {

    namespace {
        struct Traversal {
            const SplatTileSource& source;
            const SplatTileView& view;
            const std::function<bool(std::uint32_t)>& resident;
            std::vector<std::pair<float, std::uint32_t>> wanted;

            bool visible(const SplatTile& tile) const {
                const float r = tile.radius();
                return std::ranges::all_of(view.planes, [&](const glm::vec4& p) {
                    return glm::dot(glm::vec3(p), tile.center) + p.w >= -r;
                });
            }

            // Appends the tiles covering `index` to `render`; true when fully covered.
            bool visit(const std::uint32_t index, std::vector<std::uint32_t>& render) {
                const auto& tile = source.tiles()[index];
                if (!visible(tile))
                    return true;
                const float sse = view.orthographic
                                      ? tile.geometric_error * view.sse_per_error
                                      : tile.geometric_error * view.sse_per_error /
                                            std::max(tile.distance(view.camera), 1e-6f);
                const bool has = tile.splat_count > 0;
                const bool ready = has && resident(index);
                if (has)
                    wanted.emplace_back(sse, index);
                const bool refine = tile.child_count > 0 && (sse > view.max_sse || !has);
                if (!refine) {
                    if (ready)
                        render.push_back(index);
                    return !has || ready;
                }
                std::vector<std::uint32_t> children;
                bool covered = true;
                for (std::uint32_t c = 0; c < tile.child_count; ++c)
                    covered = visit(tile.first_child + c, children) && covered;
                if (tile.additive) {
                    // ADD: children only add detail on top of the drawn parent, so a missing
                    // (or failed) child leaves no hole; show the ones already loaded. Without
                    // parent content the children alone must cover the region.
                    if (ready)
                        render.push_back(index);
                    render.insert(render.end(), children.begin(), children.end());
                    return has ? ready : covered;
                }
                if (covered || !ready) {
                    render.insert(render.end(), children.begin(), children.end());
                    return covered;
                }
                render.push_back(index); // children still loading: keep the parent
                return true;
            }
        };
    } // namespace

    SplatTileSelection select_splat_tiles(const SplatTileSource& source, const SplatTileView& view,
                                          const std::function<bool(std::uint32_t)>& resident) {
        SplatTileSelection out;
        if (source.tiles().empty())
            return out;
        Traversal traversal{source, view, resident, {}};
        out.complete = traversal.visit(0, out.render);
        for (const auto index : out.render)
            out.render_splats += source.tiles()[index].splat_count;
        std::ranges::sort(traversal.wanted, std::greater{});
        for (const auto& [_, index] : traversal.wanted)
            out.wanted.push_back(index);
        return out;
    }

    lfs::Result<core::SplatData> load_splat_tile_gpu(const SplatTileSource& source,
                                                     const std::uint32_t tile) {
        auto data = source.load_tile(tile);
        if (!data)
            return data;
        using core::Device;
        data->means_raw() = data->means_raw().to(Device::GPU);
        data->sh0_raw() = data->sh0_raw().to(Device::GPU);
        if (data->shN_raw().is_valid() && data->shN_raw().numel() > 0)
            data->shN_raw() = data->shN_raw().to(Device::GPU);
        data->scaling_raw() = data->scaling_raw().to(Device::GPU);
        data->rotation_raw() = data->rotation_raw().to(Device::GPU);
        data->opacity_raw() = data->opacity_raw().to(Device::GPU);
        // Bake the tile's placement into the splats once, here, so repeated cut
        // rebuilds can concatenate the cached tiles on the fast identity path
        // instead of re-transforming and re-cloning every tile each time.
        if (const auto& placement = source.tiles()[tile].transform; placement != glm::mat4(1.0f))
            core::transform(*data, placement);
        return data;
    }

    std::unique_ptr<core::SplatData> merge_splat_tiles(
        const SplatTileSource& /*source*/, const std::span<const std::uint32_t> tiles,
        const std::function<const core::SplatData*(std::uint32_t)>& splats,
        const core::SplatTensorAllocator& allocator) {
        std::vector<const core::SplatData*> pieces;
        std::size_t total = 0;
        int max_sh = 0;
        int max_active_sh = 0;
        float total_scale = 0.0f;
        for (const auto index : tiles) {
            const auto* data = splats(index);
            if (!data || data->size() == 0)
                continue;
            pieces.push_back(data);
            total += static_cast<std::size_t>(data->size());
            if (data->shN_raw().is_valid() && data->shN_raw().numel() > 0 && data->max_sh_coeffs_rest() > 0)
                max_sh = std::max(max_sh, data->get_max_sh_degree());
            max_active_sh = std::max(max_active_sh, data->get_active_sh_degree());
            total_scale += data->get_scene_scale();
        }
        if (pieces.empty())
            return nullptr;

        // Placement is already baked in by load_splat_tile_gpu and tiles carry no
        // deletion mask, so each tile is slice-copied straight into the destination:
        // one allocation per attribute and no per-tile staging clone.
        using core::DataType;
        using core::Tensor;
        using core::TensorShape;
        const auto device = pieces.front()->means_raw().device();
        const auto alloc = [&](TensorShape shape, const std::size_t rows, const std::string_view name) {
            return allocator ? allocator(std::move(shape), rows, DataType::Float32, name)
                             : Tensor::empty(std::move(shape), device);
        };
        Tensor means = alloc(TensorShape({total, 3}), total, "SplatData.means");
        Tensor sh0 = alloc(TensorShape({total, 1, 3}), total, "SplatData.sh0");
        Tensor scaling = alloc(TensorShape({total, 3}), total, "SplatData.scaling");
        Tensor rotation = alloc(TensorShape({total, 4}), total, "SplatData.rotation");
        Tensor opacity = alloc(TensorShape({total, 1}), total, "SplatData.opacity");
        const auto rest = core::sh_rest_coefficients_for_degree(max_sh);
        const std::size_t shN_floats = core::sh_swizzled_float_count(total, rest);
        Tensor shN;
        if (shN_floats > 0) {
            // Quantized renderer storage is encoded from a float workspace afterwards.
            if (allocator && !core::sh_value_quant::enabled()) {
                shN = allocator(TensorShape({shN_floats}), shN_floats, DataType::Float32, "SplatData.shN");
                shN.zero_();
            } else {
                shN = Tensor::zeros_direct(TensorShape({shN_floats}), shN_floats, core::Device::GPU);
            }
        } else {
            shN = Tensor::zeros({0}, core::Device::GPU);
        }

        std::size_t offset = 0;
        for (const auto* piece : pieces) {
            const auto rows = static_cast<std::size_t>(piece->size());
            means.slice(0, offset, offset + rows).copy_from(piece->means_raw());
            sh0.slice(0, offset, offset + rows).copy_from(piece->sh0_raw());
            scaling.slice(0, offset, offset + rows).copy_from(piece->scaling_raw());
            rotation.slice(0, offset, offset + rows).copy_from(piece->rotation_raw());
            opacity.slice(0, offset, offset + rows).copy_from(piece->opacity_raw());
            if (rest > 0 && piece->shN_raw().is_valid() && piece->shN_raw().numel() > 0 &&
                piece->max_sh_coeffs_rest() > 0)
                core::copy_sh_coefficients(*piece, shN, offset, rest);
            offset += rows;
        }

        auto merged = std::make_unique<core::SplatData>(max_sh, std::move(means), std::move(sh0), std::move(shN),
                                                        std::move(scaling), std::move(rotation), std::move(opacity),
                                                        total_scale / static_cast<float>(pieces.size()),
                                                        core::SplatData::ShNLayout::Swizzled);
        merged->set_active_sh_degree(max_active_sh);
        if (allocator) {
            merged->set_tensor_allocator(allocator);
            if (core::sh_value_quant::enabled())
                (void)merged->apply_shN_value_quant();
        }
        return merged;
    }

} // namespace lfs::io
