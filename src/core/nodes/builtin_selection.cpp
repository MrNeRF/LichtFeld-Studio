/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "core/tensor_fused.hpp"
#include "core/tensor_spatial.hpp"
#include <limits>
namespace lfs::nodes::builtin {

    namespace {
        struct PaintSample {
            glm::vec3 position{0.0f};
            float radius = 0.0f;
            float value = 1.0f;
        };

        using PaintStroke = std::vector<PaintSample>;

        std::vector<PaintStroke> paint_strokes(const Node& node) {
            std::vector<PaintStroke> result;
            const auto data = node.properties.value("data", nlohmann::json::array());
            if (!data.is_array())
                return result;
            for (const auto& stroke : data) {
                if (!stroke.is_array())
                    continue;
                PaintStroke samples;
                for (const auto& sample : stroke) {
                    if (!sample.is_array() || sample.size() < 5)
                        continue;
                    try {
                        PaintSample value{
                            .position = {sample[0].get<float>(), sample[1].get<float>(), sample[2].get<float>()},
                            .radius = sample[3].get<float>(),
                            .value = sample[4].get<float>(),
                        };
                        if (std::isfinite(value.position.x) && std::isfinite(value.position.y) &&
                            std::isfinite(value.position.z) && std::isfinite(value.radius) &&
                            std::isfinite(value.value) && value.radius > 0.0f)
                            samples.push_back(value);
                    } catch (const nlohmann::json::exception&) {
                        // A malformed sample is ignored instead of invalidating the graph.
                    }
                }
                if (!samples.empty())
                    result.push_back(std::move(samples));
            }
            return result;
        }

        // Bool mask of the points inside [minimum, maximum].
        Tensor inside_box(const Tensor& points, const glm::vec3& minimum, const glm::vec3& maximum) {
            const auto device = points.device();
            const auto centre = vector_tensor((minimum + maximum) * 0.5f, device);
            const auto half = vector_tensor((maximum - minimum) * 0.5f, device);
            return (half - (points - centre).abs()).min(1).ge(0.0f);
        }

        // Strokes apply in order: painting raises the selection, erasing (value 0) lowers it,
        // so repainting an erased patch selects it again.
        Tensor paint_falloff(const Tensor& positions, const std::vector<PaintStroke>& strokes,
                             const float softness) {
            const std::size_t count = positions.shape()[0];
            auto result = Tensor::zeros({count}, positions.device());
            if (!count || strokes.empty())
                return result;

            glm::vec3 minimum(std::numeric_limits<float>::max());
            glm::vec3 maximum(std::numeric_limits<float>::lowest());
            for (const auto& stroke : strokes)
                for (const auto& sample : stroke) {
                    minimum = glm::min(minimum, sample.position - sample.radius);
                    maximum = glm::max(maximum, sample.position + sample.radius);
                }
            const auto eligible = inside_box(positions, minimum, maximum).nonzero().reshape({-1}).to(DataType::Int32);
            if (!eligible.numel())
                return result;
            const auto candidates = positions.index_select(0, eligible);

            // Each stroke, then each chunk of consecutive samples, is culled to its own bounds, so the
            // work stays near the brush even when strokes are far apart.
            constexpr std::size_t pair_budget = 8 * 1024 * 1024;
            constexpr std::size_t sample_chunk = 64;
            const auto bounds = [](const PaintStroke& stroke, const std::size_t begin, const std::size_t end) {
                std::pair<glm::vec3, glm::vec3> box{glm::vec3(std::numeric_limits<float>::max()),
                                                    glm::vec3(std::numeric_limits<float>::lowest())};
                for (std::size_t index = begin; index < end; ++index) {
                    box.first = glm::min(box.first, stroke[index].position - stroke[index].radius);
                    box.second = glm::max(box.second, stroke[index].position + stroke[index].radius);
                }
                return box;
            };
            const auto rows_inside = [](const Tensor& points, const std::pair<glm::vec3, glm::vec3>& box) {
                return inside_box(points, box.first, box.second).nonzero().reshape({-1}).to(DataType::Int32);
            };
            auto selected = Tensor::zeros({candidates.shape()[0]}, positions.device());
            for (const auto& stroke : strokes) {
                throw_if_evaluation_cancelled();
                const auto stroke_rows = rows_inside(candidates, bounds(stroke, 0, stroke.size()));
                if (!stroke_rows.numel())
                    continue;
                const auto stroke_points = candidates.index_select(0, stroke_rows);
                auto painted = Tensor::zeros({stroke_rows.numel()}, positions.device());
                auto erased = Tensor::zeros_like(painted);
                for (std::size_t sample_begin = 0; sample_begin < stroke.size(); sample_begin += sample_chunk) {
                    throw_if_evaluation_cancelled();
                    const std::size_t sample_end = std::min(sample_begin + sample_chunk, stroke.size());
                    const std::size_t sample_count = sample_end - sample_begin;
                    const auto rows = rows_inside(stroke_points, bounds(stroke, sample_begin, sample_end));
                    const std::size_t row_count = rows.numel();
                    if (!row_count)
                        continue;
                    std::vector<float> centres;
                    std::vector<float> radii;
                    std::vector<float> values;
                    centres.reserve(sample_count * 3);
                    radii.reserve(sample_count);
                    values.reserve(sample_count);
                    for (std::size_t index = sample_begin; index < sample_end; ++index) {
                        const auto& sample = stroke[index];
                        centres.insert(centres.end(), {sample.position.x, sample.position.y, sample.position.z});
                        radii.push_back(sample.radius);
                        values.push_back(std::clamp(sample.value, 0.0f, 1.0f));
                    }
                    const auto points = stroke_points.index_select(0, rows);
                    const auto centre_tensor = Tensor::from_vector(centres, {sample_count, 3}).to(positions.device());
                    const auto radius_tensor = Tensor::from_vector(radii, {1, sample_count}).to(positions.device());
                    const auto value_tensor = Tensor::from_vector(values, {1, sample_count}).to(positions.device());
                    const auto erase_tensor = value_tensor.eq(0).to(DataType::Float32);
                    auto chunk_painted = Tensor::zeros({row_count}, positions.device());
                    auto chunk_erased = Tensor::zeros({row_count}, positions.device());
                    const std::size_t element_tile = std::max<std::size_t>(1, pair_budget / sample_count);
                    for (std::size_t element_begin = 0; element_begin < row_count; element_begin += element_tile) {
                        const std::size_t element_count = std::min(element_tile, row_count - element_begin);
                        const auto tile = points.slice(0, element_begin, element_begin + element_count);
                        const auto delta = tile.unsqueeze(1) - centre_tensor.unsqueeze(0);
                        const auto distance = (delta * delta).sum(2).sqrt() / radius_tensor;
                        const auto weight = softness <= 0.0f
                                                ? distance.le(1.0f).to(DataType::Float32)
                                                : ((distance.neg() + 1.0f) / softness).clamp(0, 1);
                        chunk_painted.slice(0, element_begin, element_begin + element_count)
                            .copy_from((weight * value_tensor).max(1));
                        chunk_erased.slice(0, element_begin, element_begin + element_count)
                            .copy_from((weight * erase_tensor).max(1));
                    }
                    painted.index_copy_(0, rows, painted.index_select(0, rows).maximum(chunk_painted));
                    erased.index_copy_(0, rows, erased.index_select(0, rows).maximum(chunk_erased));
                }
                // Rows outside the stroke have nothing painted or erased, so they keep their value.
                selected.index_copy_(0, stroke_rows,
                                     selected.index_select(0, stroke_rows).maximum(painted).minimum(erased.neg() + 1.0f));
            }
            result.index_add_(0, eligible, selected);
            return result;
        }

        void evaluate_paint_selection(NodeContext& context) {
            const auto strokes = paint_strokes(context.node());
            const float softness = std::clamp(input_float(context, "Softness", 0.5f), 0.0f, 1.0f);
            const bool invert = property_bool(context, "invert", false);
            context.set_output(
                "Selection", operation(FLOAT_SOCKET, {position_field()}, [strokes, softness, invert](const auto& values) {
                    auto weights = paint_falloff(values[0], strokes, softness);
                    return invert ? weights.neg() + 1.0f : weights;
                }));
        }
    } // namespace

    void evaluate_box(NodeContext& context, bool ellipsoid) {
        const auto centre = input_vector(context, "Centre");
        const auto extent = input_vector(context, ellipsoid ? "Radii" : "Size");
        const auto inverse = glm::transpose(glm::mat3(rotation_matrix(input_vector(context, "Rotation"))));
        const float falloff = input_float(context, "Falloff");
        context.set_output(
            "Selection",
            operation(
                FLOAT_SOCKET, {position_field()},
                [centre, extent, inverse, falloff, ellipsoid](const std::vector<Tensor>& values) {
                    const auto device = values[0].device();
                    const auto local =
                        (values[0] - vector_tensor(centre, device)).matmul(matrix_tensor(inverse, device));
                    Tensor distance;
                    if (ellipsoid) {
                        const auto scaled = local / vector_tensor(glm::max(extent, glm::vec3(1e-8f)), device);
                        distance = (scaled * scaled).sum(1).sqrt() - 1;
                    } else {
                        distance = (local.abs() - vector_tensor(extent * 0.5f, device)).max(1);
                    }
                    return falloff <= 0 ? distance.le(0).to(DataType::Float32)
                                        : ((distance.maximum(0) / falloff).neg() + 1).clamp(0, 1);
                }));
    }

    void evaluate_colour_key(NodeContext& context) {
        const auto target = input_vector(context, "Colour");
        const auto tolerance = input_float(context, "Tolerance");
        const auto softness = input_float(context, "Softness");
        context.set_output(
            "Selection",
            operation(FLOAT_SOCKET, {colour_field()},
                      [target, tolerance, softness](const std::vector<Tensor>& values) {
                          const auto difference = values[0].clamp(0, 1) -
                                                  vector_tensor(glm::clamp(target, glm::vec3(0), glm::vec3(1)),
                                                                values[0].device());
                          const auto distance = (difference * difference).sum(1).sqrt();
                          return softness <= 0
                                     ? distance.le(tolerance).to(DataType::Float32)
                                     : ((distance.neg() + tolerance + softness) / softness).clamp(0, 1);
                      }));
    }

    void evaluate_hsv_range(NodeContext& context) {
        const float hue = input_float(context, "Hue");
        const float range = input_float(context, "Hue Range");
        const float hue_softness = input_float(context, "Hue Softness");
        const float saturation_min = input_float(context, "Saturation Min");
        const float saturation_max = input_float(context, "Saturation Max");
        const float value_min = input_float(context, "Value Min");
        const float value_max = input_float(context, "Value Max");
        const float softness = input_float(context, "Softness");
        context.set_output(
            "Selection", operation(FLOAT_SOCKET, {colour_field()}, [=](const std::vector<Tensor>& values) {
                const auto hsv = rgb_to_hsv(values[0]);
                const auto difference = (channel(hsv, 0) - hue).abs();
                const auto distance = difference.minimum(difference.neg() + 1);
                const auto hue_weight =
                    hue_softness <= 0 ? distance.le(range).to(DataType::Float32)
                                      : ((distance.neg() + range + hue_softness) / hue_softness).clamp(0, 1);
                const auto band = [softness](const Tensor& value, float low, float high) {
                    if (softness <= 0)
                        return value.ge(low).logical_and(value.le(high)).to(DataType::Float32);
                    return ((value - low) / softness)
                        .clamp(0, 1)
                        .minimum(((value.neg() + high) / softness).clamp(0, 1));
                };
                return hue_weight * band(channel(hsv, 1), saturation_min, saturation_max) *
                       band(channel(hsv, 2), value_min, value_max);
            }));
    }

    Tensor neighbour_counts(const Tensor& positions, float radius, int32_t max_count, const Tensor* queries) {
        const auto count = positions.shape()[0];
        if (!count || radius <= 0 || max_count <= 0)
            return Tensor::zeros({count}, positions.device(), DataType::Int32);
        return core::radius_neighbor_counts(positions, Tensor::full_bool({count}, true, positions.device()), radius, max_count,
                                            queries);
    }

    Tensor relative_neighbour_counts(const Tensor& positions, const Tensor& activated_scale, float radius_multiple,
                                     int32_t max_count, const Tensor* queries) {
        const auto count = positions.shape()[0];
        if (!count || radius_multiple <= 0 || max_count <= 0)
            return Tensor::zeros({count}, positions.device(), DataType::Int32);
        // A collapsed or nonfinite splat has no usable radius; it counts no neighbours.
        const auto radii = activated_scale.max(1) * radius_multiple;
        const auto usable = radii.isfinite().logical_and(radii.gt(std::numeric_limits<float>::min()));
        return core::radius_neighbor_counts(positions, Tensor::full_bool({count}, true, positions.device()),
                                            Tensor::where(usable, radii, Tensor::zeros_like(radii)), max_count, queries);
    }

    const core::fused::Kernel& ray_parity_kernel() {
        static const auto kernel = [] {
            namespace f = core::fused;
            f::Builder builder(2);
            const auto points = builder.input(DataType::Float32, 2);
            const auto vertices = builder.input(DataType::Float32, 2);
            const auto edge1 = builder.input(DataType::Float32, 2);
            const auto edge2 = builder.input(DataType::Float32, 2);
            const auto point = builder.iota(0);
            const auto triangle = builder.iota(1);
            const auto component = [&](const f::Input& input, int axis) {
                return input.gather({triangle, builder.constant(axis)});
            };
            const auto ax = component(vertices, 0);
            const auto ay = component(vertices, 1);
            const auto az = component(vertices, 2);
            const auto e1x = component(edge1, 0);
            const auto e1y = component(edge1, 1);
            const auto e1z = component(edge1, 2);
            const auto e2x = component(edge2, 0);
            const auto e2y = component(edge2, 1);
            const auto e2z = component(edge2, 2);
            // Non-axis-aligned direction avoids the shared edges of axis-aligned boxes.
            const auto px = 0.37139067f * e2z - 0.52911311f * e2y;
            const auto py = 0.52911311f * e2x - e2z;
            const auto pz = e2y - 0.37139067f * e2x;
            const auto determinant = e1x * px + e1y * py + e1z * pz;
            const auto inverse = 1.0f / determinant;
            const auto tx = points.gather({point, builder.constant(0)}) - ax;
            const auto ty = points.gather({point, builder.constant(1)}) - ay;
            const auto tz = points.gather({point, builder.constant(2)}) - az;
            const auto u = (tx * px + ty * py + tz * pz) * inverse;
            const auto qx = ty * e1z - tz * e1y;
            const auto qy = tz * e1x - tx * e1z;
            const auto qz = tx * e1y - ty * e1x;
            const auto v = (qx + 0.37139067f * qy + 0.52911311f * qz) * inverse;
            const auto distance = (e2x * qx + e2y * qy + e2z * qz) * inverse;
            const auto hit = (f::abs(determinant) > 1e-8f) && (u >= 0.0f) && (u <= 1.0f) && (v >= 0.0f) &&
                             (u + v <= 1.0f) && (distance > 1e-7f);
            builder.output(builder.fold(f::Fold::Count, hit, 1), DataType::Int32);
            return f::Kernel(builder);
        }();
        return kernel;
    }

    Tensor ray_hits(const Tensor& points, const Tensor& a, const Tensor& edge1, const Tensor& edge2) {
        if (points.device() == Device::GPU)
            return ray_parity_kernel()({points.shape()[0], a.shape()[0]}, {points, a, edge1, edge2})[0];
        // CPU tensors use the same broadcast equations through public tensor ops;
        // fused::Kernel requires at least one GPU input.
        const auto e1x = channel(edge1, 0).unsqueeze(0);
        const auto e1y = channel(edge1, 1).unsqueeze(0);
        const auto e1z = channel(edge1, 2).unsqueeze(0);
        const auto e2x = channel(edge2, 0).unsqueeze(0);
        const auto e2y = channel(edge2, 1).unsqueeze(0);
        const auto e2z = channel(edge2, 2).unsqueeze(0);
        const auto px = e2z * 0.37139067f - e2y * 0.52911311f;
        const auto py = e2x * 0.52911311f - e2z;
        const auto pz = e2y - e2x * 0.37139067f;
        const auto determinant = e1x * px + e1y * py + e1z * pz;
        const auto tx = channel(points, 0).unsqueeze(1) - channel(a, 0).unsqueeze(0);
        const auto ty = channel(points, 1).unsqueeze(1) - channel(a, 1).unsqueeze(0);
        const auto tz = channel(points, 2).unsqueeze(1) - channel(a, 2).unsqueeze(0);
        const auto u = (tx * px + ty * py + tz * pz) / determinant;
        const auto qx = ty * e1z - tz * e1y;
        const auto qy = tz * e1x - tx * e1z;
        const auto qz = tx * e1y - ty * e1x;
        const auto v = (qx + qy * 0.37139067f + qz * 0.52911311f) / determinant;
        const auto distance = (e2x * qx + e2y * qy + e2z * qz) / determinant;
        const auto hit = determinant.abs()
                             .gt(1e-8f)
                             .logical_and(u.ge(0))
                             .logical_and(u.le(1))
                             .logical_and(v.ge(0))
                             .logical_and((u + v).le(1))
                             .logical_and(distance.gt(1e-7f));
        return hit.to(DataType::Float32).sum(1).to(DataType::Int32);
    }

    void evaluate_inside_mesh(NodeContext& context) {
        const auto geometry = geometry_input(context, "Mesh");
        if (!geometry.mesh || !geometry.mesh->mesh)
            throw NodeError("Inside Mesh requires mesh geometry");
        const auto mesh = geometry.mesh->mesh;
        context.set_output(
            "Selection",
            operation(BOOL_SOCKET, {position_field()}, [mesh](const std::vector<Tensor>& values) {
                const auto& positions = values[0];
                const auto count = positions.shape()[0];
                const auto device = positions.device();
                auto result = Tensor::zeros({count}, device);
                const auto triangles = mesh->indices.shape()[0];
                if (!triangles || !count)
                    return result.ne(0);
                const auto vertices = mesh->vertices.to(device);
                const auto indices = mesh->indices.to(device).to(DataType::Int32);
                const auto minimum = vertices.min(0);
                const auto maximum = vertices.max(0);
                const auto in_bounds = positions.ge(minimum)
                                           .logical_and(positions.le(maximum))
                                           .to(DataType::Float32)
                                           .sum(1)
                                           .eq(3);
                const auto eligible = in_bounds.nonzero().reshape({-1}).to(DataType::Int32);
                if (!eligible.numel())
                    return result.ne(0);
                const auto candidates = positions.index_select(0, eligible);
                const auto a = vertices.index_select(0, channel(indices, 0));
                const auto edge1 = vertices.index_select(0, channel(indices, 1)) - a;
                const auto edge2 = vertices.index_select(0, channel(indices, 2)) - a;
                auto parity = Tensor::zeros({eligible.numel()}, device, DataType::Int32);
                // Broadcast points against triangles in bounded tiles; fused reduction
                // never materialises the point-by-triangle intermediates.
                constexpr std::size_t pair_budget = 8 * 1024 * 1024;
                for (std::size_t face = 0; face < triangles; face += 4096) {
                    const auto faces = std::min<std::size_t>(4096, triangles - face);
                    const auto points_per_chunk = std::max<std::size_t>(1, pair_budget / faces);
                    for (std::size_t point = 0; point < eligible.numel(); point += points_per_chunk) {
                        const auto points = std::min(points_per_chunk, eligible.numel() - point);
                        auto hits = ray_hits(
                            candidates.slice(0, point, point + points), a.slice(0, face, face + faces),
                            edge1.slice(0, face, face + faces), edge2.slice(0, face, face + faces));
                        auto destination = parity.slice(0, point, point + points);
                        destination.copy_from(destination + hits);
                    }
                }
                const auto hits = parity.to(DataType::Float32);
                result.index_add_(0, eligible, hits - (hits / 2).floor() * 2);
                return result.ne(0);
            }));
    }

    void evaluate_neighbour_count(NodeContext& context) {
        const auto radius = input_float(context, "Radius", 1);
        const bool relative = property_bool(context, "relative_to_size", false);
        context.set_output("Count", Field(std::string(INT_SOCKET), [radius, relative](const FieldContext& domain, FieldMemo& memo) {
                               constexpr int32_t limit = std::numeric_limits<int32_t>::max();
                               const auto positions = memo.evaluate(position_field(), domain);
                               // Points and meshes have no size, so they always count in scene units.
                               if (relative && domain.domain == Domain::Splat)
                                   return relative_neighbour_counts(positions, memo.evaluate(scale_field(), domain), radius, limit)
                                       .to(DataType::Int32);
                               return neighbour_counts(positions, radius, limit).to(DataType::Int32);
                           }));
    }
    void register_selection(NodeTypeRegistry& registry) {
        const auto geo = std::string(GEOMETRY_SOCKET);
        const auto f = std::string(FLOAT_SOCKET);
        const auto i = std::string(INT_SOCKET);
        const auto b = std::string(BOOL_SOCKET);
        const auto v = std::string(VECTOR_SOCKET);
        const auto c = std::string(COLOUR_SOCKET);
        register_type(registry, type("lfs.box_selection", "Selection",
                                     {in("Centre", v, glm::vec3(0)).step_size(0.01),
                                      in("Size", v, glm::vec3(1)).minimum(0).step_size(0.01),
                                      in("Rotation", v, glm::vec3(0)).step_size(1),
                                      in("Falloff", f, 0.0f).minimum(0).step_size(0.01)},
                                     {out("Selection", f)}, [](NodeContext& x) {
                                         evaluate_box(x, false);
                                     }));
        register_type(registry, type("lfs.ellipsoid_selection", "Selection",
                                     {in("Centre", v, glm::vec3(0)).step_size(0.01),
                                      in("Radii", v, glm::vec3(1)).minimum(0).step_size(0.01),
                                      in("Rotation", v, glm::vec3(0)).step_size(1),
                                      in("Falloff", f, 0.0f).minimum(0).step_size(0.01)},
                                     {out("Selection", f)}, [](NodeContext& x) {
                                         evaluate_box(x, true);
                                     }));
        register_type(registry,
                      type("lfs.colour_key", "Selection",
                           {in("Colour", c, glm::vec3(0)).step_size(0.01),
                            in("Tolerance", f, 0.1f).minimum(0).step_size(0.01),
                            in("Softness", f, 0.0f).minimum(0).step_size(0.01)},
                           {out("Selection", f)}, evaluate_colour_key));
        register_type(registry,
                      type("lfs.hsv_range", "Selection",
                           {in("Hue", f, 0.0f).range(0, 1).step_size(0.01),
                            in("Hue Range", f, 0.1f).range(0, 1).step_size(0.01),
                            in("Hue Softness", f, 0.0f).minimum(0).step_size(0.01),
                            in("Saturation Min", f, 0.0f).range(0, 1).step_size(0.01),
                            in("Saturation Max", f, 1.0f).range(0, 1).step_size(0.01),
                            in("Value Min", f, 0.0f).range(0, 1).step_size(0.01),
                            in("Value Max", f, 1.0f).range(0, 1).step_size(0.01),
                            in("Softness", f, 0.0f).minimum(0).step_size(0.01)},
                           {out("Selection", f)}, evaluate_hsv_range));
        register_type(registry,
                      type("lfs.paint_selection", "Selection",
                           {in("Softness", f, 0.5f).range(0, 1).step_size(0.01)},
                           {out("Selection", f)}, evaluate_paint_selection,
                           {prop("data", PropertyKind::Data, nlohmann::json::array()),
                            prop("invert", PropertyKind::Bool, false)}));
        register_type(registry, type("lfs.inside_mesh", "Selection",
                                     {in("Mesh", geo)}, {out("Selection", b)}, evaluate_inside_mesh));
        register_type(registry, type("lfs.neighbour_count", "Selection",
                                     {in("Radius", f, 1.0f).minimum(0).step_size(0.01)},
                                     {out("Count", i)}, evaluate_neighbour_count,
                                     {prop("relative_to_size", PropertyKind::Bool, false)}));
    }

} // namespace lfs::nodes::builtin
