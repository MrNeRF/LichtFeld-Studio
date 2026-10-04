/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nodes/types.hpp"

#include "core/assert.hpp"
#include "core/tensor_backend.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

namespace lfs::nodes {
    namespace {

        constexpr float kShC0 = 0.28209479177387814f;

        core::Tensor context_position(const FieldContext& context) {
            switch (context.domain) {
            case Domain::Splat:
                if (context.splats)
                    return context.splats->means;
                break;
            case Domain::Point:
                if (context.points)
                    return context.points->positions;
                break;
            case Domain::Vertex:
                if (context.mesh && context.mesh->mesh)
                    return context.mesh->mesh->vertices;
                break;
            }
            throw std::runtime_error("Position field has no component for its domain");
        }

    } // namespace

    std::size_t FieldContext::size() const {
        switch (domain) {
        case Domain::Splat: return splats && splats->means.is_valid() ? splats->means.shape()[0] : 0;
        case Domain::Point: return points && points->positions.is_valid() ? points->positions.shape()[0] : 0;
        case Domain::Vertex:
            return mesh && mesh->mesh ? static_cast<std::size_t>(mesh->mesh->vertex_count()) : 0;
        }
        return 0;
    }

    core::Device FieldContext::device() const {
        const core::Tensor positions = context_position(*this);
        return positions.is_valid() ? positions.device() : core::Device::CPU;
    }

    struct Field::Node {
        std::uint64_t id = 0;
        std::string type_id;
        EvaluateFn evaluate;
        bool context_dependent = true;
    };

    Field::Field(std::string type_id, EvaluateFn evaluate, bool context_dependent)
        : node_([&] {
              static std::atomic<std::uint64_t> next_id{1};
              return std::make_shared<Node>(Node{next_id.fetch_add(1, std::memory_order_relaxed),
                                                 std::move(type_id), std::move(evaluate), context_dependent});
          }()) {}

    bool Field::valid() const noexcept {
        return static_cast<bool>(node_);
    }
    std::string_view Field::type_id() const noexcept {
        return node_ ? node_->type_id : std::string_view{};
    }
    bool Field::context_dependent() const noexcept {
        return node_ && node_->context_dependent;
    }
    const void* Field::identity() const noexcept {
        return node_.get();
    }

    core::Tensor Field::evaluate(const FieldContext& context, FieldMemo& memo) const {
        if (!node_)
            throw std::runtime_error("Cannot evaluate an empty field");
        return memo.evaluate(*this, context);
    }

    std::size_t FieldMemo::KeyHash::operator()(const Key& key) const noexcept {
        const auto a = std::hash<std::uint64_t>{}(key.node);
        const auto b = std::hash<std::uint64_t>{}(key.context);
        return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6U) + (a >> 2U));
    }

    core::Tensor FieldMemo::evaluate(const Field& field, const FieldContext& context) {
        const Key key{field.node_->id, context.identity};
        if (const auto found = values_.find(key); found != values_.end())
            return found->second;
        std::optional<core::GpuBackendScope> backend_scope;
        if (const auto backend = core::gpu_backend_of(context_position(context)))
            backend_scope.emplace(*backend);
        core::Tensor value = field.node_->evaluate(context, *this);
        values_.emplace(key, value);
        return value;
    }

    void FieldMemo::clear() {
        values_.clear();
    }

    Field constant_field(const Value& value, std::string_view type_id) {
        const std::string type(type_id);
        return Field(
            type,
            [value, type](const FieldContext& context, FieldMemo&) {
                const auto n = context.size();
                const auto device = context.device();
                if (type == FLOAT_SOCKET) {
                    float scalar = 0.0f;
                    if (const auto* v = value.get_if<float>())
                        scalar = *v;
                    else if (const auto* v = value.get_if<std::int64_t>())
                        scalar = static_cast<float>(*v);
                    else if (const auto* v = value.get_if<bool>())
                        scalar = *v ? 1.0f : 0.0f;
                    return core::Tensor::full({n}, scalar, device);
                }
                if (type == INT_SOCKET) {
                    int scalar = 0;
                    if (const auto* v = value.get_if<std::int64_t>())
                        scalar = static_cast<int>(*v);
                    else if (const auto* v = value.get_if<float>())
                        scalar = static_cast<int>(*v);
                    else if (const auto* v = value.get_if<bool>())
                        scalar = *v ? 1 : 0;
                    return core::Tensor::full({n}, static_cast<float>(scalar), device, core::DataType::Int32);
                }
                if (type == BOOL_SOCKET) {
                    bool scalar = false;
                    if (const auto* v = value.get_if<bool>())
                        scalar = *v;
                    else if (const auto* v = value.get_if<float>())
                        scalar = *v != 0.0f;
                    else if (const auto* v = value.get_if<std::int64_t>())
                        scalar = *v != 0;
                    return core::Tensor::full_bool({n}, scalar, device);
                }
                if (type == VECTOR_SOCKET || type == COLOUR_SOCKET) {
                    glm::vec3 vector(0.0f);
                    if (const auto* v = value.get_if<glm::vec3>())
                        vector = *v;
                    else if (const auto* v = value.get_if<glm::vec4>())
                        vector = glm::vec3(*v);
                    else if (const auto* v = value.get_if<float>())
                        vector = glm::vec3(*v);
                    return core::Tensor::stack({core::Tensor::full({n}, vector.x, device),
                                                core::Tensor::full({n}, vector.y, device),
                                                core::Tensor::full({n}, vector.z, device)},
                                               1);
                }
                throw std::runtime_error(std::format("Socket type '{}' cannot be a field", type));
            },
            false);
    }

    bool can_convert_socket(std::string_view from, std::string_view to) {
        if (from == ANY_SOCKET || to == ANY_SOCKET)
            return true;
        if (from == to)
            return true;
        if (from == GEOMETRY_SOCKET || to == GEOMETRY_SOCKET || from == STRING_SOCKET || to == STRING_SOCKET)
            return false;
        if ((from == VECTOR_SOCKET || from == COLOUR_SOCKET) &&
            (to == VECTOR_SOCKET || to == COLOUR_SOCKET || to == FLOAT_SOCKET))
            return true;
        if ((to == VECTOR_SOCKET || to == COLOUR_SOCKET) && from == FLOAT_SOCKET)
            return true;
        if ((from == FLOAT_SOCKET && (to == INT_SOCKET || to == BOOL_SOCKET)) ||
            (from == INT_SOCKET && (to == FLOAT_SOCKET || to == BOOL_SOCKET)) ||
            (from == BOOL_SOCKET && to == FLOAT_SOCKET))
            return true;
        return false;
    }

    Field convert_field(const Field& field, std::string_view to_type) {
        if (field.type_id() == to_type)
            return field;
        if (!can_convert_socket(field.type_id(), to_type))
            throw std::runtime_error(
                std::format("Cannot convert field from '{}' to '{}'", field.type_id(), to_type));
        const std::string output_type(to_type);
        return Field(
            output_type,
            [field, output_type](const FieldContext& context, FieldMemo& memo) {
                core::Tensor input = field.evaluate(context, memo);
                if (output_type == BOOL_SOCKET)
                    return input.ne(0.0f);
                if (output_type == INT_SOCKET)
                    return input.to(core::DataType::Int32);
                if (output_type == FLOAT_SOCKET) {
                    input = input.to(core::DataType::Float32);
                    return input.ndim() == 2 ? input.mean(1, false) : input;
                }
                if (output_type == VECTOR_SOCKET || output_type == COLOUR_SOCKET) {
                    input = input.to(core::DataType::Float32);
                    if (input.ndim() == 2)
                        return input;
                    return core::Tensor::stack({input, input, input}, 1);
                }
                return input;
            },
            field.context_dependent());
    }

    Value convert_value(const Value& value, std::string_view from_type, std::string_view to_type) {
        if (from_type == to_type)
            return value;
        if (const auto* field = value.get_if<Field>())
            return convert_field(*field, to_type);
        return convert_field(constant_field(value, from_type), to_type);
    }

    Geometry geometry_from_splat_data(const core::SplatData& data) {
        SplatsComponent component;
        core::Tensor keep;
        const bool filtered = data.has_deleted_mask() && data.deleted().numel() == data.size();
        if (filtered)
            keep = data.deleted().logical_not();
        const auto select = [&](const core::Tensor& tensor) {
            if (!filtered)
                return tensor;
            const core::Tensor indices = keep.device() == tensor.device() ? keep : keep.to(tensor.device());
            return tensor.index_select(0, indices).contiguous();
        };
        component.means = select(data.means_raw());
        core::Tensor sh0 = data.sh0_raw();
        if (sh0.ndim() == 3)
            sh0 = sh0.squeeze(1);
        component.sh0 = select(sh0);
        core::Tensor canonical = data.shN_canonical();
        const std::size_t active_coefficients = static_cast<std::size_t>(
            (data.get_active_sh_degree() + 1) * (data.get_active_sh_degree() + 1) - 1);
        if (canonical.is_valid() && canonical.ndim() == 3 && canonical.shape()[1] > active_coefficients)
            canonical = canonical.slice(1, 0, active_coefficients).contiguous();
        if (!canonical.is_valid())
            canonical =
                core::Tensor::zeros({static_cast<std::size_t>(data.size()), 0, 3}, data.means_raw().device());
        component.shN = select(canonical);
        component.scaling = select(data.scaling_raw());
        component.rotation = select(data.rotation_raw());
        core::Tensor opacity = data.opacity_raw();
        if (opacity.ndim() == 2)
            opacity = opacity.squeeze(1);
        component.opacity = select(opacity);
        component.sh_degree = data.get_active_sh_degree();
        component.scene_scale = data.get_scene_scale();
        return Geometry{std::move(component), std::nullopt, std::nullopt};
    }

    std::unique_ptr<core::SplatData> splat_data_from_geometry(const Geometry& geometry) {
        if (!geometry.splats)
            return {};
        const auto& s = *geometry.splats;
        core::Tensor sh0 = s.sh0.ndim() == 2 ? s.sh0.unsqueeze(1) : s.sh0;
        core::Tensor opacity = s.opacity.ndim() == 1 ? s.opacity.unsqueeze(1) : s.opacity;
        auto result =
            std::make_unique<core::SplatData>(s.sh_degree, s.means, sh0, s.shN, s.scaling, s.rotation,
                                              opacity, s.scene_scale, core::SplatData::ShNLayout::Canonical);
        result->shN_set_from_canonical(s.shN);
        result->set_active_sh_degree(s.sh_degree);
        return result;
    }

    Geometry geometry_from_point_cloud(const core::PointCloud& points) {
        core::Tensor colors = points.colors;
        if (colors.is_valid() && colors.dtype() == core::DataType::UInt8)
            colors = colors.to(core::DataType::Float32) / 255.0f;
        return Geometry{std::nullopt, PointsComponent{points.means, colors, {}}, std::nullopt};
    }

    core::PointCloud point_cloud_from_geometry(const Geometry& geometry) {
        if (!geometry.points)
            return {};
        return core::PointCloud(geometry.points->positions, geometry.points->colors);
    }

    Geometry geometry_from_mesh(std::shared_ptr<const core::MeshData> mesh) {
        return Geometry{std::nullopt, std::nullopt, MeshComponent{std::move(mesh)}};
    }

    Field position_field() {
        return Field(std::string(VECTOR_SOCKET), [](const FieldContext& context, FieldMemo&) {
            return context_position(context);
        });
    }

    Field colour_field() {
        return Field(std::string(COLOUR_SOCKET), [](const FieldContext& context, FieldMemo&) {
            if (context.domain == Domain::Splat && context.splats)
                return context.splats->sh0 * kShC0 + 0.5f;
            if (context.domain == Domain::Point && context.points)
                return context.points->colors;
            if (context.domain == Domain::Vertex && context.mesh && context.mesh->mesh) {
                const auto& mesh = *context.mesh->mesh;
                if (mesh.has_colors())
                    return mesh.colors.slice(1, 0, 3);
                return core::Tensor::ones({context.size(), 3}, context.device());
            }
            throw std::runtime_error("Colour field has no component for its domain");
        });
    }

    Field opacity_field() {
        return Field(std::string(FLOAT_SOCKET), [](const FieldContext& context, FieldMemo&) {
            if (context.domain == Domain::Splat && context.splats)
                return context.splats->opacity.sigmoid();
            if (context.domain == Domain::Vertex && context.mesh && context.mesh->mesh &&
                context.mesh->mesh->has_colors())
                return context.mesh->mesh->colors.slice(1, 3, 4).squeeze(1);
            return core::Tensor::ones({context.size()}, context.device());
        });
    }

    Field scale_field() {
        return Field(std::string(VECTOR_SOCKET), [](const FieldContext& context, FieldMemo&) {
            if (context.domain == Domain::Splat && context.splats)
                return context.splats->scaling.exp();
            return core::Tensor::zeros({context.size(), 3}, context.device());
        });
    }

    Field index_field() {
        return Field(std::string(INT_SOCKET), [](const FieldContext& context, FieldMemo&) {
            return (core::Tensor::ones({context.size()}, context.device(), core::DataType::Int32).cumsum(0) -
                    1)
                .to(core::DataType::Int32);
        });
    }

    Field named_attribute_field(std::string name, std::string type_id) {
        return Field(std::move(type_id), [name = std::move(name)](const FieldContext& context, FieldMemo&) {
            const AttributeMap* attributes = nullptr;
            if (context.domain == Domain::Splat && context.splats)
                attributes = &context.splats->attributes;
            else if (context.domain == Domain::Point && context.points)
                attributes = &context.points->attributes;
            else if (context.domain == Domain::Vertex && context.mesh)
                attributes = &context.mesh->attributes;
            if (!attributes)
                throw std::runtime_error(
                    std::format("Named attribute '{}' is unavailable on this domain", name));
            const auto found = attributes->find(name);
            if (found == attributes->end())
                throw std::runtime_error(std::format("Named attribute '{}' does not exist", name));
            return found->second;
        });
    }

} // namespace lfs::nodes
