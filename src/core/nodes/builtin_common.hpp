/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/nodes/builtin.hpp"
#include "core/nodes/evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::nodes::builtin {
    using core::DataType;
    using core::Device;
    using core::Tensor;
    inline constexpr float kShC0 = 0.28209479177387814f;

    std::string property_string(const NodeContext&, std::string_view, std::string fallback = {});
    bool property_bool(const NodeContext&, std::string_view, bool fallback = false);
    int property_int(const NodeContext&, std::string_view, int fallback = 0);
    float input_float(const NodeContext&, std::string_view, float fallback = 0);
    int input_int(const NodeContext&, std::string_view, int fallback = 0);
    glm::vec3 input_vector(const NodeContext&, std::string_view);
    Geometry geometry_input(const NodeContext&, std::string_view socket = "Geometry");
    Tensor vector_tensor(glm::vec3, Device);
    Tensor matrix_tensor(const glm::mat3&, Device);
    using lfs::nodes::field_context;
    using lfs::nodes::rotation_matrix;
    Tensor selection(const NodeContext&, std::string_view, const FieldContext&, bool structural = false);
    Tensor blend(const Tensor&, const Tensor&, Tensor weight);
    Tensor safe_divide(const Tensor&, const Tensor&);
    Tensor channel(const Tensor&, int);
    Tensor rgb_to_hsv(const Tensor&);
    Tensor hsv_to_rgb(const Tensor&);
    Field operation(std::string_view type, std::vector<Field> inputs,
                    std::function<Tensor(const std::vector<Tensor>&)> evaluate);
    SplatsComponent filter_splats(const SplatsComponent&, const Tensor&);
    PointsComponent filter_points(const PointsComponent&, const Tensor&);
    std::shared_ptr<core::MeshData> copy_mesh(const core::MeshData&, Tensor vertices, Tensor indices);
    std::shared_ptr<core::MeshData> filter_mesh_faces(const core::MeshData&, const Tensor&, bool);
    // An optional Bool [N] query mask leaves the other counts at zero.
    Tensor neighbour_counts(const Tensor&, float, int32_t max_count, const Tensor* queries = nullptr);
    Tensor relative_neighbour_counts(const Tensor&, const Tensor& activated_scale, float radius_multiple, int32_t max_count,
                                     const Tensor* queries = nullptr);
    SocketDecl in(std::string, std::string, Value value = {}, bool field = false, bool multi = false);
    SocketDecl out(std::string, std::string);
    PropertyDecl prop(std::string, PropertyKind, nlohmann::json, std::vector<std::string> items = {});
    std::vector<SocketDecl> geometry_inputs(std::vector<SocketDecl>);
    NodeTypeInfo type(std::string, std::string, std::vector<SocketDecl>,
                      std::vector<SocketDecl>, std::function<void(NodeContext&)>,
                      std::vector<PropertyDecl> properties = {});
    void register_type(NodeTypeRegistry&, NodeTypeInfo);
    void register_input(NodeTypeRegistry&);
    void register_utilities(NodeTypeRegistry&);
    void register_selection(NodeTypeRegistry&);
    void register_geometry(NodeTypeRegistry&);
    void register_splat(NodeTypeRegistry&);
    void register_cleanup(NodeTypeRegistry&);
    void register_conversion(NodeTypeRegistry&);
    void register_colour(NodeTypeRegistry&);
    void register_texture(NodeTypeRegistry&);
    void register_instances(NodeTypeRegistry&);
} // namespace lfs::nodes::builtin
