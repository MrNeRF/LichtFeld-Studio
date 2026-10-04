/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace lfs::nodes {

    inline constexpr std::string_view ANY_SOCKET = "lfs.any";

    using AttributeMap = std::unordered_map<std::string, core::Tensor>;

    enum class Domain { Splat,
                        Point,
                        Vertex };

    struct SplatsComponent {
        core::Tensor means;
        core::Tensor sh0;
        core::Tensor shN;
        core::Tensor scaling;
        core::Tensor rotation;
        core::Tensor opacity;
        int sh_degree = 0;
        float scene_scale = 1.0f;
        AttributeMap attributes;
    };

    struct PointsComponent {
        core::Tensor positions;
        core::Tensor colors;
        AttributeMap attributes;
    };

    struct MeshComponent {
        std::shared_ptr<const core::MeshData> mesh;
        // Albedo RGB tensors, indexed by the mesh's one-based texture handles.
        // Immutable device uploads are shared across derived mesh components.
        std::vector<core::Tensor> textures;
        // Vertex-domain attributes remain available to later nodes and modifiers.
        // MeshData has no generic attribute payload, so Apply intentionally drops them.
        AttributeMap attributes;
    };

    struct Geometry {
        std::optional<SplatsComponent> splats;
        std::optional<PointsComponent> points;
        std::optional<MeshComponent> mesh;

        [[nodiscard]] bool empty() const noexcept {
            return !splats && !points && !mesh;
        }
    };

    inline constexpr std::string_view GEOMETRY_SOCKET = "lfs.geometry";
    inline constexpr std::string_view FLOAT_SOCKET = "lfs.float";
    inline constexpr std::string_view INT_SOCKET = "lfs.int";
    inline constexpr std::string_view BOOL_SOCKET = "lfs.bool";
    inline constexpr std::string_view VECTOR_SOCKET = "lfs.vector";
    inline constexpr std::string_view COLOUR_SOCKET = "lfs.colour";
    inline constexpr std::string_view STRING_SOCKET = "lfs.string";

    struct LFS_CORE_API FieldContext {
        Domain domain = Domain::Splat;
        const SplatsComponent* splats = nullptr;
        const PointsComponent* points = nullptr;
        const MeshComponent* mesh = nullptr;
        std::uint64_t identity = 0;

        [[nodiscard]] std::size_t size() const;
        [[nodiscard]] core::Device device() const;
    };

    // The identity covers every tensor, attribute and scalar a field can read. Tensor handle ids are
    // never reused, so equal identities mean equal contents for as long as a memo lives.
    LFS_CORE_API FieldContext field_context(const SplatsComponent&);
    LFS_CORE_API FieldContext field_context(const PointsComponent&);
    LFS_CORE_API FieldContext field_context(const MeshComponent&);
    // The first of splats, points and mesh.
    LFS_CORE_API std::optional<FieldContext> field_context(const Geometry&);

    class Field;

    struct LFS_CORE_API FieldMemo {
        core::Tensor evaluate(const Field& field, const FieldContext& context);
        void clear();

    private:
        struct Key {
            std::uint64_t node = 0;
            std::uint64_t context = 0;
            bool operator==(const Key&) const = default;
        };
        struct KeyHash {
            std::size_t operator()(const Key& key) const noexcept;
        };
        std::unordered_map<Key, core::Tensor, KeyHash> values_;
    };

    class LFS_CORE_API Field {
    public:
        using EvaluateFn = std::function<core::Tensor(const FieldContext&, FieldMemo&)>;

        Field() = default;
        Field(std::string type_id, EvaluateFn evaluate, bool context_dependent = true);

        [[nodiscard]] bool valid() const noexcept;
        [[nodiscard]] std::string_view type_id() const noexcept;
        [[nodiscard]] bool context_dependent() const noexcept;
        [[nodiscard]] core::Tensor evaluate(const FieldContext& context, FieldMemo& memo) const;
        [[nodiscard]] const void* identity() const noexcept;

    private:
        struct Node;
        std::shared_ptr<const Node> node_;
        friend struct FieldMemo;
    };

    using ValueStorage = std::variant<std::monostate, Geometry, Field, float, std::int64_t, bool, glm::vec3,
                                      glm::vec4, std::string>;

    struct Value {
        ValueStorage data;

        Value() = default;
        template <typename T>
        Value(T value) : data(std::move(value)) {}

        template <typename T>
        [[nodiscard]] const T* get_if() const {
            return std::get_if<T>(&data);
        }
        template <typename T>
        [[nodiscard]] T* get_if() {
            return std::get_if<T>(&data);
        }
    };

    LFS_CORE_API Field constant_field(const Value& value, std::string_view type_id);
    LFS_CORE_API Field convert_field(const Field& field, std::string_view to_type);
    LFS_CORE_API bool can_convert_socket(std::string_view from_type, std::string_view to_type);
    LFS_CORE_API Value convert_value(const Value& value, std::string_view from_type,
                                     std::string_view to_type);

    // Node rotation inputs in degrees, applied X, then Y, then Z (R = Rz * Ry * Rx).
    LFS_CORE_API glm::mat4 rotation_matrix(glm::vec3 degrees);
    LFS_CORE_API Geometry geometry_from_splat_data(const core::SplatData& data);
    LFS_CORE_API std::unique_ptr<core::SplatData> splat_data_from_geometry(const Geometry& geometry);
    LFS_CORE_API Geometry geometry_from_point_cloud(const core::PointCloud& points);
    LFS_CORE_API core::PointCloud point_cloud_from_geometry(const Geometry& geometry);
    LFS_CORE_API Geometry geometry_from_mesh(std::shared_ptr<const core::MeshData> mesh);

    LFS_CORE_API Field position_field();
    LFS_CORE_API Field colour_field();
    LFS_CORE_API Field opacity_field();
    LFS_CORE_API Field scale_field();
    LFS_CORE_API Field index_field();
    LFS_CORE_API Field named_attribute_field(std::string name, std::string type_id);

} // namespace lfs::nodes
