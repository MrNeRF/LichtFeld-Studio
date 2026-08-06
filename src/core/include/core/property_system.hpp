/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lfs::core::prop {

    enum class PropSource : uint8_t { CPP,
                                      PYTHON };

    struct PropertyObjectRef {
        void* ptr = nullptr;
        PropSource source = PropSource::CPP;

        static PropertyObjectRef cpp(void* p) { return {p, PropSource::CPP}; }
        static PropertyObjectRef python(void* p) { return {p, PropSource::PYTHON}; }

        [[nodiscard]] bool is_cpp() const { return source == PropSource::CPP; }
        [[nodiscard]] bool is_python() const { return source == PropSource::PYTHON; }
    };

    enum class PropType {
        Bool,
        Int,
        Float,
        String,
        Enum,
        SizeT,
        // Geometric types for animation
        Vec2,
        Vec3,
        Vec4,
        Quat,
        Mat4,
        Color3,
        Color4,
        // GPU tensor type
        Tensor,
        FloatVector,
        IntVector
    };

    enum class PropUIHint { Default,
                            Slider,
                            Drag,
                            Input,
                            Checkbox,
                            Combo,
                            Hidden };

    enum PropFlags : uint32_t {
        PROP_NONE = 0,
        PROP_READONLY = 1 << 0,
        PROP_LIVE_UPDATE = 1 << 1,
        PROP_NEEDS_RESTART = 1 << 2,
        PROP_ANIMATABLE = 1 << 3,
        PROP_OPERATOR_ARG = 1 << 4,
        PROP_ADVANCED = 1 << 5,
    };

    inline PropFlags operator|(PropFlags a, PropFlags b) {
        return static_cast<PropFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }

    inline PropFlags operator&(PropFlags a, PropFlags b) {
        return static_cast<PropFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    }

    struct EnumItem {
        std::string name;
        std::string identifier;
        int value;
        std::string locale_key;
        std::string wire_value;
    };

    using PropDefault = std::variant<bool, int64_t, double, std::string,
                                     std::array<double, 2>, std::array<double, 3>,
                                     std::array<double, 4>, std::array<double, 16>>;

    struct PropertyMeta {
        std::string id;
        std::string name;
        std::string description;
        std::string group;
        PropType type = PropType::Float;
        PropUIHint ui_hint = PropUIHint::Default;
        uint32_t flags = PROP_NONE;
        PropSource source = PropSource::CPP;

        std::string ui_locale_key;
        std::string ui_tooltip_key;
        std::optional<int> ui_precision;
        std::string json_key;
        bool json_required = false;

        std::optional<int> vector_size;
        std::optional<double> min_value;
        std::optional<double> max_value;
        double soft_min = 0.0;
        double soft_max = 1.0;
        double step = 1.0;
        std::optional<PropDefault> default_value;
        std::vector<EnumItem> enum_items;
        std::vector<std::string> strategies;
        bool strategy_applicability_explicit = false;

        std::function<std::any(const PropertyObjectRef&)> getter;
        std::function<void(PropertyObjectRef&, const std::any&)> setter;

        std::function<void(const PropertyObjectRef&, const std::any&, const std::any&)> on_update;

        [[nodiscard]] bool has_flag(PropFlags f) const { return (flags & f) != PROP_NONE; }
        [[nodiscard]] bool is_readonly() const { return has_flag(PROP_READONLY); }
        [[nodiscard]] bool is_live_update() const { return has_flag(PROP_LIVE_UPDATE); }
        [[nodiscard]] bool needs_restart() const { return has_flag(PROP_NEEDS_RESTART); }
        [[nodiscard]] bool is_animatable() const { return has_flag(PROP_ANIMATABLE); }
        [[nodiscard]] bool is_advanced() const { return has_flag(PROP_ADVANCED); }
    };

    struct PropertyGroup {
        std::string id;
        std::string name;
        std::vector<PropertyMeta> properties;

        [[nodiscard]] const PropertyMeta* find(const std::string& prop_id) const {
            for (const auto& p : properties) {
                if (p.id == prop_id)
                    return &p;
            }
            return nullptr;
        }
    };

    using PropertyCallback = std::function<void(const std::string& group_id,
                                                const std::string& prop_id,
                                                const std::any& old_value,
                                                const std::any& new_value)>;

} // namespace lfs::core::prop
