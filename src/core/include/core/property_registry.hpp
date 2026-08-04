/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include "animatable_property.hpp"
#include "property_system.hpp"

#include <cassert>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <mutex>
#include <unordered_map>

namespace lfs::core::prop {

    struct PropertyKey {
        std::string group_id;
        std::string prop_id;

        bool operator==(const PropertyKey&) const = default;
    };

    struct PropertyKeyHash {
        size_t operator()(const PropertyKey& k) const {
            size_t h1 = std::hash<std::string>{}(k.group_id);
            size_t h2 = std::hash<std::string>{}(k.prop_id);
            return h1 ^ (h2 << 1);
        }
    };

    class LFS_CORE_API PropertyRegistry {
    public:
        static PropertyRegistry& instance();

        void register_group(PropertyGroup group);
        void unregister_group(const std::string& group_id);
        [[nodiscard]] const PropertyGroup* get_group(const std::string& group_id) const;
        [[nodiscard]] std::optional<PropertyGroup> get_group_snapshot(const std::string& group_id) const;
        [[nodiscard]] std::optional<PropertyMeta> get_property(const std::string& group_id,
                                                               const std::string& prop_id) const;
        [[nodiscard]] std::vector<std::string> get_group_ids() const;

        void register_operator_args(const std::string& operator_id, std::vector<PropertyMeta> args);
        void unregister_operator_args(const std::string& operator_id);

        size_t subscribe(PropertyCallback callback);
        size_t subscribe(const std::string& group_id, const std::string& prop_id, PropertyCallback callback);
        void unsubscribe(size_t id);
        void notify(const std::string& group_id, const std::string& prop_id,
                    const std::any& old_value, const std::any& new_value);

    private:
        PropertyRegistry() = default;

        mutable std::mutex mutex_;
        std::unordered_map<std::string, PropertyGroup> groups_;
        std::unordered_map<size_t, PropertyCallback> global_subscribers_;
        std::unordered_map<PropertyKey, std::unordered_map<size_t, PropertyCallback>, PropertyKeyHash>
            prop_subscribers_;
        size_t next_id_ = 1;
    };

    template <typename StructT>
    class PropertyGroupBuilder {
    public:
        PropertyGroupBuilder(const std::string& group_id, const std::string& group_name)
            : group_{.id = group_id, .name = group_name} {}

        PropertyGroupBuilder& float_prop(float StructT::*member,
                                         const std::string& id,
                                         const std::string& name,
                                         float default_val,
                                         float min_val,
                                         float max_val,
                                         const std::string& desc = "",
                                         PropUIHint hint = PropUIHint::Slider) {
            add_prop(
                member, id, name, PropType::Float, desc, hint,
                [](float v) { return v; },
                [](const std::any& v) { return std::any_cast<float>(v); });
            auto& meta = group_.properties.back();
            meta.default_value = static_cast<double>(default_val);
            meta.min_value = min_val;
            meta.max_value = max_val;
            meta.soft_min = min_val;
            meta.soft_max = max_val;
            meta.step = (max_val - min_val) / 100.0;
            return *this;
        }

        PropertyGroupBuilder& int_prop(int StructT::*member,
                                       const std::string& id,
                                       const std::string& name,
                                       int default_val,
                                       int min_val,
                                       int max_val,
                                       const std::string& desc = "",
                                       PropUIHint hint = PropUIHint::Slider) {
            add_prop(
                member, id, name, PropType::Int, desc, hint,
                [](int v) { return v; },
                [](const std::any& v) { return std::any_cast<int>(v); });
            auto& meta = group_.properties.back();
            meta.default_value = static_cast<int64_t>(default_val);
            meta.min_value = min_val;
            meta.max_value = max_val;
            meta.soft_min = min_val;
            meta.soft_max = max_val;
            meta.step = 1.0;
            return *this;
        }

        PropertyGroupBuilder& size_prop(size_t StructT::*member,
                                        const std::string& id,
                                        const std::string& name,
                                        size_t default_val,
                                        size_t min_val,
                                        size_t max_val,
                                        const std::string& desc = "",
                                        PropUIHint hint = PropUIHint::Input) {
            add_prop(
                member, id, name, PropType::SizeT, desc, hint,
                [](size_t v) { return v; },
                [](const std::any& v) { return std::any_cast<size_t>(v); });
            auto& meta = group_.properties.back();
            meta.default_value = static_cast<int64_t>(default_val);
            meta.min_value = static_cast<double>(min_val);
            meta.max_value = static_cast<double>(max_val);
            meta.soft_min = static_cast<double>(min_val);
            meta.soft_max = static_cast<double>(max_val);
            meta.step = 1.0;
            return *this;
        }

        PropertyGroupBuilder& bool_prop(bool StructT::*member,
                                        const std::string& id,
                                        const std::string& name,
                                        bool default_val,
                                        const std::string& desc = "") {
            add_prop(
                member, id, name, PropType::Bool, desc, PropUIHint::Checkbox,
                [](bool v) { return v; },
                [](const std::any& v) { return std::any_cast<bool>(v); });
            group_.properties.back().default_value = default_val;
            return *this;
        }

        PropertyGroupBuilder& string_prop(std::string StructT::*member,
                                          const std::string& id,
                                          const std::string& name,
                                          const std::string& default_val = "",
                                          const std::string& desc = "") {
            add_prop(
                member, id, name, PropType::String, desc, PropUIHint::Input,
                [](const std::string& v) { return v; },
                [](const std::any& v) { return std::any_cast<std::string>(v); });
            group_.properties.back().default_value = default_val;
            return *this;
        }

        template <typename EnumT>
        PropertyGroupBuilder& enum_prop(EnumT StructT::*member,
                                        const std::string& id,
                                        const std::string& name,
                                        EnumT default_val,
                                        std::initializer_list<std::pair<std::string, EnumT>> items,
                                        const std::string& desc = "") {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Enum;
            meta.ui_hint = PropUIHint::Combo;
            meta.default_value = static_cast<int64_t>(default_val);

            for (const auto& [item_name, item_val] : items) {
                EnumItem ei;
                ei.name = item_name;
                ei.identifier = item_name;
                ei.value = static_cast<int>(item_val);
                meta.enum_items.push_back(std::move(ei));
            }

            meta.getter = [member](const PropertyObjectRef& ref) -> std::any {
                assert(ref.is_cpp() && "Cannot call C++ property getter with Python object");
                return static_cast<int>(static_cast<const StructT*>(ref.ptr)->*member);
            };
            meta.setter = [member](PropertyObjectRef& ref, const std::any& val) {
                assert(ref.is_cpp() && "Cannot call C++ property setter with Python object");
                static_cast<StructT*>(ref.ptr)->*member = static_cast<EnumT>(std::any_cast<int>(val));
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        // AnimatableProperty<T> with undo/animation support
        template <typename T>
        PropertyGroupBuilder& animatable_prop(AnimatableProperty<T> StructT::*member,
                                              const std::string& id,
                                              const std::string& name,
                                              const T& default_val,
                                              const std::string& desc = "") {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropertyTraits<T>::type;
            meta.flags = PROP_ANIMATABLE;
            meta.supports_descriptor = true;

            meta.getter = [member](const PropertyObjectRef& ref) -> std::any {
                assert(ref.is_cpp() && "Cannot call C++ property getter with Python object");
                return (static_cast<const StructT*>(ref.ptr)->*member).get();
            };
            meta.setter = [member](PropertyObjectRef& ref, const std::any& val) {
                assert(ref.is_cpp() && "Cannot call C++ property setter with Python object");
                (static_cast<StructT*>(ref.ptr)->*member) = std::any_cast<T>(val);
            };
            meta.get_animatable_ptr = [member](const PropertyObjectRef& ref) -> void* {
                assert(ref.is_cpp() && "Cannot call C++ animatable_ptr with Python object");
                return &(static_cast<StructT*>(const_cast<void*>(ref.ptr))->*member);
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        PropertyGroupBuilder& vec3_prop(glm::vec3 StructT::*member,
                                        const std::string& id,
                                        const std::string& name,
                                        const glm::vec3& default_val,
                                        const std::string& desc = "") {
            add_prop(
                member, id, name, PropType::Vec3, desc, PropUIHint::Default,
                [](const glm::vec3& v) { return std::array<float, 3>{v.x, v.y, v.z}; },
                [](const std::any& v) {
                    auto arr = std::any_cast<std::array<float, 3>>(v);
                    return glm::vec3{arr[0], arr[1], arr[2]};
                });
            group_.properties.back().default_value =
                std::array<double, 3>{default_val.x, default_val.y, default_val.z};
            return *this;
        }

        PropertyGroupBuilder& quat_prop(glm::quat StructT::*member,
                                        const std::string& id,
                                        const std::string& name,
                                        const glm::quat& default_val,
                                        const std::string& desc = "") {
            add_prop(
                member, id, name, PropType::Quat, desc, PropUIHint::Default,
                [](const glm::quat& q) { return std::array<float, 4>{q.w, q.x, q.y, q.z}; },
                [](const std::any& v) {
                    auto arr = std::any_cast<std::array<float, 4>>(v);
                    return glm::quat{arr[0], arr[1], arr[2], arr[3]};
                });
            group_.properties.back().default_value =
                std::array<double, 4>{default_val.w, default_val.x, default_val.y, default_val.z};
            return *this;
        }

        PropertyGroupBuilder& mat4_prop(glm::mat4 StructT::*member,
                                        const std::string& id,
                                        const std::string& name,
                                        const std::string& desc = "") {
            add_prop(
                member, id, name, PropType::Mat4, desc, PropUIHint::Default,
                [](const glm::mat4& m) {
                    std::array<float, 16> arr;
                    for (int i = 0; i < 4; ++i)
                        for (int j = 0; j < 4; ++j)
                            arr[i * 4 + j] = m[i][j];
                    return arr;
                },
                [](const std::any& v) {
                    auto arr = std::any_cast<std::array<float, 16>>(v);
                    glm::mat4 m;
                    for (int i = 0; i < 4; ++i)
                        for (int j = 0; j < 4; ++j)
                            m[i][j] = arr[i * 4 + j];
                    return m;
                });
            return *this;
        }

        PropertyGroupBuilder& color3_prop(glm::vec3 StructT::*member,
                                          const std::string& id,
                                          const std::string& name,
                                          const glm::vec3& default_val,
                                          const std::string& desc = "") {
            add_prop(
                member, id, name, PropType::Color3, desc, PropUIHint::Default,
                [](const glm::vec3& v) { return std::array<float, 3>{v.x, v.y, v.z}; },
                [](const std::any& v) {
                    auto arr = std::any_cast<std::array<float, 3>>(v);
                    return glm::vec3{arr[0], arr[1], arr[2]};
                });
            group_.properties.back().default_value =
                std::array<double, 3>{default_val.x, default_val.y, default_val.z};
            return *this;
        }

        PropertyGroupBuilder& color3_prop(std::array<float, 3> StructT::*member,
                                          const std::string& id,
                                          const std::string& name,
                                          const std::array<float, 3>& default_val,
                                          const std::string& desc = "") {
            add_prop(
                member, id, name, PropType::Color3, desc, PropUIHint::Default,
                [](const std::array<float, 3>& v) { return v; },
                [](const std::any& v) { return std::any_cast<std::array<float, 3>>(v); });
            group_.properties.back().default_value =
                std::array<double, 3>{default_val[0], default_val[1], default_val[2]};
            return *this;
        }

        template <typename CollT>
        PropertyGroupBuilder& collection_prop(CollT StructT::*member,
                                              const std::string& id,
                                              const std::string& item_type) {
            PropertyMeta meta;
            meta.id = id;
            meta.is_collection = true;
            meta.collection_item_type = item_type;

            meta.collection_size = [member](const PropertyObjectRef& ref) -> size_t {
                assert(ref.is_cpp() && "Cannot call C++ collection_size with Python object");
                return (static_cast<const StructT*>(ref.ptr)->*member).size();
            };
            meta.collection_get = [member](const PropertyObjectRef& ref, size_t i) -> PropertyObjectRef {
                assert(ref.is_cpp() && "Cannot call C++ collection_get with Python object");
                auto& coll = static_cast<StructT*>(const_cast<void*>(ref.ptr))->*member;
                if (i < coll.size()) {
                    return PropertyObjectRef::cpp(&coll[i]);
                }
                return PropertyObjectRef{};
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        PropertyGroupBuilder& flags(uint32_t f) {
            if (!group_.properties.empty()) {
                group_.properties.back().flags = f;
            }
            return *this;
        }

        PropertyGroupBuilder& on_update(
            std::function<void(StructT*, const std::any&, const std::any&)> cb) {
            if (!group_.properties.empty()) {
                group_.properties.back().on_update =
                    [cb](const PropertyObjectRef& ref, const std::any& old_val, const std::any& new_val) {
                        assert(ref.is_cpp() && "Cannot call C++ on_update with Python object");
                        cb(static_cast<StructT*>(const_cast<void*>(ref.ptr)), old_val, new_val);
                    };
            }
            return *this;
        }

        void build() { PropertyRegistry::instance().register_group(std::move(group_)); }
        [[nodiscard]] PropertyGroup get() const { return group_; }

    private:
        template <typename MemberT, typename GetFn, typename SetFn>
        PropertyGroupBuilder& add_prop(MemberT StructT::*member,
                                       const std::string& id,
                                       const std::string& name,
                                       PropType type,
                                       const std::string& desc,
                                       PropUIHint hint,
                                       GetFn get_fn,
                                       SetFn set_fn) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = type;
            meta.ui_hint = hint;

            meta.getter = [member, get_fn](const PropertyObjectRef& ref) -> std::any {
                assert(ref.is_cpp() && "Cannot call C++ property getter with Python object");
                return get_fn(static_cast<const StructT*>(ref.ptr)->*member);
            };
            meta.setter = [member, set_fn](PropertyObjectRef& ref, const std::any& val) {
                assert(ref.is_cpp() && "Cannot call C++ property setter with Python object");
                static_cast<StructT*>(ref.ptr)->*member = set_fn(val);
            };

            group_.properties.push_back(std::move(meta));
            return *this;
        }

        PropertyGroup group_;
    };

    inline PropertyMeta operator_arg(std::string id, std::string description, const PropType type) {
        PropertyMeta meta;
        meta.id = std::move(id);
        meta.name = meta.id;
        meta.description = std::move(description);
        meta.type = type;
        meta.flags = PROP_OPERATOR_ARG;
        return meta;
    }

    inline PropertyMeta arg_string(std::string id, std::string description) {
        return operator_arg(std::move(id), std::move(description), PropType::String);
    }

    inline PropertyMeta arg_bool(std::string id, std::string description) {
        return operator_arg(std::move(id), std::move(description), PropType::Bool);
    }

    inline PropertyMeta arg_float(std::string id, std::string description) {
        return operator_arg(std::move(id), std::move(description), PropType::Float);
    }

    inline PropertyMeta arg_int(std::string id, std::string description) {
        return operator_arg(std::move(id), std::move(description), PropType::Int);
    }

    inline PropertyMeta arg_enum(std::string id, std::string description, std::vector<EnumItem> items) {
        auto meta = operator_arg(std::move(id), std::move(description), PropType::Enum);
        meta.enum_items = std::move(items);
        return meta;
    }

    inline PropertyMeta arg_float_vector(std::string id, std::string description,
                                         std::optional<int> size = std::nullopt) {
        auto meta = operator_arg(std::move(id), std::move(description), PropType::FloatVector);
        meta.vector_size = size;
        return meta;
    }

    inline PropertyMeta arg_int_vector(std::string id, std::string description,
                                       std::optional<int> size = std::nullopt) {
        auto meta = operator_arg(std::move(id), std::move(description), PropType::IntVector);
        meta.vector_size = size;
        return meta;
    }

    inline PropertyMeta arg_tensor(std::string id, std::string description) {
        return operator_arg(std::move(id), std::move(description), PropType::Tensor);
    }

} // namespace lfs::core::prop
