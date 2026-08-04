/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/property_system.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <any>
#include <array>
#include <cassert>

namespace nb = nanobind;

namespace lfs::python {

    using ToPythonFn = nb::object (*)(const std::any&);
    using FromPythonFn = std::any (*)(nb::object);

    namespace detail {

        template <core::prop::PropType T>
        struct PropTraits;

        template <>
        struct PropTraits<core::prop::PropType::Bool> {
            static nb::object to_python(const std::any& v) {
                return nb::cast(std::any_cast<bool>(v));
            }
            static std::any from_python(nb::object v) {
                return nb::cast<bool>(v);
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Int> {
            static nb::object to_python(const std::any& v) {
                return nb::cast(std::any_cast<int>(v));
            }
            static std::any from_python(nb::object v) {
                return nb::cast<int>(v);
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Float> {
            static nb::object to_python(const std::any& v) {
                return nb::cast(std::any_cast<float>(v));
            }
            static std::any from_python(nb::object v) {
                return nb::cast<float>(v);
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::String> {
            static nb::object to_python(const std::any& v) {
                return nb::cast(std::any_cast<std::string>(v));
            }
            static std::any from_python(nb::object v) {
                return nb::cast<std::string>(v);
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Enum> {
            static nb::object to_python(const std::any& v) {
                if (v.type() == typeid(std::string)) {
                    return nb::cast(std::any_cast<std::string>(v));
                }
                return nb::cast(std::any_cast<int>(v));
            }
            static std::any from_python(nb::object v) {
                if (nb::isinstance<nb::str>(v)) {
                    return nb::cast<std::string>(v);
                }
                return nb::cast<int>(v);
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::SizeT> {
            static nb::object to_python(const std::any& v) {
                return nb::cast(std::any_cast<size_t>(v));
            }
            static std::any from_python(nb::object v) {
                return nb::cast<size_t>(v);
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Vec2> {
            static nb::object to_python(const std::any& v) {
                const auto arr = std::any_cast<std::array<float, 2>>(v);
                return nb::make_tuple(arr[0], arr[1]);
            }
            static std::any from_python(nb::object v) {
                const auto t = nb::cast<nb::tuple>(v);
                return std::array<float, 2>{nb::cast<float>(t[0]), nb::cast<float>(t[1])};
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Vec3> {
            static nb::object to_python(const std::any& v) {
                const auto arr = std::any_cast<std::array<float, 3>>(v);
                return nb::make_tuple(arr[0], arr[1], arr[2]);
            }
            static std::any from_python(nb::object v) {
                const auto t = nb::cast<nb::tuple>(v);
                return std::array<float, 3>{
                    nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2])};
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Vec4> {
            static nb::object to_python(const std::any& v) {
                const auto arr = std::any_cast<std::array<float, 4>>(v);
                return nb::make_tuple(arr[0], arr[1], arr[2], arr[3]);
            }
            static std::any from_python(nb::object v) {
                const auto t = nb::cast<nb::tuple>(v);
                return std::array<float, 4>{nb::cast<float>(t[0]), nb::cast<float>(t[1]),
                                            nb::cast<float>(t[2]), nb::cast<float>(t[3])};
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Quat> {
            static nb::object to_python(const std::any& v) {
                const auto arr = std::any_cast<std::array<float, 4>>(v);
                return nb::make_tuple(arr[0], arr[1], arr[2], arr[3]);
            }
            static std::any from_python(nb::object v) {
                const auto t = nb::cast<nb::tuple>(v);
                return std::array<float, 4>{nb::cast<float>(t[0]), nb::cast<float>(t[1]),
                                            nb::cast<float>(t[2]), nb::cast<float>(t[3])};
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Mat4> {
            static nb::object to_python(const std::any& v) {
                const auto arr = std::any_cast<std::array<float, 16>>(v);
                nb::list rows;
                for (int i = 0; i < 4; ++i) {
                    rows.append(nb::make_tuple(arr[i * 4 + 0], arr[i * 4 + 1],
                                               arr[i * 4 + 2], arr[i * 4 + 3]));
                }
                return rows;
            }
            static std::any from_python(nb::object v) {
                const auto rows = nb::cast<nb::list>(v);
                std::array<float, 16> arr;
                for (int i = 0; i < 4; ++i) {
                    const auto row = nb::cast<nb::tuple>(rows[i]);
                    for (int j = 0; j < 4; ++j) {
                        arr[i * 4 + j] = nb::cast<float>(row[j]);
                    }
                }
                return arr;
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Color3> {
            static nb::object to_python(const std::any& v) {
                const auto arr = std::any_cast<std::array<float, 3>>(v);
                return nb::make_tuple(arr[0], arr[1], arr[2]);
            }
            static std::any from_python(nb::object v) {
                const auto t = nb::cast<nb::tuple>(v);
                return std::array<float, 3>{
                    nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2])};
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Color4> {
            static nb::object to_python(const std::any& v) {
                const auto arr = std::any_cast<std::array<float, 4>>(v);
                return nb::make_tuple(arr[0], arr[1], arr[2], arr[3]);
            }
            static std::any from_python(nb::object v) {
                const auto t = nb::cast<nb::tuple>(v);
                return std::array<float, 4>{nb::cast<float>(t[0]), nb::cast<float>(t[1]),
                                            nb::cast<float>(t[2]), nb::cast<float>(t[3])};
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::Tensor> {
            static nb::object to_python(const std::any& v) {
                return std::any_cast<nb::object>(v);
            }
            static std::any from_python(nb::object v) {
                return v;
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::FloatVector> {
            static nb::object to_python(const std::any& v) {
                return nb::cast(std::any_cast<std::vector<float>>(v));
            }
            static std::any from_python(nb::object v) {
                return nb::cast<std::vector<float>>(v);
            }
        };

        template <>
        struct PropTraits<core::prop::PropType::IntVector> {
            static nb::object to_python(const std::any& v) {
                return nb::cast(std::any_cast<std::vector<int>>(v));
            }
            static std::any from_python(nb::object v) {
                return nb::cast<std::vector<int>>(v);
            }
        };

    } // namespace detail

    inline constexpr std::array<ToPythonFn, 16> g_to_python_table = {
        detail::PropTraits<core::prop::PropType::Bool>::to_python,
        detail::PropTraits<core::prop::PropType::Int>::to_python,
        detail::PropTraits<core::prop::PropType::Float>::to_python,
        detail::PropTraits<core::prop::PropType::String>::to_python,
        detail::PropTraits<core::prop::PropType::Enum>::to_python,
        detail::PropTraits<core::prop::PropType::SizeT>::to_python,
        detail::PropTraits<core::prop::PropType::Vec2>::to_python,
        detail::PropTraits<core::prop::PropType::Vec3>::to_python,
        detail::PropTraits<core::prop::PropType::Vec4>::to_python,
        detail::PropTraits<core::prop::PropType::Quat>::to_python,
        detail::PropTraits<core::prop::PropType::Mat4>::to_python,
        detail::PropTraits<core::prop::PropType::Color3>::to_python,
        detail::PropTraits<core::prop::PropType::Color4>::to_python,
        detail::PropTraits<core::prop::PropType::Tensor>::to_python,
        detail::PropTraits<core::prop::PropType::FloatVector>::to_python,
        detail::PropTraits<core::prop::PropType::IntVector>::to_python,
    };

    inline constexpr std::array<FromPythonFn, 16> g_from_python_table = {
        detail::PropTraits<core::prop::PropType::Bool>::from_python,
        detail::PropTraits<core::prop::PropType::Int>::from_python,
        detail::PropTraits<core::prop::PropType::Float>::from_python,
        detail::PropTraits<core::prop::PropType::String>::from_python,
        detail::PropTraits<core::prop::PropType::Enum>::from_python,
        detail::PropTraits<core::prop::PropType::SizeT>::from_python,
        detail::PropTraits<core::prop::PropType::Vec2>::from_python,
        detail::PropTraits<core::prop::PropType::Vec3>::from_python,
        detail::PropTraits<core::prop::PropType::Vec4>::from_python,
        detail::PropTraits<core::prop::PropType::Quat>::from_python,
        detail::PropTraits<core::prop::PropType::Mat4>::from_python,
        detail::PropTraits<core::prop::PropType::Color3>::from_python,
        detail::PropTraits<core::prop::PropType::Color4>::from_python,
        detail::PropTraits<core::prop::PropType::Tensor>::from_python,
        detail::PropTraits<core::prop::PropType::FloatVector>::from_python,
        detail::PropTraits<core::prop::PropType::IntVector>::from_python,
    };

    inline nb::object any_to_python(const std::any& val, core::prop::PropType type) {
        const auto idx = static_cast<size_t>(type);
        assert(idx < g_to_python_table.size());
        return g_to_python_table[idx](val);
    }

    inline std::any python_to_any(nb::object val, core::prop::PropType type) {
        const auto idx = static_cast<size_t>(type);
        assert(idx < g_from_python_table.size());
        return g_from_python_table[idx](std::move(val));
    }

    inline const char* prop_type_string(core::prop::PropType type) {
        static constexpr std::array<const char*, 16> names = {
            "bool", "int", "float", "string", "enum", "size_t",
            "vec2", "vec3", "vec4", "quat", "mat4", "color3", "color4", "tensor",
            "float_vector", "int_vector"};
        const auto idx = static_cast<size_t>(type);
        assert(idx < names.size());
        return names[idx];
    }

    inline nb::object prop_default_to_python(const core::prop::PropDefault& value) {
        return std::visit([](const auto& item) { return nb::cast(item); }, value);
    }

} // namespace lfs::python
