/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/include/core/parameters.hpp"
#include "core/include/core/property_registry.hpp"
#include "python/lfs/py_params.hpp"
#include <any>
#include <gtest/gtest.h>
#include <string>

using lfs::core::prop::PropertyObjectRef;
using lfs::core::prop::PropertyRegistry;
using lfs::core::prop::PropType;

namespace {

    class TrainingParametersTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::python::register_optimization_properties();
        }

        void TearDown() override {
            PropertyRegistry::instance().unregister_group("optimization");
        }
    };

    /* Registry defaults and OptimizationParameters member defaults are written
    independently; reset serves the registry copy, so drift must fail here. */

    TEST_F(TrainingParametersTest, RegistryDefaultsMatchStructDefaults) {
        auto group = PropertyRegistry::instance().get_group("optimization");
        ASSERT_NE(group, nullptr);
        ASSERT_FALSE(group->properties.empty());

        lfs::core::param::OptimizationParameters defaults;
        const auto ref = PropertyObjectRef::cpp(&defaults);

        for (const auto& meta : group->properties) {
            SCOPED_TRACE(meta.id);

            if (!meta.getter) {
                ADD_FAILURE() << "property has no getter";
                continue;
            }
            const std::any from_struct = meta.getter(ref);

            switch (meta.type) {
            case PropType::Float:
                EXPECT_FLOAT_EQ(static_cast<float>(meta.default_value),
                                std::any_cast<float>(from_struct));
                break;
            case PropType::Int:
                EXPECT_EQ(static_cast<int>(meta.default_value),
                          std::any_cast<int>(from_struct));
                break;
            case PropType::SizeT:
                EXPECT_EQ(static_cast<size_t>(meta.default_value),
                          std::any_cast<size_t>(from_struct));
                break;
            case PropType::Bool:
                EXPECT_EQ(meta.default_value > 0.5,
                          std::any_cast<bool>(from_struct));
                break;
            case PropType::String:
                EXPECT_EQ(meta.default_string,
                          std::any_cast<std::string>(from_struct));
                break;
            case PropType::Enum:
                EXPECT_EQ(meta.default_enum,
                          std::any_cast<int>(from_struct));
                break;
            default:
                ADD_FAILURE() << "unhandled property type";
                break;
            }
        }
    }

} // namespace