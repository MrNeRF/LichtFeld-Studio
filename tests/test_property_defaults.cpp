/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/property_defaults.hpp"
#include "core/property_registry.hpp"

#include <algorithm>
#include <gtest/gtest.h>

using namespace lfs::core::prop;

namespace {
    PropertyMeta sparse_enum() {
        PropertyMeta meta;
        meta.id = "resolution";
        meta.type = PropType::Enum;
        meta.enum_items = {{"Small", "small", 512}, {"Medium", "medium", 2048}, {"Large", "large", 4096}};
        meta.default_value = int64_t{2048};
        return meta;
    }
} // namespace

TEST(PropertyDefaults, ResolvesValueRatherThanPosition) {
    auto meta = sparse_enum();
    EXPECT_EQ(enum_default_identifier(meta), "medium");
    std::ranges::reverse(meta.enum_items);
    EXPECT_EQ(enum_default_identifier(meta), "medium");
    meta.default_value = int64_t{1};
    EXPECT_FALSE(enum_default_identifier(meta));
}

TEST(PropertyDefaults, SupportsZeroAndNegativeValues) {
    auto meta = sparse_enum();
    meta.enum_items = {{"Auto", "auto", -1}, {"Off", "off", 0}, {"On", "on", 7}};
    meta.default_value = int64_t{-1};
    EXPECT_EQ(enum_default_identifier(meta), "auto");
    meta.default_value = int64_t{0};
    EXPECT_EQ(enum_default_identifier(meta), "off");
    EXPECT_EQ(enum_identifier(meta, 7), "on");
}

TEST(PropertyDefaults, MissingDefaultIsDifferentFromZero) {
    auto meta = sparse_enum();
    meta.default_value.reset();
    EXPECT_FALSE(enum_default_identifier(meta));
}

TEST(PropertyDefaults, RejectsWrongVariantAndUnknownContextValues) {
    auto meta = sparse_enum();
    for (const PropDefault& value : {PropDefault{2.0}, PropDefault{std::string("medium")}, PropDefault{false}}) {
        meta.default_value = value;
        EXPECT_FALSE(enum_default_identifier(meta));
    }
    EXPECT_THROW((void)enum_identifier(meta, 100), std::invalid_argument);
    meta.type = PropType::Int;
    EXPECT_FALSE(enum_default_identifier(meta));
    EXPECT_THROW((void)enum_identifier(meta, 2048), std::invalid_argument);
}

TEST(PropertyDefaults, AmbiguousIdentifiersAndValuesDisableAutomaticDefaults) {
    auto meta = sparse_enum();
    meta.enum_items.push_back({"Alias", "medium", 8192});
    EXPECT_FALSE(enum_default_identifier(meta));
    meta.enum_items.back() = {"Alias", "alias", 2048};
    EXPECT_FALSE(enum_default_identifier(meta));
}

TEST(PropertyDefaults, BuilderPreservesUnderlyingEnumValues) {
    enum class Resolution { Small = 512,
                            Medium = 2048 };
    struct Settings {
        Resolution resolution = Resolution::Medium;
    } settings;
    const auto group = PropertyGroupBuilder<Settings>("tests.builder_defaults", "Defaults")
                           .enum_prop(&Settings::resolution, "resolution", "Resolution", Resolution::Medium,
                                      {{"Small", Resolution::Small}, {"Medium", Resolution::Medium}})
                           .get();
    const auto& meta = group.properties.front();
    EXPECT_EQ(std::get<int64_t>(*meta.default_value), 2048);
    EXPECT_EQ(enum_default_identifier(meta), "Medium");
    auto ref = PropertyObjectRef::cpp(&settings);
    EXPECT_EQ(std::any_cast<int>(meta.getter(ref)), 2048);
}
