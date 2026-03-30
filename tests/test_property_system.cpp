/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/property_registry.hpp"
#include "core/property_system.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <set>
#include <thread>
#include <vector>

using namespace lfs::core::prop;

// ============================================================================
// Test Structures
// ============================================================================

namespace {

    // Simple test struct with various property types
    struct TestParams {
        float learning_rate = 0.001f;
        int max_iterations = 1000;
        bool enabled = true;
        size_t buffer_size = 4096;
        std::string name = "default";
    };

    // Enum for testing
    enum class TestStrategy { Fast,
                              Accurate,
                              Balanced };

    struct TestParamsWithEnum {
        float value = 1.0f;
        TestStrategy strategy = TestStrategy::Balanced;
    };

    // Counter for tracking callback invocations
    struct CallbackTracker {
        std::atomic<int> call_count{0};
        std::string last_group;
        std::string last_prop;
        std::any last_old_value;
        std::any last_new_value;

        void reset() {
            call_count = 0;
            last_group.clear();
            last_prop.clear();
            last_old_value.reset();
            last_new_value.reset();
        }
    };

} // anonymous namespace

// ============================================================================
// PropertyMeta Tests
// ============================================================================

class PropertyMetaTest : public ::testing::Test {};

TEST_F(PropertyMetaTest, DefaultConstructedMeta) {
    PropertyMeta meta;
    EXPECT_TRUE(meta.id.empty());
    EXPECT_TRUE(meta.name.empty());
    EXPECT_EQ(meta.type, PropType::Float);
    EXPECT_EQ(meta.ui_hint, PropUIHint::Default);
    EXPECT_EQ(meta.flags, PROP_NONE);
    EXPECT_FALSE(meta.is_readonly());
    EXPECT_FALSE(meta.is_live_update());
    EXPECT_FALSE(meta.needs_restart());
}

TEST_F(PropertyMetaTest, FlagsOperations) {
    PropertyMeta meta;

    // Test ReadOnly flag
    meta.flags = PROP_READONLY;
    EXPECT_TRUE(meta.is_readonly());
    EXPECT_FALSE(meta.is_live_update());
    EXPECT_FALSE(meta.needs_restart());

    // Test LiveUpdate flag
    meta.flags = PROP_LIVE_UPDATE;
    EXPECT_FALSE(meta.is_readonly());
    EXPECT_TRUE(meta.is_live_update());
    EXPECT_FALSE(meta.needs_restart());

    // Test NeedsRestart flag
    meta.flags = PROP_NEEDS_RESTART;
    EXPECT_FALSE(meta.is_readonly());
    EXPECT_FALSE(meta.is_live_update());
    EXPECT_TRUE(meta.needs_restart());

    // Test combined flags
    meta.flags = PROP_READONLY | PROP_LIVE_UPDATE;
    EXPECT_TRUE(meta.is_readonly());
    EXPECT_TRUE(meta.is_live_update());
    EXPECT_FALSE(meta.needs_restart());
}

TEST_F(PropertyMetaTest, HasFlagMethod) {
    PropertyMeta meta;
    meta.flags = PROP_LIVE_UPDATE | PROP_ANIMATABLE;

    EXPECT_FALSE(meta.has_flag(PROP_NONE));
    EXPECT_FALSE(meta.has_flag(PROP_READONLY));
    EXPECT_TRUE(meta.has_flag(PROP_LIVE_UPDATE));
    EXPECT_FALSE(meta.has_flag(PROP_NEEDS_RESTART));
    EXPECT_TRUE(meta.has_flag(PROP_ANIMATABLE));
}

// ============================================================================
// PropertyGroup Tests
// ============================================================================

class PropertyGroupTest : public ::testing::Test {};

TEST_F(PropertyGroupTest, FindProperty) {
    PropertyGroup group;
    group.id = "test_group";
    group.name = "Test Group";

    PropertyMeta meta1;
    meta1.id = "prop1";
    meta1.name = "Property 1";

    PropertyMeta meta2;
    meta2.id = "prop2";
    meta2.name = "Property 2";

    group.properties.push_back(meta1);
    group.properties.push_back(meta2);

    // Find existing properties
    const PropertyMeta* found1 = group.find("prop1");
    ASSERT_NE(found1, nullptr);
    EXPECT_EQ(found1->name, "Property 1");

    const PropertyMeta* found2 = group.find("prop2");
    ASSERT_NE(found2, nullptr);
    EXPECT_EQ(found2->name, "Property 2");

    // Find non-existing property
    const PropertyMeta* not_found = group.find("nonexistent");
    EXPECT_EQ(not_found, nullptr);
}

TEST_F(PropertyGroupTest, EmptyGroup) {
    PropertyGroup group;
    group.id = "empty";
    group.name = "Empty Group";

    EXPECT_EQ(group.find("anything"), nullptr);
    EXPECT_TRUE(group.properties.empty());
}

// ============================================================================
// PropertyGroupBuilder Tests
// ============================================================================

class PropertyGroupBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear registry state for each test (if needed)
    }
};

TEST_F(PropertyGroupBuilderTest, FloatProperty) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("test", "Test Group");
    builder.float_prop(&TestParams::learning_rate,
                       "learning_rate", "Learning Rate",
                       0.001f, 0.0f, 1.0f,
                       "Rate of learning");

    PropertyGroup group = builder.get();

    ASSERT_EQ(group.properties.size(), 1);
    const auto& prop = group.properties[0];

    EXPECT_EQ(prop.id, "learning_rate");
    EXPECT_EQ(prop.name, "Learning Rate");
    EXPECT_EQ(prop.description, "Rate of learning");
    EXPECT_EQ(prop.type, PropType::Float);
    EXPECT_DOUBLE_EQ(prop.min_value, 0.0);
    EXPECT_DOUBLE_EQ(prop.max_value, 1.0);
    // default_value is stored as double but from float, use tolerance
    EXPECT_NEAR(prop.default_value, 0.001, 1e-6);

    // Test getter
    ASSERT_TRUE(prop.getter);
    std::any val = prop.getter(&params);
    ASSERT_TRUE(val.has_value());
    EXPECT_FLOAT_EQ(std::any_cast<float>(val), 0.001f);

    // Test setter
    ASSERT_TRUE(prop.setter);
    prop.setter(&params, 0.5f);
    EXPECT_FLOAT_EQ(params.learning_rate, 0.5f);
}

TEST_F(PropertyGroupBuilderTest, IntProperty) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("test", "Test Group");
    builder.int_prop(&TestParams::max_iterations,
                     "max_iterations", "Max Iterations",
                     1000, 1, 1000000,
                     "Maximum iteration count");

    PropertyGroup group = builder.get();

    ASSERT_EQ(group.properties.size(), 1);
    const auto& prop = group.properties[0];

    EXPECT_EQ(prop.id, "max_iterations");
    EXPECT_EQ(prop.type, PropType::Int);
    EXPECT_DOUBLE_EQ(prop.min_value, 1.0);
    EXPECT_DOUBLE_EQ(prop.max_value, 1000000.0);
    EXPECT_DOUBLE_EQ(prop.default_value, 1000.0);

    // Test getter
    std::any val = prop.getter(&params);
    EXPECT_EQ(std::any_cast<int>(val), 1000);

    // Test setter
    prop.setter(&params, 5000);
    EXPECT_EQ(params.max_iterations, 5000);
}

TEST_F(PropertyGroupBuilderTest, BoolProperty) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("test", "Test Group");
    builder.bool_prop(&TestParams::enabled,
                      "enabled", "Enabled",
                      true, "Toggle feature on/off");

    PropertyGroup group = builder.get();

    ASSERT_EQ(group.properties.size(), 1);
    const auto& prop = group.properties[0];

    EXPECT_EQ(prop.id, "enabled");
    EXPECT_EQ(prop.type, PropType::Bool);
    EXPECT_EQ(prop.ui_hint, PropUIHint::Checkbox);

    // Test getter
    std::any val = prop.getter(&params);
    EXPECT_TRUE(std::any_cast<bool>(val));

    // Test setter
    prop.setter(&params, false);
    EXPECT_FALSE(params.enabled);
}

TEST_F(PropertyGroupBuilderTest, SizeTProperty) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("test", "Test Group");
    builder.size_prop(&TestParams::buffer_size,
                      "buffer_size", "Buffer Size",
                      4096, 256, 1048576,
                      "Size of internal buffer");

    PropertyGroup group = builder.get();

    ASSERT_EQ(group.properties.size(), 1);
    const auto& prop = group.properties[0];

    EXPECT_EQ(prop.id, "buffer_size");
    EXPECT_EQ(prop.type, PropType::SizeT);

    // Test getter
    std::any val = prop.getter(&params);
    EXPECT_EQ(std::any_cast<size_t>(val), 4096);

    // Test setter
    prop.setter(&params, size_t{8192});
    EXPECT_EQ(params.buffer_size, 8192);
}

TEST_F(PropertyGroupBuilderTest, StringProperty) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("test", "Test Group");
    builder.string_prop(&TestParams::name,
                        "name", "Name",
                        "default", "Object name");

    PropertyGroup group = builder.get();

    ASSERT_EQ(group.properties.size(), 1);
    const auto& prop = group.properties[0];

    EXPECT_EQ(prop.id, "name");
    EXPECT_EQ(prop.type, PropType::String);
    EXPECT_EQ(prop.default_string, "default");

    // Test getter
    std::any val = prop.getter(&params);
    EXPECT_EQ(std::any_cast<std::string>(val), "default");

    // Test setter
    prop.setter(&params, std::string("custom_name"));
    EXPECT_EQ(params.name, "custom_name");
}

TEST_F(PropertyGroupBuilderTest, EnumProperty) {
    TestParamsWithEnum params;

    PropertyGroupBuilder<TestParamsWithEnum> builder("test", "Test Group");
    builder.enum_prop(&TestParamsWithEnum::strategy,
                      "strategy", "Strategy",
                      TestStrategy::Balanced,
                      {{"Fast", TestStrategy::Fast},
                       {"Accurate", TestStrategy::Accurate},
                       {"Balanced", TestStrategy::Balanced}},
                      "Algorithm strategy");

    PropertyGroup group = builder.get();

    ASSERT_EQ(group.properties.size(), 1);
    const auto& prop = group.properties[0];

    EXPECT_EQ(prop.id, "strategy");
    EXPECT_EQ(prop.type, PropType::Enum);
    EXPECT_EQ(prop.ui_hint, PropUIHint::Combo);
    ASSERT_EQ(prop.enum_items.size(), 3);
    EXPECT_EQ(prop.enum_items[0].name, "Fast");
    EXPECT_EQ(prop.enum_items[1].name, "Accurate");
    EXPECT_EQ(prop.enum_items[2].name, "Balanced");

    // Test getter (returns int representation)
    std::any val = prop.getter(&params);
    EXPECT_EQ(std::any_cast<int>(val), static_cast<int>(TestStrategy::Balanced));

    // Test setter
    prop.setter(&params, static_cast<int>(TestStrategy::Fast));
    EXPECT_EQ(params.strategy, TestStrategy::Fast);
}

TEST_F(PropertyGroupBuilderTest, FlagsChaining) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("test", "Test Group");
    builder.float_prop(&TestParams::learning_rate, "lr", "LR", 0.001f, 0.0f, 1.0f)
        .flags(PROP_LIVE_UPDATE | PROP_ANIMATABLE);

    PropertyGroup group = builder.get();
    const auto& prop = group.properties[0];

    EXPECT_TRUE(prop.has_flag(PROP_LIVE_UPDATE));
    EXPECT_TRUE(prop.has_flag(PROP_ANIMATABLE));
    EXPECT_FALSE(prop.has_flag(PROP_READONLY));
}

TEST_F(PropertyGroupBuilderTest, CategoryChaining) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("test", "Test Group");
    builder.float_prop(&TestParams::learning_rate, "lr", "LR", 0.001f, 0.0f, 1.0f)
        .category("optimization");

    PropertyGroup group = builder.get();
    const auto& prop = group.properties[0];

    EXPECT_EQ(prop.group, "optimization");
}

TEST_F(PropertyGroupBuilderTest, MultiplePropertiesChained) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("test", "Test Group");
    builder.float_prop(&TestParams::learning_rate, "lr", "LR", 0.001f, 0.0f, 1.0f)
        .flags(PROP_LIVE_UPDATE)
        .category("rates")
        .int_prop(&TestParams::max_iterations, "iters", "Iterations", 1000, 1, 1000000)
        .category("control")
        .bool_prop(&TestParams::enabled, "enabled", "Enabled", true)
        .flags(PROP_READONLY);

    PropertyGroup group = builder.get();

    ASSERT_EQ(group.properties.size(), 3);

    // First property
    EXPECT_EQ(group.properties[0].id, "lr");
    EXPECT_TRUE(group.properties[0].has_flag(PROP_LIVE_UPDATE));
    EXPECT_EQ(group.properties[0].group, "rates");

    // Second property
    EXPECT_EQ(group.properties[1].id, "iters");
    EXPECT_EQ(group.properties[1].group, "control");

    // Third property
    EXPECT_EQ(group.properties[2].id, "enabled");
    EXPECT_TRUE(group.properties[2].has_flag(PROP_READONLY));
}

// ============================================================================
// PropertyRegistry Tests
// ============================================================================

class PropertyRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // The registry is a singleton, we'll use unique group IDs per test
        test_group_id_ = "test_group_" + std::to_string(test_counter_++);
    }

    std::string test_group_id_;
    static int test_counter_;
};

int PropertyRegistryTest::test_counter_ = 0;

TEST_F(PropertyRegistryTest, RegisterAndRetrieveGroup) {
    PropertyGroup group;
    group.id = test_group_id_;
    group.name = "Test Group";

    PropertyMeta meta;
    meta.id = "test_prop";
    meta.name = "Test Property";
    meta.type = PropType::Float;
    group.properties.push_back(meta);

    auto& registry = PropertyRegistry::instance();
    registry.register_group(std::move(group));

    const PropertyGroup* retrieved = registry.get_group(test_group_id_);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->id, test_group_id_);
    EXPECT_EQ(retrieved->name, "Test Group");
    ASSERT_EQ(retrieved->properties.size(), 1);
    EXPECT_EQ(retrieved->properties[0].id, "test_prop");
}

TEST_F(PropertyRegistryTest, GetNonExistentGroup) {
    auto& registry = PropertyRegistry::instance();
    const PropertyGroup* group = registry.get_group("nonexistent_group_xyz");
    EXPECT_EQ(group, nullptr);
}

TEST_F(PropertyRegistryTest, GetProperty) {
    PropertyGroup group;
    group.id = test_group_id_;
    group.name = "Test Group";

    PropertyMeta meta1, meta2;
    meta1.id = "prop1";
    meta1.name = "Property 1";
    meta2.id = "prop2";
    meta2.name = "Property 2";
    group.properties.push_back(meta1);
    group.properties.push_back(meta2);

    auto& registry = PropertyRegistry::instance();
    registry.register_group(std::move(group));

    const PropertyMeta* found1 = registry.get_property(test_group_id_, "prop1");
    ASSERT_NE(found1, nullptr);
    EXPECT_EQ(found1->name, "Property 1");

    const PropertyMeta* found2 = registry.get_property(test_group_id_, "prop2");
    ASSERT_NE(found2, nullptr);
    EXPECT_EQ(found2->name, "Property 2");

    // Non-existent property
    const PropertyMeta* not_found = registry.get_property(test_group_id_, "nonexistent");
    EXPECT_EQ(not_found, nullptr);

    // Non-existent group
    const PropertyMeta* no_group = registry.get_property("nonexistent_group", "prop1");
    EXPECT_EQ(no_group, nullptr);
}

TEST_F(PropertyRegistryTest, GetGroupIds) {
    PropertyGroup group;
    group.id = test_group_id_;
    group.name = "Test Group";

    auto& registry = PropertyRegistry::instance();
    registry.register_group(std::move(group));

    auto ids = registry.get_group_ids();
    EXPECT_FALSE(ids.empty());

    // Our test group should be in there
    auto it = std::find(ids.begin(), ids.end(), test_group_id_);
    EXPECT_NE(it, ids.end());
}

TEST_F(PropertyRegistryTest, GlobalSubscription) {
    CallbackTracker tracker;

    auto& registry = PropertyRegistry::instance();

    // Register callback
    size_t sub_id = registry.subscribe(
        [&tracker](const std::string& group, const std::string& prop,
                   const std::any& old_val, const std::any& new_val) {
            tracker.call_count++;
            tracker.last_group = group;
            tracker.last_prop = prop;
            tracker.last_old_value = old_val;
            tracker.last_new_value = new_val;
        });

    // Trigger notification
    registry.notify("test_group", "test_prop", 1.0f, 2.0f);

    EXPECT_EQ(tracker.call_count, 1);
    EXPECT_EQ(tracker.last_group, "test_group");
    EXPECT_EQ(tracker.last_prop, "test_prop");
    EXPECT_FLOAT_EQ(std::any_cast<float>(tracker.last_old_value), 1.0f);
    EXPECT_FLOAT_EQ(std::any_cast<float>(tracker.last_new_value), 2.0f);

    // Unsubscribe
    registry.unsubscribe(sub_id);

    // Trigger again - should not call
    tracker.reset();
    registry.notify("test_group", "test_prop", 2.0f, 3.0f);
    EXPECT_EQ(tracker.call_count, 0);
}

TEST_F(PropertyRegistryTest, PropertySpecificSubscription) {
    CallbackTracker tracker;

    auto& registry = PropertyRegistry::instance();

    // Register callback for specific property
    size_t sub_id = registry.subscribe(
        "my_group", "my_prop",
        [&tracker](const std::string& group, const std::string& prop,
                   const std::any& old_val, const std::any& new_val) {
            tracker.call_count++;
            tracker.last_group = group;
            tracker.last_prop = prop;
        });

    // Notify matching property
    registry.notify("my_group", "my_prop", 1, 2);
    EXPECT_EQ(tracker.call_count, 1);

    // Notify different property - should not call
    tracker.reset();
    registry.notify("my_group", "other_prop", 1, 2);
    EXPECT_EQ(tracker.call_count, 0);

    // Notify different group - should not call
    registry.notify("other_group", "my_prop", 1, 2);
    EXPECT_EQ(tracker.call_count, 0);

    registry.unsubscribe(sub_id);
}

TEST_F(PropertyRegistryTest, MultipleSubscribers) {
    std::atomic<int> count1{0}, count2{0}, count3{0};

    auto& registry = PropertyRegistry::instance();

    size_t sub1 = registry.subscribe(
        [&count1](const std::string&, const std::string&,
                  const std::any&, const std::any&) { count1++; });

    size_t sub2 = registry.subscribe(
        [&count2](const std::string&, const std::string&,
                  const std::any&, const std::any&) { count2++; });

    size_t sub3 = registry.subscribe(
        "specific", "prop",
        [&count3](const std::string&, const std::string&,
                  const std::any&, const std::any&) { count3++; });

    // Notify generic property
    registry.notify("any_group", "any_prop", 0, 0);
    EXPECT_EQ(count1, 1);
    EXPECT_EQ(count2, 1);
    EXPECT_EQ(count3, 0);

    // Notify specific property
    registry.notify("specific", "prop", 0, 0);
    EXPECT_EQ(count1, 2);
    EXPECT_EQ(count2, 2);
    EXPECT_EQ(count3, 1);

    registry.unsubscribe(sub1);
    registry.unsubscribe(sub2);
    registry.unsubscribe(sub3);
}

TEST_F(PropertyRegistryTest, CallbackExceptionSafety) {
    std::atomic<int> good_count{0};

    auto& registry = PropertyRegistry::instance();

    // First callback throws
    size_t sub1 = registry.subscribe(
        [](const std::string&, const std::string&,
           const std::any&, const std::any&) {
            throw std::runtime_error("callback error");
        });

    // Second callback should still be called
    size_t sub2 = registry.subscribe(
        [&good_count](const std::string&, const std::string&,
                      const std::any&, const std::any&) { good_count++; });

    // Notify - first throws, second should still run
    registry.notify("group", "prop", 0, 0);
    EXPECT_EQ(good_count, 1);

    registry.unsubscribe(sub1);
    registry.unsubscribe(sub2);
}

TEST_F(PropertyRegistryTest, UnsubscribeIdempotent) {
    auto& registry = PropertyRegistry::instance();

    size_t sub_id = registry.subscribe(
        [](const std::string&, const std::string&,
           const std::any&, const std::any&) {});

    // Unsubscribe multiple times should be safe
    registry.unsubscribe(sub_id);
    registry.unsubscribe(sub_id);
    registry.unsubscribe(sub_id);

    // Unsubscribe non-existent ID should be safe
    registry.unsubscribe(999999);
}

// ============================================================================
// PropertyGroupBuilder with Registry Integration
// ============================================================================

class PropertyBuilderRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_group_id_ = "builder_test_" + std::to_string(test_counter_++);
    }

    std::string test_group_id_;
    static int test_counter_;
};

int PropertyBuilderRegistryTest::test_counter_ = 0;

TEST_F(PropertyBuilderRegistryTest, BuildAndRegister) {
    TestParams params;

    PropertyGroupBuilder<TestParams>(test_group_id_, "Test Group")
        .float_prop(&TestParams::learning_rate, "lr", "LR", 0.001f, 0.0f, 1.0f)
        .int_prop(&TestParams::max_iterations, "iters", "Iterations", 1000, 1, 1000000)
        .build();

    auto& registry = PropertyRegistry::instance();
    const PropertyGroup* group = registry.get_group(test_group_id_);

    ASSERT_NE(group, nullptr);
    EXPECT_EQ(group->id, test_group_id_);
    ASSERT_EQ(group->properties.size(), 2);
}

// ============================================================================
// Edge Case and Error Tests
// ============================================================================

class PropertyEdgeCaseTest : public ::testing::Test {};

TEST_F(PropertyEdgeCaseTest, EmptyPropertyId) {
    PropertyMeta meta;
    meta.id = "";
    meta.name = "Empty ID Property";

    PropertyGroup group;
    group.id = "edge_cases";
    group.name = "Edge Cases";
    group.properties.push_back(meta);

    // Should be findable with empty string
    const PropertyMeta* found = group.find("");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name, "Empty ID Property");
}

TEST_F(PropertyEdgeCaseTest, ExtremeFloatValues) {
    struct ExtremeParams {
        float tiny = 1e-38f;
        float huge = 1e38f;
        float negative = -1e10f;
    };

    ExtremeParams params;

    PropertyGroupBuilder<ExtremeParams> builder("extreme", "Extreme Values");
    builder.float_prop(&ExtremeParams::tiny, "tiny", "Tiny", 1e-38f, 0.0f, 1e-30f)
        .float_prop(&ExtremeParams::huge, "huge", "Huge", 1e38f, 0.0f, 1e38f)
        .float_prop(&ExtremeParams::negative, "negative", "Negative", -1e10f, -1e20f, 0.0f);

    PropertyGroup group = builder.get();

    // Test getter/setter for extreme values
    for (const auto& prop : group.properties) {
        ASSERT_TRUE(prop.getter);
        ASSERT_TRUE(prop.setter);
        std::any val = prop.getter(&params);
        EXPECT_TRUE(val.has_value());
    }

    // Setter should handle extreme values
    group.properties[0].setter(&params, 1e-45f);
    EXPECT_FLOAT_EQ(params.tiny, 1e-45f);
}

TEST_F(PropertyEdgeCaseTest, ZeroRangeProperty) {
    struct ZeroRange {
        float locked = 0.5f;
    };

    ZeroRange params;

    PropertyGroupBuilder<ZeroRange> builder("zero_range", "Zero Range");
    builder.float_prop(&ZeroRange::locked, "locked", "Locked", 0.5f, 0.5f, 0.5f);

    PropertyGroup group = builder.get();
    ASSERT_EQ(group.properties.size(), 1);

    // Min equals max equals default
    EXPECT_DOUBLE_EQ(group.properties[0].min_value, 0.5);
    EXPECT_DOUBLE_EQ(group.properties[0].max_value, 0.5);
    EXPECT_DOUBLE_EQ(group.properties[0].default_value, 0.5);
}

TEST_F(PropertyEdgeCaseTest, UnicodePropertyNames) {
    PropertyMeta meta;
    meta.id = "unicode_prop";
    meta.name = "日本語プロパティ";
    meta.description = "Beschreibung mit Umlauten: äöü";

    EXPECT_EQ(meta.name, "日本語プロパティ");
    EXPECT_EQ(meta.description, "Beschreibung mit Umlauten: äöü");
}

TEST_F(PropertyEdgeCaseTest, TypeMismatchInAnyCast) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("type_mismatch", "Type Mismatch");
    builder.float_prop(&TestParams::learning_rate, "lr", "LR", 0.001f, 0.0f, 1.0f);

    PropertyGroup group = builder.get();
    const auto& prop = group.properties[0];

    // Attempt to set wrong type should throw std::bad_any_cast
    EXPECT_THROW(prop.setter(&params, 42), std::bad_any_cast);
    EXPECT_THROW(prop.setter(&params, std::string("wrong")), std::bad_any_cast);
}

TEST_F(PropertyEdgeCaseTest, NullObjectPointer) {
    TestParams params;

    PropertyGroupBuilder<TestParams> builder("null_test", "Null Test");
    builder.float_prop(&TestParams::learning_rate, "lr", "LR", 0.001f, 0.0f, 1.0f);

    PropertyGroup group = builder.get();
    const auto& prop = group.properties[0];

    // Passing nullptr should cause undefined behavior or crash
    // We test that it at least throws or doesn't corrupt memory
    // In practice, the code assumes valid pointers - this is testing documentation
    // EXPECT_DEATH(prop.getter(nullptr), ".*");  // Uncomment if death tests are desired
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

class PropertyThreadSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_group_id_ = "thread_test_" + std::to_string(test_counter_++);
    }

    std::string test_group_id_;
    static int test_counter_;
};

int PropertyThreadSafetyTest::test_counter_ = 0;

TEST_F(PropertyThreadSafetyTest, ConcurrentSubscribeNotify) {
    auto& registry = PropertyRegistry::instance();
    std::atomic<int> total_calls{0};
    std::vector<size_t> sub_ids;
    std::mutex sub_mutex;

    constexpr int NUM_THREADS = 8;
    constexpr int OPERATIONS_PER_THREAD = 100;

    std::vector<std::thread> threads;

    // Start threads that subscribe and notify concurrently
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                if (i % 2 == 0) {
                    // Subscribe
                    size_t id = registry.subscribe(
                        [&total_calls](const std::string&, const std::string&,
                                       const std::any&, const std::any&) {
                            total_calls++;
                        });
                    std::lock_guard lock(sub_mutex);
                    sub_ids.push_back(id);
                } else {
                    // Notify
                    registry.notify(test_group_id_, "prop_" + std::to_string(i), i - 1, i);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Cleanup
    for (size_t id : sub_ids) {
        registry.unsubscribe(id);
    }

    // Should have some calls (exact number depends on timing)
    EXPECT_GT(total_calls, 0);
}

TEST_F(PropertyThreadSafetyTest, ConcurrentRegistration) {
    auto& registry = PropertyRegistry::instance();

    constexpr int NUM_THREADS = 4;
    constexpr int GROUPS_PER_THREAD = 10;

    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < GROUPS_PER_THREAD; ++i) {
                PropertyGroup group;
                group.id = test_group_id_ + "_t" + std::to_string(t) + "_" + std::to_string(i);
                group.name = "Thread " + std::to_string(t) + " Group " + std::to_string(i);
                registry.register_group(std::move(group));
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Verify all groups are registered
    auto ids = registry.get_group_ids();
    int our_groups = 0;
    for (const auto& id : ids) {
        if (id.find(test_group_id_) == 0) {
            our_groups++;
        }
    }

    EXPECT_EQ(our_groups, NUM_THREADS * GROUPS_PER_THREAD);
}

// ============================================================================
// Integration Tests
// ============================================================================

class PropertyIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_group_id_ = "integration_test_" + std::to_string(test_counter_++);
    }

    std::string test_group_id_;
    static int test_counter_;
};

int PropertyIntegrationTest::test_counter_ = 0;

TEST_F(PropertyIntegrationTest, FullWorkflow) {
    // 1. Define a struct with properties
    struct TrainingConfig {
        float learning_rate = 0.001f;
        int epochs = 100;
        bool use_gpu = true;
        std::string model_name = "default";
    };

    TrainingConfig config;

    // 2. Build and register property group
    PropertyGroupBuilder<TrainingConfig>(test_group_id_, "Training Configuration")
        .float_prop(&TrainingConfig::learning_rate, "learning_rate", "Learning Rate",
                    0.001f, 0.0f, 1.0f, "Step size for optimization")
        .flags(PROP_LIVE_UPDATE)
        .category("optimization")
        .int_prop(&TrainingConfig::epochs, "epochs", "Epochs",
                  100, 1, 10000, "Number of training epochs")
        .bool_prop(&TrainingConfig::use_gpu, "use_gpu", "Use GPU",
                   true, "Enable GPU acceleration")
        .flags(PROP_NEEDS_RESTART)
        .string_prop(&TrainingConfig::model_name, "model_name", "Model Name",
                     "default", "Name of the model")
        .build();

    auto& registry = PropertyRegistry::instance();

    // 3. Retrieve the group
    const PropertyGroup* group = registry.get_group(test_group_id_);
    ASSERT_NE(group, nullptr);
    ASSERT_EQ(group->properties.size(), 4);

    // 4. Subscribe to changes
    CallbackTracker tracker;
    size_t sub_id = registry.subscribe(
        test_group_id_, "learning_rate",
        [&tracker](const std::string& grp, const std::string& prop,
                   const std::any& old_val, const std::any& new_val) {
            tracker.call_count++;
            tracker.last_old_value = old_val;
            tracker.last_new_value = new_val;
        });

    // 5. Modify property and notify
    const PropertyMeta* lr_meta = registry.get_property(test_group_id_, "learning_rate");
    ASSERT_NE(lr_meta, nullptr);

    float old_lr = std::any_cast<float>(lr_meta->getter(&config));
    lr_meta->setter(&config, 0.01f);
    float new_lr = std::any_cast<float>(lr_meta->getter(&config));

    registry.notify(test_group_id_, "learning_rate", old_lr, new_lr);

    // 6. Verify callback was triggered
    EXPECT_EQ(tracker.call_count, 1);
    EXPECT_FLOAT_EQ(std::any_cast<float>(tracker.last_old_value), 0.001f);
    EXPECT_FLOAT_EQ(std::any_cast<float>(tracker.last_new_value), 0.01f);

    // 7. Verify property flags
    EXPECT_TRUE(lr_meta->has_flag(PROP_LIVE_UPDATE));
    EXPECT_FALSE(lr_meta->has_flag(PROP_NEEDS_RESTART));

    const PropertyMeta* gpu_meta = registry.get_property(test_group_id_, "use_gpu");
    ASSERT_NE(gpu_meta, nullptr);
    EXPECT_TRUE(gpu_meta->has_flag(PROP_NEEDS_RESTART));

    // 8. Cleanup
    registry.unsubscribe(sub_id);
}

TEST_F(PropertyIntegrationTest, PropertyEnumeration) {
    struct Settings {
        float a = 1.0f;
        float b = 2.0f;
        int c = 3;
    };

    Settings settings;

    PropertyGroupBuilder<Settings>(test_group_id_, "Settings")
        .float_prop(&Settings::a, "a", "A", 1.0f, 0.0f, 10.0f)
        .category("cat1")
        .float_prop(&Settings::b, "b", "B", 2.0f, 0.0f, 10.0f)
        .category("cat1")
        .int_prop(&Settings::c, "c", "C", 3, 0, 100)
        .category("cat2")
        .build();

    auto& registry = PropertyRegistry::instance();
    const PropertyGroup* group = registry.get_group(test_group_id_);
    ASSERT_NE(group, nullptr);

    // Count properties by category
    std::map<std::string, int> category_counts;
    for (const auto& prop : group->properties) {
        category_counts[prop.group]++;
    }

    EXPECT_EQ(category_counts["cat1"], 2);
    EXPECT_EQ(category_counts["cat2"], 1);
}

TEST_F(PropertyIntegrationTest, DefaultValueRestoration) {
    struct Params {
        float value = 42.0f;
    };

    Params params;

    PropertyGroupBuilder<Params>(test_group_id_, "Params")
        .float_prop(&Params::value, "value", "Value", 42.0f, 0.0f, 100.0f)
        .build();

    auto& registry = PropertyRegistry::instance();
    const PropertyMeta* meta = registry.get_property(test_group_id_, "value");
    ASSERT_NE(meta, nullptr);

    // Modify value
    meta->setter(&params, 77.0f);
    EXPECT_FLOAT_EQ(params.value, 77.0f);

    // Restore default from metadata
    float default_val = static_cast<float>(meta->default_value);
    meta->setter(&params, default_val);
    EXPECT_FLOAT_EQ(params.value, 42.0f);
}

// ============================================================================
// OptimizationParameters Registration Tests
// ============================================================================

// Helper function to register optimization properties for tests
// This mirrors what py_params.cpp does
namespace {
    void register_test_optimization_properties() {
        using namespace lfs::core::param;

        PropertyGroupBuilder<OptimizationParameters>("optimization", "Optimization")
            // Training control
            .size_prop(&OptimizationParameters::iterations,
                       "iterations", "Max Iterations", 30000, 1, 1000000,
                       "Maximum number of training iterations")
            .int_prop(&OptimizationParameters::sh_degree,
                      "sh_degree", "SH Degree", 3, 0, 3,
                      "Spherical harmonics degree (0-3)")
            .int_prop(&OptimizationParameters::max_cap,
                      "max_cap", "Max Gaussians", 1000000, 1000, 10000000,
                      "Maximum number of gaussians")

            // Learning rates
            .float_prop(&OptimizationParameters::means_lr,
                        "means_lr", "Position LR", 0.000016f, 0.0f, 0.001f,
                        "Learning rate for gaussian positions")
            .flags(PROP_LIVE_UPDATE)
            .category("learning_rates")
            .float_prop(&OptimizationParameters::shs_lr,
                        "shs_lr", "SH LR", 0.0025f, 0.0f, 0.1f,
                        "Learning rate for spherical harmonics")
            .flags(PROP_LIVE_UPDATE)
            .category("learning_rates")
            .float_prop(&OptimizationParameters::opacity_lr,
                        "opacity_lr", "Opacity LR", 0.05f, 0.0f, 1.0f,
                        "Learning rate for opacity")
            .flags(PROP_LIVE_UPDATE)
            .category("learning_rates")

            // Loss parameters
            .float_prop(&OptimizationParameters::lambda_dssim,
                        "lambda_dssim", "DSSIM Weight", 0.2f, 0.0f, 1.0f,
                        "Weight for structural similarity loss")
            .category("loss")

            // Refinement
            .size_prop(&OptimizationParameters::refine_every,
                       "refine_every", "Refine Every", 100, 1, 1000,
                       "Interval for adaptive density control")
            .category("refinement")
            .float_prop(&OptimizationParameters::min_opacity,
                        "min_opacity", "Min Opacity", 0.005f, 0.0f, 0.1f,
                        "Minimum opacity for pruning")
            .category("refinement")

            // Mask parameters
            .enum_prop(&OptimizationParameters::mask_mode,
                       "mask_mode", "Mask Mode", MaskMode::None,
                       {{"None", MaskMode::None},
                        {"Segment", MaskMode::Segment},
                        {"Ignore", MaskMode::Ignore},
                        {"AlphaConsistent", MaskMode::AlphaConsistent}},
                       "Attention mask behavior during training")
            .category("mask")
            .bool_prop(&OptimizationParameters::invert_masks,
                       "invert_masks", "Invert Masks", false,
                       "Swap object and background in masks")
            .category("mask")

            // Bilateral grid
            .bool_prop(&OptimizationParameters::use_bilateral_grid,
                       "use_bilateral_grid", "Bilateral Grid", false,
                       "Enable bilateral grid color correction")
            .flags(PROP_NEEDS_RESTART)
            .category("bilateral_grid")

            // Strategy
            .string_prop(&OptimizationParameters::strategy,
                         "strategy", "Strategy", "mrnf",
                         "Optimization strategy: mcmc, mrnf, or igs+")
            .flags(PROP_NEEDS_RESTART)

            // Headless (read-only)
            .bool_prop(&OptimizationParameters::headless,
                       "headless", "Headless", false,
                       "Run without visualization")
            .flags(PROP_READONLY)

            .build();
    }
} // anonymous namespace

class OptimizationParamsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register properties once (idempotent due to map overwrite)
        static bool registered = false;
        if (!registered) {
            register_test_optimization_properties();
            registered = true;
        }
    }
};

TEST_F(OptimizationParamsTest, GroupRegistered) {
    auto& registry = PropertyRegistry::instance();
    const PropertyGroup* group = registry.get_group("optimization");
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(group->id, "optimization");
    EXPECT_EQ(group->name, "Optimization");
    EXPECT_GT(group->properties.size(), 10);
}

TEST_F(OptimizationParamsTest, IterationsProperty) {
    auto& registry = PropertyRegistry::instance();
    const PropertyMeta* meta = registry.get_property("optimization", "iterations");
    ASSERT_NE(meta, nullptr);

    EXPECT_EQ(meta->type, PropType::SizeT);
    EXPECT_DOUBLE_EQ(meta->min_value, 1.0);
    EXPECT_DOUBLE_EQ(meta->max_value, 1000000.0);
    EXPECT_NEAR(meta->default_value, 30000.0, 0.001);

    // Test getter/setter
    lfs::core::param::OptimizationParameters params;
    params.iterations = 50000;

    std::any val = meta->getter(&params);
    EXPECT_EQ(std::any_cast<size_t>(val), 50000);

    meta->setter(&params, size_t{75000});
    EXPECT_EQ(params.iterations, 75000);
}

TEST_F(OptimizationParamsTest, LearningRateProperties) {
    auto& registry = PropertyRegistry::instance();

    // means_lr
    const PropertyMeta* means_lr = registry.get_property("optimization", "means_lr");
    ASSERT_NE(means_lr, nullptr);
    EXPECT_EQ(means_lr->type, PropType::Float);
    EXPECT_TRUE(means_lr->has_flag(PROP_LIVE_UPDATE));
    EXPECT_EQ(means_lr->group, "learning_rates");

    // shs_lr
    const PropertyMeta* shs_lr = registry.get_property("optimization", "shs_lr");
    ASSERT_NE(shs_lr, nullptr);
    EXPECT_TRUE(shs_lr->has_flag(PROP_LIVE_UPDATE));

    // opacity_lr
    const PropertyMeta* opacity_lr = registry.get_property("optimization", "opacity_lr");
    ASSERT_NE(opacity_lr, nullptr);
    EXPECT_TRUE(opacity_lr->has_flag(PROP_LIVE_UPDATE));

    // Test setter
    lfs::core::param::OptimizationParameters params;
    means_lr->setter(&params, 0.0001f);
    EXPECT_FLOAT_EQ(params.means_lr, 0.0001f);
}

TEST_F(OptimizationParamsTest, EnumMaskMode) {
    auto& registry = PropertyRegistry::instance();
    const PropertyMeta* meta = registry.get_property("optimization", "mask_mode");
    ASSERT_NE(meta, nullptr);

    EXPECT_EQ(meta->type, PropType::Enum);
    EXPECT_EQ(meta->ui_hint, PropUIHint::Combo);
    EXPECT_EQ(meta->group, "mask");
    ASSERT_EQ(meta->enum_items.size(), 4);

    // Check enum items
    EXPECT_EQ(meta->enum_items[0].name, "None");
    EXPECT_EQ(meta->enum_items[1].name, "Segment");
    EXPECT_EQ(meta->enum_items[2].name, "Ignore");
    EXPECT_EQ(meta->enum_items[3].name, "AlphaConsistent");

    // Test getter/setter
    lfs::core::param::OptimizationParameters params;
    params.mask_mode = lfs::core::param::MaskMode::Ignore;

    std::any val = meta->getter(&params);
    EXPECT_EQ(std::any_cast<int>(val), static_cast<int>(lfs::core::param::MaskMode::Ignore));

    meta->setter(&params, static_cast<int>(lfs::core::param::MaskMode::Segment));
    EXPECT_EQ(params.mask_mode, lfs::core::param::MaskMode::Segment);
}

TEST_F(OptimizationParamsTest, BooleanProperties) {
    auto& registry = PropertyRegistry::instance();

    const PropertyMeta* invert_masks = registry.get_property("optimization", "invert_masks");
    ASSERT_NE(invert_masks, nullptr);
    EXPECT_EQ(invert_masks->type, PropType::Bool);
    EXPECT_EQ(invert_masks->group, "mask");

    const PropertyMeta* bilateral = registry.get_property("optimization", "use_bilateral_grid");
    ASSERT_NE(bilateral, nullptr);
    EXPECT_TRUE(bilateral->has_flag(PROP_NEEDS_RESTART));

    // Test getter/setter
    lfs::core::param::OptimizationParameters params;
    invert_masks->setter(&params, true);
    EXPECT_TRUE(params.invert_masks);

    std::any val = invert_masks->getter(&params);
    EXPECT_TRUE(std::any_cast<bool>(val));
}

TEST_F(OptimizationParamsTest, StringProperty) {
    auto& registry = PropertyRegistry::instance();
    const PropertyMeta* meta = registry.get_property("optimization", "strategy");
    ASSERT_NE(meta, nullptr);

    EXPECT_EQ(meta->type, PropType::String);
    EXPECT_EQ(meta->default_string, "mrnf");
    EXPECT_TRUE(meta->has_flag(PROP_NEEDS_RESTART));

    lfs::core::param::OptimizationParameters params;
    params.strategy = "igs+";

    std::any val = meta->getter(&params);
    EXPECT_EQ(std::any_cast<std::string>(val), "igs+");

    meta->setter(&params, std::string("mrnf"));
    EXPECT_EQ(params.strategy, "mrnf");
}

TEST_F(OptimizationParamsTest, ReadOnlyProperty) {
    auto& registry = PropertyRegistry::instance();
    const PropertyMeta* meta = registry.get_property("optimization", "headless");
    ASSERT_NE(meta, nullptr);

    EXPECT_TRUE(meta->is_readonly());
    EXPECT_TRUE(meta->has_flag(PROP_READONLY));

    // Getter should still work
    lfs::core::param::OptimizationParameters params;
    params.headless = true;
    std::any val = meta->getter(&params);
    EXPECT_TRUE(std::any_cast<bool>(val));

    // Setter should also work at C++ level (readonly is enforced at Python level)
    meta->setter(&params, false);
    EXPECT_FALSE(params.headless);
}

TEST_F(OptimizationParamsTest, PropertyCategories) {
    auto& registry = PropertyRegistry::instance();
    const PropertyGroup* group = registry.get_group("optimization");
    ASSERT_NE(group, nullptr);

    std::set<std::string> categories;
    for (const auto& prop : group->properties) {
        if (!prop.group.empty()) {
            categories.insert(prop.group);
        }
    }

    // Should have these categories
    EXPECT_TRUE(categories.count("learning_rates") > 0);
    EXPECT_TRUE(categories.count("loss") > 0);
    EXPECT_TRUE(categories.count("refinement") > 0);
    EXPECT_TRUE(categories.count("mask") > 0);
    EXPECT_TRUE(categories.count("bilateral_grid") > 0);
}

TEST_F(OptimizationParamsTest, PropertyCallback) {
    auto& registry = PropertyRegistry::instance();

    std::string changed_prop;
    float old_val = 0, new_val = 0;

    size_t sub_id = registry.subscribe(
        "optimization", "means_lr",
        [&](const std::string& grp, const std::string& prop,
            const std::any& ov, const std::any& nv) {
            changed_prop = prop;
            old_val = std::any_cast<float>(ov);
            new_val = std::any_cast<float>(nv);
        });

    // Simulate property change notification
    registry.notify("optimization", "means_lr", 0.000016f, 0.0001f);

    EXPECT_EQ(changed_prop, "means_lr");
    EXPECT_FLOAT_EQ(old_val, 0.000016f);
    EXPECT_FLOAT_EQ(new_val, 0.0001f);

    registry.unsubscribe(sub_id);
}

TEST_F(OptimizationParamsTest, UnknownPropertyReturnsNull) {
    auto& registry = PropertyRegistry::instance();

    const PropertyMeta* meta = registry.get_property("optimization", "nonexistent_property");
    EXPECT_EQ(meta, nullptr);

    const PropertyMeta* meta2 = registry.get_property("nonexistent_group", "iterations");
    EXPECT_EQ(meta2, nullptr);
}
