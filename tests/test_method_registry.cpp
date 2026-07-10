/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/training/method_registry.hpp"

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

namespace {

    using namespace std::chrono_literals;

    class DummySession final : public lfs::vis::IMethodSession {
    public:
        std::expected<void, std::string> initialize(
            const lfs::vis::MethodCreateContext&) override {
            ++initialize_calls;
            return {};
        }

        void run(const std::stop_token stop_token) override {
            ++run_calls;
            while (!stop_token.stop_requested()) {
                if (pause_requested.load()) {
                    paused.store(true);
                    std::this_thread::sleep_for(1ms);
                    continue;
                }
                paused.store(false);
                ++iteration;
                std::this_thread::sleep_for(1ms);
            }
            paused.store(false);
        }

        void request_pause() override {
            ++pause_calls;
            pause_requested.store(true);
        }

        void request_resume() override {
            ++resume_calls;
            pause_requested.store(false);
        }

        bool is_paused() const override { return paused.load(); }

        lfs::vis::MethodStatus status() const override {
            return {
                .iteration = iteration.load(),
                .total_iterations = 100,
                .loss = 0.5F,
                .primitive_count = 12,
                .phase = paused.load() ? "paused" : "training",
            };
        }

        void shutdown() override { ++shutdown_calls; }

        std::atomic<int> initialize_calls{0};
        std::atomic<int> run_calls{0};
        std::atomic<int> pause_calls{0};
        std::atomic<int> resume_calls{0};
        std::atomic<int> shutdown_calls{0};

    private:
        std::atomic<bool> pause_requested{false};
        std::atomic<bool> paused{false};
        std::atomic<int> iteration{0};
    };

    lfs::vis::MethodDescriptor dummy_descriptor(const std::string& id = "dummy") {
        return {
            .id = id,
            .display_name = "Dummy Method",
            .primitive_noun = "primitives",
            .capabilities = {},
            .options = {},
            .version = "1.0",
        };
    }

    lfs::vis::MethodDescriptor options_descriptor() {
        using Type = lfs::vis::MethodOptionSpec::Type;
        return {
            .id = "options",
            .display_name = "Option Test",
            .primitive_noun = "primitives",
            .capabilities = {},
            .options = {
                {.key = "enabled", .display_name = "Enabled", .type = Type::Bool, .default_value = false},
                {.key = "count", .display_name = "Count", .type = Type::Int, .default_value = std::int64_t{3}, .min_value = 1.0, .max_value = 8.0},
                {.key = "rate", .display_name = "Rate", .type = Type::Float, .default_value = 0.25, .min_value = 0.0, .max_value = 1.0},
                {.key = "label", .display_name = "Label", .type = Type::String, .default_value = std::string{"default"}},
                {.key = "path", .display_name = "Path", .type = Type::Path, .default_value = std::string{"."}},
                {.key = "mode", .display_name = "Mode", .type = Type::Enum, .default_value = std::string{"fast"}, .choices = {"fast", "quality"}},
            },
            .version = "1.0",
        };
    }

} // namespace

TEST(MethodRegistry, RegistersCreatesListsAndDescribesMethods) {
    lfs::vis::MethodRegistry registry;
    DummySession* created_session = nullptr;

    EXPECT_TRUE(registry.register_method(dummy_descriptor(), [&created_session] {
        auto session = std::make_unique<DummySession>();
        created_session = session.get();
        return session;
    }));
    EXPECT_TRUE(registry.has("dummy"));

    const auto descriptor = registry.descriptor("dummy");
    ASSERT_TRUE(descriptor.has_value());
    EXPECT_EQ(descriptor->display_name, "Dummy Method");

    const auto descriptors = registry.list();
    ASSERT_EQ(descriptors.size(), 1U);
    EXPECT_EQ(descriptors.front().id, "dummy");

    auto session = registry.create("dummy");
    ASSERT_TRUE(session.has_value()) << session.error();
    EXPECT_EQ(session->get(), created_session);
}

TEST(MethodRegistry, RejectsDuplicateRegistration) {
    lfs::vis::MethodRegistry registry;
    const auto factory = [] { return std::make_unique<DummySession>(); };

    EXPECT_TRUE(registry.register_method(dummy_descriptor(), factory));
    EXPECT_FALSE(registry.register_method(dummy_descriptor(), factory));
    EXPECT_EQ(registry.list().size(), 1U);
}

TEST(MethodRegistry, UnknownMethodErrorListsBuiltInAndRegisteredMethods) {
    lfs::vis::MethodRegistry registry;
    ASSERT_TRUE(registry.register_method(
        dummy_descriptor("alpha"), [] { return std::make_unique<DummySession>(); }));

    const auto result = registry.create("missing");
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("missing"), std::string::npos);
    EXPECT_NE(result.error().find("alpha"), std::string::npos);
    EXPECT_NE(result.error().find("3dgs"), std::string::npos);
}

TEST(MethodRegistry, DummySessionHonorsControlAndStop) {
    DummySession session;
    std::jthread worker([&session](const std::stop_token stop_token) {
        session.run(stop_token);
    });

    while (session.status().iteration == 0) {
        std::this_thread::yield();
    }
    session.request_pause();
    while (!session.is_paused()) {
        std::this_thread::yield();
    }
    session.request_resume();
    worker.request_stop();
    worker.join();

    EXPECT_EQ(session.run_calls.load(), 1);
    EXPECT_EQ(session.pause_calls.load(), 1);
    EXPECT_EQ(session.resume_calls.load(), 1);
}

TEST(MethodOptions, AppliesDefaults) {
    const auto resolved = lfs::vis::resolve_method_options(options_descriptor(), {});
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_EQ(std::get<bool>(resolved->at("enabled")), false);
    EXPECT_EQ(std::get<std::int64_t>(resolved->at("count")), 3);
    EXPECT_DOUBLE_EQ(std::get<double>(resolved->at("rate")), 0.25);
    EXPECT_EQ(std::get<std::string>(resolved->at("mode")), "fast");
}

TEST(MethodOptions, CoercesDeclaredTypesAndPassesPathsThrough) {
    const auto resolved = lfs::vis::resolve_method_options(
        options_descriptor(),
        {{"enabled", "1"},
         {"count", "7"},
         {"rate", "0.75"},
         {"label", "hello"},
         {"path", "folder/value=kept"},
         {"mode", "quality"}});
    ASSERT_TRUE(resolved.has_value()) << resolved.error();
    EXPECT_TRUE(std::get<bool>(resolved->at("enabled")));
    EXPECT_EQ(std::get<std::int64_t>(resolved->at("count")), 7);
    EXPECT_DOUBLE_EQ(std::get<double>(resolved->at("rate")), 0.75);
    EXPECT_EQ(std::get<std::string>(resolved->at("label")), "hello");
    EXPECT_EQ(std::get<std::string>(resolved->at("path")), "folder/value=kept");
    EXPECT_EQ(std::get<std::string>(resolved->at("mode")), "quality");
}

TEST(MethodOptions, CoercesAllBooleanSpellings) {
    for (const auto* raw_value : {"true", "1"}) {
        const auto resolved = lfs::vis::resolve_method_options(
            options_descriptor(), {{"enabled", raw_value}});
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        EXPECT_TRUE(std::get<bool>(resolved->at("enabled")));
    }

    for (const auto* raw_value : {"false", "0"}) {
        const auto resolved = lfs::vis::resolve_method_options(
            options_descriptor(), {{"enabled", raw_value}});
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        EXPECT_FALSE(std::get<bool>(resolved->at("enabled")));
    }
}

TEST(MethodOptions, RejectsIntegerAndFloatRangeViolations) {
    const auto bad_integer = lfs::vis::resolve_method_options(
        options_descriptor(), {{"count", "9"}});
    ASSERT_FALSE(bad_integer.has_value());
    EXPECT_NE(bad_integer.error().find("count"), std::string::npos);

    const auto bad_float = lfs::vis::resolve_method_options(
        options_descriptor(), {{"rate", "-0.1"}});
    ASSERT_FALSE(bad_float.has_value());
    EXPECT_NE(bad_float.error().find("rate"), std::string::npos);
}

TEST(MethodOptions, RejectsUnknownKeysAndListsValidKeys) {
    const auto resolved = lfs::vis::resolve_method_options(
        options_descriptor(), {{"unknown", "value"}});
    ASSERT_FALSE(resolved.has_value());
    EXPECT_NE(resolved.error().find("unknown"), std::string::npos);
    EXPECT_NE(resolved.error().find("enabled"), std::string::npos);
    EXPECT_NE(resolved.error().find("mode"), std::string::npos);
}

TEST(MethodOptions, RejectsInvalidEnumChoice) {
    const auto resolved = lfs::vis::resolve_method_options(
        options_descriptor(), {{"mode", "slow"}});
    ASSERT_FALSE(resolved.has_value());
    EXPECT_NE(resolved.error().find("fast"), std::string::npos);
    EXPECT_NE(resolved.error().find("quality"), std::string::npos);
}
