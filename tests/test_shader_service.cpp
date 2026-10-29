/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "user_shader_test_access.hpp"

// The shader service with a scheduled fake linker: which composition's
// programs draws should run while links land late, fail or go stale,
// bindings across node removal and undo, independent export leases,
// and the effect state of a generated scene. The Slang linker is checked
// once against the shipped package.

#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/editor/python_editor.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/reveal_render_state.hpp"
#include "shaders/shader_service.hpp"

#include "custom_effect_hosts.hpp"
#include "user_shader_linker.hpp"
#include "user_shader_sources.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    namespace shaders = lfs::vis::shaders;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    using lfs::core::Uuid;
    using lfs::rendering::GaussianCustomEffectState;
    using shaders::ShaderPrograms;
    using shaders::ShaderService;
    using shaders::ShaderSlot;
    using shaders::ShaderState;
    using Kind = ShaderSlot::Kind;
    using namespace std::chrono_literals;

    constexpr char kHost[] = "projection_forward_custom";
    constexpr char kGutHost[] = "projection_forward_3dgut_custom";
    constexpr char kQuantHost[] = "projection_forward_quant_custom";
    constexpr char kSurvivorsHost[] = "projection_forward_survivors_custom";
    using NodeBindings = std::unordered_map<Uuid, std::uint32_t>;

    using lfs::test::Schedule;
    using FakeLinker = lfs::test::ScheduledLinker;

    std::unique_ptr<SplatData> splat(const float x) {
        constexpr std::size_t n = 4;
        std::vector<float> means, rotations;
        for (std::size_t i = 0; i < n; ++i) {
            means.insert(means.end(), {x + 0.1f * static_cast<float>(i), 0.2f * static_cast<float>(i), -4.0f});
            rotations.insert(rotations.end(), {1.0f, 0.0f, 0.0f, 0.0f});
        }
        return std::make_unique<SplatData>(
            1, Tensor::from_vector(means, {n, std::size_t{3}}, Device::CUDA),
            Tensor::zeros({n, std::size_t{1}, std::size_t{3}}, Device::CUDA),
            Tensor::zeros({n, std::size_t{3}, std::size_t{3}}, Device::CUDA),
            Tensor::full({n, std::size_t{3}}, -3.0f, Device::CUDA),
            Tensor::from_vector(rotations, {n, std::size_t{4}}, Device::CUDA),
            Tensor::zeros({n, std::size_t{1}}, Device::CUDA), 1.0f);
    }

    // The effect index of `asset` in the composition of `assets`.
    std::uint32_t effectIndex(std::vector<Uuid> assets, const Uuid& asset) {
        std::ranges::sort(assets, {}, &Uuid::bytes);
        return static_cast<std::uint32_t>(std::ranges::find(assets, asset) - assets.begin());
    }

    lfs::Error renderError(std::string message) {
        return lfs::make_error(lfs::ErrorInit{.code = lfs::ErrorCode::Internal,
                                              .domain = lfs::ErrorDomain::Rendering,
                                              .user_message = std::move(message),
                                              .detection = LFS_SOURCE_SITE_CURRENT()});
    }

    class ShaderServiceTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                GTEST_SKIP() << "CUDA device unavailable";
            lfs::event::EventBridge::instance().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();
            scene_ = std::make_unique<lfs::vis::SceneManager>();
            rendering_ = std::make_unique<lfs::vis::RenderingManager>();
            lfs::vis::services().set(scene_.get());
            lfs::vis::services().set(rendering_.get());
            for (int i = 0; i < 3; ++i) {
                Uuid uuid;
                const auto id = scene().addSplat(std::format("node{}", i), splat(1.5f * static_cast<float>(i - 1)),
                                                 lfs::core::NULL_NODE, &uuid);
                ASSERT_NE(id, lfs::core::NULL_NODE);
                nodes_.push_back(uuid);
            }
            group_ = scene().getNodeById(scene().addGroup("group"))->uuid;
            scene_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
            service_ = std::make_unique<ShaderService>(
                scene(),
                [this] {
                    const std::lock_guard lock(schedule_.mutex);
                    ++schedule_.linkers;
                    return std::make_unique<FakeLinker>(schedule_);
                },
                [this] { wakes_.fetch_add(1); });
        }
        void TearDown() override {
            schedule_.gate(false);
            service_.reset();
            lfs::vis::op::undoHistory().clear();
            lfs::vis::services().clear();
            lfs::event::EventBridge::instance().clear_all();
            rendering_.reset();
            scene_.reset();
        }

        lfs::core::Scene& scene() { return scene_->getScene(); }
        ShaderService& service() { return *service_; }
        lfs::vis::ShaderCommands commands() { return {service()}; }

        Uuid create(const std::string& source = "effect") {
            auto asset = commands().createAsset("Effect", source);
            EXPECT_TRUE(asset) << lfs::format_for_developer(asset.error());
            return asset ? *asset : Uuid{};
        }
        Uuid requireGlobal(const std::string& source = "effect") {
            const auto asset = create(source);
            bind({.kind = Kind::Global}, asset);
            demand(kHost);
            return asset;
        }
        void bind(const ShaderSlot& slot, const Uuid& asset) {
            const auto bound = commands().bind(slot, asset);
            ASSERT_TRUE(bound) << lfs::format_for_developer(bound.error());
        }
        std::vector<Uuid> capacityNodes(const std::size_t count) {
            auto nodes = nodes_;
            while (nodes.size() < count) {
                Uuid uuid;
                EXPECT_NE(scene().addSplat(std::format("capacity{}", nodes.size()), splat(0.0f),
                                           lfs::core::NULL_NODE, &uuid),
                          lfs::core::NULL_NODE);
                nodes.push_back(uuid);
            }
            return nodes;
        }
        std::uint64_t links(const Uuid& asset) {
            const auto status = service().status(asset);
            return status ? status->links : 0;
        }
        // Waits until the worker returned `count` links with `asset` in total.
        bool waitLinks(const Uuid& asset, const std::uint64_t count) {
            const auto deadline = std::chrono::steady_clock::now() + 5s;
            while (links(asset) < count) {
                if (std::chrono::steady_clock::now() > deadline)
                    return false;
                std::this_thread::sleep_for(1ms);
            }
            return true;
        }
        bool waitIdle() {
            const auto changes = service().changes();
            const auto deadline = std::chrono::steady_clock::now() + 5s;
            do {
                const auto generation = service().generation();
                if (service().work().work.empty())
                    return true;
                changes->wait(generation, deadline);
            } while (std::chrono::steady_clock::now() < deadline);
            return false;
        }
        ShaderState state(const Uuid& asset) {
            auto status = service().status(asset);
            EXPECT_TRUE(status);
            return status ? status->state : ShaderState::Unbound;
        }
        void demand(const std::string& host) {
            if (!std::ranges::contains(hosts_, host))
                hosts_.push_back(host);
            service().demandLive(hosts_, display_ ? display_->snapshot : nullptr);
        }
        std::shared_ptr<const ShaderPrograms> readyPrograms() {
            auto snapshot = service().candidate();
            if (!snapshot && display_)
                snapshot = service().restrictFallback(display_->snapshot);
            return snapshot ? snapshot->programs : nullptr;
        }
        std::shared_ptr<const ShaderPrograms> show() {
            auto snapshot = service().candidate();
            if (!snapshot && display_)
                snapshot = service().restrictFallback(display_->snapshot);
            display_ = std::make_shared<const shaders::DisplayRecord>(shaders::DisplayRecord{snapshot, bool(snapshot)});
            service().observeDisplay(display_);
            return snapshot ? snapshot->programs : nullptr;
        }
        std::vector<std::string> hosts_;
        std::shared_ptr<const shaders::DisplayRecord> display_;

        Schedule schedule_;
        std::atomic<int> wakes_ = 0;
        std::unique_ptr<lfs::vis::SceneManager> scene_;
        std::unique_ptr<lfs::vis::RenderingManager> rendering_;
        std::vector<Uuid> nodes_;
        Uuid group_;
        std::unique_ptr<ShaderService> service_;
    };

    TEST_F(ShaderServiceTest, CommandsValidateBeforeMutationAndUndoWithMonotonicRevisions) {
        auto created = service().command(shaders::SetShaderSource{{}, "first", "Effect", {}});
        ASSERT_TRUE(created);
        const auto id = *created;
        const auto original = service().assets();
        const std::pair<const char*, shaders::SetShaderSource> rejected[] = {
            {"stale revision", {id, "second", "Renamed", 0}},
            {"empty name", {id, "second", "", {}}},
            {"source cap", {id, std::string(shaders::kMaxShaderSourceBytes + 1, 'x'), {}, {}}}};
        for (const auto& [name, command] : rejected) {
            SCOPED_TRACE(name);
            EXPECT_FALSE(service().command(command));
            EXPECT_EQ(service().assets()[0].source, original[0].source);
            EXPECT_EQ(service().assets()[0].name, original[0].name);
            EXPECT_EQ(service().assets()[0].revision, original[0].revision);
        }
        ASSERT_TRUE(service().command(shaders::SetShaderSource{id, "second", "Renamed", 1}));
        auto& history = lfs::vis::op::undoHistory();
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(service().assets()[0].source, "first");
        EXPECT_EQ(service().assets()[0].revision, 3);
        ASSERT_TRUE(history.redo().success);
        EXPECT_EQ(service().assets()[0].source, "second");
        EXPECT_EQ(service().assets()[0].revision, 4);
        ASSERT_TRUE(service().command(shaders::RemoveShader{id}));
        EXPECT_TRUE(service().assets().empty());
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(service().assets()[0].id, id);
        EXPECT_EQ(service().assets()[0].revision, 5);
        service().clear();
        EXPECT_FALSE(history.redo().success);
        EXPECT_TRUE(service().assets().empty());
    }

    TEST_F(ShaderServiceTest, RunCommandValidatesItsTargetBeforeSourceAndUndoesAsOneEdit) {
        const auto previous = create("previous");
        bind({}, previous);
        EXPECT_FALSE(service().command(shaders::RunShader{previous, "changed", {Kind::Node, group_}}));
        EXPECT_FALSE(service().command(shaders::RunShader{{}, "new", {Kind::Node, group_}}));
        ASSERT_EQ(service().assets().size(), 1u);
        EXPECT_EQ(service().assets()[0].source, "previous");
        EXPECT_EQ(service().assets()[0].revision, 1u);
        const auto run = service().command(shaders::RunShader{{}, "new", {}});
        ASSERT_TRUE(run);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(service().boundAsset({}), previous);
        EXPECT_FALSE(service().status(*run));
        ASSERT_TRUE(lfs::vis::op::undoHistory().redo().success);
        EXPECT_EQ(service().boundAsset({}), *run);
        EXPECT_EQ(service().status(*run)->revision, 2u);
        EXPECT_FALSE(service().command(shaders::UnbindShader{{}, previous}));
        ASSERT_TRUE(service().command(shaders::BindShader{{}, previous}));
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(service().boundAsset({}), *run);
        ASSERT_TRUE(lfs::vis::op::undoHistory().redo().success);
        EXPECT_EQ(service().boundAsset({}), previous);
        EXPECT_FALSE(service().command(shaders::RemoveShader{previous}));
    }

    TEST_F(ShaderServiceTest, InspectionAndEventsDescribeDemandedWorkAndRevisionProvenance) {
        std::mutex events_mutex;
        std::vector<shaders::ShaderEvent> events;
        lfs::event::ScopedHandler handlers;
        auto* observed_service = service_.get();
        handlers.subscribe<shaders::ShaderEvent>([&, observed_service](const auto& event) {
            (void)observed_service->inspect(); // event delivery cannot hold the service mutex
            const std::lock_guard lock(events_mutex);
            events.push_back(event);
        });
        schedule_.gate(true);
        const auto id = create();
        bind({}, id);
        EXPECT_TRUE(service().inspect().work.empty());
        demand(kHost);
        ASSERT_TRUE(schedule_.waitStarted(1));
        auto snapshot = service().inspect();
        ASSERT_EQ(snapshot.work.size(), 1u);
        EXPECT_TRUE(snapshot.work[0].compiling);
        EXPECT_EQ(snapshot.work[0].members[0].asset, id);
        EXPECT_EQ(snapshot.work[0].members[0].revision, 1u);
        EXPECT_TRUE(snapshot.displayed_bindings.empty());
        schedule_.gate(false);
        ASSERT_TRUE(waitLinks(id, 1));
        show();
        EXPECT_EQ(service().status(id)->schema_revision, 1u);
        ASSERT_EQ(service().inspect().displayed_bindings.size(), 1u);
        bind({Kind::Node, nodes_[0]}, id);
        EXPECT_EQ(service().inspect().bindings.size(), 2u);
        EXPECT_EQ(service().inspect().displayed_bindings.size(), 1u);
        show();
        EXPECT_EQ(service().inspect().displayed_bindings.size(), 2u);
        ASSERT_TRUE(service().command(shaders::SetShaderParameter{id, "gain", {1.5f}, 1}));
        EXPECT_EQ(service().status(id)->links, 1u);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(service().status(id)->values[0][0], 1.f);
        EXPECT_EQ(service().status(id)->links, 1u);
        ASSERT_TRUE(service().command(shaders::SetShaderSource{id, "broken", {}, 1}));
        ASSERT_TRUE(waitLinks(id, 2));
        const auto failed = service().status(id);
        ASSERT_TRUE(failed);
        EXPECT_EQ(failed->schema_revision, 1u);
        EXPECT_EQ(failed->diagnostics_revision, 2u);
        EXPECT_EQ(failed->displayed_revision, 1u);
        EXPECT_EQ(snapshot.work[0].members[0].revision, 1u); // copied, not a live view
        // Drain the worker before releasing the event observer's captures.
        service_->stop();
        service_.reset();
        const std::lock_guard lock(events_mutex);
        EXPECT_TRUE(
            std::ranges::any_of(events, [](const auto& e) { return e.kind == shaders::ShaderEvent::Kind::Started; }));
        EXPECT_TRUE(
            std::ranges::any_of(events, [](const auto& e) { return e.kind == shaders::ShaderEvent::Kind::Completed; }));
        EXPECT_TRUE(std::ranges::any_of(events, [](const auto& e) {
            return e.kind == shaders::ShaderEvent::Kind::Failed && e.members[0].revision == 2;
        }));
        EXPECT_TRUE(
            std::ranges::any_of(events, [](const auto& e) { return e.kind == shaders::ShaderEvent::Kind::Applied; }));
    }

    TEST_F(ShaderServiceTest, NothingLinksBeforeADrawNeedsAHost) {
        const std::size_t threads = lfs::test::threadCount();
        const auto asset = create();
        bind({.kind = Kind::Global}, asset);
        // A bound asset alone starts no worker and no compiler.
        EXPECT_TRUE(lfs::test::threadsSettleAt(threads));
        EXPECT_EQ(schedule_.linkers, 0);
        EXPECT_EQ(links(asset), 0u);
        EXPECT_EQ(state(asset), ShaderState::Pending);
        EXPECT_EQ(readyPrograms(), nullptr);
        EXPECT_EQ(service().candidate(), nullptr);

        demand(kHost);
        ASSERT_TRUE(waitLinks(asset, 1));
        EXPECT_EQ(schedule_.linkers, 1);
        if (threads != 0)
            EXPECT_TRUE(lfs::test::threadsSettleAt(threads + 1));
        const auto programs = readyPrograms();
        ASSERT_NE(programs, nullptr);
        ASSERT_EQ(programs->hosts.size(), 1u);
        EXPECT_EQ(programs->hosts[0]->label, kHost);
        // Linked is not displayed.
        EXPECT_EQ(state(asset), ShaderState::Pending);
        show();
        EXPECT_EQ(state(asset), ShaderState::Applied);
        EXPECT_EQ(service().status(asset)->displayed_revision, 1u);
    }

    TEST_F(ShaderServiceTest, ResultsForDroppedBindingsAndProjectsNeverApply) {
        const auto a = create("a");
        const auto b = create("b");
        bind({.kind = Kind::Global}, a);
        schedule_.gate(true);
        demand(kHost);
        ASSERT_TRUE(schedule_.waitStarted(1));
        // `a` is unbound while it links; its result must not draw for `b`.
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        const auto work = service().inspect().work;
        ASSERT_EQ(work.size(), 1u);
        EXPECT_TRUE(work[0].compiling);
        EXPECT_EQ(work[0].members[0].asset, a);
        EXPECT_EQ(state(a), ShaderState::Unbound);
        bind({.kind = Kind::Global}, b);
        demand(kHost);
        schedule_.release();
        ASSERT_TRUE(schedule_.waitStarted(2));
        EXPECT_EQ(readyPrograms(), nullptr);
        schedule_.release();
        ASSERT_TRUE(waitLinks(b, 1));
        EXPECT_EQ(links(a), 0u);
        EXPECT_TRUE(service().inspect().work.empty());
        const auto programs = show();
        ASSERT_NE(programs, nullptr);
        EXPECT_EQ(programs->effects[0].asset, b);

        // A project switch returns while a link runs, and drops its result.
        ASSERT_TRUE(commands().updateSource(b, "b2"));
        ASSERT_TRUE(schedule_.waitStarted(3));
        auto cleared = std::async(std::launch::async, [this] { service().clear(); });
        ASSERT_EQ(cleared.wait_for(5s), std::future_status::ready);
        EXPECT_EQ(readyPrograms(), nullptr);
        EXPECT_EQ(service().candidate(), nullptr);
        EXPECT_FALSE(service().status(b));
        schedule_.gate(false);
        // The worker goes on with the next project.
        const auto c = create("c");
        bind({.kind = Kind::Global}, c);
        demand(kHost);
        ASSERT_TRUE(waitLinks(c, 1));
        ASSERT_NE(readyPrograms(), nullptr);
        EXPECT_EQ(readyPrograms()->effects[0].asset, c);
    }

    TEST_F(ShaderServiceTest, ParameterChangesNeverLink) {
        const auto asset = requireGlobal();
        ASSERT_TRUE(waitLinks(asset, 1));
        show();
        const auto before = service().candidate();
        ASSERT_EQ(before->bindings.size(), 1u);
        EXPECT_EQ(before->bindings[0].values[0], glm::vec4(1.0f));
        EXPECT_EQ(before->bindings[0].values[1], glm::vec4(0.5f));
        const int woken = wakes_.load();
        for (int i = 0; i < 50; ++i) {
            const float gain = 0.02f * static_cast<float>(i);
            ASSERT_TRUE(commands().setParameter(asset, "gain", std::span(&gain, 1)));
        }
        const std::array tint{0.1f, 0.2f, 0.3f};
        ASSERT_TRUE(commands().setParameter(asset, "tint", tint));
        EXPECT_EQ(links(asset), 1u);
        EXPECT_EQ(schedule_.started.size(), 1u);
        // Every change asks the viewer for a frame.
        EXPECT_GE(wakes_.load(), woken + 51);
        const auto after = service().candidate();
        EXPECT_NE(after, before);
        EXPECT_FLOAT_EQ(after->bindings[0].values[0].x, 0.98f);
        EXPECT_EQ(glm::vec3(after->bindings[0].values[1]), glm::vec3(0.1f, 0.2f, 0.3f));
        EXPECT_EQ(glm::vec3(service().status(asset)->values[1]), glm::vec3(0.1f, 0.2f, 0.3f));

        const float out_of_range = 3.0f;
        EXPECT_EQ(commands().setParameter(asset, "gain", std::span(&out_of_range, 1)).error().code(),
                  lfs::ErrorCode::InvalidArgument);
        EXPECT_EQ(commands().setParameter(asset, "tint", std::span(&out_of_range, 1)).error().code(),
                  lfs::ErrorCode::InvalidArgument);
        EXPECT_EQ(commands().setParameter(asset, "speed", std::span(&out_of_range, 1)).error().code(),
                  lfs::ErrorCode::NotFound);
        EXPECT_EQ(links(asset), 1u);
        // Undo stores named authoring values, never a cached reflected Asset.
        auto& history = lfs::vis::op::undoHistory();
        history.clear();
        ASSERT_TRUE(service().command(shaders::SetShaderParameter{asset, "gain", {.25f}, {}}));
        ASSERT_TRUE(service().command(shaders::SetShaderSource{asset, "reordered", {}, {}}));
        ASSERT_TRUE(waitIdle());
        ASSERT_EQ(service().status(asset)->parameters.front().name, "tint");
        EXPECT_FLOAT_EQ(service().status(asset)->values[1].x, .25f);
        schedule_.gate(true);
        ASSERT_TRUE(history.undo().success); // source back, compiler still has the new schema
        ASSERT_TRUE(history.undo().success); // parameter undo cannot overwrite that schema
        EXPECT_EQ(service().status(asset)->parameters.front().name, "tint");
        EXPECT_FLOAT_EQ(service().status(asset)->values[1].x, .98f);
        schedule_.gate(false);
        ASSERT_TRUE(waitIdle());
        EXPECT_EQ(service().status(asset)->parameters.front().name, "gain");
        EXPECT_FLOAT_EQ(service().status(asset)->values[0].x, .98f);
        ASSERT_TRUE(history.redo().success);
        ASSERT_TRUE(history.redo().success);
        ASSERT_TRUE(waitIdle());
        EXPECT_EQ(service().status(asset)->parameters.front().name, "tint");
        EXPECT_FLOAT_EQ(service().status(asset)->values[1].x, .25f);
    }

    TEST_F(ShaderServiceTest, EditsLinkOnlyTheFamiliesFramesStillDraw) {
        const auto asset = requireGlobal("v1");
        ASSERT_TRUE(waitLinks(asset, 1));
        show();
        demand(kGutHost);
        ASSERT_TRUE(waitLinks(asset, 2));
        ASSERT_EQ(show()->hosts.size(), 2u);

        // 3DGUT toggled off: frames draw only the default family, so an
        // edit links that one and publishes it alone.
        service().demandLive(std::vector<std::string>{kHost});
        ASSERT_TRUE(commands().updateSource(asset, "v2"));
        ASSERT_TRUE(waitLinks(asset, 3));
        const auto edited = show();
        ASSERT_NE(edited, nullptr);
        EXPECT_EQ(edited->effects[0].revision, 2u);
        ASSERT_EQ(edited->hosts.size(), 1u);
        EXPECT_EQ(edited->hosts[0]->label, kHost);
        EXPECT_EQ(schedule_.started.back(), std::string("v2@") + kHost);
        EXPECT_EQ(schedule_.started.size(), 3u);

        // Toggled on again, the family joins the edited set on first use.
        demand(kGutHost);
        ASSERT_TRUE(waitIdle());
        EXPECT_EQ(readyPrograms()->hosts.size(), 2u);
        EXPECT_EQ(links(asset), 4u);
    }

    TEST_F(ShaderServiceTest, BindingsComposeValidateAndKeepNamedParameters) {
        const auto a = create();
        const auto b = create();
        EXPECT_EQ(commands().createAsset("", "x").error().code(), lfs::ErrorCode::InvalidArgument);
        bind({.kind = Kind::Global}, a);
        bind({.kind = Kind::Node, .node = nodes_[1]}, a);
        // One asset in several slots runs one effect, a binding each.
        demand(kHost);
        ASSERT_TRUE(waitIdle());
        auto snapshot = service().candidate();
        ASSERT_NE(snapshot, nullptr);
        ASSERT_EQ(snapshot->bindings.size(), 2u);
        EXPECT_TRUE(std::ranges::all_of(snapshot->bindings, [](const auto& binding) { return binding.effect == 0; }));
        ASSERT_EQ(snapshot->nodes.size(), 1u);
        EXPECT_EQ(snapshot->nodes.begin()->first, nodes_[1]);
        // Another asset joins the composition with an effect of its own.
        bind({.kind = Kind::Node, .node = nodes_[0]}, b);
        ASSERT_TRUE(waitIdle());
        snapshot = service().candidate();
        EXPECT_EQ(snapshot->bindings.size(), 3u);
        ASSERT_EQ(snapshot->nodes.size(), 2u);
        EXPECT_EQ(links(a), 2u);
        EXPECT_EQ(links(b), 1u); // one new host link carries both members
        const float a_gain = .25f, b_gain = 1.5f;
        ASSERT_TRUE(commands().setParameter(a, "gain", std::span(&a_gain, 1)));
        ASSERT_TRUE(commands().setParameter(b, "gain", std::span(&b_gain, 1)));
        const auto c = create("c");
        std::vector<Uuid> members{a, b};
        const std::pair<const char*, std::function<void()>> rows[] = {
            {"two parameter blocks", [] {}},
            {"source edit", [&] { EXPECT_TRUE(commands().updateSource(b, "b2")); }},
            {"third member", [&] { bind({Kind::Global}, c); members.push_back(c); }}};
        for (std::size_t row = 0; row < std::size(rows); ++row) {
            SCOPED_TRACE(rows[row].first);
            rows[row].second();
            ASSERT_TRUE(waitIdle());
            const auto programs = show();
            ASSERT_NE(programs, nullptr);
            ASSERT_EQ(programs->effects.size(), members.size());
            for (const auto& member : members) {
                const auto index = effectIndex(members, member);
                EXPECT_EQ(programs->effects[index].asset, member);
                EXPECT_EQ(programs->effects[index].revision, member == b && row > 0 ? 2u : 1u);
                EXPECT_EQ(state(member), ShaderState::Applied);
            }
            EXPECT_EQ(links(a), row + 2);
            EXPECT_EQ(links(b), row + 1);
            EXPECT_EQ(schedule_.started.size(), row + 2);
            snapshot = service().candidate();
            for (const auto& [binding, member] : std::vector<std::pair<std::uint32_t, Uuid>>{
                     {snapshot->global, row == 2 ? c : a},
                     {snapshot->nodes.at(nodes_[1]), a},
                     {snapshot->nodes.at(nodes_[0]), b}}) {
                EXPECT_EQ(snapshot->bindings[binding].effect, effectIndex(members, member));
                EXPECT_FLOAT_EQ(snapshot->bindings[binding].values[0].x, member == a ? a_gain : member == b ? b_gain
                                                                                                            : 1.f);
            }
        }
        EXPECT_EQ(links(c), 1u);
        EXPECT_EQ(commands().bind({.kind = Kind::Node, .node = group_}, a).error().code(),
                  lfs::ErrorCode::InvalidArgument);
        EXPECT_EQ(commands().bind({.kind = Kind::Node, .node = lfs::core::generate_uuid_v4()}, a).error().code(),
                  lfs::ErrorCode::NotFound);
        EXPECT_EQ(commands().removeAsset(a).error().code(), lfs::ErrorCode::FailedPrecondition);

        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        ASSERT_TRUE(commands().unbind({.kind = Kind::Node, .node = nodes_[1]}));
        EXPECT_EQ(commands().unbind({.kind = Kind::Global}).error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_EQ(state(a), ShaderState::Unbound);
        ASSERT_TRUE(waitIdle());
        ASSERT_NE(service().candidate(), nullptr);
        EXPECT_EQ(service().candidate()->bindings.size(), 1u);
        EXPECT_EQ(service().candidate()->nodes, (NodeBindings{{nodes_[0], 0u}}));
        EXPECT_EQ(service().candidate()->global, shaders::ShaderFrameSnapshot::kNone);
        ASSERT_TRUE(commands().unbind({.kind = Kind::Node, .node = nodes_[0]}));
        EXPECT_EQ(service().candidate(), nullptr);
        ASSERT_TRUE(commands().renameAsset(a, "Renamed"));
        EXPECT_EQ(commands().renameAsset(a, "").error().code(), lfs::ErrorCode::InvalidArgument);
        ASSERT_TRUE(commands().removeAsset(a));
        EXPECT_EQ(service().status(a).error().code(), lfs::ErrorCode::NotFound);
        bind({.kind = Kind::Global}, b);
        EXPECT_GT(links(b), 0u);
    }

    TEST_F(ShaderServiceTest, AssetDiagnosticsAndBlocking) {
        for (const bool displayed : {true, false}) {
            SCOPED_TRACE(displayed ? "displayed fallback" : "new composition blocked");
            service().clear();
            display_.reset();
            const auto good = create("good"), first = create("broken first"), second = create("broken second");
            bind({}, good);
            demand(kHost);
            std::shared_ptr<const ShaderPrograms> shown;
            if (displayed) {
                ASSERT_TRUE(waitIdle());
                shown = show();
                ASSERT_NE(shown, nullptr);
            }
            bind({Kind::Node, nodes_[0]}, first);
            if (!displayed)
                bind({Kind::Node, nodes_[1]}, second);
            ASSERT_TRUE(waitIdle());
            for (const auto& bad : displayed ? std::vector{first} : std::vector{first, second}) {
                const auto status = service().status(bad);
                EXPECT_EQ(status->state, ShaderState::Failed);
                ASSERT_EQ(status->diagnostics.entries.size(), 1u);
                EXPECT_EQ(status->diagnostics.entries[0].file, bad.to_string() + ".slang");
                EXPECT_EQ(status->diagnostics.entries[0].line, 2u);
            }
            auto status = service().status(good);
            EXPECT_EQ(status->state, displayed ? ShaderState::Applied : ShaderState::Blocked);
            EXPECT_TRUE(status->diagnostics.entries.empty());
            if (displayed)
                EXPECT_EQ(readyPrograms(), shown);
            else {
                auto blockers = std::vector{first, second};
                std::ranges::sort(blockers, {}, &Uuid::bytes);
                std::ranges::sort(status->blocked_by, {}, &Uuid::bytes);
                EXPECT_EQ(status->blocked_by, blockers);
            }
            ASSERT_TRUE(commands().updateSource(first, "fixed"));
            demand(kHost);
            ASSERT_TRUE(waitIdle());
            if (displayed) {
                ASSERT_EQ(show()->effects.size(), 2u);
                EXPECT_EQ(state(first), ShaderState::Applied);
                EXPECT_EQ(state(good), ShaderState::Applied);
            } else {
                EXPECT_EQ(state(first), ShaderState::Blocked);
                EXPECT_EQ(state(good), ShaderState::Blocked);
                EXPECT_EQ(service().status(good)->blocked_by, std::vector{second});
                EXPECT_EQ(state(second), ShaderState::Failed);
            }
        }
    }

    TEST_F(ShaderServiceTest, CompilerContractFailuresStayWithTheirAsset) {
        service_ = std::make_unique<ShaderService>(
            scene(), shaders::slangHostLinker(fs::path(LFS_SHADER_PACKAGE_DIR) / "manifest.json"), [] {});
        const auto linked = [&](const Uuid& asset, const std::uint64_t count) {
            const auto deadline = std::chrono::steady_clock::now() + 60s;
            while (links(asset) < count && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(5ms);
            return links(asset) >= count;
        };
        const auto healthy = create(lfs::test::kIdentity);
        const auto broken = create("import lfs;\nLfsElement shade(LfsElement e, LfsContext ctx) { return e; }\n");
        bind({.kind = Kind::Global}, healthy);
        bind({.kind = Kind::Node, .node = nodes_[0]}, broken);
        demand(kHost);
        ASSERT_TRUE(linked(broken, 1));
        auto status = service().status(broken);
        EXPECT_EQ(status->state, ShaderState::Failed);
        ASSERT_EQ(status->diagnostics.entries.size(), 1u);
        EXPECT_EQ(status->diagnostics.entries[0].file, broken.to_string() + ".slang");
        EXPECT_TRUE(status->diagnostics.entries[0].message.starts_with("Missing LfsElement effect")) << status->diagnostics.entries[0].message;
        status = service().status(healthy);
        EXPECT_EQ(status->state, ShaderState::Blocked);
        EXPECT_TRUE(status->diagnostics.entries.empty());
        EXPECT_EQ(status->blocked_by, std::vector{broken});

        // Two broken assets keep a message each.
        ASSERT_TRUE(commands().updateSource(
            healthy, "import lfs;\nRWStructuredBuffer<float> data;\nLfsElement effect(LfsElement e, LfsContext ctx) { return e; }\n"));
        demand(kHost);
        ASSERT_TRUE(linked(healthy, 2));
        for (const auto& asset : {healthy, broken}) {
            status = service().status(asset);
            EXPECT_EQ(status->state, ShaderState::Failed);
            ASSERT_EQ(status->diagnostics.entries.size(), 1u);
            EXPECT_EQ(status->diagnostics.entries[0].file, asset.to_string() + ".slang");
        }
        EXPECT_NE(service().status(healthy)->diagnostics.entries[0].message,
                  service().status(broken)->diagnostics.entries[0].message);
    }

    TEST_F(ShaderServiceTest, BoundShadersAreCapped) {
        const auto nodes = capacityNodes(shaders::kMaxBoundShaders + 1);
        ASSERT_FALSE(HasFailure());
        // The effects of the snapshot's composition.
        const auto effects = [&] {
            demand(kHost);
            EXPECT_TRUE(waitIdle());
            const auto snapshot = service().candidate();
            return std::ranges::max(snapshot->bindings, {}, &GaussianCustomEffectState::Binding::effect).effect + 1;
        };
        std::vector<Uuid> assets;
        for (std::size_t i = 0; i < shaders::kMaxBoundShaders; ++i) {
            assets.push_back(create());
            bind({.kind = Kind::Node, .node = nodes[i]}, assets.back());
        }
        const auto extra = create();
        const auto refused = commands().bind({.kind = Kind::Node, .node = nodes.back()}, extra);
        ASSERT_FALSE(refused);
        EXPECT_EQ(refused.error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_EQ(refused.error().user_message(),
                  std::format("At most {} shaders can be bound at once; unbind one first.", shaders::kMaxBoundShaders));
        // Another slot of a bound asset adds no effect.
        bind({.kind = Kind::Node, .node = nodes.back()}, assets[0]);
        EXPECT_EQ(effects(), shaders::kMaxBoundShaders);

        // A removed node's binding comes back with undo only within the cap.
        scene_->removeNode(scene().getNodeByUuid(nodes[1])->id, false);
        EXPECT_EQ(effects(), shaders::kMaxBoundShaders - 1);
        bind({.kind = Kind::Global}, extra);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        ASSERT_NE(scene().getNodeByUuid(nodes[1]), nullptr);
        EXPECT_EQ(effects(), shaders::kMaxBoundShaders);
        EXPECT_FALSE(std::ranges::contains(service().candidate()->nodes, nodes[1], &NodeBindings::value_type::first));
        // Nor on a later scene change once there is room.
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        lfs::core::events::state::PLYAdded{}.emit();
        EXPECT_EQ(effects(), shaders::kMaxBoundShaders - 1);
        EXPECT_FALSE(std::ranges::contains(service().candidate()->nodes, nodes[1], &NodeBindings::value_type::first));
    }

    TEST_F(ShaderServiceTest, NodeRemovalDropsTheBindingAndUndoRestoresIt) {
        const auto asset = create();
        bind({.kind = Kind::Node, .node = nodes_[1]}, asset);
        scene_->removeNode(scene().getNodeByUuid(nodes_[1])->id, false);
        ASSERT_EQ(scene().getNodeByUuid(nodes_[1]), nullptr);
        EXPECT_EQ(service().candidate(), nullptr);
        EXPECT_EQ(state(asset), ShaderState::Unbound);
        const auto other = create();
        bind({.kind = Kind::Global}, other);

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        ASSERT_NE(scene().getNodeByUuid(nodes_[1]), nullptr);
        // The restored node's binding joins the asset bound meanwhile.
        demand(kHost);
        ASSERT_TRUE(waitIdle());
        const auto snapshot = service().candidate();
        ASSERT_NE(snapshot, nullptr);
        ASSERT_EQ(snapshot->nodes.size(), 1u);
        EXPECT_EQ(snapshot->nodes.begin()->first, nodes_[1]);
        EXPECT_EQ(snapshot->bindings[snapshot->nodes.begin()->second].effect, effectIndex({asset, other}, asset));
        EXPECT_EQ(state(asset), ShaderState::Pending);

        // A detached reference remains protected until its node returns and is unbound.
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        scene_->removeNode(scene().getNodeByUuid(nodes_[1])->id, false);
        EXPECT_FALSE(commands().removeAsset(asset));
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(service().boundAsset({.kind = Kind::Node, .node = nodes_[1]}), asset);
        ASSERT_TRUE(commands().unbind({.kind = Kind::Node, .node = nodes_[1]}));
        ASSERT_TRUE(commands().removeAsset(asset));
        EXPECT_EQ(service().candidate(), nullptr);
    }

    TEST_F(ShaderServiceTest, DocumentStopGuardsMultipleSlotsAndForgetsDetachedNodeUndo) {
        lfs::vis::editor::PythonEditor editor;
        const auto document = editor.newDocument("draft", lfs::vis::editor::DocumentKind::Shader);
        const ShaderSlot first{Kind::Node, nodes_[0]}, second{Kind::Node, nodes_[1]};
        for (const auto slot : {first, second, first}) {
            ASSERT_TRUE(editor.setDocumentTarget(document, slot, service()));
            ASSERT_TRUE(editor.runShader(service()));
        }
        ASSERT_EQ(editor.inspectDocument(document, &service())->owned_bindings.size(), 2u);
        const auto other = create();
        bind(second, other);
        scene_->removeNode(scene().getNodeByUuid(nodes_[0])->id, false);
        (void)editor.pollShaders(service());
        EXPECT_EQ(editor.inspectDocument(document, &service())->owned_bindings, std::vector{first});
        EXPECT_FALSE(service().command(shaders::UnbindShader{first, other}));
        EXPECT_EQ(editor.stopShader(service()), std::vector{first});
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        ASSERT_NE(scene().getNodeByUuid(nodes_[0]), nullptr);
        EXPECT_FALSE(service().boundAsset(first));
        EXPECT_EQ(service().boundAsset(second), other);
        for (const auto slot : {first, second}) {
            ASSERT_TRUE(editor.setDocumentTarget(document, slot, service()));
            ASSERT_TRUE(editor.runShader(service()));
        }
        EXPECT_EQ(editor.stopShader(service()), (std::vector{first, second}));
        EXPECT_FALSE(service().boundAsset(first));
        EXPECT_FALSE(service().boundAsset(second));
    }

    TEST_F(ShaderServiceTest, EffectStateNumbersVisibleSlotsAndBoundsTheOriginalGeometry) {
        GaussianCustomEffectState::Parameters block{};
        block[3] = glm::vec4(0.25f);
        const shaders::ShaderFrameSnapshot snapshot{.bindings = {{block}}, .global = 0u, .nodes = {{nodes_[2], 0u}}};
        auto state = shaders::makeCustomEffectState(snapshot, shaders::captureEffectScene(scene(), 3), 1.5f);
        ASSERT_TRUE(state);
        EXPECT_EQ(state->time, 1.5f);
        ASSERT_EQ(state->bindings.size(), 1u);
        EXPECT_EQ(state->bindings[0].values[3], glm::vec4(0.25f));
        EXPECT_EQ(state->global_binding, 0u);
        EXPECT_EQ(state->reveal_binding, GaussianCustomEffectState::kNone);
        auto slots = scene().getVisibleSplatNodeSlots();
        ASSERT_EQ(slots.size(), 3u);
        ASSERT_EQ(state->objects.size(), 3u);
        for (std::uint32_t i = 0; i < 3; ++i) {
            const auto& object = state->objects[slots[i].slot_index];
            EXPECT_EQ(object.id, i);
            EXPECT_EQ(object.binding, slots[i].node->uuid == nodes_[2] ? 0u : GaussianCustomEffectState::kNone);
        }
        glm::vec3 lo, hi;
        ASSERT_TRUE(lfs::vis::visibleSplatBounds(scene(), lo, hi));
        EXPECT_EQ(state->bounds_min, lo);
        EXPECT_EQ(state->bounds_max, hi);
        // The reveal spreads over the same bounds.
        lfs::sequencer::RevealClip clip;
        clip.end = 2.0f;
        const auto reveal = lfs::vis::makeRevealRenderState(scene(), clip, 1.0f);
        ASSERT_TRUE(reveal);
        EXPECT_EQ(reveal->min_y, lo.y);
        EXPECT_EQ(reveal->max_y, hi.y);

        // A hidden node takes no id, and its geometry leaves the bounds.
        scene().setNodeVisibility(scene().getNodeByUuid(nodes_[0])->id, false);
        slots = scene().getVisibleSplatNodeSlots();
        ASSERT_EQ(slots.size(), 2u);
        state = shaders::makeCustomEffectState(snapshot, shaders::captureEffectScene(scene(), 3), 0.0f);
        ASSERT_TRUE(state);
        EXPECT_EQ(state->objects[slots[0].slot_index].id, 0u);
        EXPECT_EQ(state->objects[slots[1].slot_index].id, 1u);
        glm::vec3 visible_lo, visible_hi;
        ASSERT_TRUE(lfs::vis::visibleSplatBounds(scene(), visible_lo, visible_hi));
        EXPECT_GT(visible_lo.x, lo.x);
        EXPECT_EQ(state->bounds_min, visible_lo);
        EXPECT_EQ(state->bounds_max, visible_hi);
    }

    TEST_F(ShaderServiceTest, EffectStateOnlyWhereABindingApplies) {
        const GaussianCustomEffectState::Parameters block{};
        const auto scene_state = shaders::captureEffectScene(scene(), 3);
        // A node's slot applies while the node is visible.
        const shaders::ShaderFrameSnapshot node{.bindings = {{block}}, .nodes = {{nodes_[1], 0u}}};
        EXPECT_TRUE(shaders::makeCustomEffectState(node, scene_state, 0.0f));
        scene().setNodeVisibility(scene().getNodeByUuid(nodes_[1])->id, false);
        EXPECT_FALSE(shaders::makeCustomEffectState(node, shaders::captureEffectScene(scene(), 3), 0.0f));
    }

    TEST_F(ShaderServiceTest, WholeCompositionAdoptionAndFailure) {
        const auto first = requireGlobal("broken first");
        ASSERT_TRUE(waitLinks(first, 1));
        EXPECT_FALSE(service().candidate());
        EXPECT_EQ(state(first), ShaderState::Failed);
        ASSERT_TRUE(commands().updateSource(first, "good"));
        ASSERT_TRUE(waitLinks(first, 2));
        EXPECT_EQ(state(first), ShaderState::Pending);
        ASSERT_NE(show(), nullptr);
        const auto complete = display_->snapshot;
        const auto replacement = create("broken replacement");
        bind({}, replacement);
        ASSERT_TRUE(waitLinks(replacement, 1));
        EXPECT_FALSE(service().candidate());
        EXPECT_EQ(service().restrictFallback(complete), complete); // no shared UUID exception
        EXPECT_EQ(show(), complete->programs);
        EXPECT_EQ(service().status(first)->displayed_revision, 2u);
        EXPECT_EQ(service().status(replacement)->displayed_revision, 0u);
        ASSERT_TRUE(commands().unbind({}));
        EXPECT_FALSE(service().restrictFallback(complete));
        EXPECT_FALSE(show());
        EXPECT_TRUE(service().inspect().displayed_bindings.empty());

        bind({}, replacement);
        ASSERT_TRUE(commands().updateSource(replacement, "good replacement"));
        ASSERT_TRUE(waitIdle());
        ASSERT_NE(service().candidate(), nullptr);
        const auto candidate = service().candidate();
        service().failCandidate(candidate, renderError("pipeline creation failed"));
        EXPECT_FALSE(service().candidate());
        EXPECT_EQ(state(replacement), ShaderState::Failed);
        EXPECT_EQ(service().status(replacement)->diagnostics.entries[0].message, "pipeline creation failed");
        service().clear();
        service().observeDisplay(std::make_shared<const shaders::DisplayRecord>(shaders::DisplayRecord{complete, true}));
        EXPECT_TRUE(service().inspect().displayed_bindings.empty());

        // A failed edit keeps both the old schema and its values; an explicit
        // removal still takes effect while the other binding remains failed.
        const auto global = requireGlobal("stable");
        bind({Kind::Node, nodes_[0]}, global);
        ASSERT_TRUE(waitIdle());
        ASSERT_NE(show(), nullptr);
        const auto previous = display_->snapshot;
        ASSERT_TRUE(commands().setParameter(global, "gain", std::array{.3f}));
        ASSERT_TRUE(commands().updateSource(global, "broken edit"));
        ASSERT_TRUE(waitIdle());
        EXPECT_EQ(service().restrictFallback(previous)->bindings, previous->bindings);
        ASSERT_TRUE(commands().unbind({Kind::Node, nodes_[0]}));
        const auto restricted = service().restrictFallback(previous);
        ASSERT_TRUE(restricted);
        EXPECT_TRUE(restricted->nodes.empty());
        EXPECT_EQ(restricted->global, previous->global);
        EXPECT_EQ(restricted->parameters, previous->parameters);
        EXPECT_EQ(restricted->slots.size(), 1u);
        display_ = std::make_shared<const shaders::DisplayRecord>(shaders::DisplayRecord{restricted, true});
        service().observeDisplay(display_);
        EXPECT_EQ(service().inspect().displayed_bindings.size(), 1u);

        // A later family extends the entire retained composition, never the
        // failed candidate. Old-only work cannot increment live counters.
        const auto linked = links(global);
        schedule_.gate(true);
        service().demandLive(std::array{std::string(kQuantHost)}, restricted);
        EXPECT_FALSE(service().candidate());
        EXPECT_EQ(service().restrictFallback(restricted), restricted);
        schedule_.gate(false);
        ASSERT_TRUE(waitIdle());
        const auto extended = service().restrictFallback(restricted);
        ASSERT_TRUE(extended);
        EXPECT_EQ(links(global), linked);
        EXPECT_TRUE(std::ranges::contains(extended->programs->hosts, std::string(kQuantHost), &shaders::HostProgram::label));
        EXPECT_EQ(extended->parameters, previous->parameters);
        EXPECT_EQ(extended->content, previous->content);
    }

    TEST_F(ShaderServiceTest, ExportLeaseIsIndependentAndComplete) {
        const auto id = requireGlobal("version one");
        ASSERT_TRUE(waitLinks(id, 1));
        ASSERT_NE(show(), nullptr);
        const auto live = display_;
        auto first = service().prepareExport(std::array{std::string(kHost), std::string(kGutHost)});
        ASSERT_TRUE(commands().setParameter(id, "gain", std::array{0.25f}));
        ASSERT_TRUE(commands().updateSource(id, "version two"));
        auto second = service().prepareExport(std::array{std::string(kHost)});
        ASSERT_TRUE(waitIdle());
        const auto a = service().exportSnapshot(first), b = service().exportSnapshot(second);
        ASSERT_TRUE(a && *a);
        ASSERT_TRUE(b && *b);
        EXPECT_NE((*a)->content, (*b)->content);
        EXPECT_EQ((*a)->programs->hosts.size(), 2u);
        EXPECT_EQ((*b)->programs->hosts.size(), 1u);
        EXPECT_EQ((*a)->programs->effects[0].revision, 1u);
        EXPECT_EQ((*b)->programs->effects[0].revision, 2u);
        EXPECT_FLOAT_EQ((*a)->bindings[0].values[0].x, 1.f);
        EXPECT_FLOAT_EQ((*b)->bindings[0].values[0].x, .25f);
        EXPECT_EQ(service().status(id)->displayed_revision, 1u); // readiness is not presentation
        ASSERT_NE(show(), nullptr);
        EXPECT_EQ(service().status(id)->displayed_revision, 2u);
        EXPECT_EQ((*a)->programs->effects[0].revision, 1u);
        // A frame extends the original composition after a live source/parameter edit.
        first.snapshot = *a;
        schedule_.gate(true);
        const std::array extension{std::string(kQuantHost)};
        for (int poll = 0; poll < 3; ++poll) {
            const auto pending = service().exportSnapshot(first, extension);
            ASSERT_TRUE(pending);
            EXPECT_FALSE(*pending);
            EXPECT_EQ(first.snapshot, *a);
        }
        schedule_.gate(false);
        ASSERT_TRUE(waitIdle());
        const auto extended = service().exportSnapshot(first, extension);
        ASSERT_TRUE(extended && *extended);
        EXPECT_EQ((*extended)->content, (*a)->content);
        EXPECT_EQ((*extended)->epoch, (*a)->epoch);
        EXPECT_EQ((*extended)->programs->hosts.size(), 3u);
        EXPECT_FLOAT_EQ((*extended)->bindings[0].values[0].x, 1.f);
        EXPECT_EQ((*extended)->programs->hosts.front(), (*a)->programs->hosts.front());
        {
            const std::lock_guard lock(schedule_.mutex);
            EXPECT_EQ(std::ranges::count(schedule_.started, std::string("version one@") + kQuantHost), 1);
            EXPECT_EQ(std::ranges::count_if(schedule_.started, [](const auto& link) { return link.starts_with("version one@"); }), 3);
            schedule_.failed_host = kSurvivorsHost;
        }
        EXPECT_TRUE(service().exportSnapshot(first, std::array{std::string(kSurvivorsHost)}));
        ASSERT_TRUE(waitIdle());
        const auto failed_extension = service().exportSnapshot(first);
        ASSERT_FALSE(failed_extension);
        EXPECT_EQ(failed_extension.error().user_message(), "Effect(2:3): broken effect");
        EXPECT_EQ(first.snapshot, *a); // a failed extension never publishes a subset
        ASSERT_TRUE(commands().updateSource(id, "broken"));
        const auto failed = service().prepareExport(std::array{std::string(kHost)});
        ASSERT_TRUE(waitIdle());
        const auto diagnostic = service().exportSnapshot(failed);
        ASSERT_FALSE(diagnostic);
        EXPECT_EQ(diagnostic.error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_EQ(diagnostic.error().user_message(), "Effect(2:3): broken effect");
        schedule_.gate(true);
        EXPECT_TRUE(service().exportSnapshot(second, extension));
        service().clear();
        EXPECT_TRUE(first.ended());
        EXPECT_FALSE(service().exportSnapshot(first, extension));
        EXPECT_FALSE(service().exportSnapshot(second, extension));
        schedule_.gate(false);
        service_.reset(); // surviving leases never keep the scene/service alive
        EXPECT_TRUE(second.ended());
    }

    TEST_F(ShaderServiceTest, OuterWorkerFailureCancelsWaiters) {
        const auto owner = std::this_thread::get_id();
        service_ = std::make_unique<ShaderService>(scene(), [this] { return std::make_unique<FakeLinker>(schedule_); }, [owner] {
            if (std::this_thread::get_id() != owner) throw std::runtime_error("worker wake fault"); });
        const auto changes = service().changes();
        const auto origin = service().generation();
        requireGlobal();
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto generation = service().generation();
            if (changes->cancelled(origin))
                break;
            changes->wait(generation, deadline);
        }
        EXPECT_TRUE(changes->cancelled(origin));
        EXPECT_TRUE(service().work().work.empty());
        EXPECT_FALSE(service().command(shaders::SetShaderSource{{}, "after shutdown", {}, {}}));
    }

    TEST_F(ShaderServiceTest, ThrowingLinkerFactorySettlesAndKeepsWorkerUsable) {
        service_ = std::make_unique<ShaderService>(scene(), []() -> std::unique_ptr<shaders::HostLinker> { throw std::runtime_error("factory fault"); }, [] {});
        const auto id = requireGlobal();
        ASSERT_TRUE(waitIdle());
        EXPECT_EQ(state(id), ShaderState::Failed);
        EXPECT_FALSE(service().status(id)->diagnostics.entries.empty());
        ASSERT_TRUE(commands().updateSource(id, "retry"));
        demand(kHost);
        ASSERT_TRUE(waitIdle());
        EXPECT_EQ(service().status(id)->links, 2u);
    }

    TEST(ShaderChanges, NotificationsBeforeWaitCancellationAndClosureCannotBeLost) {
        auto changes = std::make_shared<shaders::ShaderChanges>();
        const auto before = changes->generation.load();
        changes->notify(); // the observation-to-sleep race
        EXPECT_TRUE(changes->wait(before, std::chrono::steady_clock::now()));
        const auto current = changes->generation.load();
        EXPECT_FALSE(changes->wait(current, std::chrono::steady_clock::now()));
        std::stop_source stop;
        auto waiting = std::async(std::launch::async, [&] { return changes->wait(current, std::chrono::steady_clock::now() + 5s, stop.get_token()); });
        stop.request_stop();
        EXPECT_FALSE(waiting.get());
        changes->notify(false, true);
        EXPECT_TRUE(changes->cancelled(current));
        EXPECT_FALSE(changes->cancelled(changes->generation.load()));
        changes->notify(true);
        EXPECT_TRUE(changes->cancelled(changes->generation.load()));
    }

    TEST(ShaderServiceSlangLinker, LinksAPackageHostWithItsInterface) {
        const fs::path manifest = fs::path(LFS_SHADER_PACKAGE_DIR) / "manifest.json";
        auto package = shaders::SlangCompiler::load_package(manifest);
        ASSERT_TRUE(package) << lfs::format_for_developer(package.error());
        const auto& host = *std::ranges::find((*package)->hosts, std::string(kHost),
                                              [](const shaders::HostStage& stage) { return stage.module.name; });
        const shaders::EffectSource gain{lfs::test::kGain, "gain.slang"};
        const shaders::EffectSource tint{lfs::test::kTint, "tint.slang"};
        // The compiler is thread affine: the linker is made and used on one thread.
        std::thread([&] {
            auto linker = shaders::slangHostLinker(manifest)();
            shaders::CompileDiagnostics diagnostics;
            auto linked = linker->link(std::vector{gain}, kHost, diagnostics);
            ASSERT_TRUE(linked) << diagnostics.raw;
            EXPECT_EQ(linked->program.label, kHost);
            EXPECT_EQ(linked->program.workgroup, host.workgroup);
            EXPECT_EQ(linked->program.subgroup_size, host.subgroup_size);
            ASSERT_EQ(linked->program.bindings.size(), host.bindings.size());
            for (std::size_t i = 0; i < host.bindings.size(); ++i)
                EXPECT_EQ(linked->program.bindings[i], static_cast<int>(host.bindings[i].binding));
            EXPECT_FALSE(linked->program.spirv.empty());
            ASSERT_EQ(linked->schemas.size(), 1u);
            ASSERT_EQ(linked->schemas[0].size(), 1u);
            EXPECT_EQ(linked->schemas[0][0].name, "gain");

            // A composition links into one host program, each effect with its schema.
            auto composed = linker->link(std::vector{gain, tint}, kHost, diagnostics);
            ASSERT_TRUE(composed) << diagnostics.raw;
            ASSERT_EQ(composed->schemas.size(), 2u);
            EXPECT_EQ(composed->schemas[0][0].name, "gain");
            EXPECT_EQ(composed->schemas[1][0].name, "tint");
            EXPECT_NE(composed->program.spirv, linked->program.spirv);

            EXPECT_FALSE(linker->link(
                std::vector<shaders::EffectSource>{
                    {"import lfs;\nLfsElement effect(LfsElement e, LfsContext ctx) { return f; }\n"}},
                kHost, diagnostics));
            EXPECT_FALSE(diagnostics.entries.empty());

            auto missing = shaders::slangHostLinker(manifest.parent_path() / "absent.json")();
            EXPECT_FALSE(missing->link(std::vector<shaders::EffectSource>{{"import lfs;"}}, kHost, diagnostics));
            ASSERT_EQ(diagnostics.entries.size(), 1u);
            EXPECT_EQ(diagnostics.entries[0].code, "LFS");
        }).join();
    }

} // namespace
