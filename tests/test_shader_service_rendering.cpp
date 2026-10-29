/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define LFS_USER_SHADER_SERVICE_FIXTURE
#include "core/mesh_data.hpp"
#include "gui/editor/python_editor.hpp"
#include "user_shader_test_access.hpp"
#include "user_shader_test_support.hpp"

namespace lfs::test::service_rendering {
    TEST_F(ShaderServiceRenderingTest, CustomRevealResolvesIdentityAndGatesAtClipEnd) {
        const auto normal = drawn();
        const auto normal_preview = capture();
        const auto asset = bound(kReveal, {.kind = Kind::Reveal});
        const lfs::sequencer::RevealClip clip{1, 3, .25f, lfs::sequencer::RevealEffect::Custom, asset};
        EXPECT_FALSE(lfs::vis::makeRevealRenderState(scene(), clip, 2));
        manager_->setSequencerReveal(std::nullopt, lfs::vis::makeCustomRevealContext(clip, 0));
        ASSERT_TRUE(drawing(asset));
        const auto before = drawn();
        EXPECT_GT(maxDifference(before, normal), .1f);
        EXPECT_GT(maxDifference(capture(), normal_preview), .1f);
        manager_->setSequencerReveal(std::nullopt, lfs::vis::makeCustomRevealContext(clip, 1));
        EXPECT_LT(maxDifference(drawn(), before), kOneStep);
        manager_->setSequencerReveal(std::nullopt, lfs::vis::makeCustomRevealContext(clip, 2));
        const auto middle = drawn();
        EXPECT_GT(maxDifference(middle, before), .05f);
        EXPECT_GT(maxDifference(middle, normal), .05f);
        const auto count = links(asset);
        manager_->setSequencerReveal(std::nullopt, lfs::vis::makeCustomRevealContext(clip, 3));
        EXPECT_LT(maxDifference(drawn(), normal), kOneStep);
        EXPECT_LT(maxDifference(capture(), normal_preview), kOneStep);
        EXPECT_EQ(service().boundAsset({.kind = Kind::Reveal}), asset);
        EXPECT_FALSE(commands().removeAsset(asset));
        auto wrong = clip;
        wrong.shader_id = lfs::core::generate_uuid_v4();
        manager_->setSequencerReveal(std::nullopt, lfs::vis::makeCustomRevealContext(wrong, 2));
        EXPECT_LT(maxDifference(drawn(), normal), kOneStep);
        EXPECT_EQ(links(asset), count);
    }

    TEST_F(ShaderServiceRenderingTest, NothingBoundDrawsLikeNoServiceAndStartsNothing) {
        const auto counted = [&] {
            Access::binds = Access::dispatches = 0;
            redraw();
            return std::array{Access::binds, Access::dispatches};
        };
        manager_->setShaderService(nullptr);
        for (int i = 0; i < 3; ++i)
            redraw();
        const auto without = counted();
        const auto reference = image();
        const std::size_t threads = lfs::test::threadCount();

        manager_->setShaderService(service_.get());
        const auto with = counted();
        EXPECT_EQ(with, without);
        EXPECT_GT(with[1], 0u);
        EXPECT_EQ(image(), reference);
        EXPECT_TRUE(lfs::test::threadsSettleAt(threads));
        EXPECT_EQ(linkers_.load(), 0);
        EXPECT_EQ(schedule_.started.size(), 0u);
        EXPECT_EQ(manager_->customEffectGeneration(), 0u);
        // An open shader tab and unbound library do not create demand either.
        lfs::vis::editor::PythonEditor editor;
        editor.newDocument(kShift, lfs::vis::editor::DocumentKind::Shader);
        ASSERT_TRUE(commands().createAsset("Unbound", kShift));
        for (int i = 0; i < 3; ++i) {
            EXPECT_TRUE(editor.pollShaders(service()).empty());
            EXPECT_EQ(counted(), without);
            EXPECT_EQ(image(), reference);
        }
        EXPECT_EQ(linkers_.load(), 0);
        EXPECT_TRUE(schedule_.started.empty());
        // Idle frames stay idle.
        frame();
        EXPECT_FALSE(manager_->pollDirtyState());
    }

    TEST_F(ShaderServiceRenderingTest, GlobalNodeAndRevealBindingsChainInOrder) {
        const auto masks = nodeMasks();
        const auto legacy = drawn();
        for (const auto& mask : masks)
            EXPECT_LT(std::abs(excess(legacy, &mask)), 0.02f);

        const auto asset = bound(kShift);
        ASSERT_TRUE(applied(asset));
        shift(asset, kRed);
        EXPECT_EQ(linkers_.load(), 1);
        const auto global = drawn();
        for (const auto& mask : masks)
            EXPECT_GT(excess(global, &mask), 0.2f);

        // Every node bound directly draws as the global binding does.
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        for (const auto& uuid : nodes_)
            ASSERT_TRUE(commands().bind({.kind = Kind::Node, .node = uuid}, asset));
        frame();
        EXPECT_LT(maxDifference(drawn(), global), kOneStep);

        // Only the middle node bound.
        for (const auto& uuid : {nodes_[0], nodes_[2]})
            ASSERT_TRUE(commands().unbind({.kind = Kind::Node, .node = uuid}));
        frame();
        const auto middle = drawn();
        EXPECT_LT(std::abs(excess(middle, &masks[0])), 0.02f);
        EXPECT_GT(excess(middle, &masks[1]), 0.2f);
        EXPECT_LT(std::abs(excess(middle, &masks[2])), 0.02f);

        // Global, then the node's own: the middle node shifts twice.
        ASSERT_TRUE(commands().bind({.kind = Kind::Global}, asset));
        frame();
        const auto chained = drawn();
        EXPECT_NEAR(excess(chained, &masks[0]), excess(global, &masks[0]), 0.01f);
        EXPECT_GT(excess(chained, &masks[1]), excess(global, &masks[1]) + 0.1f);

        // The reveal binding runs only while a reveal is set.
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        ASSERT_TRUE(commands().unbind({.kind = Kind::Node, .node = nodes_[1]}));
        ASSERT_TRUE(commands().bind({.kind = Kind::Reveal}, asset));
        frame();
        EXPECT_LT(maxDifference(drawn(), legacy), kOneStep);
        lfs::sequencer::RevealClip clip;
        clip.end = 10.0f;
        clip.effect = lfs::sequencer::RevealEffect::Custom;
        clip.shader_id = asset;
        manager_->setSequencerReveal(std::nullopt, lfs::vis::makeCustomRevealContext(clip, 9.9f));
        const auto revealed = drawn();
        ASSERT_TRUE(commands().unbind({.kind = Kind::Reveal}));
        ASSERT_TRUE(commands().bind({.kind = Kind::Global}, asset));
        frame();
        EXPECT_LT(maxDifference(drawn(), revealed), kOneStep);
        for (const auto& mask : masks)
            EXPECT_GT(excess(revealed, &mask), 0.2f);
        EXPECT_EQ(linkers_.load(), 1);
        EXPECT_EQ(links(asset), 1u);
    }

    TEST_F(ShaderServiceRenderingTest, DifferentAssetsRunWithTheirOwnParameters) {
        const auto masks = nodeMasks();
        const auto red = bound(kShift);
        const auto green = bound(kGreen, {.kind = Kind::Node, .node = nodes_[1]});
        ASSERT_TRUE(drawing(red));
        ASSERT_TRUE(drawing(green));
        shift(red, kRed);
        const float lift = 0.6f;
        set(green, "green", std::span(&lift, 1));
        // Red everywhere; the middle node also lifts green, its own asset's parameter.
        const auto both = drawn();
        EXPECT_GT(excess(both, &masks[0]), 0.4f);
        EXPECT_LT(std::abs(excess(both, &masks[1])), 0.1f);
        EXPECT_GT(excess(both, &masks[1], 2), 0.4f);
        EXPECT_GT(excess(both, &masks[2]), 0.4f);
        // One link per host runs both.
        EXPECT_EQ(links(red), links(green));
        const float none = 0.0f;
        set(green, "green", std::span(&none, 1));
        EXPECT_GT(excess(drawn(), &masks[1]), 0.4f);

        // The reveal slot with another asset than the global one.
        ASSERT_TRUE(commands().unbind({.kind = Kind::Node, .node = nodes_[1]}));
        ASSERT_TRUE(commands().bind({.kind = Kind::Reveal}, green));
        set(green, "green", std::span(&lift, 1));
        frame();
        for (const auto& mask : masks)
            EXPECT_GT(excess(drawn(), &mask), 0.4f);
        lfs::sequencer::RevealClip clip;
        clip.end = 10.0f;
        clip.effect = lfs::sequencer::RevealEffect::Custom;
        clip.shader_id = green;
        manager_->setSequencerReveal(std::nullopt, lfs::vis::makeCustomRevealContext(clip, 9.9f));
        const auto revealed = drawn();
        for (const auto& mask : masks) {
            EXPECT_LT(std::abs(excess(revealed, &mask)), 0.1f);
            EXPECT_GT(excess(revealed, &mask, 2), 0.4f);
        }
        // The same assets in other slots compose as before: nothing linked again.
        EXPECT_EQ(links(red), links(green));
    }

    TEST_F(ShaderServiceRenderingTest, AnotherAssetDrawsOnceTheShownOneIsUnbound) {
        const auto masks = nodeMasks();
        const auto shown = bound(kShift);
        ASSERT_TRUE(drawing(shown));
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        redraw();
        // Bound to the reveal slot before a reveal plays: frames show some
        // programs meanwhile, never the unbound asset's in its place.
        const auto green = bound(kGreen, {.kind = Kind::Reveal});
        ASSERT_TRUE(settle([&] { return shownHosts() > 0; }));
        lfs::sequencer::RevealClip clip;
        clip.end = 10.0f;
        clip.effect = lfs::sequencer::RevealEffect::Custom;
        clip.shader_id = green;
        manager_->setSequencerReveal(std::nullopt, lfs::vis::makeCustomRevealContext(clip, 9.9f));
        ASSERT_TRUE(drawing(green));
        const float lift = 0.6f;
        set(green, "green", std::span(&lift, 1));
        EXPECT_LT(excess(drawn(), nullptr, 1), -0.4f);
        EXPECT_EQ(service().status(shown)->displayed_revision, 0u);

        // Bound to a node shown only once programs are.
        ASSERT_TRUE(commands().unbind({.kind = Kind::Reveal}));
        manager_->setSequencerReveal(std::nullopt);
        redraw();
        scene().setNodeVisibility(id(nodes_[0]), false);
        const auto red = bound(kShift, {.kind = Kind::Node, .node = nodes_[0]});
        ASSERT_TRUE(settle([&] { return shownHosts() > 0; }));
        scene().setNodeVisibility(id(nodes_[0]), true);
        ASSERT_TRUE(drawing(red));
        shift(red, kRed);
        const auto image = drawn();
        EXPECT_GT(excess(image, &masks[0]), 0.2f);
        EXPECT_LT(std::abs(excess(image, &masks[1])), 0.02f);
    }

    TEST_F(ShaderServiceRenderingTest, CaptureAndOverlayRerenderDrawTheEffect) {
        const auto legacy_capture = rgbOf(manager_->renderPreviewImage(
            scene_.get(), viewport_.getRotationMatrix(), viewport_.getTranslation(), settings_.focal_length_mm, kSize,
            kSize));
        EXPECT_LT(std::abs(excess(legacy_capture)), 0.02f);
        const auto asset = bound(kShift);
        ASSERT_TRUE(applied(asset));
        shift(asset, kRed);
        const auto main = drawn();
        const float main_excess = excess(main);
        EXPECT_GT(main_excess, 0.2f);

        const auto capture = rgbOf(manager_->renderPreviewImage(
            scene_.get(), viewport_.getRotationMatrix(), viewport_.getTranslation(), settings_.focal_length_mm, kSize,
            kSize));
        EXPECT_NEAR(excess(capture), main_excess, 0.02f);

        // A selection-only frame may not reuse a built-in projection.
        manager_->markDirty(DirtyFlag::SELECTION, FrameReason::SettingsChange);
        frame();
        EXPECT_LT(maxDifference(image(), main), kOneStep);
    }

    // Dev draws independent cameras as separate 3D views (editor areas) rather than a dual split.
    TEST_F(ShaderServiceRenderingTest, IndependentViewsDrawWithTheirOwnCameras) {
        const auto asset = bound(kCameraSide);
        ASSERT_TRUE(applied(asset));
        const auto serial = Access::programSetSerial(*manager_);
        const auto first = views_.activeView();
        const lfs::vis::ViewId second =
            views_.screen().split(lfs::vis::screen::AreaId{first}, lfs::vis::screen::SplitAxis::Columns, 0.5f).value;
        manager_->retainVisibleViews({first, second});
        auto left_camera = viewport_;
        look(left_camera, {-2.0f, 0.0f, 6.0f});
        auto right_camera = viewport_;
        look(right_camera, {2.0f, 0.0f, 6.0f});
        const auto draw = [&](const lfs::vis::ViewId view, Viewport& camera) {
            manager_->syncCustomPrograms();
            manager_->markDirty(DirtyFlag::SPLATS, FrameReason::SettingsChange);
            return manager_->renderFrame({.view = view,
                                          .viewport = camera,
                                          .settings = settings_,
                                          .scene_manager = scene_.get(),
                                          .graphics_context = graphics_.get(),
                                          .on_screen = true})
                .rendered;
        };
        ASSERT_TRUE(settle([&] { return draw(first, left_camera); }));
        ASSERT_TRUE(settle([&] { return draw(second, right_camera); }));
        const auto left = Access::viewImage(*manager_, *context_, first);
        const auto right = Access::viewImage(*manager_, *context_, second);
        // Red against blue: one side sees x > 0, the other x < 0.
        const float left_side = excess(left, nullptr, 2);
        const float right_side = excess(right, nullptr, 2);
        EXPECT_GT(std::abs(left_side), 0.5f);
        EXPECT_GT(std::abs(right_side), 0.5f);
        EXPECT_LT(left_side * right_side, 0.0f);
        EXPECT_EQ(Access::programSetSerial(*manager_), serial);
        EXPECT_EQ(links(asset), 1u);
    }

    class ShaderComparisonTest : public ShaderServiceRenderingTest, public ::testing::WithParamInterface<bool> {};
    TEST_P(ShaderComparisonTest, PanelsKeepTheirNodeBindings) {
        const bool consolidated = GetParam();
        auto settings = settings_;
        settings.split_view_mode = lfs::vis::SplitViewMode::PLYComparison;
        if (consolidated) {
            ASSERT_EQ(scene().consolidateNodeModels(), nodes_.size());
            apply(settings);
            redraw();
        }
        const auto asset = bound(kShift, {.kind = Kind::Node, .node = nodes_[0]});
        ASSERT_TRUE(drawing(asset));
        shift(asset, kRed);
        const auto linked = links(asset);
        const auto hosts = shownHosts();
        // The consolidated row is in comparison already; a settings change would defer the next frame.
        if (!consolidated)
            apply(settings);
        ASSERT_TRUE(settle([&] { return redraw().rendered; }));
        // Entering the comparison keeps the renderer and its warmed chain, so the panels
        // draw the family the single view linked.
        EXPECT_EQ(shownHosts(), hosts);
        EXPECT_EQ(links(asset), linked);
        redraw();
        // Each panel draws one node's model alone, with that node's record.
        EXPECT_GT(excess(image(Slot::SplitLeft)), 0.2f);
        EXPECT_LT(std::abs(excess(image(Slot::SplitRight))), 0.02f);

        ASSERT_TRUE(commands().unbind({.kind = Kind::Node, .node = nodes_[0]}));
        ASSERT_TRUE(commands().bind({.kind = Kind::Node, .node = nodes_[1]}, asset));
        redraw();
        EXPECT_LT(std::abs(excess(image(Slot::SplitLeft))), 0.02f);
        EXPECT_GT(excess(image(Slot::SplitRight)), 0.2f);
        EXPECT_EQ(links(asset), linked);
    }
    INSTANTIATE_TEST_SUITE_P(Storage, ShaderComparisonTest, ::testing::Bool(),
                             [](const auto& row) { return row.param ? "CombinedModel" : "OwnedModels"; });

    TEST_F(ShaderServiceRenderingTest, TogglingGutDrawsBuiltInThenLinksOneFamily) {
        auto gut = settings_;
        gut.gut = true;
        apply(gut);
        const auto legacy_gut = drawn();
        gut.gut = false;
        apply(gut);

        const auto asset = bound(kShift);
        ASSERT_TRUE(applied(asset));
        shift(asset, kRed);
        const auto linked = links(asset);
        const auto hosts = shownHosts();

        gut.gut = true;
        apply(gut);
        frame();
        // The set lacks the family: this draw ran the built-in programs.
        EXPECT_FALSE(Access::ranEveryProgram(*manager_));
        EXPECT_LT(maxDifference(image(), legacy_gut), kOneStep);
        ASSERT_TRUE(settle([&] { return shownHosts() == hosts + 1; }));
        EXPECT_EQ(links(asset), linked + 1);
        EXPECT_GT(excess(drawn()), 0.2f);
        EXPECT_TRUE(Access::ranEveryProgram(*manager_));
        EXPECT_EQ(service().status(asset)->state, ShaderState::Applied);

        // Toggled off again, an edit links only the family frames still draw.
        gut.gut = false;
        apply(gut);
        redraw();
        ASSERT_TRUE(commands().updateSource(asset, std::string(kShift) + "\n// edited\n"));
        ASSERT_TRUE(applied(asset));
        EXPECT_EQ(links(asset), linked + 2);
        EXPECT_EQ(shownHosts(), hosts);
        EXPECT_TRUE(Access::ranEveryProgram(*manager_));
    }

    TEST_F(ShaderServiceRenderingTest, WholeCompositionAdoptionAndFailure) {
        const auto builtins = drawn();
        const auto broken = bound(kBroken);
        ASSERT_TRUE(settle([&] { return service().status(broken)->state == ShaderState::Failed; }));
        EXPECT_LT(maxDifference(drawn(), builtins), kOneStep);
        EXPECT_EQ(service().status(broken)->displayed_revision, 0u);
        ASSERT_TRUE(commands().unbind({}));
        redraw();
        const auto asset = bound(kShift);
        ASSERT_TRUE(applied(asset));
        shift(asset, kRed);
        // A reset's first frame uploads the scene again through the legacy
        // chain, whose family links once.
        Access::resetRenderer(*manager_);
        ASSERT_TRUE(drawing(asset));
        // The last complete programs draw until frames run the candidate's, which hold that family too.
        ASSERT_TRUE(settle([&] {
            const auto candidate = service().candidate();
            return candidate && Access::displayPrograms(*manager_) == candidate->programs && Access::ranEveryProgram(*manager_);
        }));
        const auto linked = links(asset);
        const auto before = drawn();

        Access::resetRenderer(*manager_);
        // A reset stays pending until a real publication, even before initialization.
        manager_->syncCustomPrograms();
        EXPECT_EQ(Access::displayPrograms(*manager_), nullptr);
        EXPECT_EQ(service().status(asset)->state, ShaderState::Pending);
        // The reset dropped the set; the CPU programs stay.
        frame();
        manager_->syncCustomPrograms();
        EXPECT_EQ(service().status(asset)->state, ShaderState::Pending);
        ASSERT_TRUE(drawing(asset));
        EXPECT_EQ(links(asset), linked);
        EXPECT_LT(maxDifference(drawn(), before), kOneStep);
        const auto good = Access::displayPrograms(*manager_);
        gate(true);
        ASSERT_TRUE(commands().updateSource(asset, std::string(kShift) + "\n// new revision\n"));
        Access::resetRenderer(*manager_);
        // The reset's first frame resets the renderer once more; it builds the last good set while the revision links.
        frame();
        Access::waitForBuilds(Access::renderer(*manager_));
        Access::fail_pipeline = true; // the revision's set is the next custom job
        gate(false);
        const auto deadline = std::chrono::steady_clock::now() + 30s;
        while ((!service().candidate() || service().candidate()->programs == good) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(1ms);
        ASSERT_NE(service().candidate(), nullptr);
        ASSERT_NE(service().candidate()->programs, good);
        manager_->syncCustomPrograms();
        EXPECT_EQ(Access::displayPrograms(*manager_), nullptr);
        EXPECT_EQ(service().status(asset)->displayed_revision, 0u);
        Access::waitForBuilds(Access::renderer(*manager_));
        manager_->syncCustomPrograms();
        ASSERT_EQ(service().status(asset)->state, ShaderState::Failed);
        // Rebuilding last-good pipelines can exceed the frame polling deadline
        // with a full driver cache. Synchronize the worker, then publish once.
        manager_->syncCustomPrograms();
        Access::waitForBuilds(Access::renderer(*manager_));
        manager_->syncCustomPrograms();
        redraw();
        ASSERT_EQ(Access::displayPrograms(*manager_), good);
        EXPECT_EQ(service().status(asset)->displayed_revision, 1u);
        EXPECT_LT(maxDifference(drawn(), before), kOneStep);
        // Failure of an unrelated replacement keeps the whole last-complete image.
        const auto replacement = bound(kBroken);
        ASSERT_TRUE(settle([&] { return service().status(replacement)->state == ShaderState::Failed; }));
        EXPECT_EQ(Access::displayPrograms(*manager_), good);
        EXPECT_EQ(service().status(asset)->displayed_revision, 1u);
        EXPECT_EQ(service().status(replacement)->displayed_revision, 0u);
        const auto failed_links = links(replacement);
        for (int i = 0; i < 4; ++i)
            EXPECT_LT(maxDifference(drawn(), before), kOneStep);
        EXPECT_EQ(links(replacement), failed_links); // no retry/warning loop
    }

    TEST_F(ShaderServiceRenderingTest, UnbindRetiresAndRebindReusesOnlyCpuPrograms) {
        const auto asset = bound(kShift);
        ASSERT_TRUE(applied(asset));
        shift(asset, kRed);
        const auto before = drawn();
        const auto linked = links(asset);
        const std::weak_ptr set{Access::displaySet(*manager_)};
        ASSERT_FALSE(set.expired());
        redraw();
        const auto last = Access::completion(*manager_);
        ASSERT_NE(last.completion_semaphore, VkSemaphore{VK_NULL_HANDLE});
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        manager_->syncCustomPrograms();
        redraw();
        EXPECT_EQ(Access::displayPrograms(*manager_), nullptr);
        std::uint64_t reached = 0;
        ASSERT_EQ(vkGetSemaphoreCounterValue(context_->device(), last.completion_semaphore, &reached), VK_SUCCESS);
        if (Access::retiredSets(*manager_) == 0) {
            // Destroyed at once only when nothing in flight used it.
            EXPECT_GE(reached, last.completion_value);
        } else {
            const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                           .semaphoreCount = 1,
                                           .pSemaphores = &last.completion_semaphore,
                                           .pValues = &last.completion_value};
            ASSERT_EQ(vkWaitSemaphores(context_->device(), &wait, 5'000'000'000ull), VK_SUCCESS);
        }
        // A later render boundary frees what the finished submission used.
        redraw();
        redraw();
        EXPECT_EQ(Access::retiredSets(*manager_), 0u);
        ASSERT_TRUE(settle([&] {
            redraw();
            return set.expired() && Access::retiredSets(Access::renderer(*manager_)) == 0;
        }));
        ASSERT_TRUE(commands().bind({}, asset));
        ASSERT_TRUE(drawing(asset));
        EXPECT_EQ(links(asset), linked);
        EXPECT_LT(maxDifference(drawn(), before), kOneStep);
        ASSERT_TRUE(commands().unbind({}));
        redraw();
        ASSERT_TRUE(commands().removeAsset(asset));
    }

    TEST_F(ShaderServiceRenderingTest, AnEditKeepsTheFamilyItsLastCompleteCompositionWasLinking) {
        const auto asset = bound(kShift);
        ASSERT_TRUE(drawing(asset));
        shift(asset, kRed);
        ASSERT_GT(excess(drawn()), 0.2f);
        gate(true);
        const auto started = schedule_.started.size();
        auto gut = settings_;
        gut.gut = true;
        apply(gut);
        frame();
        ASSERT_TRUE(schedule_.waitStarted(started + 1));
        // The edit replaces the composition while its GUT family links.
        ASSERT_TRUE(commands().updateSource(asset, std::string(kShift) + "\n// edited\n"));
        schedule_.release();
        ASSERT_TRUE(schedule_.waitStarted(started + 2));
        // The owner thread takes the replaced link's result before any frame retains it.
        (void)service().candidate();
        // The last complete composition draws that family while the edit links.
        ASSERT_TRUE(settle([&] { return excess(drawn()) > 0.2f; }));
        schedule_.release();
        ASSERT_TRUE(drawing(asset));
        EXPECT_GT(excess(drawn()), 0.2f);
        // Nothing linked the family again for the last complete composition.
        EXPECT_FALSE(schedule_.waitStarted(started + 3, 500ms));
    }

    TEST_F(ShaderServiceRenderingTest, SelectionWaitsOnlyWhileTheEffectIsDisplayed) {
        lfs::vis::SelectionService selection(scene_.get(), manager_.get());
        const std::string blocked = LOC("selection.shader_effect_blocked");
        const auto select = [&](const bool displayed) {
            using Mode = lfs::vis::SelectionMode;
            const std::pair<const char*, std::function<std::string()>> entries[] = {
                {"rect", [&] { return selection.selectRect(0, 0, kSize, kSize, Mode::Replace).error; }},
                {"brush", [&] { return selection.selectBrush(kSize / 2, kSize / 2, 10, Mode::Replace).error; }},
                {"click", [&] { return selection.selectRing(kSize / 2, kSize / 2, Mode::Replace).error; }},
                {"lasso", [&] { return selection.selectLasso({{0, 0}, {kSize, 0}, {kSize, kSize}}, Mode::Replace).error; }},
                {"color", [&] { return selection.selectByColorAt(kSize / 2, kSize / 2, Mode::Replace).error; }},
                {"filtered all", [&] { return selection.selectAllFiltered().error; }}};
            auto settings = settings_;
            settings.crop_filter_for_selection = true;
            manager_->updateSettings(settings);
            for (const auto& [name, invoke] : entries) {
                SCOPED_TRACE(name);
                EXPECT_EQ(invoke() == blocked, displayed);
            }
            manager_->updateSettings(settings_);
        };
        using Motion = RenderingManager::SplatMotion;

        // A broken shader never displays.
        const auto broken = bound(kBroken);
        ASSERT_TRUE(settle([&] { return service().status(broken)->state == ShaderState::Failed; }));
        EXPECT_EQ(manager_->displayedSplatMotion(), Motion::None);
        select(false);
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));

        // Pending: still the built-in draws; a queued link forces no redraw.
        gate(true);
        const auto asset = bound(kShift);
        frame();
        frame();
        EXPECT_EQ(service().status(asset)->state, ShaderState::Pending);
        EXPECT_FALSE(manager_->pollDirtyState());
        EXPECT_EQ(manager_->displayedSplatMotion(), Motion::None);
        select(false);
        gate(false);
        ASSERT_TRUE(applied(asset));
        EXPECT_EQ(manager_->displayedSplatMotion(), Motion::CustomEffect);
        select(true);

        // Unbound, the last frame of the effect stays on screen until the
        // next one; a frame drawn off screen is not that.
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        select(true);
        manager_->syncCustomPrograms();
        select(true);
        manager_->markDirty(DirtyFlag::SPLATS, FrameReason::SettingsChange);
        ASSERT_TRUE(frame(false).rendered);
        select(true);
        redraw();
        EXPECT_EQ(manager_->displayedSplatMotion(), Motion::None);
        select(false);

        // Bindings that reach no draw block nothing: the reveal slot without
        // a reveal, and a hidden node's slot.
        bound(kShift, {.kind = Kind::Reveal});
        redraw();
        redraw();
        EXPECT_EQ(manager_->displayedSplatMotion(), Motion::None);
        select(false);
        ASSERT_TRUE(commands().unbind({.kind = Kind::Reveal}));
        scene().setNodeVisibility(id(nodes_[0]), false);
        bound(kShift, {.kind = Kind::Node, .node = nodes_[0]});
        redraw();
        redraw();
        EXPECT_EQ(manager_->displayedSplatMotion(), Motion::None);
        select(false);
        ASSERT_TRUE(commands().unbind({.kind = Kind::Node, .node = nodes_[0]}));
        scene().setNodeVisibility(id(nodes_[0]), true);

        // An independent lease never keeps a viewport binding active.
        ASSERT_TRUE(commands().bind({.kind = Kind::Global}, asset));
        ASSERT_TRUE(applied(asset));
        redraw();
        select(true);
        const auto pin = manager_->prepareShaderExport(kFrame);
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        redraw();
        EXPECT_EQ(Access::displayPrograms(*manager_), nullptr);
        EXPECT_EQ(manager_->displayedSplatMotion(), Motion::None);
        select(false);
    }

    // Runs work on another thread, as an export or a script does, while this
    // thread acts as the viewer: it runs what the work posts and keeps
    // drawing frames, or runs `loop` instead. A run that has not ended after
    // 30 s stops the service.

    TEST_F(ShaderServiceRenderingTest, ExportFramesKeepTheSceneOfTheirStart) {
        const auto asset = bound(kObjectBounds);
        const auto reveal = bound(kReveal, {.kind = Kind::Reveal});
        ASSERT_TRUE(applied(asset));
        lfs::sequencer::RevealClip clip{0, 2, .1f, lfs::sequencer::RevealEffect::Custom, reveal};
        std::vector<std::vector<float>> reference;
        {
            const auto pin = manager_->prepareShaderExport(kFrame);
            auto frames = ExportRun(*this, pin, clip).run();
            ASSERT_TRUE(frames) << frames.error();
            reference = std::move(*frames);
        }
        EXPECT_GT(maxDifference(reference[1], reference[2]), 0.1f);
        // The reveal binding runs: the effect alone keeps green above red.
        EXPECT_GT(excess(reference[2]), 0.05f);
        // A node hidden, then another removed, mid-export: the frames keep
        // the objects, bounds and reveal of the start.
        {
            const auto pin = manager_->prepareShaderExport(kFrame);
            auto frames = ExportRun(*this, pin, clip).run([&](const int index) {
                if (index == 0) {
                    scene().setNodeVisibility(id(nodes_[0]), false);
                    EXPECT_TRUE(commands().unbind({.kind = Kind::Reveal}));
                    manager_->setSequencerReveal(std::nullopt);
                }
                if (index == 1)
                    scene_->removeNode(id(nodes_[2]), false);
            });
            ASSERT_TRUE(frames) << frames.error();
            EXPECT_EQ(*frames, reference);
        }
        // An export started now draws the changed scene, which the effect shows.
        const auto pin = manager_->prepareShaderExport(kFrame);
        auto changed = ExportRun(*this, pin, clip).run();
        ASSERT_TRUE(changed) << changed.error();
        EXPECT_GT(maxDifference(changed->back(), reference.back()), 0.1f);
    }

    TEST_F(ShaderServiceRenderingTest, StillExportsOnTheViewerThreadFailAsNotReady) {
        const auto asset = bound(kShift);
        ASSERT_TRUE(applied(asset));
        shift(asset, kRed);
        // An export captured during an edit waits for its complete frozen revision.
        gate(true);
        ASSERT_TRUE(commands().updateSource(asset, std::string(kShift) + "\n// edited\n"));
        const auto pin = manager_->prepareShaderExport(kFrame);
        // The viewer thread cannot wait for the lease programs.
        const auto unready = manager_->drawWithShaders(&pin, kInline, [&]() -> Image { return capture(); });
        ASSERT_FALSE(unready);
        EXPECT_EQ(unready.error(), LOC("runtime.export_shader_not_ready"));
        gate(false);
        ASSERT_TRUE(settle([&] { return manager_->prepareShaderExport(pin).value_or(false); }));
        const auto ready = manager_->drawWithShaders(&pin, kInline, [&]() -> Image { return capture(); });
        ASSERT_TRUE(ready) << ready.error();
        EXPECT_GT(excess(*ready), 0.2f);
    }

    TEST_F(ShaderServiceRenderingTest, TransparentFallbackKeepsBothPassesFrozenAndReportsShaderFailures) {
        const auto asset = bound(kShift);
        ASSERT_TRUE(drawing(asset));
        shift(asset, {0.15f, -0.15f, -0.15f});
        const RenderingManager::ExportImageRequest request{
            .rotation = viewport_.getRotationMatrix(),
            .translation = viewport_.getTranslation(),
            .focal_length_mm = settings_.focal_length_mm,
            .width = kSize,
            .height = kSize,
            .mode = lfs::vis::ExportPostProcessMode::Transparent};
        const auto owner = std::this_thread::get_id();
        const auto fallback = [&] {
            EXPECT_EQ(std::this_thread::get_id(), owner);
            return manager_->renderTransparentExportFallback(scene_.get(), request);
        };
        OffViewer reference_viewer(*this);
        auto reference = reference_viewer.run([&] { return manager_->drawWithShaders(true, reference_viewer.post(), fallback, kFrameOf); });
        ASSERT_TRUE(reference) << lfs::format_for_developer(reference.error());

        shift(asset, kRed);
        std::string edited = kShift;
        edited.replace(edited.find("e.color += p.shift;"), std::string("e.color += p.shift;").size(),
                       "e.color += 0.5 * p.shift;");
        gate(true);
        ASSERT_TRUE(commands().updateSource(asset, std::move(edited)));
        int draws = 0;
        OffViewer viewer(*this, [&] {
            gate(false);
            frame();
        });
        const auto image = viewer.run([&] {
            return manager_->drawWithShaders(true, viewer.post(), [&] {
                ++draws;
                shift(asset, {-0.3f, 0.3f, -0.3f});
                return fallback(); }, kFrameOf);
        });
        ASSERT_TRUE(image) << lfs::format_for_developer(image.error());
        EXPECT_EQ(draws, 1);
        ASSERT_EQ(image->shape(), reference->shape());
        const auto* actual = image->ptr<uint8_t>();
        const auto* expected = reference->ptr<uint8_t>();
        for (std::size_t i = 0; i < image->numel(); ++i)
            ASSERT_LE(std::abs(int(actual[i]) - int(expected[i])), 1) << i;

        ASSERT_TRUE(commands().updateSource(asset, kBroken));
        const auto failed = viewer.run([&] { return manager_->drawWithShaders(true, viewer.post(), fallback, kFrameOf); });
        ASSERT_FALSE(failed);
        EXPECT_EQ(failed.error().code(), lfs::ErrorCode::FailedPrecondition);
        EXPECT_FALSE(failed.error().user_message().empty());
    }

    TEST_F(ShaderServiceRenderingTest, ExportDrawsEndAsCancelledWhenTheViewerStops) {
        const auto asset = bound(kShift);
        ASSERT_TRUE(applied(asset));
        // Capture an edit still linking; cancellation must precede drawing.
        gate(true);
        ASSERT_TRUE(commands().updateSource(asset, std::string(kShift) + "\n// edited\n"));
        const auto pin = manager_->prepareShaderExport(kFrame);
        // An image with an Error keeps the code of the wait.
        std::stop_source cancel;
        int draws = 0;
        OffViewer viewer(*this, [&] { cancel.request_stop(); manager_->syncCustomPrograms(); });
        const auto cancelled = viewer.run([&] {
            return manager_->drawWithShaders(&pin, viewer.post(), [&]() -> Image { ++draws; return capture(); }, cancel.get_token());
        });
        ASSERT_FALSE(cancelled);
        EXPECT_EQ(cancelled.error(), LOC("runtime.export_cancelled"));
        EXPECT_EQ(draws, 0);
        service().stop();
        EXPECT_TRUE(pin.ended());
        const auto stopped = viewer.run([&] { return manager_->drawWithShaders(&pin, viewer.post(), [&]() -> Image { ++draws; return capture(); }); });
        ASSERT_FALSE(stopped);
        EXPECT_EQ(stopped.error(), LOC("runtime.export_stopped"));
        EXPECT_EQ(draws, 0);
    }

    TEST_F(ShaderServiceRenderingTest, ExportLeaseIsIndependentAndComplete) {
        const auto asset = bound(kShift);
        ASSERT_TRUE(applied(asset));
        shift(asset, kRed);
        manager_->syncCustomPrograms();
        gate(true);
        // The edit leaves no complete viewport set: the lease must link its frame's host itself.
        ASSERT_TRUE(commands().updateSource(asset, std::string(kShift) + "\n// edited"));
        const auto first = manager_->prepareShaderExport(kFrame);
        int draws = 0;
        const auto pending = manager_->drawWithShaders(&first, kInline, [&]() -> Image { ++draws; return capture(); });
        EXPECT_FALSE(pending);
        EXPECT_EQ(draws, 0);
        gate(false);
        OffViewer viewer(*this);
        const auto draw = [&](const auto& lease) {
            return viewer.run([&] { return manager_->drawWithShaders(&lease, viewer.post(), [&]() -> Image { ++draws; return capture(); }); });
        };
        const auto red = draw(first);
        ASSERT_TRUE(red) << red.error();
        EXPECT_EQ(draws, 1);
        EXPECT_GT(excess(*red), .2f);
        shift(asset, {-0.3f, .3f, -.3f});
        ASSERT_TRUE(settle([&] { return redraw().rendered && excess(drawn()) < -.2f; }));
        const auto second = manager_->prepareShaderExport(kFrame);
        const auto green = draw(second);
        ASSERT_TRUE(green) << green.error();
        EXPECT_LT(excess(*green), -.2f);
        const auto again = draw(first);
        ASSERT_TRUE(again);
        EXPECT_LT(maxDifference(*again, *red), kOneStep);
        EXPECT_EQ(draws, 3);
        // Reset before submission rebuilds the frozen set before the callback runs.
        Access::resetRenderer(*manager_);
        const auto rebuilt = draw(first);
        ASSERT_TRUE(rebuilt) << rebuilt.error();
        EXPECT_LT(maxDifference(*rebuilt, *red), kOneStep);
        EXPECT_EQ(draws, 4);
        ASSERT_TRUE(commands().updateSource(asset, kBroken));
        const auto failed = manager_->prepareShaderExport(kFrame);
        EXPECT_FALSE(draw(failed));
        EXPECT_EQ(draws, 4);
        // Background live reads settle before one callback, and a broken edit
        // uses the viewport's complete prior composition.
        int live_draws = 0;
        const auto live = viewer.run([&] { return manager_->drawWithShaders(false, viewer.post(), [&]() -> Image { ++live_draws; return capture(); }, kFrameOf); });
        ASSERT_TRUE(live);
        EXPECT_EQ(live_draws, 1);
        EXPECT_LT(excess(*live), -.2f);
        ASSERT_TRUE(commands().unbind({.kind = Kind::Global}));
        const auto frozen = draw(second);
        ASSERT_TRUE(frozen);
        EXPECT_LT(maxDifference(*frozen, *green), kOneStep);
        EXPECT_EQ(draws, 5);
        service().clear();
        EXPECT_FALSE(draw(first));
        EXPECT_EQ(draws, 5);
        // Empty requests bypass the compiler even if another desired shader failed.
        const RenderingManager::ExportLease empty;
        ASSERT_TRUE(draw(empty));
        EXPECT_EQ(draws, 6);

        // Callback-driven storage changes: one full-image GUT host per encoding, two in total.
        ASSERT_TRUE(Access::renderer(Access::renderer(*manager_)).supportsFloat16Storage());
        settings_.gut = true;
        settings_.sh_degree = 1;
        manager_->updateSettings(settings_);
        settings_ = manager_->getSettings(); // exports freeze the backend the renderer resolved
        const auto base = node(0.f);
        const auto make_model = [&](const bool half) {
            const SplatData canonical(1, base->means(), base->sh0(),
                                      Tensor::zeros({static_cast<std::size_t>(base->size()), 3, 3}, Device::GPU),
                                      base->scaling_raw(), base->rotation_raw(), base->opacity_raw(), 1.f);
            const auto values = half ? canonical.shN_raw().to(lfs::core::DataType::Float16) : canonical.shN_raw();
            return std::make_shared<SplatData>(1, base->means(), base->sh0(),
                                               externalTensor(*context_, values, values.shape()[0], "export-storage"),
                                               base->scaling_raw(), base->rotation_raw(), base->opacity_raw(), 1.f, SplatData::ShNLayout::Swizzled);
        };
        const std::array models{make_model(false), make_model(true)};
        for (const bool frozen_backend : {true, false}) {
            SCOPED_TRACE(frozen_backend ? "dynamic storage" : "no frozen backend");
            service().clear();
            const std::string source = std::string(kShift) + (frozen_backend ? "\n// dynamic storage" : "\n// per draw backend");
            const auto dynamic_asset = bound(source);
            const auto dynamic = manager_->prepareShaderExport(kFrame, nullptr, frozen_backend ? &settings_ : nullptr, true);
            EXPECT_FALSE(dynamic.snapshot);
            EXPECT_TRUE(service().work().work.empty()); // no broad union before frame 1
            ASSERT_TRUE(commands().updateSource(dynamic_asset, kShift));
            lfs::vis::gui::VideoExportSceneSnapshot frame_snapshot;
            frame_snapshot.model_transforms = {glm::mat4(1.f)};
            frame_snapshot.effect_scene.objects.resize(1);
            std::vector<std::string> expected_hosts;
            int callbacks = 0, frame_draws = 0;
            const auto callback = [&](const int frame) {
                ++callbacks;
                frame_snapshot.combined_model = models[frame % 2];
            };
            for (int frame = 0; frame < 4; ++frame) {
                callback(frame); // once, before the preflight/wait loop
                const auto host = Access::renderer(*manager_).hostsFor({false, false, frame % 2 != 0, true, false, false}).host;
                if (!std::ranges::contains(expected_hosts, host))
                    expected_hosts.emplace_back(host);
                const auto preflight = [&] { return manager_->prepareShaderExport(dynamic, frame_snapshot.gaussianModel(), &settings_); };
                const auto draw_frame = [&]() -> Image {
                    ++frame_draws;
                    EXPECT_EQ(dynamic.snapshot->programs->hosts.size(), expected_hosts.size());
                    EXPECT_EQ(dynamic.snapshot->programs->effects[0].revision, 1u);
                    auto image = exportFrame(frame_snapshot, dynamic, static_cast<float>(frame), {});
                    EXPECT_EQ(Access::encodedSh(Access::renderer(*manager_))[0], frame % 2 != 0);
                    EXPECT_TRUE(Access::customProjection(Access::renderer(*manager_)));
                    return image;
                };
                if (frame == 1) {
                    gate(true);
                    const auto pending_frame = manager_->drawWithShaders(&dynamic, kInline, draw_frame, {}, preflight);
                    EXPECT_FALSE(pending_frame);
                    EXPECT_EQ(frame_draws, 1);
                    std::stop_source cancelled;
                    cancelled.request_stop();
                    EXPECT_FALSE(manager_->drawWithShaders(&dynamic, kInline, draw_frame, cancelled.get_token(), preflight));
                    EXPECT_EQ(frame_draws, 1);
                    Access::resetRenderer(*manager_); // extension also survives device-generation invalidation
                    gate(false);
                }
                const auto image = viewer.run([&] { return manager_->drawWithShaders(&dynamic, viewer.post(), draw_frame, {}, preflight); });
                ASSERT_TRUE(image) << image.error();
                EXPECT_EQ(callbacks, frame + 1);
                EXPECT_EQ(frame_draws, frame + 1);
                const std::lock_guard lock(schedule_.mutex);
                std::vector<std::string> linked;
                for (const auto& link : schedule_.started)
                    if (link.starts_with(source + "@"))
                        linked.push_back(link.substr(source.size() + 1));
                EXPECT_EQ(linked.size(), expected_hosts.size());
                for (const auto& host : expected_hosts)
                    EXPECT_EQ(std::ranges::count(linked, host), 1) << host;
            }
            EXPECT_EQ(expected_hosts.size(), 2u);
            ASSERT_TRUE(applied(dynamic_asset)); // the viewport adopts live edits during this lease
            auto failed_settings = settings_;
            failed_settings.raster_backend = lfs::rendering::GaussianRasterBackend::ThreeDgs;
            failed_settings.gut = false;
            {
                const std::lock_guard lock(schedule_.mutex);
                schedule_.failed_host = Access::renderer(*manager_).hostsFor({false, false, true, false, true, false}).host;
            }
            const auto failed_frame = viewer.run([&] {
                return manager_->drawWithShaders(&dynamic, viewer.post(), [&]() -> Image { ++frame_draws; return capture(); }, {}, [&] { return manager_->prepareShaderExport(dynamic, models[1].get(), &failed_settings); });
            });
            ASSERT_FALSE(failed_frame);
            // The export fails with the extension's diagnostic, which the preflight reports.
            const auto diagnostic = manager_->prepareShaderExport(dynamic, models[1].get(), &failed_settings);
            ASSERT_FALSE(diagnostic);
            EXPECT_EQ(diagnostic.error().user_message(), "Effect(2:3): broken effect");
            EXPECT_EQ(failed_frame.error(), LOCF(lichtfeld::Strings::Runtime::EXPORT_SHADER_UNAVAILABLE, std::string(diagnostic.error().user_message())));
            EXPECT_EQ(frame_draws, 4);
            EXPECT_EQ(dynamic.snapshot->programs->hosts.size(), 2u);
            {
                const std::lock_guard lock(schedule_.mutex);
                schedule_.failed_host.clear();
            }
        }
    }

    // An export links the one host its frame size selects. A draw beyond that plan, as the
    // banded retry of a tile-instance overflow, links its host and draws that frame again.
    TEST_F(ShaderServiceRenderingTest, ExportLinksTheHostItsFrameSizeSelects) {
        ASSERT_TRUE(Access::renderer(Access::renderer(*manager_)).supportsFloat16Storage());
        const auto asset = bound(kShift);
        ASSERT_TRUE(applied(asset));
        shift(asset, kRed);
        // An edit leaves no complete viewport set for the lease to start from.
        ASSERT_TRUE(commands().updateSource(asset, std::string(kShift) + "\n// edited"));
        auto snapshot = lfs::vis::gui::captureVideoExportSceneSnapshot(*scene_);
        ASSERT_TRUE(snapshot);
        ASSERT_TRUE(lfs::io::migrateSplatTensorsToAllocator(*snapshot->combined_model, manager_->makeSplatTensorAllocator()));
        const auto lease = manager_->prepareShaderExport({.width = 8192, .height = 4096}, snapshot->gaussianModel(), &settings_);
        ASSERT_TRUE(settle([&] { return manager_->prepareShaderExport(lease).value_or(false); }));
        const auto host = [&](const bool banded) { return std::string(Access::renderer(*manager_).hostsFor({false, false, false, false, true, banded}).host); };
        const auto hosts = [&] {
            std::vector<std::string> labels;
            for (const auto& program : lease.snapshot->programs->hosts)
                labels.push_back(program->label);
            return labels;
        };
        EXPECT_EQ(hosts(), std::vector{host(true)}); // a tiled frame draws bands only
        OffViewer viewer(*this);
        int draws = 0;
        const auto draw = [&] {
            return viewer.run([&] {
                return manager_->drawWithShaders(&lease, viewer.post(), [&]() -> Image {
                    ++draws; // as a video frame: a failed draw is an error, not an image
                    const auto image = manager_->renderPreviewImage(*snapshot->gaussianModel(), lfs::vis::gui::makeVideoExportGaussianSceneState(*snapshot, {}, lease.snapshot.get(), 0.f),
                                                                    viewport_.getRotationMatrix(), viewport_.getTranslation(), settings_.focal_length_mm, kSize, kSize);
                    if (!image)
                        return std::unexpected("the frame drew no image");
                    return rgbOf(image);
                });
            });
        };
        const auto full = draw();
        ASSERT_TRUE(full) << full.error();
        EXPECT_EQ(draws, 2);
        EXPECT_GT(excess(*full), .2f);
        EXPECT_EQ(hosts(), (std::vector{host(true), host(false)}));
        const auto again = draw();
        ASSERT_TRUE(again) << again.error();
        EXPECT_EQ(draws, 3);
        EXPECT_LT(maxDifference(*again, *full), kOneStep);
    }
    // Scripted renders (Python, MCP) link the hosts of the frame they describe, nothing more.
    TEST_F(ShaderServiceRenderingTest, ScriptedRendersLinkOnlyTheHostsTheyDraw) {
        ASSERT_TRUE(Access::renderer(Access::renderer(*manager_)).supportsFloat16Storage());
        const auto plain = capture();
        // Bound without a viewport frame: nothing is demanded or linked before the scripted renders.
        const std::string source = std::string(kTint) + "\n// scripted";
        bound(source);
        EXPECT_TRUE(service().work().work.empty());
        const auto linked = [&] {
            const std::lock_guard lock(schedule_.mutex);
            std::vector<std::string> hosts;
            for (const auto& link : schedule_.started)
                if (link.starts_with(source + "@"))
                    hosts.push_back(link.substr(source.size() + 1));
            return hosts;
        };
        const auto host = [&](const bool survivors, const bool banded) {
            return std::string(Access::renderer(*manager_).hostsFor({false, false, false, false, survivors, banded}).host);
        };
        OffViewer viewer(*this, [&] { service().acceptResults(); });
        int draws = 0;
        // Each render runs on another thread, as a script does, and gets its own lease.
        const auto scripted = [&](auto draw, const RenderingManager::ExportFrame frame, const bool exporting = false) {
            const auto before = linked().size();
            auto image = viewer.run([&] {
                return manager_->drawWithShaders(exporting, viewer.post(), [&] { ++draws; return draw(); }, [frame] { return frame; });
            });
            auto hosts = linked();
            hosts.erase(hosts.begin(), hosts.begin() + static_cast<std::ptrdiff_t>(before));
            return std::pair{std::move(image), std::move(hosts)};
        };

        // render_view: the full-image preview host.
        const auto [view, view_hosts] = scripted([&]() -> Image { return capture(); }, kFrame);
        ASSERT_TRUE(view) << view.error();
        EXPECT_EQ(draws, 1);
        EXPECT_GT(maxDifference(*view, plain), .1f);
        EXPECT_EQ(view_hosts, std::vector{host(true, false)});

        // render_view_and_depth: a depth capture draws the per-pixel chain alone.
        const auto [depth, depth_hosts] = scripted([&]() -> Image {
            const auto rgbd = manager_->renderPreviewImageAndDepth(scene_.get(), viewport_.getRotationMatrix(), viewport_.getTranslation(),
                                                                   settings_.focal_length_mm, kSize, kSize);
            if (!rgbd.image || !rgbd.depth || !rgbd.depth->is_valid())
                return std::unexpected("the depth capture drew nothing");
            return rgbOf(rgbd.image); }, {.width = kSize, .height = kSize, .depth = true});
        ASSERT_TRUE(depth) << depth.error();
        EXPECT_EQ(draws, 2);
        EXPECT_GT(maxDifference(*depth, plain), .1f);
        EXPECT_EQ(depth_hosts, std::vector{host(false, false)});

        // A render that cannot describe its frame discovers its host: one extra callback, no other host.
        const auto [unframed, unframed_hosts] = scripted([&]() -> Image {
            const auto image = manager_->renderPreviewImage(scene_.get(), viewport_.getRotationMatrix(), viewport_.getTranslation(),
                                                            settings_.focal_length_mm, kSize, kSize);
            if (!image)
                return std::unexpected("the discovery draw drew no image");
            return rgbOf(image); }, {});
        ASSERT_TRUE(unframed) << unframed.error();
        EXPECT_EQ(draws, 4);
        EXPECT_LT(maxDifference(*unframed, *view), kOneStep);
        EXPECT_EQ(unframed_hosts, std::vector{host(true, false)});

        // A viewport export banding splat depth under an on-screen mesh adds the banded host.
        const RenderingManager::ExportImageRequest request{
            .rotation = viewport_.getRotationMatrix(),
            .translation = viewport_.getTranslation(),
            .focal_length_mm = settings_.focal_length_mm,
            .width = kSize,
            .height = kSize};
        EXPECT_FALSE(manager_->exportImageFrame(scene_.get(), request).bands);
        auto mesh = std::make_shared<lfs::core::MeshData>();
        mesh->vertices = Tensor::from_vector(std::vector<float>{-.5f, -.5f, 0.f, .5f, -.5f, 0.f, 0.f, .5f, 0.f}, {3, 3}, Device::CPU);
        mesh->indices = Tensor::from_vector(std::vector<int32_t>{0, 1, 2}, {1, 3}, Device::CPU);
        scene().addMesh("mesh", mesh);
        const auto frame = manager_->exportImageFrame(scene_.get(), request);
        EXPECT_TRUE(frame.bands);
        auto aside = request;
        aside.translation.x += 20.f;
        EXPECT_FALSE(manager_->exportImageFrame(scene_.get(), aside).bands);
        const auto [exported, export_hosts] = scripted([&] { return manager_->renderExportImage(scene_.get(), request); }, frame, true);
        ASSERT_TRUE(exported) << exported.error();
        EXPECT_EQ(draws, 5);
        EXPECT_EQ(export_hosts, (std::vector{host(true, false), host(true, true)}));
    }

} // namespace lfs::test::service_rendering
