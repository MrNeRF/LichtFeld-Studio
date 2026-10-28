/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include <RmlUi/Core.h>
#include <gtest/gtest.h>
#include <string>

namespace lfs::vis {
    class SequencerKeyframeGestureTest : public testing::TestWithParam<float> {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            scale_ = GetParam();
            ui_.snap_to_grid = true;
            ui_.snap_interval = .5f;
            controller_.setClipDuration(10.f);
            for (const float time : {0.f, 2.13f, 8.f}) {
                sequencer::Keyframe key;
                key.time = time;
                const auto id = controller_.addKeyframeAtTime(key, time);
                if (time == 2.13f)
                    key_ = id;
            }
            panel_.cached_dp_ratio_ = scale_;
            ASSERT_FLOAT_EQ(controller_.timeline().clipDuration(), 10.f);
            panel_.cached_panel_width_ = width_;
            panel_.cached_total_height_ = 200.f * scale_;
            controller_.setKeyframeTimeCommitCallback([this](auto, float before, float after) {
                ++commits_;
                committed_before_ = before;
                committed_after_ = after;
            });
            core::events::cmd::SequencerSelectKeyframe::when([this](const auto& event) {
                EXPECT_TRUE(controller_.selectKeyframe(event.keyframe_index));
            });
            core::events::cmd::SequencerGoToKeyframe::when([this](const auto&) { ++visits_; });
        }
        void TearDown() override { lfs::event::EventBridge::instance().clear_all(); }
        float x(float time) const { return time * width_ / 10.f; }
        float time() const { return controller_.timeline().getKeyframeById(key_)->time; }
        void input(float mouse_x, bool clicked, bool down, bool ctrl = false) {
            PanelInputState state;
            state.mouse_x = mouse_x;
            state.mouse_y = 50.f * scale_;
            state.mouse_clicked[0] = clicked;
            state.mouse_down[0] = down;
            state.key_ctrl = ctrl;
            state.time = clock_;
            if (clicked)
                clock_ += 1.f;
            panel_.handleTimelineInteraction({0.f, 0.f}, width_, 80.f * scale_, state);
        }
        gui::RmlUIManager manager_;
        SequencerController controller_;
        gui::panels::SequencerUIState ui_;
        RmlSequencerPanel panel_{controller_, ui_, &manager_};
        sequencer::KeyframeId key_ = sequencer::INVALID_KEYFRAME_ID;
        float scale_ = 1.f;
        float clock_ = 10.f;
        static constexpr float width_ = 1000.f;
        int commits_ = 0;
        int visits_ = 0;
        float committed_before_ = 0.f;
        float committed_after_ = 0.f;
    };

    TEST_P(SequencerKeyframeGestureTest, ClickingOffCenterAndSmallJitterNeverRetimesOrCommits) {
        for (const float offset : {-4.f, 0.f, 4.f}) {
            const auto revision = controller_.timelineRevision();
            const float press_x = x(2.13f) + offset * scale_;
            input(press_x, true, true);
            EXPECT_FLOAT_EQ(time(), 2.13f);
            input(press_x + 2.f * scale_, false, true);
            EXPECT_FLOAT_EQ(time(), 2.13f);
            input(press_x, false, false);
            EXPECT_FLOAT_EQ(time(), 2.13f);
            EXPECT_EQ(controller_.timelineRevision(), revision);
            EXPECT_EQ(commits_, 0);
        }
        EXPECT_EQ(visits_, 3);
        EXPECT_EQ(controller_.selectedKeyframeId(), key_);
    }

    TEST_P(SequencerKeyframeGestureTest, DragPreservesGrabOffsetAndCommitsOnce) {
        const float press_x = x(2.13f) + 4.f * scale_;
        // Ctrl disables snapping after the initial selection click.
        input(press_x, true, true);
        input(press_x + 20.f * scale_, false, true, true);
        EXPECT_NEAR(time(), 2.13f + .2f * scale_, .00001f);
        input(press_x + 40.f * scale_, false, true, true);
        EXPECT_EQ(commits_, 0);
        input(press_x + 40.f * scale_, false, false, true);
        EXPECT_EQ(commits_, 1);
        EXPECT_FLOAT_EQ(committed_before_, 2.13f);
        EXPECT_NEAR(committed_after_, 2.13f + .4f * scale_, .00001f);
        EXPECT_EQ(visits_, 0);
    }

    INSTANTIATE_TEST_SUITE_P(Scale, SequencerKeyframeGestureTest, testing::Values(1.f, 2.f));
} // namespace lfs::vis

namespace {
    class SequencerKeyframeLayoutTest : public testing::TestWithParam<float> {
    protected:
        class Renderer final : public Rml::RenderInterface {
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        } renderer_;
        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            context_ = Rml::CreateContext("keyframe-layout", {800, 200}, &renderer_);
            ASSERT_NE(context_, nullptr);
            context_->SetDensityIndependentPixelRatio(GetParam());
        }
        void TearDown() override {
            Rml::RemoveContext("keyframe-layout");
            Rml::Shutdown();
        }
        Rml::Context* context_ = nullptr;
    };

    TEST_P(SequencerKeyframeLayoutTest, SelectionKeepsTheDiamondSizeAndCenter) {
        const auto css = std::string(PROJECT_ROOT_PATH) + "/src/visualizer/gui/rmlui/resources/sequencer.rcss";
        auto* doc = context_->LoadDocumentFromMemory(
            "<rml><head><link type='text/rcss' href='" + css + "'/></head><body>"
                                                               "<div style='position:relative;width:600px;height:100px;'>"
                                                               "<div id='diamond' class='keyframe' style='left:200px;background-color:#6699cc;border-color:#ffffff;'/></div></body></rml>");
        ASSERT_NE(doc, nullptr);
        doc->Show();
        context_->Update();
        auto* diamond = doc->GetElementById("diamond");
        ASSERT_NE(diamond, nullptr);
        const auto size = diamond->GetBox().GetSize(Rml::BoxArea::Border);
        const auto offset = diamond->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto center = offset + size * .5f;
        EXPECT_FLOAT_EQ(size.x, 14.f * GetParam());
        for (const bool selected : {true, false, true}) {
            diamond->SetClass("selected", selected);
            context_->Update();
            const auto selected_size = diamond->GetBox().GetSize(Rml::BoxArea::Border);
            const auto selected_center = diamond->GetAbsoluteOffset(Rml::BoxArea::Border) + selected_size * .5f;
            EXPECT_EQ(selected_size, size);
            EXPECT_EQ(selected_center, center);
        }
    }
    INSTANTIATE_TEST_SUITE_P(Scale, SequencerKeyframeLayoutTest, testing::Values(1.f, 2.f));
} // namespace
