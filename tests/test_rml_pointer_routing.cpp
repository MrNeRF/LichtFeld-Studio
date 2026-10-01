/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_input.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/rml_progress_overlay.hpp"
#include "gui/rml_viewport_overlay.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_pointer_dispatch.hpp"
#include "gui/string_keys.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/key_codes.hpp"
#include "input/view_targets.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/depth_window_ops.hpp"
#include "rendering/render_pass.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/rendering_types.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "selection/selection_service.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/rendering/depth_window_state.hpp"
#include "visualizer_impl.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/RenderInterface.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lfs::vis {
    // SDL batches events, so motion after a press already appears in frame-end
    // mouse_x/mouse_y. Classify each DOWN using its own recorded coordinates.
    namespace {
        // Use non-default timestamps and click counts to prove capture/copy preserves
        // them. Zero would also pass if either field were silently dropped to its
        // default.
        SDL_Event mouseDownEvent(const int sdl_button, const float x, const float y,
                                 const Uint64 timestamp = 0, const int clicks = 0) {
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.button = static_cast<Uint8>(sdl_button);
            event.button.x = x;
            event.button.y = y;
            event.button.timestamp = timestamp;
            event.button.clicks = static_cast<Uint8>(clicks);
            return event;
        }

        SDL_Event mouseUpEvent(const int sdl_button, const float x, const float y,
                               const Uint64 timestamp = 0, const int clicks = 0) {
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_BUTTON_UP;
            event.button.button = static_cast<Uint8>(sdl_button);
            event.button.x = x;
            event.button.y = y;
            event.button.timestamp = timestamp;
            event.button.clicks = static_cast<Uint8>(clicks);
            return event;
        }

        SDL_Event mouseMotionEvent(const float x, const float y) {
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_MOTION;
            event.motion.x = x;
            event.motion.y = y;
            return event;
        }

        // Everything finalize() does that matters here. finalize() itself needs
        // an SDL_Window: it samples the LIVE cursor into mouse_x/mouse_y, which
        // is precisely the position these tests must NOT be classified from.
        void settleLiveCursor(lfs::vis::FrameInputBuffer& buffer,
                              const float x, const float y) {
            buffer.mouse_x = x;
            buffer.mouse_y = y;
        }
    } // namespace

    TEST(ViewportPressOwnershipTest, PressPointSurvivesMotionCoalescedBehindThePress) {
        using lfs::vis::gui::buildPanelInputFromSDL;
        const auto pointInsideViewport = [](glm::vec2 point, glm::vec2 pos, glm::vec2 size) {
            return ViewTarget{.pos = pos, .size = size}.contains(point.x, point.y);
        };

        const glm::vec2 pos{320.0f, 40.0f};
        const glm::vec2 size{800.0f, 600.0f};

        // Use synthetic SDL through FrameInputBuffer intake and the production
        // PanelInputState copy. Hand-seeding the destination would leave both failure
        // sites untested.
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        // DOWN on the right half of the viewport...
        buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 900.0f, 300.0f));
        // ...then the cursor is dragged onto the left dock inside the same
        // buffered frame. SDL delivers both to the same frame.
        buffer.processEvent(mouseMotionEvent(100.0f, 300.0f));
        settleLiveCursor(buffer, 100.0f, 300.0f);

        const auto input = buildPanelInputFromSDL(buffer);
        ASSERT_TRUE(input.mouse_clicked[0]);
        const auto* const press = input.lastPress(0);
        ASSERT_NE(press, nullptr)
            << "the button-down did not survive capture and copy";
        EXPECT_EQ(glm::vec2(press->x, press->y), glm::vec2(900.0f, 300.0f))
            << "the button-down coordinates did not survive capture and copy";
        EXPECT_TRUE(pointInsideViewport({press->x, press->y}, pos, size));
        // The latest cursor position -- what the code used to read -- would have
        // classified this press as a dock press and resolved the wrong panel.
        EXPECT_FALSE(pointInsideViewport({input.mouse_x, input.mouse_y}, pos, size));

        // The inverse: a press on the dock followed by motion into the viewport
        // must not become a viewport press. Asymmetric coordinates, so a copy
        // that swaps X and Y cannot pass either direction.
        lfs::vis::FrameInputBuffer inverse_buffer;
        inverse_buffer.beginFrame();
        inverse_buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 100.0f, 500.0f));
        inverse_buffer.processEvent(mouseMotionEvent(900.0f, 300.0f));
        settleLiveCursor(inverse_buffer, 900.0f, 300.0f);
        const auto inverse = buildPanelInputFromSDL(inverse_buffer);
        const auto* const inverse_press = inverse.lastPress(0);
        ASSERT_NE(inverse_press, nullptr);
        EXPECT_EQ(glm::vec2(inverse_press->x, inverse_press->y), glm::vec2(100.0f, 500.0f));
        EXPECT_FALSE(pointInsideViewport({inverse_press->x, inverse_press->y}, pos, size));

        // One frame may contain DOWN/UP or two complete presses. Preserve every DOWN's
        // point and arrival order, apply focus per press, and let lastPress inspect the
        // last without changing the stream.
        lfs::vis::FrameInputBuffer double_press;
        double_press.beginFrame();
        double_press.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        double_press.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 120.0f, 310.0f));
        settleLiveCursor(double_press, 120.0f, 310.0f);
        const auto doubled = buildPanelInputFromSDL(double_press);
        ASSERT_EQ(doubled.mouse_button_events.size(), 2u)
            << "the canonical stream was coalesced";
        EXPECT_FLOAT_EQ(doubled.mouse_button_events[0].x, 700.0f);
        EXPECT_FLOAT_EQ(doubled.mouse_button_events[1].x, 120.0f);
        ASSERT_NE(doubled.lastPress(0), nullptr);
        EXPECT_FLOAT_EQ(doubled.lastPress(0)->x, 120.0f);

        // Per-button, not per-frame: a right press must not answer for the left.
        lfs::vis::FrameInputBuffer right_only;
        right_only.beginFrame();
        right_only.processEvent(mouseDownEvent(SDL_BUTTON_RIGHT, 640.0f, 200.0f));
        settleLiveCursor(right_only, 700.0f, 300.0f);
        const auto right_input = buildPanelInputFromSDL(right_only);
        ASSERT_NE(right_input.lastPress(1), nullptr);
        EXPECT_EQ(glm::vec2(right_input.lastPress(1)->x, right_input.lastPress(1)->y),
                  glm::vec2(640.0f, 200.0f));
        // With no press for that button there is no press event to hand back,
        // and callers get nothing rather than another button's coordinates.
        EXPECT_EQ(right_input.lastPress(0), nullptr);
        EXPECT_EQ(right_input.lastPress(-1), nullptr);
        EXPECT_EQ(right_input.lastPress(3), nullptr);
    }

    // Record GUI ownership at every BUTTON_DOWN, including repeated same-button DOWNs.
    // Do not re-derive it from frame-time rectangles or retain only the first press's
    // verdict.
    TEST(ViewportPressOwnershipTest, PressOwnershipIsRecordedAtTheButtonDown) {
        using lfs::vis::gui::buildPanelInputFromSDL;

        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        // What WindowManager does at the event, from GuiManager::hitTestMouseButton.
        buffer.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        // A second verdict for the SAME press is ignored: the owner and the
        // coordinates it was taken at must describe one event.
        buffer.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/false);
        settleLiveCursor(buffer, 320.0f, 300.0f);

        auto input = buildPanelInputFromSDL(buffer);
        ASSERT_NE(input.lastPress(0), nullptr);
        EXPECT_TRUE(input.lastPress(0)->gui_owned);
        EXPECT_EQ(input.lastPress(1), nullptr) << "ownership leaked across buttons";

        // A DOWN whose verdict was never recorded stays unowned rather than
        // inheriting the previous press's.
        buffer.beginFrame();
        buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        settleLiveCursor(buffer, 700.0f, 300.0f);
        input = buildPanelInputFromSDL(buffer);
        ASSERT_NE(input.lastPress(0), nullptr);
        EXPECT_FALSE(input.lastPress(0)->gui_owned)
            << "last frame's dock-edge verdict vetoed this frame's viewport press";

        // A stray verdict with no DOWN awaiting one owns nothing.
        buffer.beginFrame();
        buffer.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        EXPECT_TRUE(buildPanelInputFromSDL(buffer).mouse_button_events.empty());

        // EVERY DOWN gets its own verdict, including a second DOWN for the same
        // button in the same frame. "First DOWN of the frame wins" would have
        // given the second press the first one's owner.
        lfs::vis::FrameInputBuffer twice;
        twice.beginFrame();
        twice.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        twice.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        twice.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        twice.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        twice.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/false);
        settleLiveCursor(twice, 700.0f, 300.0f);
        const auto twice_input = buildPanelInputFromSDL(twice);
        ASSERT_EQ(twice_input.mouse_button_events.size(), 3u);
        EXPECT_TRUE(twice_input.mouse_button_events[0].gui_owned);
        EXPECT_TRUE(twice_input.mouse_button_events[1].gui_owned)
            << "the UP did not follow its own DOWN's owner";
        EXPECT_FALSE(twice_input.mouse_button_events[2].gui_owned)
            << "the second press inherited the first press's owner";
        ASSERT_NE(twice_input.lastPress(0), nullptr);
        EXPECT_FALSE(twice_input.lastPress(0)->gui_owned);
    }

    // A press LIFECYCLE outlives the frame it started in: the release must find
    // its own DOWN's verdict however many frames later it arrives, and an
    // unmatched release must find nothing.
    TEST(ViewportPressOwnershipTest, ReleaseFollowsItsOwnDownAcrossFrames) {
        using lfs::vis::gui::buildPanelInputFromSDL;

        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        buffer.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        buffer.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        ASSERT_EQ(buffer.mouse_button_events.size(), 1u);
        EXPECT_TRUE(buffer.mouse_button_events[0].gui_owned);

        // ...held across an idle frame...
        buffer.beginFrame();
        EXPECT_TRUE(buffer.mouse_button_events.empty());

        // ...and released in a third frame, far outside the control it started
        // on. The release still belongs to the press that opened it.
        buffer.beginFrame();
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 1100.0f, 620.0f));
        const auto released = buildPanelInputFromSDL(buffer);
        ASSERT_EQ(released.mouse_button_events.size(), 1u);
        EXPECT_FALSE(released.mouse_button_events[0].down);
        EXPECT_TRUE(released.mouse_button_events[0].gui_owned)
            << "a release outside the pressed control lost its own DOWN's owner";

        // A SECOND release with no press behind it inherits nothing.
        buffer.beginFrame();
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 1100.0f, 620.0f));
        const auto unmatched = buildPanelInputFromSDL(buffer);
        ASSERT_EQ(unmatched.mouse_button_events.size(), 1u);
        EXPECT_FALSE(unmatched.mouse_button_events[0].gui_owned)
            << "an unmatched release inherited an earlier same-button press";

        // ...and neither does a release of a button that never went down.
        lfs::vis::FrameInputBuffer other;
        other.beginFrame();
        other.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 320.0f, 300.0f));
        other.notePressOwner(SDL_BUTTON_LEFT, /*gui_owned=*/true);
        other.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 320.0f, 300.0f));
        ASSERT_EQ(other.mouse_button_events.size(), 2u);
        EXPECT_FALSE(other.mouse_button_events[1].gui_owned)
            << "ownership leaked from the left press to the right release";
    }

    // Preserve arrival order across buttons. Reversing right-then-left invents motion
    // after the left DOWN and can start a drag. RmlPointerReplayTest observes this
    // consequence on a real RmlUi context.
    TEST(ViewportPressOwnershipTest, PressArrivalOrderIsRecordedAtTheButtonDown) {
        using lfs::vis::gui::buildPanelInputFromSDL;

        // RIGHT first, then LEFT, in one buffered frame.
        lfs::vis::FrameInputBuffer right_first;
        right_first.beginFrame();
        right_first.processEvent(mouseDownEvent(SDL_BUTTON_RIGHT, 640.0f, 200.0f));
        right_first.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        settleLiveCursor(right_first, 700.0f, 300.0f);
        const auto right_input = buildPanelInputFromSDL(right_first);
        ASSERT_EQ(right_input.mouse_button_events.size(), 2u);
        EXPECT_EQ(right_input.mouse_button_events[0].button, 1)
            << "the right press arrived first and the stream does not say so";
        EXPECT_EQ(right_input.mouse_button_events[1].button, 0);

        // The same two buttons the other way round must read the other way
        // round: an order that merely encoded the button index would pass the
        // case above and fail here.
        lfs::vis::FrameInputBuffer left_first;
        left_first.beginFrame();
        left_first.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        left_first.processEvent(mouseDownEvent(SDL_BUTTON_RIGHT, 640.0f, 200.0f));
        settleLiveCursor(left_first, 640.0f, 200.0f);
        const auto left_input = buildPanelInputFromSDL(left_first);
        ASSERT_EQ(left_input.mouse_button_events.size(), 2u);
        EXPECT_EQ(left_input.mouse_button_events[0].button, 0);
        EXPECT_EQ(left_input.mouse_button_events[1].button, 1);

        // A repeated press is a THIRD event at its own point, not a re-stamp of
        // the first: nothing is collapsed and nothing moves.
        lfs::vis::FrameInputBuffer repeated;
        repeated.beginFrame();
        repeated.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 700.0f, 300.0f));
        repeated.processEvent(mouseDownEvent(SDL_BUTTON_RIGHT, 640.0f, 200.0f));
        repeated.processEvent(mouseDownEvent(SDL_BUTTON_LEFT, 660.0f, 310.0f));
        settleLiveCursor(repeated, 660.0f, 310.0f);
        const auto repeated_input = buildPanelInputFromSDL(repeated);
        ASSERT_EQ(repeated_input.mouse_button_events.size(), 3u);
        EXPECT_EQ(repeated_input.mouse_button_events[0].button, 0);
        EXPECT_FLOAT_EQ(repeated_input.mouse_button_events[0].x, 700.0f);
        EXPECT_EQ(repeated_input.mouse_button_events[1].button, 1);
        EXPECT_EQ(repeated_input.mouse_button_events[2].button, 0);
        EXPECT_FLOAT_EQ(repeated_input.mouse_button_events[2].x, 660.0f);

        // Last frame's events are cleared, so no ordering can ever be compared
        // across frames.
        right_first.beginFrame();
        EXPECT_TRUE(right_first.mouse_button_events.empty());
    }

    // ------------------------------------------------------------------
    // Escape contract on real RmlUi elements. wantsTextInput excludes ordinary selects,
    // so a text-only outer gate can make select handling unreachable. Whole-file
    // searches miss this. Pin the shared cancellation decision and IME composition's
    // claim on Escape.
    // ------------------------------------------------------------------
    class OverlayEscapeContractTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
        }

        Rml::Element* make(const Rml::String& tag) {
            auto element = document_.CreateElement(tag);
            EXPECT_TRUE(element) << "RmlUi could not instance <" << tag << ">";
            return document_.AppendChild(std::move(element));
        }

        Rml::ElementDocument document_{"body"};
    };

    TEST_F(OverlayEscapeContractTest, CancelTargetsAreTextEditablesAndSelects) {
        using lfs::vis::gui::rml_input::isEscapeCancelTarget;

        auto* const text = make("input");
        ASSERT_NE(text, nullptr);
        text->SetAttribute("type", "text");
        EXPECT_TRUE(isEscapeCancelTarget(text));

        auto* const textarea = make("textarea");
        ASSERT_NE(textarea, nullptr);
        EXPECT_TRUE(isEscapeCancelTarget(textarea));

        // The case the viewport overlay could not reach. A select is NOT a text
        // input -- wantsTextInput() rejects it -- yet Escape must still close it,
        // which is what the sidebar host does.
        auto* const select = make("select");
        ASSERT_NE(select, nullptr);
        EXPECT_FALSE(lfs::vis::gui::rml_input::wantsTextInput(select))
            << "a select must not be treated as a text input";
        EXPECT_TRUE(isEscapeCancelTarget(select));

        // Focus inside an open dropdown lands on a descendant, not the select.
        auto option = document_.CreateElement("div");
        ASSERT_TRUE(option);
        auto* const inside = select->AppendChild(std::move(option));
        ASSERT_NE(inside, nullptr);
        EXPECT_TRUE(isEscapeCancelTarget(inside));

        // Everything else keeps Escape's ordinary meaning.
        EXPECT_FALSE(isEscapeCancelTarget(make("button")));
        EXPECT_FALSE(isEscapeCancelTarget(make("div")));
        EXPECT_FALSE(isEscapeCancelTarget(nullptr));
    }

    TEST_F(OverlayEscapeContractTest, ImeCompositionKeepsEscape) {
        using lfs::vis::gui::rml_input::shouldCancelOnEscape;

        auto* const text = make("input");
        ASSERT_NE(text, nullptr);
        text->SetAttribute("type", "text");

        EXPECT_TRUE(shouldCancelOnEscape(text, /*composing=*/false));
        // While an IME composition is in flight Escape aborts the composition.
        // No host may steal it to revert the field, or the composition can never
        // be cancelled without also throwing the edit away.
        EXPECT_FALSE(shouldCancelOnEscape(text, /*composing=*/true));

        auto* const select = make("select");
        ASSERT_NE(select, nullptr);
        EXPECT_TRUE(shouldCancelOnEscape(select, /*composing=*/false));
        EXPECT_FALSE(shouldCancelOnEscape(select, /*composing=*/true));

        EXPECT_FALSE(shouldCancelOnEscape(nullptr, /*composing=*/false));
    }

    // ------------------------------------------------------------------
    // Pointer delivery through production replay and a headless Rml::Context. Exercise
    // rml_pointer_dispatch.hpp as used by processInput, observing RmlUi dispatches.
    // This does not execute a complete GUI frame.
    //
    // Keep the canonical stream intact. Delivered transitions retain their own point,
    // button and arrival order; skip unowned events without disturbing that order.
    // Ownership must not leak across buttons or repeated presses.
    // ------------------------------------------------------------------
    namespace {

        // The minimum RmlUi needs to instance a context. Nothing is drawn.
        class NullRenderInterface final : public Rml::RenderInterface {
        public:
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>,
                                                        Rml::Span<const int>) override {
                return Rml::CompiledGeometryHandle(1);
            }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f,
                                Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
                dimensions = Rml::Vector2i(1, 1);
                return Rml::TextureHandle(1);
            }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>,
                                               Rml::Vector2i) override {
                return Rml::TextureHandle(1);
            }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        };

        // Record element:event order and each RmlUi button parameter (-1 when absent,
        // such as motion/hover). Ordering and counts alone would let a host replay both
        // buttons as 0; assert identity independently of loop index.
        class PointerEventRecorder final : public Rml::EventListener {
        public:
            void ProcessEvent(Rml::Event& event) override {
                auto* const target = event.GetTargetElement();
                log_.push_back((target ? target->GetId() : Rml::String("<null>")) + ":" +
                               event.GetType());
                buttons_.push_back(event.GetParameter<int>("button", -1));
            }

            void listen(Rml::Element* element) {
                // Observe dragstart: primary DOWN arms drag and later movement starts
                // it. Reversing button order can manufacture that movement and a drag.
                for (const char* type : {"mouseover", "mouseout", "mousemove", "mousedown",
                                         "mouseup", "click", "dragstart", "drag"})
                    element->AddEventListener(type, this);
            }

            void clear() {
                log_.clear();
                buttons_.clear();
            }
            const std::vector<Rml::String>& log() const { return log_; }

            // The `button` parameter of the entry at `index`, or -1 when the
            // index is out of range (so a missing event reads as "no button"
            // rather than crashing the assertion that names it).
            int buttonAt(int index) const {
                if (index < 0 || static_cast<std::size_t>(index) >= buttons_.size())
                    return -1;
                return buttons_[static_cast<std::size_t>(index)];
            }

            // How many entries equal `entry` AND were delivered with `button`.
            int countOfWithButton(const Rml::String& entry, int button) const {
                int count = 0;
                for (std::size_t i = 0; i < log_.size(); ++i) {
                    if (log_[i] == entry && buttons_[i] == button)
                        ++count;
                }
                return count;
            }

            // Index of the first entry equal to `entry`, or -1.
            int indexOf(const Rml::String& entry) const {
                for (std::size_t i = 0; i < log_.size(); ++i) {
                    if (log_[i] == entry)
                        return static_cast<int>(i);
                }
                return -1;
            }

            // How many entries equal `entry`. Distinguishes "this element was
            // pressed once" from "it was pressed once per button".
            int countOf(const Rml::String& entry) const {
                int count = 0;
                for (const auto& logged : log_) {
                    if (logged == entry)
                        ++count;
                }
                return count;
            }

            // Index of the first entry naming `id`, or -1.
            int firstIndexFor(const Rml::String& id) const {
                for (std::size_t i = 0; i < log_.size(); ++i) {
                    if (log_[i].rfind(id + ":", 0) == 0)
                        return static_cast<int>(i);
                }
                return -1;
            }

            // Diagnostic only. Button-carrying entries are suffixed "#<button>"
            // so a failure shows WHICH button was delivered, not just that one
            // was.
            Rml::String joined() const {
                Rml::String out;
                for (std::size_t i = 0; i < log_.size(); ++i) {
                    if (!out.empty())
                        out += " -> ";
                    out += log_[i];
                    if (buttons_[i] >= 0)
                        out += "#" + std::to_string(buttons_[i]);
                }
                return out;
            }

        private:
            std::vector<Rml::String> log_;
            std::vector<int> buttons_;
        };

    } // namespace

    class RmlPointerReplayTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            render_interface_ = new NullRenderInterface();
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
            delete render_interface_;
            render_interface_ = nullptr;
        }

        void SetUp() override {
            context_ = Rml::CreateContext("pointer_delivery", Rml::Vector2i(400, 300),
                                          render_interface_);
            ASSERT_NE(context_, nullptr) << "headless RmlUi context could not be created";

            // Two non-overlapping absolute boxes, far enough apart that a press
            // in one and a cursor in the other cannot be confused.
            static constexpr const char* kDocument =
                "<rml><head><style>"
                "body { width: 400px; height: 300px; }"
                "div { position: absolute; top: 0px; width: 100px; height: 100px; }"
                "#alpha { left: 0px; }"
                "#beta { left: 200px; }"
                // A DRAGGABLE third box, well clear of the other two. Kept
                // separate rather than making `alpha` draggable so the ordering
                // and ownership tests above stay free of drag semantics.
                "#dragger { left: 0px; top: 150px; drag: drag; }"
                "</style></head>"
                "<body><div id=\"alpha\"/><div id=\"beta\"/><div id=\"dragger\"/></body></rml>";

            document_ = context_->LoadDocumentFromMemory(kDocument);
            ASSERT_NE(document_, nullptr);
            document_->Show();
            context_->Update();

            alpha_ = document_->GetElementById("alpha");
            beta_ = document_->GetElementById("beta");
            dragger_ = document_->GetElementById("dragger");
            ASSERT_NE(alpha_, nullptr);
            ASSERT_NE(beta_, nullptr);
            ASSERT_NE(dragger_, nullptr);

            // Guard the fixture itself: if layout did not run, every hover
            // assertion below would pass vacuously against a null hover.
            ASSERT_EQ(context_->GetElementAtPoint(Rml::Vector2f(50.0f, 50.0f)), alpha_);
            ASSERT_EQ(context_->GetElementAtPoint(Rml::Vector2f(250.0f, 50.0f)), beta_);
            ASSERT_EQ(context_->GetElementAtPoint(Rml::Vector2f(50.0f, 200.0f)), dragger_);
            // ...and that `dragger` really is draggable: every dragstart
            // assertion below would pass vacuously against a box RmlUi refuses
            // to drag.
            ASSERT_EQ(dragger_->GetComputedValues().drag(), Rml::Style::Drag::Drag);

            recorder_.listen(alpha_);
            recorder_.listen(beta_);
            recorder_.listen(dragger_);
            recorder_.clear();
        }

        void TearDown() override {
            if (context_)
                Rml::RemoveContext(context_->GetName());
            context_ = nullptr;
            document_ = nullptr;
            alpha_ = nullptr;
            beta_ = nullptr;
            dragger_ = nullptr;
        }

        template <typename OwnsElementFn>
        bool replay(bool (&down)[3], const std::vector<FrameMouseButtonEvent>& events,
                    const OwnsElementFn& owns, const bool capture = false) {
            return gui::rml_input::replayButtonEvents(*context_, down, events, {0.f, 0.f},
                                                      {400.f, 300.f}, 0, capture, owns);
        }

        void press(const int x, const int y) {
            context_->ProcessMouseMove(x, y, 0);
            context_->ProcessMouseButtonDown(0, 0);
        }

        static inline NullRenderInterface* render_interface_ = nullptr;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* alpha_ = nullptr;
        Rml::Element* beta_ = nullptr;
        Rml::Element* dragger_ = nullptr;
        PointerEventRecorder recorder_;
    };

    namespace {

        // Use the production admission question at each point. Boxes represent
        // interactive overlay chrome; nullptr outside the context is never owned.
        auto ownsOverlayBox(std::vector<Rml::Element*> owned) {
            return [owned = std::move(owned)](const Rml::Element* const element) {
                return element != nullptr &&
                       std::find(owned.begin(), owned.end(), element) != owned.end();
            };
        }

        // Everything the SDL intake would have produced, without an SDL_Window:
        // one buffered frame's worth of transitions, each with its own point and
        // its own ownership verdict.
        void pressAt(lfs::vis::FrameInputBuffer& buffer, const int sdl_button,
                     const float x, const float y, const bool gui_owned,
                     const Uint64 timestamp = 0, const int clicks = 0) {
            buffer.processEvent(mouseDownEvent(sdl_button, x, y, timestamp, clicks));
            buffer.notePressOwner(sdl_button, gui_owned);
        }

        // The host's per-button press lifecycle, fresh for one test. The
        // production owner is RmlViewportOverlay::pointer_down_delivered_.
        struct ReplayState {
            bool down_delivered[3] = {};
        };

    } // namespace

    // The host hover pass moves to frame-end position before replay. Every transition
    // must still use its own point: alpha's DOWN must reach alpha even though the frame
    // ended over beta.
    TEST_F(RmlPointerReplayTest, EachEventIsDeliveredAtItsOwnPoint) {
        // The host's own hover move to the frame-end cursor, as processInput
        // makes it before the replay.
        context_->ProcessMouseMove(250, 50, 0); // beta
        recorder_.clear();

        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false); // alpha

        const bool replayed = replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_}));
        EXPECT_TRUE(replayed);

        const auto trace = recorder_.joined();
        const int down = recorder_.indexOf("alpha:mousedown");
        ASSERT_GE(down, 0) << "the DOWN did not land on the pressed element: " << trace;
        EXPECT_EQ(recorder_.buttonAt(down), 0) << trace;
        EXPECT_EQ(context_->GetHoverElement(), alpha_) << trace;
    }

    // Skip unowned dragger while preserving later owned events. Alpha's DOWN must still
    // arrive in its original position as button 0.
    TEST_F(RmlPointerReplayTest, UnownedEventsAreSkippedWithoutDisturbingTheRest) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_RIGHT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        pressAt(buffer, SDL_BUTTON_LEFT, 250.0f, 50.0f, /*gui_owned=*/false);  // beta

        const bool replayed = replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_}));
        EXPECT_TRUE(replayed);

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOf("dragger:mousedown"), 0)
            << "an unowned press was delivered: " << trace;
        EXPECT_EQ(recorder_.countOfWithButton("beta:mousedown", 0), 1) << trace;

        // ...and an event outside the context's rectangle is not owned either.
        recorder_.clear();
        ReplayState outside_state;
        lfs::vis::FrameInputBuffer outside;
        outside.beginFrame();
        pressAt(outside, SDL_BUTTON_LEFT, 900.0f, 900.0f, /*gui_owned=*/false);
        EXPECT_FALSE(replay(outside_state.down_delivered, outside.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(recorder_.log().empty()) << recorder_.joined();
    }

    // A capture this host already owns (the VRAM HUD, the toolbar mid-drag)
    // delivers its events wherever they land -- that is what keeps a drag alive
    // once the pointer leaves the control.
    TEST_F(RmlPointerReplayTest, CaptureDeliversEventsThatLandOnNothingOfOurs) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_}), /*capture=*/true));
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mousedown", 0), 1)
            << recorder_.joined();
    }

    // Replay right-then-left in that order. Reversing them moves onto dragger after
    // primary DOWN arms drag, manufacturing dragstart without user motion.
    TEST_F(RmlPointerReplayTest, RightBeforeLeftStartsNoDragOnTheLeftTarget) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false);  // dragger

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_, dragger_})));

        const auto trace = recorder_.joined();
        const int right_down = recorder_.indexOf("beta:mousedown");
        const int left_down = recorder_.indexOf("dragger:mousedown");
        ASSERT_GE(right_down, 0) << trace;
        ASSERT_GE(left_down, 0) << trace;
        EXPECT_LT(right_down, left_down)
            << "the recorded arrival order was not the delivery order: " << trace;
        EXPECT_EQ(recorder_.buttonAt(right_down), 1) << trace;
        EXPECT_EQ(recorder_.buttonAt(left_down), 0) << trace;
        EXPECT_EQ(recorder_.countOf("dragger:dragstart"), 0)
            << "a drag was manufactured after the left DOWN: " << trace;
    }

    // ...and the same two buttons the other way round, so an implementation
    // that hard-coded one order cannot pass both.
    TEST_F(RmlPointerReplayTest, LeftBeforeRightIsDeliveredLeftFirst) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false);   // alpha
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        const int left_down = recorder_.indexOf("alpha:mousedown");
        const int right_down = recorder_.indexOf("beta:mousedown");
        ASSERT_GE(left_down, 0) << trace;
        ASSERT_GE(right_down, 0) << trace;
        EXPECT_LT(left_down, right_down) << trace;
        EXPECT_EQ(recorder_.buttonAt(left_down), 0) << trace;
        EXPECT_EQ(recorder_.buttonAt(right_down), 1) << trace;
    }

    // A matching UP follows its delivered DOWN's owner, even outside the control. Press
    // owned alpha and release over unowned dragger: hit-testing UP alone skips the
    // release needed to clear RmlUi's active/drag state.
    TEST_F(RmlPointerReplayTest, OwnedReleaseSurvivesTheCursorLeavingTheControl) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false); // alpha
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));   // dragger

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mousedown", 0), 1) << trace;
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mouseup", 0), 1)
            << "the release outside the pressed control was withheld: " << trace;
        EXPECT_FALSE(state.down_delivered[0])
            << "the press lifecycle stayed open after its release";
    }

    // Reject an UP whose DOWN this host never delivered. It cannot borrow another
    // button's ownership, revive a refused press, or end a press merely because release
    // lands on chrome.
    TEST_F(RmlPointerReplayTest, ReleaseIsDeliveredOnlyForAMatchingDown) {
        // (a) a bare release, no press behind it at all.
        ReplayState bare_state;
        lfs::vis::FrameInputBuffer bare;
        bare.beginFrame();
        bare.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 50.0f)); // over alpha
        EXPECT_FALSE(replay(bare_state.down_delivered, bare.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_EQ(recorder_.countOf("alpha:mouseup"), 0) << recorder_.joined();

        // (b) a press this host REFUSED, released over chrome it owns. The
        // refusal is what makes the release unmatched.
        recorder_.clear();
        ReplayState refused_state;
        lfs::vis::FrameInputBuffer refused;
        refused.beginFrame();
        pressAt(refused, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        refused.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 50.0f));     // alpha
        EXPECT_FALSE(replay(refused_state.down_delivered, refused.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(recorder_.log().empty())
            << "a release with no delivered press of its own was replayed: "
            << recorder_.joined();
        EXPECT_FALSE(refused_state.down_delivered[0]);

        // (c) a stale open press cannot lend its right to a LATER press this
        // host refused: the refused DOWN clears the lifecycle.
        recorder_.clear();
        ReplayState stale_state;
        stale_state.down_delivered[0] = true; // an UP eaten by a focus loss
        lfs::vis::FrameInputBuffer stale;
        stale.beginFrame();
        pressAt(stale, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger, refused
        stale.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));
        EXPECT_FALSE(replay(stale_state.down_delivered, stale.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(recorder_.log().empty()) << recorder_.joined();
        EXPECT_FALSE(stale_state.down_delivered[0]);
    }

    // An unowned dragger press followed by frame-end hover over owned alpha must
    // deliver nothing. A nonempty canonical vector forbids aggregate replay, which
    // would invent an alpha press at the wrong point and order. The overlay fallback
    // therefore requires an empty vector.
    TEST_F(RmlPointerReplayTest, UnownedPressThenChromeHoverDeliversNoButtonEvent) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));
        settleLiveCursor(buffer, 50.0f, 50.0f); // ...and the cursor ends over alpha

        const auto input = lfs::vis::gui::buildPanelInputFromSDL(buffer);
        // The frame the fallback must NOT fire on: real events exist, and the
        // aggregate bits that describe them are set.
        ASSERT_FALSE(input.mouse_button_events.empty());
        ASSERT_TRUE(input.mouse_clicked[0]);
        ASSERT_TRUE(input.mouse_released[0]);

        // The host's own hover pass, as processInput makes it before the replay.
        context_->ProcessMouseMove(50, 50, 0);
        recorder_.clear();

        EXPECT_FALSE(replay(state.down_delivered, input.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_EQ(recorder_.countOf("alpha:mousedown"), 0)
            << "a press on something else was manufactured on chrome: "
            << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("alpha:mouseup"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("alpha:click"), 0) << recorder_.joined();
        EXPECT_FALSE(state.down_delivered[0]);
    }

    // Exercise mixed owners across buttons in both directions. A frame-wide some-owned
    // flag would pass only one ordering. The rejected button must acquire nothing while
    // the accepted lifecycle arrives intact.
    TEST_F(RmlPointerReplayTest, RejectedLeftLeavesTheAcceptedRightUntouched) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 200.0f, /*gui_owned=*/false);  // dragger
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 250.0f, 50.0f));

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOf("dragger:mousedown"), 0)
            << "the rejected left press was delivered: " << trace;
        EXPECT_EQ(recorder_.countOf("dragger:mouseup"), 0) << trace;
        EXPECT_EQ(recorder_.countOfWithButton("beta:mousedown", 1), 1) << trace;
        EXPECT_EQ(recorder_.countOfWithButton("beta:mouseup", 1), 1) << trace;
        EXPECT_FALSE(state.down_delivered[0]) << "the rejected press opened a lifecycle";
        EXPECT_FALSE(state.down_delivered[1]);
    }

    TEST_F(RmlPointerReplayTest, RejectedRightTakesNoOwnershipBesideAnAcceptedLeft) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false);   // alpha
        pressAt(buffer, SDL_BUTTON_RIGHT, 50.0f, 200.0f, /*gui_owned=*/false); // dragger
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 50.0f, 50.0f));     // over alpha
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 50.0f));

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mousedown", 0), 1) << trace;
        EXPECT_EQ(recorder_.countOf("dragger:mousedown"), 0) << trace;
        // The right release lands on chrome this host owns, but its press was
        // refused, so it must not ride the left press's lifecycle in.
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mouseup", 1), 0)
            << "the rejected right press took ownership from the accepted left: " << trace;
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mouseup", 0), 1) << trace;
        EXPECT_FALSE(state.down_delivered[0]);
        EXPECT_FALSE(state.down_delivered[1]);
    }

    // Two independently admitted buttons run two independent lifecycles: each
    // release follows ITS OWN press, and neither closes the other's.
    TEST_F(RmlPointerReplayTest, BothAdmittedPressesReleaseTheirOwnButtons) {
        ReplayState state;
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        pressAt(buffer, SDL_BUTTON_LEFT, 50.0f, 50.0f, /*gui_owned=*/false);   // alpha
        pressAt(buffer, SDL_BUTTON_RIGHT, 250.0f, 50.0f, /*gui_owned=*/false); // beta

        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));
        EXPECT_TRUE(state.down_delivered[0]);
        EXPECT_TRUE(state.down_delivered[1]);

        // Released in a LATER frame, each outside the box it started on.
        recorder_.clear();
        buffer.beginFrame();
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_RIGHT, 50.0f, 200.0f)); // dragger
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, 50.0f, 200.0f));  // dragger
        EXPECT_TRUE(replay(state.down_delivered, buffer.mouse_button_events, ownsOverlayBox({alpha_, beta_})));

        const auto trace = recorder_.joined();
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mouseup", 1), 1)
            << "the right release did not follow its own press: " << trace;
        EXPECT_EQ(recorder_.countOfWithButton("dragger:mouseup", 0), 1)
            << "the left release did not follow its own press: " << trace;
        const int right_up = recorder_.indexOf("dragger:mouseup");
        ASSERT_GE(right_up, 0) << trace;
        EXPECT_EQ(recorder_.buttonAt(right_up), 1)
            << "the releases were reordered into button-index order: " << trace;
        EXPECT_FALSE(state.down_delivered[0]);
        EXPECT_FALSE(state.down_delivered[1]);
    }

    // ------------------------------------------------------------------
    // Two complete same-button presses at distinct owned/unowned points within one
    // frame, in both orders. First-DOWN or frame-end coalescing must fail.
    //
    // Derive verdicts from real elements and the replay admission predicate; fixture
    // assertions reject equivalent points. Production intake, copy and replay must
    // preserve all four events, their DOWN/UP identities and their distinct,
    // non-default coordinates, timestamps and click counts.
    //
    // Compare the full delivery sequence except replay-generated mousemove. Reject
    // extra events and ownership leakage: each UP follows its own DOWN, unowned presses
    // are not delivered, and the host lifecycle ends closed.
    // ------------------------------------------------------------------
    namespace {

        struct DoubleClickCase {
            const char* name;
            bool owned_press_first;
        };

        // One press: DOWN then UP at the same point, with its own timestamp and
        // click count, and its ownership verdict taken from `gui_owned`.
        void clickAt(lfs::vis::FrameInputBuffer& buffer, const float x, const float y,
                     const bool gui_owned, const Uint64 timestamp, const int clicks) {
            pressAt(buffer, SDL_BUTTON_LEFT, x, y, gui_owned, timestamp, clicks);
            buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, x, y, timestamp + 1, clicks));
        }

        // Remove only replay-generated mousemove from the delivered sequence: it
        // positions the pointer for each event. Keep every other event so extra
        // presses, releases, clicks and manufactured drags fail the whole-sequence
        // comparison.
        std::vector<Rml::String> withoutCursorMoves(const std::vector<Rml::String>& log) {
            std::vector<Rml::String> out;
            for (const auto& entry : log) {
                if (entry.size() >= 10 && entry.compare(entry.size() - 10, 10, ":mousemove") == 0)
                    continue;
                out.push_back(entry);
            }
            return out;
        }

    } // namespace

    class RmlPointerReplayDoubleClickTest : public RmlPointerReplayTest,
                                            public ::testing::WithParamInterface<DoubleClickCase> {
    };

    TEST_P(RmlPointerReplayDoubleClickTest, TwoPressesOneButtonKeepTheirOwnPointsAndOwners) {
        const auto& params = GetParam();

        // `alpha` is chrome this host owns; `dragger` is not in its owned set.
        // Different x AND different y, so a copy that swapped the two axes could
        // not pass either press.
        constexpr float kOwnedX = 60.0f;
        constexpr float kOwnedY = 50.0f;
        constexpr float kUnownedX = 50.0f;
        constexpr float kUnownedY = 200.0f;

        const auto owns = ownsOverlayBox({alpha_, beta_});
        const auto guiOwnsPoint = [&](const float x, const float y) {
            return owns(context_->GetElementAtPoint(Rml::Vector2f(x, y)));
        };
        // The case is only meaningful if the two points really are on opposite
        // sides of the admission question.
        ASSERT_TRUE(guiOwnsPoint(kOwnedX, kOwnedY));
        ASSERT_FALSE(guiOwnsPoint(kUnownedX, kUnownedY));

        const float first_x = params.owned_press_first ? kOwnedX : kUnownedX;
        const float first_y = params.owned_press_first ? kOwnedY : kUnownedY;
        const float second_x = params.owned_press_first ? kUnownedX : kOwnedX;
        const float second_y = params.owned_press_first ? kUnownedY : kOwnedY;
        const bool first_owned = guiOwnsPoint(first_x, first_y);
        const bool second_owned = guiOwnsPoint(second_x, second_y);
        ASSERT_NE(first_owned, second_owned);

        // Buffer two same-button presses and their releases at distinct points in one
        // frame. Non-default timestamps and click counts distinguish preserved fields
        // from silently dropped defaults.
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        clickAt(buffer, first_x, first_y, first_owned, /*timestamp=*/1100, /*clicks=*/1);
        clickAt(buffer, second_x, second_y, second_owned, /*timestamp=*/1300, /*clicks=*/2);
        settleLiveCursor(buffer, second_x, second_y);

        // --- the COMPLETE canonical stream, through the production copy ------
        const auto input = lfs::vis::gui::buildPanelInputFromSDL(buffer);
        ASSERT_EQ(input.mouse_button_events.size(), 4u)
            << "the stream was coalesced or truncated";
        const auto& events = input.mouse_button_events;

        EXPECT_TRUE(events[0].down);
        EXPECT_FALSE(events[1].down);
        EXPECT_TRUE(events[2].down);
        EXPECT_FALSE(events[3].down);
        for (const auto& event : events)
            EXPECT_EQ(event.button, 0) << "an event changed button identity";

        // EVERY canonical field of every event, in order.
        const float kExpectedX[4] = {first_x, first_x, second_x, second_x};
        const float kExpectedY[4] = {first_y, first_y, second_y, second_y};
        const std::uint64_t kExpectedTimestamp[4] = {1100u, 1101u, 1300u, 1301u};
        const int kExpectedClicks[4] = {1, 1, 2, 2};
        for (int i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(events[i].x, kExpectedX[i]) << "event " << i;
            EXPECT_FLOAT_EQ(events[i].y, kExpectedY[i]) << "event " << i;
            EXPECT_EQ(events[i].timestamp, kExpectedTimestamp[i]) << "event " << i;
            EXPECT_EQ(events[i].clicks, kExpectedClicks[i]) << "event " << i;
        }

        // --- NO OWNERSHIP LEAKAGE -------------------------------------------
        EXPECT_EQ(events[0].gui_owned, first_owned);
        EXPECT_EQ(events[1].gui_owned, first_owned)
            << "the first UP did not follow its own DOWN";
        EXPECT_EQ(events[2].gui_owned, second_owned)
            << "the second press inherited the first press's owner";
        EXPECT_EQ(events[3].gui_owned, second_owned)
            << "the second UP did not follow its own DOWN";

        // --- the COMPLETE GUI-DELIVERY SEQUENCE ------------------------------
        ReplayState state;
        EXPECT_TRUE(replay(state.down_delivered, input.mouse_button_events, owns));

        const auto trace = recorder_.joined();
        // The owned press is delivered whole, at its own point; the unowned one
        // is delivered NOWHERE, in either position in the frame. Compared as the
        // whole sequence, so an extra transition anywhere fails.
        const std::vector<Rml::String> kExpectedDelivery{
            "alpha:mouseover", "alpha:mousedown", "alpha:mouseup", "alpha:click"};
        EXPECT_EQ(withoutCursorMoves(recorder_.log()), kExpectedDelivery)
            << "the delivered sequence was not exactly the owned press: " << trace;

        // ...and the identity of the one press that was delivered.
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mousedown", 0), 1) << trace;
        EXPECT_EQ(recorder_.countOfWithButton("alpha:mouseup", 0), 1) << trace;

        // --- NO STATE LEAKAGE ------------------------------------------------
        EXPECT_FALSE(state.down_delivered[0])
            << "the host still owes RmlUi an UP after both presses closed";
        EXPECT_FALSE(state.down_delivered[1]);
        EXPECT_FALSE(state.down_delivered[2]);
    }

    INSTANTIATE_TEST_SUITE_P(
        BothOwnershipOrderings, RmlPointerReplayDoubleClickTest,
        ::testing::Values(DoubleClickCase{"GuiOwnedFirst", /*owned_press_first=*/true},
                          DoubleClickCase{"ViewportOwnedFirst", /*owned_press_first=*/false}),
        [](const ::testing::TestParamInfo<DoubleClickCase>& info) {
            return std::string(info.param.name);
        });

    // Exercise the compiled overlay entry point used by GuiManager, including
    // its blockers and early returns. Only the headless context attachment is
    // fixture access; ownership, coordinates and delivery are production code.
    class RmlViewportInputRoutingTest : public RmlPointerReplayTest {
    protected:
        void seedToolbarCaptureForCancel() {
            overlay_->viewport_toolbar_position_ = "free";
            overlay_->toolbar_drag_active_ = true;
            // No preference write: this probe never moves the toolbar itself.
            overlay_->toolbar_drag_moved_ = false;
            bar_->AddEventListener(Rml::EventId::Dragend, &overlay_->toolbar_drag_listener_);
        }
        bool toolbarCaptureActive() const { return overlay_->toolbar_drag_active_; }

        void SetUp() override {
            RmlPointerReplayTest::SetUp();
            gui::guiFocusState().reset();
            static constexpr const char* kControls =
                "<rml><head><style>"
                "body { width: 400px; height: 300px; }"
                "input { position: absolute; left: 20px; top: 20px; width: 300px; height: 24px; }"
                "slidertrack { height: 24px; }"
                "sliderbar { width: 20px; height: 24px; }"
                "sliderarrowdec, sliderarrowinc { width: 0px; height: 0px; }"
                "button { position: absolute; left: 20px; top: 100px; width: 100px; height: 40px; }"
                "#text { top: 180px; }"
                "</style></head><body>"
                "<input id=\"range\" type=\"range\" min=\"0\" max=\"100\" step=\"1\" value=\"50\"/>"
                "<button id=\"action\"/><input id=\"text\" type=\"text\" value=\"keep\"/>"
                "</body></rml>";
            controls_ = context_->LoadDocumentFromMemory(kControls);
            ASSERT_NE(controls_, nullptr);
            controls_->Show();
            context_->Update();
            range_ = dynamic_cast<Rml::ElementFormControlInput*>(controls_->GetElementById("range"));
            text_ = dynamic_cast<Rml::ElementFormControlInput*>(controls_->GetElementById("text"));
            button_ = controls_->GetElementById("action");
            ASSERT_NE(range_, nullptr);
            ASSERT_NE(text_, nullptr);
            ASSERT_NE(button_, nullptr);
            bar_ = findElementByTag(range_, "sliderbar");
            ASSERT_NE(bar_, nullptr);
            bar_->SetId("range-bar");
            const auto bar_offset = bar_->GetAbsoluteOffset();
            range_point_ = {bar_offset.x + bar_->GetOffsetWidth() / 2,
                            bar_offset.y + bar_->GetOffsetHeight() / 2};
            ASSERT_EQ(context_->GetElementAtPoint({range_point_.x, range_point_.y}), bar_);
            ASSERT_EQ(context_->GetElementAtPoint({button_point_.x, button_point_.y}), button_);
            ASSERT_EQ(bar_->GetComputedValues().drag(), Rml::Style::Drag::Drag);
            recorder_.listen(bar_);
            recorder_.listen(button_);
            for (const char* event : {"dragend", "dragdrop", "mousescroll"})
                bar_->AddEventListener(event, &recorder_);
            range_->AddEventListener("change", &recorder_);
            text_->AddEventListener("keydown", &recorder_);
            overlay_ = std::make_unique<gui::RmlViewportOverlay>();
            overlay_->rml_context_ = context_;
            overlay_->document_ = controls_;
            overlay_->setViewportBounds(origin_, {400.0f, 300.0f}, {0.0f, 0.0f});
            recorder_.clear();
        }

        void TearDown() override {
            if (overlay_) {
                overlay_->rml_context_ = nullptr;
                overlay_->document_ = nullptr;
                overlay_.reset();
            }
            gui::guiFocusState().reset();
            RmlPointerReplayTest::TearDown();
        }

        static Rml::Element* findElementByTag(Rml::Element* element, const Rml::String& tag) {
            if (element->GetTagName() == tag)
                return element;
            for (int i = 0; i < element->GetNumChildren(true); ++i) {
                if (auto* found = findElementByTag(element->GetChild(i), tag))
                    return found;
            }
            return nullptr;
        }

        gui::PanelInputState inputAt(const glm::vec2 local) const {
            gui::PanelInputState input;
            input.mouse_x = local.x + origin_.x;
            input.mouse_y = local.y + origin_.y;
            return input;
        }

        FrameMouseButtonEvent transition(const bool down, const glm::vec2 local,
                                         const uint8_t button = 0) const {
            return {.button = button,
                    .down = down,
                    .x = local.x + origin_.x,
                    .y = local.y + origin_.y,
                    .timestamp = down ? 1001u : 1002u,
                    .clicks = 1,
                    .gui_owned = true};
        }

        void route(const gui::PanelInputState& input,
                   const gui::ViewportOverlayInputBlockers blockers = {}) {
            const auto original = input.mouse_button_events;
            overlay_->processInput(input, blockers);
            ASSERT_EQ(input.mouse_button_events.size(), original.size());
            for (std::size_t i = 0; i < original.size(); ++i) {
                const auto& event = input.mouse_button_events[i];
                EXPECT_EQ(event.button, original[i].button);
                EXPECT_EQ(event.down, original[i].down);
                EXPECT_FLOAT_EQ(event.x, original[i].x);
                EXPECT_FLOAT_EQ(event.y, original[i].y);
                EXPECT_EQ(event.timestamp, original[i].timestamp);
                EXPECT_EQ(event.clicks, original[i].clicks);
                EXPECT_EQ(event.gui_owned, original[i].gui_owned);
            }
        }

        void arm(const glm::vec2 point, const uint8_t button = 0) {
            auto input = inputAt(point);
            input.mouse_button_events = {transition(true, point, button)};
            input.mouse_clicked[button] = true;
            input.mouse_down[button] = true;
            route(input);
            ASSERT_TRUE(owns(button));
            recorder_.clear();
        }

        bool owns(const uint8_t button = 0) const { return overlay_->pointer_down_delivered_[button]; }
        bool hoveredInteractive() const { return overlay_->hovered_interactive_; }
        bool hasCoordinateOrigin() const { return overlay_->last_valid_input_origin_.has_value(); }

        void expectRangeInert() {
            ASSERT_FALSE(owns());
            EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
            const auto value = range_->GetValue();
            recorder_.clear();
            route(inputAt(range_point_ + glm::vec2(80.0f, 0.0f)));
            EXPECT_EQ(range_->GetValue(), value);
            EXPECT_EQ(recorder_.countOf("range-bar:dragstart"), 0) << recorder_.joined();
            EXPECT_EQ(recorder_.countOf("range:change"), 0) << recorder_.joined();
        }

        std::unique_ptr<gui::RmlViewportOverlay> overlay_;
        Rml::ElementDocument* controls_ = nullptr;
        Rml::ElementFormControlInput* range_ = nullptr;
        Rml::ElementFormControlInput* text_ = nullptr;
        Rml::Element* bar_ = nullptr;
        Rml::Element* button_ = nullptr;
        const glm::vec2 origin_{40.0f, 60.0f};
        const glm::vec2 button_point_{70.0f, 120.0f};
        glm::vec2 range_point_{};
    };

    struct OverlayBlockerCase {
        const char* name;
        gui::ViewportOverlayInputBlockers blockers;
    };

    class RmlViewportInputRoutingBlockerTest : public RmlViewportInputRoutingTest,
                                               public ::testing::WithParamInterface<OverlayBlockerCase> {};

    TEST_P(RmlViewportInputRoutingBlockerTest, HeldRangeDoesNotReceiveMaskedMotionAndTrueReleaseDisarms) {
        arm(range_point_);
        ASSERT_EQ(range_->GetValue(), "50.000000");
        auto held = inputAt({390.0f, 280.0f});
        held.mouse_down[0] = true;
        held.mouse_wheel = 3.0f;
        held.keys_pressed = {SDL_SCANCODE_RIGHT};
        held.text_inputs = {"x"};
        route(held, GetParam().blockers);
        EXPECT_TRUE(owns());
        EXPECT_TRUE(bar_->IsPseudoClassSet("active"));
        EXPECT_EQ(range_->GetValue(), "50.000000");
        EXPECT_TRUE(recorder_.log().empty()) << recorder_.joined();

        auto released = inputAt({390.0f, 280.0f});
        released.mouse_button_events = {transition(false, range_point_)};
        route(released, GetParam().blockers);
        EXPECT_EQ(recorder_.countOf("range-bar:click"), 1) << recorder_.joined();
        EXPECT_EQ(range_->GetValue(), "50.000000");
        expectRangeInert();
    }

    TEST_P(RmlViewportInputRoutingBlockerTest, ButtonReleaseCompletesButNewBlockedPressCannotFocusOrClick) {
        arm(button_point_);
        auto released = inputAt(button_point_);
        released.mouse_button_events = {transition(false, button_point_)};
        route(released, GetParam().blockers);
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf("action:click"), 1) << recorder_.joined();
        recorder_.clear();
        released.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        released.mouse_clicked[0] = true;
        released.mouse_released[0] = true;
        route(released, GetParam().blockers);
        EXPECT_FALSE(owns());
        EXPECT_TRUE(recorder_.log().empty()) << recorder_.joined();
    }

    INSTANTIATE_TEST_SUITE_P(
        GuiBlockers, RmlViewportInputRoutingBlockerTest,
        ::testing::Values(OverlayBlockerCase{"Startup", {.startup = true}},
                          OverlayBlockerCase{"Progress", {.progress = true}},
                          OverlayBlockerCase{"Modal", {.modal = true}},
                          OverlayBlockerCase{"PendingModal", {.pending_modal = true}},
                          OverlayBlockerCase{"ContextMenu", {.context_menu = true}},
                          OverlayBlockerCase{"MenuPointer", {.menu_pointer = true}},
                          OverlayBlockerCase{"FloatingPanel", {.floating_panel = true}}),
        [](const ::testing::TestParamInfo<OverlayBlockerCase>& info) { return info.param.name; });

    TEST_F(RmlViewportInputRoutingTest, CanonicalReleaseBypassesIdleShortcutWithoutAggregateBits) {
        arm(range_point_);
        auto released = inputAt(range_point_);
        released.mouse_button_events = {transition(false, range_point_)};
        route(released);
        EXPECT_EQ(recorder_.countOf("range-bar:click"), 1) << recorder_.joined();
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, BlockedSameButtonInterleavingsKeepTheirOwnLifecycle) {
        arm(button_point_);
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(false, button_point_),
                                     transition(true, button_point_),
                                     transition(false, button_point_)};
        route(input, {.modal = true});
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf("action:click"), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:mousedown"), 0);

        arm(button_point_);
        input.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        route(input, {.modal = true});
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf("action:click"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:mouseup"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:mousedown"), 0) << recorder_.joined();
        // A replacement DOWN cancels the old press without borrowing its UP.
        EXPECT_FALSE(button_->IsPseudoClassSet("active"));
    }

    TEST_F(RmlViewportInputRoutingTest, BlockedReleasesRemainIndependentAcrossButtons) {
        arm(range_point_);
        arm(range_point_, 1);
        auto input = inputAt(range_point_);
        input.mouse_button_events = {transition(false, range_point_, 2), transition(false, range_point_, 1)};
        route(input, {.menu_pointer = true});
        EXPECT_TRUE(owns(0));
        EXPECT_FALSE(owns(1));
        EXPECT_FALSE(owns(2));
        EXPECT_EQ(recorder_.countOfWithButton("range-bar:mouseup", 1), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOfWithButton("range-bar:mouseup", 2), 0);
        input.mouse_button_events = {transition(false, range_point_)};
        route(input, {.menu_pointer = true});
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, OutsideAndMovedReleasesUseEventCoordinatesInsteadOfLatestHover) {
        arm(button_point_);
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(false, {450.0f, 350.0f})};
        route(input, {.floating_panel = true});
        EXPECT_FALSE(owns());
        EXPECT_FALSE(button_->IsPseudoClassSet("active"));
        EXPECT_EQ(recorder_.countOf("action:click"), 0) << recorder_.joined();

        arm(range_point_);
        input = inputAt(range_point_);
        input.mouse_button_events = {transition(false, range_point_ + glm::vec2(60.0f, 0.0f))};
        route(input, {.floating_panel = true});
        EXPECT_EQ(recorder_.countOf("range-bar:dragstart"), 1) << recorder_.joined();
        EXPECT_GE(recorder_.countOf("range:change"), 1);
        EXPECT_NE(range_->GetValue(), "50.000000");
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, CollapsedBoundsUseRetainedLiveContextOriginAndResetOnShutdown) {
        arm(button_point_);
        const auto dimensions = context_->GetDimensions();
        ASSERT_EQ(dimensions, Rml::Vector2i(400, 300));
        overlay_->setViewportBounds({900.0f, 800.0f}, {0.0f, 0.0f}, {0.0f, 0.0f});
        EXPECT_EQ(context_->GetDimensions(), dimensions);
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(false, button_point_)};
        route(input);
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf("action:click"), 1) << recorder_.joined();
        EXPECT_TRUE(hasCoordinateOrigin());
        overlay_->shutdown();
        EXPECT_FALSE(hasCoordinateOrigin());
        EXPECT_FALSE(owns());
        recorder_.clear();
        input.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        route(input);
        EXPECT_TRUE(recorder_.log().empty());
    }

    TEST_F(RmlViewportInputRoutingTest, ExternalCaptureCannotSkipOwnedReleaseOutsideOverlay) {
        arm(range_point_);
        auto outside = inputAt({-50.0f, -50.0f});
        outside.mouse_down[0] = true;
        route(outside);
        ASSERT_FALSE(hoveredInteractive());
        ASSERT_TRUE(owns());
        recorder_.clear();
        gui::guiFocusState().want_capture_mouse = true;
        outside.mouse_down[0] = false;
        outside.mouse_button_events = {transition(false, {-50.0f, -50.0f})};
        route(outside);
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, ExternalCapturePreservesEarlierViewportTextDismissClassification) {
        const glm::vec2 bare_point{350.0f, 250.0f};
        const glm::vec2 outside_point{450.0f, 250.0f};
        auto* const bare_element = context_->GetElementAtPoint({bare_point.x, bare_point.y});
        ASSERT_NE(bare_element, nullptr);
        recorder_.listen(bare_element);
        text_->AddEventListener("blur", &recorder_);
        ASSERT_TRUE(text_->Focus());
        route(inputAt(bare_point));
        ASSERT_FALSE(hoveredInteractive());
        ASSERT_EQ(context_->GetFocusElement(), text_);
        recorder_.clear();

        // A completed viewport click retains its event-time ownership even
        // when later motion leaves the overlay for a capturing GUI panel.
        FrameInputBuffer buffer;
        buffer.beginFrame();
        const auto press_point = bare_point + origin_;
        pressAt(buffer, SDL_BUTTON_LEFT, press_point.x, press_point.y, false, 1001, 1);
        buffer.processEvent(mouseUpEvent(SDL_BUTTON_LEFT, press_point.x, press_point.y, 1002, 1));
        const auto latest_point = outside_point + origin_;
        buffer.processEvent(mouseMotionEvent(latest_point.x, latest_point.y));
        settleLiveCursor(buffer, latest_point.x, latest_point.y);
        const auto input = gui::buildPanelInputFromSDL(buffer);
        ASSERT_EQ(input.mouse_button_events.size(), 2u);
        ASSERT_FALSE(input.mouse_button_events[0].gui_owned);
        ASSERT_FALSE(input.mouse_button_events[1].gui_owned);
        ASSERT_FALSE(input.mouse_down[0]);
        gui::guiFocusState().want_capture_mouse = true;
        route(input);

        EXPECT_NE(context_->GetFocusElement(), text_);
        EXPECT_EQ(recorder_.countOf("text:blur"), 1) << recorder_.joined();
        EXPECT_FALSE(owns());
        EXPECT_EQ(recorder_.countOf(bare_element->GetId() + ":mousedown"), 0) << recorder_.joined();
    }

    TEST_F(RmlViewportInputRoutingTest, BlockedTextInputNeitherBlursNorReceivesKeys) {
        ASSERT_TRUE(text_->Focus());
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        input.keys_pressed = {SDL_SCANCODE_BACKSPACE};
        input.text_inputs = {"x"};
        input.text_codepoints = {U'x'};
        const auto value = text_->GetValue();
        route(input, {.pending_modal = true});
        EXPECT_EQ(context_->GetFocusElement(), text_);
        EXPECT_EQ(text_->GetValue(), value);
        EXPECT_EQ(recorder_.countOf("text:keydown"), 0);
        input.mouse_button_events.clear();
        input.keys_pressed.clear();
        route(input);
        EXPECT_NE(text_->GetValue(), value);
    }

} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(RmlViewportInputRoutingTest, RefusedSameButtonDownMustCancelRangeWithoutHoverEdits) {
        arm(range_point_);
        gui::guiFocusState().reset();
        ASSERT_TRUE(bar_->IsPseudoClassSet("active"));
        const auto before = range_->GetValue();
        const glm::vec2 bare{350.0f, 270.0f};
        auto input = inputAt(bare);
        input.mouse_button_events = {transition(true, bare), transition(false, bare)};
        input.mouse_button_events[0].gui_owned = false;
        input.mouse_button_events[1].gui_owned = false;
        route(input);
        EXPECT_FALSE(overlay_->wantsInput());
        EXPECT_FALSE(gui::guiFocusState().want_capture_mouse);
        EXPECT_FALSE(owns());
        EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
        recorder_.clear();
        route(inputAt({range_point_.x + 60.0f, range_point_.y}));
        EXPECT_EQ(range_->GetValue(), before);
        EXPECT_EQ(recorder_.countOf("range:change"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:dragstart"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:drag"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:click"), 0) << recorder_.joined();
    }

    TEST_F(RmlViewportInputRoutingTest, StartedRangeReplacementCancelsBeforeFrameEndMove) {
        arm(range_point_);
        auto moved = inputAt(range_point_ + glm::vec2(20.0f, 0.0f));
        moved.mouse_down[0] = true;
        route(moved);
        ASSERT_GT(recorder_.countOf("range:change"), 0) << recorder_.joined();
        const auto before = range_->GetValue();
        recorder_.clear();
        const glm::vec2 bare{350.0f, 270.0f};
        auto replacement = inputAt({5.0f, 270.0f});
        replacement.mouse_button_events = {transition(true, bare), transition(false, bare)};
        for (auto& event : replacement.mouse_button_events)
            event.gui_owned = false;
        route(replacement);
        context_->ProcessMouseButtonCancel(0, 0); // Repeated cancellation must stay inert.
        EXPECT_EQ(range_->GetValue(), before);
        EXPECT_EQ(recorder_.countOf("range:change"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:dragend"), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:dragdrop"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:click"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("range-bar:mouseup"), 0) << recorder_.joined();
        expectRangeInert();

        context_->Update();
        const auto offset = bar_->GetAbsoluteOffset();
        const glm::vec2 current{offset.x + bar_->GetOffsetWidth() / 2, offset.y + bar_->GetOffsetHeight() / 2};
        arm(current);
        moved = inputAt(current - glm::vec2(20.0f, 0.0f));
        moved.mouse_down[0] = true;
        route(moved);
        EXPECT_NE(range_->GetValue(), before);
        moved.mouse_down[0] = false;
        moved.mouse_button_events = {transition(false, current - glm::vec2(20.0f, 0.0f))};
        route(moved);
        expectRangeInert();
    }

    TEST_F(RmlViewportInputRoutingTest, AcceptedReplacementClicksOnlyNewPress) {
        arm(button_point_);
        auto replacement = inputAt(button_point_);
        replacement.mouse_button_events = {transition(true, button_point_), transition(false, button_point_)};
        route(replacement);
        EXPECT_FALSE(owns());
        EXPECT_FALSE(button_->IsPseudoClassSet("active"));
        EXPECT_EQ(recorder_.countOf("action:mousedown"), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:mouseup"), 1) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("action:click"), 1) << recorder_.joined();
    }

    TEST_F(RmlViewportInputRoutingTest, CancellationMustRefreshToolbarCaptureBeforeRefusingReplacement) {
        arm(range_point_);
        auto drag = inputAt({range_point_.x + 20.0f, range_point_.y});
        drag.mouse_down[0] = true;
        route(drag);
        ASSERT_TRUE(bar_->IsPseudoClassSet("active"));
        seedToolbarCaptureForCancel();
        ASSERT_TRUE(toolbarCaptureActive());
        PointerEventRecorder counts;
        auto* body = range_->GetParentNode();
        for (auto id : {Rml::EventId::Mousedown, Rml::EventId::Mouseup, Rml::EventId::Click})
            body->AddEventListener(id, &counts);
        gui::guiFocusState().reset();
        const glm::vec2 bare{350, 270};
        auto replacement = inputAt(bare);
        replacement.mouse_button_events = {transition(true, bare), transition(false, bare)};
        for (auto& event : replacement.mouse_button_events)
            event.gui_owned = false;
        route(replacement);
        // The real toolbar Dragend handler ran during cancellation and released capture.
        ASSERT_FALSE(toolbarCaptureActive());
        EXPECT_TRUE(counts.log().empty()) << counts.joined();
        EXPECT_FALSE(overlay_->wantsInput());
        EXPECT_FALSE(gui::guiFocusState().want_capture_mouse);
        EXPECT_FALSE(owns());
        for (auto id : {Rml::EventId::Mousedown, Rml::EventId::Mouseup, Rml::EventId::Click})
            body->RemoveEventListener(id, &counts);
    }
} // namespace lfs::vis

namespace lfs::vis {
    class CancelPseudoElement final : public Rml::Element {
    public:
        explicit CancelPseudoElement(const Rml::String& tag) : Rml::Element(tag) {}
        std::function<void()> on_deactivate;
        void OnPseudoClassChange(const Rml::String& pseudo, const bool activate) override {
            if (pseudo == "active" && !activate && on_deactivate) {
                auto callback = std::move(on_deactivate);
                callback();
            }
        }
    };

    enum class CancelCallbackAction { Move,
                                      Up,
                                      FreshDown,
                                      Remove };
    class RmlCancelCallbackTest : public RmlPointerReplayTest,
                                  public ::testing::WithParamInterface<CancelCallbackAction> {
    protected:
        void SetUp() override {
            RmlPointerReplayTest::SetUp();
            Rml::Factory::RegisterElementInstancer("cancel-probe", &instancer_);
            auto element = Rml::Factory::InstanceElement(document_, "cancel-probe", "cancel-probe", {});
            probe_ = static_cast<CancelPseudoElement*>(element.get());
            probe_->SetId("cancel-probe");
            for (const auto& [name, value] : {std::pair{"position", "absolute"}, {"left", "200px"}, {"top", "150px"}, {"width", "100px"}, {"height", "100px"}, {"drag", "drag"}})
                probe_->SetProperty(name, value);
            document_->AppendChild(std::move(element));
            context_->Update();
            ASSERT_EQ(context_->GetElementAtPoint({250.f, 200.f}), probe_);
            recorder_.listen(probe_);
            press(250, 200);
            ASSERT_TRUE(probe_->IsPseudoClassSet("active"));
            recorder_.clear();
        }
        void TearDown() override {
            if (probe_)
                probe_->on_deactivate = {};
            RmlPointerReplayTest::TearDown();
        }
        static inline Rml::ElementInstancerGeneric<CancelPseudoElement> instancer_;
        CancelPseudoElement* probe_ = nullptr;
    };

    TEST_P(RmlCancelCallbackTest, DetachesOldPressBeforeActiveCallback) {
        bool invoked = false;
        probe_->on_deactivate = [&] {
            invoked = true;
            switch (GetParam()) {
            case CancelCallbackAction::Move: context_->ProcessMouseMove(260, 200, 0); break;
            case CancelCallbackAction::Up: context_->ProcessMouseButtonUp(0, 0); break;
            case CancelCallbackAction::FreshDown:
                press(250, 200);
                break;
            case CancelCallbackAction::Remove: {
                auto* old = probe_;
                probe_ = nullptr;
                old->GetParentNode()->RemoveChild(old).reset();
                break;
            }
            }
        };
        context_->ProcessMouseButtonCancel(0, 0);
        EXPECT_TRUE(invoked);
        EXPECT_EQ(recorder_.countOf("cancel-probe:dragstart"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("cancel-probe:drag"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf("cancel-probe:click"), 0) << recorder_.joined();
        if (GetParam() == CancelCallbackAction::FreshDown) {
            ASSERT_NE(probe_, nullptr);
            EXPECT_EQ(context_->GetHoverElement(), probe_);
            EXPECT_TRUE(probe_->IsPseudoClassSet("hover"));
            EXPECT_TRUE(probe_->IsPseudoClassSet("active"));
            context_->ProcessMouseMove(260, 200, 0);
            context_->ProcessMouseButtonUp(0, 0);
            EXPECT_EQ(recorder_.countOf("cancel-probe:dragstart"), 1) << recorder_.joined();
            EXPECT_EQ(recorder_.countOf("cancel-probe:click"), 1) << recorder_.joined();
        } else {
            press(50, 50);
            context_->ProcessMouseButtonUp(0, 0);
            EXPECT_EQ(recorder_.countOf("alpha:click"), 1) << recorder_.joined();
        }
    }
    INSTANTIATE_TEST_SUITE_P(ReentrantInput, RmlCancelCallbackTest,
                             ::testing::Values(CancelCallbackAction::Move, CancelCallbackAction::Up,
                                               CancelCallbackAction::FreshDown, CancelCallbackAction::Remove));

    class CancelTestClock final : public Rml::SystemInterface {
    public:
        double GetElapsedTime() override { return now; }
        double now = 0;
    };
    class RmlDisabledCancelTest : public RmlViewportInputRoutingTest,
                                  public ::testing::WithParamInterface<bool> {
    protected:
        static void SetUpTestSuite() {
            Rml::SetSystemInterface(&clock_);
            RmlPointerReplayTest::SetUpTestSuite();
        }
        static void TearDownTestSuite() {
            RmlPointerReplayTest::TearDownTestSuite();
            Rml::SetSystemInterface(nullptr);
        }

        static inline CancelTestClock clock_;
    };
    TEST_P(RmlDisabledCancelTest, DisabledSliderTerminatesAndRemainsUsable) {
        const bool arrow_press = GetParam();
        auto* arrow = findElementByTag(range_, "sliderarrowinc");
        ASSERT_NE(arrow, nullptr);
        if (arrow_press) {
            arrow->SetId("cancel-arrow");
            recorder_.listen(arrow);
            arrow->SetProperty("width", "20px");
            arrow->SetProperty("height", "24px");
            context_->Update();
        }
        const auto offset = arrow->GetAbsoluteOffset();
        const int x = arrow_press ? int(offset.x + arrow->GetOffsetWidth() / 2) : int(range_point_.x + 65);
        const int y = arrow_press ? int(offset.y + arrow->GetOffsetHeight() / 2) : int(range_point_.y);
        press(x, y);
        if (!arrow_press) {
            context_->ProcessMouseMove(x + 20, y, 0);
            ASSERT_TRUE(bar_->IsPseudoClassSet("active"));
        }
        const auto before = range_->GetValue();
        ASSERT_NE(before, "50.000000");
        range_->SetDisabled(true);
        recorder_.clear();
        context_->ProcessMouseButtonCancel(0, 0);
        clock_.now += 1.0;
        context_->Update();
        EXPECT_EQ(range_->GetValue(), before);
        EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
        EXPECT_EQ(recorder_.countOf("range:change"), 0) << recorder_.joined();
        const char* target = arrow_press ? "cancel-arrow" : "range-bar";
        EXPECT_EQ(recorder_.countOf(Rml::String(target) + ":click"), 0) << recorder_.joined();
        EXPECT_EQ(recorder_.countOf(Rml::String(target) + ":mouseup"), 0) << recorder_.joined();
        range_->SetDisabled(false);
        clock_.now += 1.0;
        context_->Update();
        EXPECT_EQ(range_->GetValue(), before);
        if (arrow_press) {
            press(x, y);
            context_->ProcessMouseButtonUp(0, 0);
        } else {
            const auto at = bar_->GetAbsoluteOffset();
            const int bx = int(at.x + bar_->GetOffsetWidth() / 2);
            press(bx, y);
            context_->ProcessMouseMove(bx - 20, y, 0);
            context_->ProcessMouseButtonUp(0, 0);
        }
        EXPECT_NE(range_->GetValue(), before);
        EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
    }
    INSTANTIATE_TEST_SUITE_P(SliderParts, RmlDisabledCancelTest, ::testing::Bool());

    TEST_F(RmlPointerReplayTest, CancellationBalancesDragTargetWithoutDrop) {
        dragger_->SetProperty("drag", "drag-drop");
        context_->Update();
        PointerEventRecorder listener;
        for (const auto id : {Rml::EventId::Dragover, Rml::EventId::Dragout, Rml::EventId::Dragdrop})
            beta_->AddEventListener(id, &listener);
        const auto start_drag = [&] {
            press(50, 200);
            context_->ProcessMouseMove(250, 50, 0);
        };
        start_drag();
        ASSERT_FALSE(listener.log().empty());
        ASSERT_EQ(listener.log().back(), "beta:dragover");
        context_->ProcessMouseButtonCancel(0, 0);
        EXPECT_EQ(listener.log(), (std::vector<Rml::String>{"beta:dragover", "beta:dragout"}));
        start_drag();
        ASSERT_FALSE(listener.log().empty());
        ASSERT_EQ(listener.log().back(), "beta:dragover");
        context_->ProcessMouseButtonUp(0, 0);
        EXPECT_EQ(listener.log(), (std::vector<Rml::String>{"beta:dragover", "beta:dragout",
                                                            "beta:dragover", "beta:dragdrop", "beta:dragout"}));
        for (const auto id : {Rml::EventId::Dragover, Rml::EventId::Dragout, Rml::EventId::Dragdrop})
            beta_->RemoveEventListener(id, &listener);
    }
} // namespace lfs::vis

namespace lfs::vis {
    class SliderCancelCallback final : public Rml::EventListener {
    public:
        std::function<void(Rml::Event&)> fn;
        void ProcessEvent(Rml::Event& event) override { fn(event); }
    };
    class RmlSharedSliderCancelTest : public RmlViewportInputRoutingTest,
                                      public ::testing::WithParamInterface<bool> {};

    TEST_P(RmlSharedSliderCancelTest, CancellationPreservesFreshInteractionOnOtherSliderPart) {
        const bool fresh_track = GetParam();
        const auto phase = fresh_track ? Rml::EventId::Dragend : Rml::EventId::Mouseout;
        const int old_x = int(range_point_.x) + (fresh_track ? 0 : 65);
        const int y = int(range_point_.y);
        press(old_x, y);
        context_->ProcessMouseMove(old_x + 20, y, 0);
        ASSERT_TRUE(bar_->IsPseudoClassSet("active"));
        SliderCancelCallback callback;
        int callbacks = 0, releases = 0, clicks = 0, drops = 0, new_x = 0;
        callback.fn = [&](Rml::Event& event) {
            releases += event.GetId() == Rml::EventId::Mouseup;
            clicks += event.GetId() == Rml::EventId::Click;
            drops += event.GetId() == Rml::EventId::Dragdrop;
            if (event.GetId() != phase || callbacks)
                return;
            ++callbacks;
            context_->Update();
            const auto at = bar_->GetAbsoluteOffset();
            new_x = fresh_track ? 290 : int(at.x + bar_->GetOffsetWidth() / 2);
            context_->ProcessMouseMove(new_x, y, 0);
            EXPECT_EQ(context_->GetHoverElement() == bar_, !fresh_track);
            context_->ProcessMouseButtonDown(0, 0);
            context_->ProcessMouseMove(new_x - 10, y, 0);
            EXPECT_TRUE(bar_->IsPseudoClassSet("active"));
        };
        for (const auto id : {phase, Rml::EventId::Mouseup, Rml::EventId::Click, Rml::EventId::Dragdrop})
            range_->AddEventListener(id, &callback);
        context_->ProcessMouseButtonCancel(0, 0);
        EXPECT_EQ(callbacks, 1);
        EXPECT_EQ(releases, 0);
        EXPECT_EQ(clicks, 0);
        EXPECT_EQ(drops, 0);
        EXPECT_TRUE(bar_->IsPseudoClassSet("active"));
        const auto before = range_->GetValue();
        context_->ProcessMouseMove(new_x - 30, y, 0);
        EXPECT_NE(range_->GetValue(), before);
        for (const auto id : {phase, Rml::EventId::Mouseup, Rml::EventId::Click, Rml::EventId::Dragdrop})
            range_->RemoveEventListener(id, &callback);
        context_->ProcessMouseButtonUp(0, 0);
        EXPECT_FALSE(bar_->IsPseudoClassSet("active"));
    }
    INSTANTIATE_TEST_SUITE_P(CleanupCallbacks, RmlSharedSliderCancelTest, ::testing::Bool());

    // Attach a headless context to the real progress overlay and use its action
    // listener, so these tests exercise the same click delivery as GuiManager.
    class RmlProgressInputRoutingTest : public RmlPointerReplayTest {
    protected:
        void SetUp() override {
            RmlPointerReplayTest::SetUp();
            document_->Hide();
            gui::guiFocusState().reset();
            controls_ = context_->LoadDocumentFromMemory(
                "<rml><head><style>body{width:400px;height:300px;}"
                "#progress-action{position:absolute;left:20px;top:100px;width:100px;height:40px;}"
                "</style></head><body><button id='progress-action'/></body></rml>");
            ASSERT_NE(controls_, nullptr);
            for (const char* id : {"progress-backdrop", "progress-dialog", "progress-title", "progress-path",
                                   "progress-row", "progress-value", "progress-text", "progress-stage",
                                   "progress-detail", "progress-error", "progress-actions"}) {
                auto element = controls_->CreateElement("div");
                element->SetId(id);
                controls_->AppendChild(std::move(element));
            }
            controls_->Show();
            context_->Update();
            ASSERT_EQ(context_->GetElementAtPoint({50, 120})->GetId(), "progress-action");
            overlay_ = std::make_unique<gui::RmlProgressOverlay>(
                &manager_, [this] { ++dismissals_; }, [this] { ++cancels_; });
            overlay_->rml_manager_ = nullptr;
            overlay_->rml_context_ = context_;
            overlay_->document_ = controls_;
            overlay_->cacheElements();
            ASSERT_TRUE(overlay_->elements_cached_);
            showVideo();
        }

        void TearDown() override {
            if (overlay_) {
                overlay_->cancelPointerInput();
                context_->UnloadDocument(controls_);
                context_->Update();
                overlay_->rml_context_ = nullptr;
                overlay_->document_ = nullptr;
                overlay_.reset();
            }
            app_store().video_export_overlay_state.set({});
            app_store().import_overlay_state.set({});
            gui::guiFocusState().reset();
            RmlPointerReplayTest::TearDown();
        }

        void showVideo() {
            AppStore::VideoExportOverlayState video;
            video.active = true;
            app_store().video_export_overlay_state.set(video);
            overlay_->presentation_ = gui::makeProgressOverlayPresentation({}, video);
        }

        void showFailedImport() {
            app_store().video_export_overlay_state.set({});
            AppStore::ImportOverlayState state;
            state.show_completion = true;
            state.error = "Invalid cameras";
            app_store().import_overlay_state.set(state);
            overlay_->presentation_ = gui::makeProgressOverlayPresentation(state, {});
        }

        gui::PanelInputState inputAt(const glm::vec2 point) const {
            gui::PanelInputState input;
            input.screen_x = 30.0f;
            input.screen_y = 40.0f;
            input.mouse_x = point.x + input.screen_x;
            input.mouse_y = point.y + input.screen_y;
            return input;
        }

        FrameMouseButtonEvent transition(const bool down, const glm::vec2 point) const {
            return {.button = 0, .down = down, .x = point.x + 30.0f, .y = point.y + 40.0f};
        }

        gui::PanelInputState clickAt(const glm::vec2 point, const glm::vec2 final_point) const {
            auto input = inputAt(final_point);
            input.mouse_clicked[0] = true;
            input.mouse_released[0] = true;
            input.mouse_button_events = {transition(true, point), transition(false, point)};
            return input;
        }

        bool ownsPress() const { return overlay_->pointer_down_delivered_[0]; }
        const glm::vec2 button_point_{50.0f, 120.0f};
        const glm::vec2 outside_point_{350.0f, 250.0f};
        gui::RmlUIManager manager_;
        std::unique_ptr<gui::RmlProgressOverlay> overlay_;
        Rml::ElementDocument* controls_ = nullptr;
        int cancels_ = 0;
        int dismissals_ = 0;
    };

    TEST_F(RmlProgressInputRoutingTest, RecordedCancelClickSurvivesMotionAway) {
        overlay_->processInput(clickAt(button_point_, outside_point_));
        EXPECT_EQ(cancels_, 1);
        EXPECT_FALSE(ownsPress());
        EXPECT_NE(context_->GetHoverElement()->GetId(), "progress-action");
    }

    TEST_F(RmlProgressInputRoutingTest, RecordedFailedImportDismissalSurvivesMotionAway) {
        showFailedImport();
        overlay_->processInput(clickAt(button_point_, outside_point_));
        EXPECT_EQ(dismissals_, 1);
        EXPECT_EQ(cancels_, 0);
    }

    TEST_F(RmlProgressInputRoutingTest, MotionOntoButtonCannotMoveAnOutsideClick) {
        overlay_->processInput(clickAt(outside_point_, button_point_));
        EXPECT_EQ(cancels_, 0);
        EXPECT_EQ(context_->GetHoverElement()->GetId(), "progress-action");
    }

    TEST_F(RmlProgressInputRoutingTest, CanonicalEventsWorkWithoutAggregateFlagsAndDoNotDuplicateClicks) {
        auto input = clickAt(button_point_, button_point_);
        input.mouse_clicked[0] = false;
        input.mouse_released[0] = false;
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 1);
        input.mouse_clicked[0] = true;
        input.mouse_released[0] = true;
        input.mouse_button_events.push_back(transition(true, button_point_));
        input.mouse_button_events.push_back(transition(false, button_point_));
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 3);
        EXPECT_FALSE(ownsPress());
    }

    TEST_F(RmlProgressInputRoutingTest, AggregateOnlyClickRemainsSupported) {
        auto input = clickAt(button_point_, button_point_);
        input.mouse_button_events.clear();
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 1);
        EXPECT_FALSE(ownsPress());
    }

    TEST_F(RmlProgressInputRoutingTest, PressSpansFramesAndOutsideReleaseDisarms) {
        auto input = inputAt(button_point_);
        input.mouse_button_events = {transition(true, button_point_)};
        overlay_->processInput(input);
        ASSERT_TRUE(ownsPress());
        input.mouse_button_events = {transition(false, button_point_)};
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 1);
        input.mouse_button_events = {transition(true, button_point_)};
        overlay_->processInput(input);
        input = inputAt({-100.0f, -100.0f});
        input.mouse_button_events = {transition(false, {-100.0f, -100.0f})};
        overlay_->processInput(input);
        EXPECT_EQ(cancels_, 1);
        EXPECT_FALSE(ownsPress());
        overlay_->processInput(clickAt(button_point_, button_point_));
        EXPECT_EQ(cancels_, 2);
    }

    TEST_F(RmlProgressInputRoutingTest, HiddenOrBlockedOverlayCannotRetainAnArmedAction) {
        for (const bool blocked : {false, true}) {
            auto input = inputAt(button_point_);
            input.mouse_button_events = {transition(true, button_point_)};
            overlay_->processInput(input);
            ASSERT_TRUE(ownsPress());
            if (!blocked)
                app_store().video_export_overlay_state.set({});
            overlay_->processInput(inputAt(button_point_), blocked);
            EXPECT_FALSE(ownsPress());
            showVideo();
            input.mouse_button_events = {transition(false, button_point_)};
            overlay_->processInput(input);
            EXPECT_EQ(cancels_, 0);
        }
        overlay_->processInput(clickAt(button_point_, button_point_));
        EXPECT_EQ(cancels_, 1);
    }

} // namespace lfs::vis
