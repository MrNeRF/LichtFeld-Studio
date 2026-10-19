/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/editor_context.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "input/sdl_key_mapping.hpp"
#include "internal/viewport.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/selection_ops.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/selection_tool.hpp"
#include "tools/tool_base.hpp"

#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_scancode.h>

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace lfs::vis {
    namespace {

        struct PersistOff {
            PersistOff() { input::InputBindings::setPersistenceEnabled(false); }
            ~PersistOff() { input::InputBindings::setPersistenceEnabled(true); }
        };

        // Level-1 letters the rows actually press. Physical scancode, then the printed key.
        struct Letter {
            SDL_Scancode scan;
            int logical;
        };

        constexpr Letter kColemakE{SDL_SCANCODE_E, input::KEY_F};
        constexpr Letter kColemakT{SDL_SCANCODE_T, input::KEY_G};
        constexpr Letter kColemakS{SDL_SCANCODE_S, input::KEY_R};
        constexpr Letter kUsR{SDL_SCANCODE_R, input::KEY_R};
        constexpr Letter kDvorakR{SDL_SCANCODE_O, input::KEY_R};

        std::unique_ptr<lfs::core::SplatData> makeChordSplat() {
            constexpr size_t count = 3;
            auto zeros = [](std::initializer_list<size_t> shape) {
                return lfs::core::Tensor::zeros(shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            };
            std::vector<float> rotation(count * 4, 0.0f);
            rotation[0] = rotation[4] = rotation[8] = 1.0f;
            return std::make_unique<lfs::core::SplatData>(
                1, zeros({count, 3}), zeros({count, 1, 3}), zeros({count, 3, 3}), zeros({count, 3}),
                lfs::core::Tensor::from_vector(rotation, {count, size_t{4}}, lfs::core::Device::CUDA)
                    .to(lfs::core::DataType::Float32),
                zeros({count, 1}), 1.0f);
        }

        class InputLayoutChordTest : public ::testing::Test {
        protected:
            PersistOff persist_;
            Viewport viewport_{1920, 1080};
            InputController controller_{nullptr, viewport_};
            std::unique_ptr<RenderingManager> rendering_;
            std::unique_ptr<SceneManager> scene_;
            std::shared_ptr<tools::SelectionTool> tool_ = std::make_shared<tools::SelectionTool>();
            ToolContext tool_context_{nullptr, nullptr, nullptr, nullptr};

            void SetUp() override {
                lfs::event::EventBridge::instance().clear_all();
                gui::guiFocusState().reset();
                lfs::vis::services().clear();
                lfs::vis::op::undoHistory().clear();
                rendering_ = std::make_unique<RenderingManager>();
                scene_ = std::make_unique<SceneManager>();
                lfs::vis::services().set(rendering_.get());
                lfs::vis::services().set(scene_.get());
                scene_->getScene().addSplat("chord", makeChordSplat());
                scene_->initSelectionService();
                auto* const service = scene_->getSelectionService();
                service->setTestingViewport({.x = 0, .y = 0, .width = 1920, .height = 1080, .render_width = 1920, .render_height = 1080});
                service->setTestingScreenPositions(std::make_shared<lfs::core::Tensor>(
                    lfs::core::Tensor::from_vector(std::vector<float>{100, 100, 180, 100, 400, 100}, {size_t{3}, size_t{2}}, lfs::core::Device::CUDA)
                        .to(lfs::core::DataType::Float32)));
                lfs::vis::op::operators().setSceneManager(scene_.get());
                lfs::vis::op::registerSelectionOperators();
                controller_.initialize();
                tool_->setEnabled(true);
                controller_.setSelectionTool(tool_);
                tool_context_ = ToolContext(rendering_.get(), scene_.get(), &viewport_, nullptr);
                controller_.setToolContext(&tool_context_);
            }

            void TearDown() override {
                lfs::vis::op::operators().cancelModalOperator();
                lfs::vis::op::operators().unregisterOperator(lfs::vis::op::BuiltinOp::SelectionStroke);
                lfs::vis::op::operators().setSceneManager(nullptr);
                lfs::vis::services().clear();
                lfs::vis::op::undoHistory().clear();
                gui::guiFocusState().reset();
                lfs::event::EventBridge::instance().clear_all();
            }

            void key(Letter letter, int action) {
                controller_.handleKey(input::sdlScancodeToAppKey(letter.scan), letter.logical,
                                      static_cast<int>(letter.scan), action, input::KEYMOD_NONE);
            }

            [[nodiscard]] bool selectionActive() const {
                return scene_->getSelectionService()->isInteractiveSelectionActive() &&
                       lfs::vis::op::operators().activeModalId() ==
                           lfs::vis::op::to_string(lfs::vis::op::BuiltinOp::SelectionStroke);
            }

            [[nodiscard]] std::vector<uint8_t> mask() const {
                const auto value = scene_->getScene().getSelectionMask();
                if (!value || !value->is_valid())
                    return {};
                return value->cpu().to_vector_uint8();
            }
        };

        void expectWheel(InputController& controller, Viewport& viewport, bool rolls) {
            const auto rotation = viewport.camera.R;
            const auto translation = viewport.camera.t;
            controller.handleScroll(0.0, 1.0);
            float rolled = 0.0f;
            for (int column = 0; column < 3; ++column)
                rolled = std::max(rolled, glm::distance(rotation[column], viewport.camera.R[column]));
            const float moved = glm::distance(translation, viewport.camera.t);
            if (rolls) {
                EXPECT_GT(rolled, 1e-3f);
                EXPECT_LT(moved, 1e-4f);
            } else {
                EXPECT_LT(rolled, 1e-4f);
                EXPECT_GT(moved, 1e-3f);
            }
        }

        // Colemak physical E prints F (focus). The rebound row prints G (GT compare).
        TEST_F(InputLayoutChordTest, DivergentMovementPressRepeatRelease) {
            struct Row {
                Letter letter;
                int rebind;
                input::Action shortcut;
            };
            const Row rows[] = {
                {kColemakE, 0, input::Action::CAMERA_FOCUS_SELECTION},
                {kColemakT, input::KEY_T, input::Action::TOGGLE_GT_COMPARISON},
            };
            for (const auto& row : rows) {
                if (row.rebind != 0) {
                    controller_.getBindings().setBinding(
                        input::ToolMode::GLOBAL, input::Action::CAMERA_MOVE_DOWN,
                        input::KeyTrigger{row.rebind, input::MODIFIER_NONE, true});
                }
                EXPECT_EQ(controller_.getBindings().getActionForKey(
                              input::ToolMode::GLOBAL, row.letter.logical, input::KEYMOD_NONE),
                          row.shortcut);
                lfs::event::ScopedHandler handlers;
                int comparisons = 0;
                handlers.subscribe<core::events::cmd::ToggleGTComparison>([&](const auto&) { ++comparisons; });
                const auto translation = viewport_.camera.t;
                for (const int action : {input::ACTION_PRESS, input::ACTION_REPEAT}) {
                    key(row.letter, action);
                    EXPECT_TRUE(controller_.isContinuousInputActive()) << row.letter.logical;
                }
                EXPECT_EQ(comparisons, 0);
                EXPECT_LT(glm::distance(translation, viewport_.camera.t), 1e-4f);
                key(row.letter, input::ACTION_RELEASE);
                EXPECT_FALSE(controller_.isContinuousInputActive()) << row.letter.logical;
            }
        }

        TEST_F(InputLayoutChordTest, PhysicalMovementDoesNotArmScrollChord) {
            struct Row {
                const char* layout;
                Letter letter;
                bool zooms;
            };
            const Row rows[] = {
                {"colemak", kColemakS, true},
                {"us", kUsR, false},
                {"azerty", kUsR, false},
                {"dvorak", kDvorakR, false},
            };
            for (const auto& row : rows) {
                key(row.letter, input::ACTION_PRESS);
                expectWheel(controller_, viewport_, !row.zooms);
                EXPECT_EQ(controller_.isContinuousInputActive(), row.zooms) << row.layout;
                key(row.letter, input::ACTION_RELEASE);
            }
        }

        TEST_F(InputLayoutChordTest, SelectionAndPanButtonOrder) {
            constexpr int left = static_cast<int>(input::AppMouseButton::LEFT);
            constexpr int right = static_cast<int>(input::AppMouseButton::RIGHT);
            const auto translation = viewport_.camera.t;

            controller_.handleMouseButton(left, input::ACTION_PRESS, 100.0, 100.0);
            controller_.handleMouseMove(140.0, 100.0);
            ASSERT_TRUE(selectionActive());
            controller_.handleMouseButton(right, input::ACTION_PRESS, 140.0, 100.0);
            controller_.handleMouseMove(180.0, 100.0);
            EXPECT_TRUE(selectionActive());
            EXPECT_FALSE(controller_.isCameraDragging());
            EXPECT_LT(glm::distance(translation, viewport_.camera.t), 1e-5f);
            controller_.handleMouseButton(left, input::ACTION_RELEASE, 180.0, 100.0);
            controller_.handleMouseButton(right, input::ACTION_RELEASE, 180.0, 100.0);
            const auto selected = mask();
            ASSERT_EQ(selected.size(), 3u);
            EXPECT_EQ(selected[0], 1u);
            EXPECT_EQ(selected[1], 1u);
            EXPECT_EQ(selected[2], 0u);

            controller_.handleMouseButton(right, input::ACTION_PRESS, 100.0, 100.0);
            ASSERT_TRUE(controller_.isCameraDragging());
            const auto origin = viewport_.camera.t;
            controller_.handleMouseMove(110.0, 100.0);
            const float step = glm::distance(origin, viewport_.camera.t);
            const auto parked = viewport_.camera.t;
            controller_.handleMouseButton(left, input::ACTION_PRESS, 110.0, 100.0);
            ASSERT_TRUE(selectionActive());
            controller_.handleMouseMove(400.0, 100.0);
            EXPECT_LT(glm::distance(parked, viewport_.camera.t), 1e-5f);
            EXPECT_TRUE(controller_.isCameraDragging());
            controller_.handleMouseButton(left, input::ACTION_RELEASE, 400.0, 100.0);
            const auto after = mask();
            ASSERT_EQ(after.size(), 3u);
            EXPECT_EQ(after[2], 1u);
            controller_.handleMouseMove(410.0, 100.0);
            const float resume = glm::distance(parked, viewport_.camera.t);
            EXPECT_GT(resume, step * 0.4f);
            EXPECT_LT(resume, step * 2.5f);
        }

        TEST_F(InputLayoutChordTest, AltGrModeRejectsAndCtrlRightAltCaptures) {
            auto& bindings = controller_.getBindings();
            const auto scale_before = bindings.getTriggerDescription(input::Action::TOOL_SCALE, input::ToolMode::GLOBAL);
            const auto down_before = bindings.getTriggerDescription(input::Action::CAMERA_MOVE_DOWN, input::ToolMode::GLOBAL);
            const int mode = input::sdlModsToAppMods(SDL_KMOD_MODE);
            const int ctrl_right_alt = input::sdlModsToAppMods(static_cast<SDL_Keymod>(SDL_KMOD_RALT | SDL_KMOD_CTRL));

            // German AltGr+4 types U+00BC. The number row still resolves to 4, and that chord must not select Scale.
            const int quarter = input::sdlShortcutKeyToAppKey(SDL_SCANCODE_4, static_cast<SDL_Keycode>(0x00BC));
            EXPECT_EQ(quarter, input::KEY_4);
            EXPECT_EQ(bindings.getActionForKey(input::ToolMode::GLOBAL, quarter, mode), input::Action::NONE);
            lfs::event::ScopedHandler tools;
            int tool = -1;
            tools.subscribe<core::events::tools::SetToolbarTool>([&](const auto& event) { tool = event.tool_mode; });
            const int digit = input::sdlScancodeToAppKey(SDL_SCANCODE_4);
            controller_.handleKey(digit, quarter, static_cast<int>(SDL_SCANCODE_4), input::ACTION_PRESS, mode);
            EXPECT_EQ(tool, -1);
            controller_.handleKey(digit, input::KEY_4, static_cast<int>(SDL_SCANCODE_4), input::ACTION_PRESS, input::KEYMOD_NONE);
            EXPECT_EQ(tool, static_cast<int>(ToolType::Scale));

            bindings.startCapture(input::ToolMode::GLOBAL, input::Action::TOOL_SCALE);
            bindings.captureKey(input::KEY_E, input::KEY_E, mode);
            EXPECT_TRUE(bindings.isCapturing());
            EXPECT_TRUE(bindings.takeAltGrCaptureRejection());
            EXPECT_EQ(bindings.getTriggerDescription(input::Action::TOOL_SCALE, input::ToolMode::GLOBAL), scale_before);

            bindings.captureKey(input::KEY_E, input::KEY_E, ctrl_right_alt);
            EXPECT_FALSE(bindings.isCapturing());
            EXPECT_FALSE(bindings.takeAltGrCaptureRejection());
            EXPECT_EQ(bindings.getTriggerDescription(input::Action::CAMERA_MOVE_DOWN, input::ToolMode::GLOBAL), down_before);
            const auto captured = bindings.getTriggerForAction(input::Action::TOOL_SCALE, input::ToolMode::GLOBAL);
            ASSERT_TRUE(captured.has_value());
            const auto* captured_key = std::get_if<input::KeyTrigger>(&*captured);
            ASSERT_NE(captured_key, nullptr);
            EXPECT_EQ(captured_key->key, input::KEY_E);
            EXPECT_EQ(captured_key->modifiers & input::KEYMOD_CTRL, input::KEYMOD_CTRL);
            EXPECT_EQ(captured_key->modifiers & input::KEYMOD_ALT, input::KEYMOD_ALT);
            EXPECT_EQ(captured_key->modifiers & input::KEYMOD_ALTGR, 0);

            bindings.startCapture(input::ToolMode::GLOBAL, input::Action::TOOL_SCALE);
            bindings.captureKey(input::KEY_E, input::KEY_E, mode);
            EXPECT_TRUE(bindings.isCapturing());
            EXPECT_TRUE(bindings.takeAltGrCaptureRejection());
            const auto kept = bindings.getTriggerForAction(input::Action::TOOL_SCALE, input::ToolMode::GLOBAL);
            ASSERT_TRUE(kept.has_value());
            const auto* kept_key = std::get_if<input::KeyTrigger>(&*kept);
            ASSERT_NE(kept_key, nullptr);
            EXPECT_EQ(kept_key->key, captured_key->key);
            EXPECT_EQ(kept_key->modifiers, captured_key->modifiers);
        }

    } // namespace
} // namespace lfs::vis
