/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

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

    } // namespace
} // namespace lfs::vis
