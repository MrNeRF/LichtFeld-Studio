/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "input/input_types.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::op {

    struct ModalEvent {
        enum class Type : uint8_t {
            NONE,
            MOUSE_BUTTON,
            MOUSE_MOVE,
            MOUSE_SCROLL,
            KEY,
        };

        Type type = Type::NONE;
        std::variant<std::monostate, MouseButtonEvent, MouseMoveEvent, MouseScrollEvent, KeyEvent> data;

        template <typename T>
        [[nodiscard]] const T* as() const {
            return std::get_if<T>(&data);
        }
    };

    class LFS_VIS_API OperatorContext {
    public:
        explicit OperatorContext(SceneManager& scene);

        [[nodiscard]] SceneManager& scene() { return scene_; }
        [[nodiscard]] const SceneManager& scene() const { return scene_; }

        [[nodiscard]] bool hasSelection() const;
        [[nodiscard]] std::vector<std::string> selectedNodes() const;
        [[nodiscard]] std::string activeNode() const;

        void setModalEvent(const ModalEvent& event);
        [[nodiscard]] const ModalEvent* event() const { return current_event_.get(); }
        [[nodiscard]] glm::vec2 mousePosition() const { return last_mouse_pos_; }
        [[nodiscard]] glm::vec2 mouseDelta() const;

    private:
        SceneManager& scene_;
        std::unique_ptr<ModalEvent> current_event_;
        glm::vec2 last_mouse_pos_{0.0f};
    };

} // namespace lfs::vis::op
