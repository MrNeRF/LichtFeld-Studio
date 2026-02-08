/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace lfs::core {
    class Scene;
}

namespace lfs::vis {
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis::op {

    class UndoEntry {
    public:
        virtual ~UndoEntry() = default;
        virtual void undo() = 0;
        virtual void redo() = 0;
        [[nodiscard]] virtual std::string name() const = 0;
    };

    using UndoEntryPtr = std::unique_ptr<UndoEntry>;

    enum class ModifiesFlag : uint8_t {
        NONE = 0,
        SELECTION = 1 << 0,
        TRANSFORMS = 1 << 1,
        TOPOLOGY = 1 << 2
    };

    inline ModifiesFlag operator|(ModifiesFlag a, ModifiesFlag b) {
        return static_cast<ModifiesFlag>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }

    inline ModifiesFlag operator&(ModifiesFlag a, ModifiesFlag b) {
        return static_cast<ModifiesFlag>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
    }

    inline bool hasFlag(ModifiesFlag flags, ModifiesFlag flag) {
        return (static_cast<uint8_t>(flags) & static_cast<uint8_t>(flag)) != 0;
    }

    class SceneSnapshot : public UndoEntry {
    public:
        explicit SceneSnapshot(SceneManager& scene, std::string name = "Operation");

        void captureSelection();
        void captureTransforms(const std::vector<std::string>& nodes);
        void captureTopology();
        void captureAfter();

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string name() const override { return name_; }

    private:
        SceneManager& scene_;
        std::string name_;

        std::shared_ptr<lfs::core::Tensor> selection_before_;
        std::shared_ptr<lfs::core::Tensor> selection_after_;

        std::unordered_map<std::string, glm::mat4> transforms_before_;
        std::unordered_map<std::string, glm::mat4> transforms_after_;

        std::shared_ptr<lfs::core::Tensor> deleted_mask_before_;
        std::shared_ptr<lfs::core::Tensor> deleted_mask_after_;

        ModifiesFlag captured_ = ModifiesFlag::NONE;
    };

} // namespace lfs::vis::op
