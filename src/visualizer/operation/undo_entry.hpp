/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "core/tensor.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>

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
        [[nodiscard]] bool captureTransformsBefore(const std::vector<std::string>& nodes,
                                                   const std::vector<glm::mat4>& transforms);
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

    class CropBoxUndoEntry : public UndoEntry {
    public:
        CropBoxUndoEntry(SceneManager& scene, std::string node_name,
                         lfs::core::CropBoxData before, glm::mat4 transform_before);

        void undo() override;
        void redo() override;
        [[nodiscard]] bool hasChanges() const;
        [[nodiscard]] std::string name() const override { return "cropbox.transform"; }

    private:
        void captureAfter();

        SceneManager& scene_;
        std::string node_name_;
        lfs::core::CropBoxData before_;
        lfs::core::CropBoxData after_;
        glm::mat4 transform_before_;
        glm::mat4 transform_after_;
    };

    class EllipsoidUndoEntry : public UndoEntry {
    public:
        EllipsoidUndoEntry(SceneManager& scene, std::string node_name,
                           lfs::core::EllipsoidData before, glm::mat4 transform_before);

        void undo() override;
        void redo() override;
        [[nodiscard]] bool hasChanges() const;
        [[nodiscard]] std::string name() const override { return "ellipsoid.transform"; }

    private:
        void captureAfter();

        SceneManager& scene_;
        std::string node_name_;
        lfs::core::EllipsoidData before_;
        lfs::core::EllipsoidData after_;
        glm::mat4 transform_before_;
        glm::mat4 transform_after_;
    };

} // namespace lfs::vis::op
