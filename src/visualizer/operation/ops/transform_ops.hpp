/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "operation/operation.hpp"

namespace lfs::vis::op {

    class LFS_VIS_API TransformTranslate : public Operation {
    public:
        OperationResult execute(SceneManager& scene,
                                const OperatorProperties& props,
                                const std::any& input) override;

        [[nodiscard]] bool poll(SceneManager& scene) const override;
        [[nodiscard]] std::string id() const override { return "transform.translate"; }
        [[nodiscard]] std::string label() const override { return "Translate"; }
        [[nodiscard]] ModifiesFlag modifies() const override { return ModifiesFlag::TRANSFORMS; }
    };

    class LFS_VIS_API TransformRotate : public Operation {
    public:
        OperationResult execute(SceneManager& scene,
                                const OperatorProperties& props,
                                const std::any& input) override;

        [[nodiscard]] bool poll(SceneManager& scene) const override;
        [[nodiscard]] std::string id() const override { return "transform.rotate"; }
        [[nodiscard]] std::string label() const override { return "Rotate"; }
        [[nodiscard]] ModifiesFlag modifies() const override { return ModifiesFlag::TRANSFORMS; }
    };

    class LFS_VIS_API TransformScale : public Operation {
    public:
        OperationResult execute(SceneManager& scene,
                                const OperatorProperties& props,
                                const std::any& input) override;

        [[nodiscard]] bool poll(SceneManager& scene) const override;
        [[nodiscard]] std::string id() const override { return "transform.scale"; }
        [[nodiscard]] std::string label() const override { return "Scale"; }
        [[nodiscard]] ModifiesFlag modifies() const override { return ModifiesFlag::TRANSFORMS; }
    };

    class LFS_VIS_API TransformSet : public Operation {
    public:
        OperationResult execute(SceneManager& scene,
                                const OperatorProperties& props,
                                const std::any& input) override;

        [[nodiscard]] bool poll(SceneManager& scene) const override;
        [[nodiscard]] std::string id() const override { return "transform.set"; }
        [[nodiscard]] std::string label() const override { return "Set Transform"; }
        [[nodiscard]] ModifiesFlag modifies() const override { return ModifiesFlag::TRANSFORMS; }
    };

} // namespace lfs::vis::op
