/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include <any>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis {
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis::op {

    struct LFS_VIS_API UndoMetadata {
        std::string id;
        std::string label;
        std::string source = "system";
        std::string scope = "general";
    };

    class LFS_VIS_API UndoEntry {
    public:
        virtual ~UndoEntry() = default;
        virtual void undo() = 0;
        virtual void redo() = 0;
        [[nodiscard]] virtual std::string name() const = 0;
        [[nodiscard]] virtual UndoMetadata metadata() const {
            const auto entry_name = name();
            return UndoMetadata{
                .id = entry_name,
                .label = entry_name,
            };
        }
        [[nodiscard]] virtual size_t estimatedBytes() const { return 0; }
        virtual bool tryMerge(const UndoEntry& incoming) {
            (void)incoming;
            return false;
        }
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
        [[nodiscard]] UndoMetadata metadata() const override;
        [[nodiscard]] bool hasChanges() const;

    private:
        SceneManager& scene_;
        std::string name_;

        lfs::core::Scene::SelectionStateSnapshot selection_before_;
        lfs::core::Scene::SelectionStateSnapshot selection_after_;

        std::unordered_map<std::string, glm::mat4> transforms_before_;
        std::unordered_map<std::string, glm::mat4> transforms_after_;

        std::unordered_map<std::string, std::shared_ptr<lfs::core::Tensor>> deleted_masks_before_;
        std::unordered_map<std::string, std::shared_ptr<lfs::core::Tensor>> deleted_masks_after_;

        ModifiesFlag captured_ = ModifiesFlag::NONE;

        void captureDeletedMasks(std::unordered_map<std::string, std::shared_ptr<lfs::core::Tensor>>& target);
        void restoreDeletedMasks(const std::unordered_map<std::string, std::shared_ptr<lfs::core::Tensor>>& source);
        [[nodiscard]] size_t selectionBytes(const lfs::core::Scene::SelectionStateSnapshot& snapshot) const;

    public:
        [[nodiscard]] size_t estimatedBytes() const override;
    };

    class CropBoxUndoEntry : public UndoEntry {
    public:
        CropBoxUndoEntry(SceneManager& scene, std::string node_name,
                         lfs::core::CropBoxData before, glm::mat4 transform_before);

        void undo() override;
        void redo() override;
        [[nodiscard]] bool hasChanges() const;
        [[nodiscard]] std::string name() const override { return "cropbox.transform"; }
        [[nodiscard]] UndoMetadata metadata() const override;
        [[nodiscard]] size_t estimatedBytes() const override { return sizeof(*this) + node_name_.size(); }

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
        [[nodiscard]] UndoMetadata metadata() const override;
        [[nodiscard]] size_t estimatedBytes() const override { return sizeof(*this) + node_name_.size(); }

    private:
        void captureAfter();

        SceneManager& scene_;
        std::string node_name_;
        lfs::core::EllipsoidData before_;
        lfs::core::EllipsoidData after_;
        glm::mat4 transform_before_;
        glm::mat4 transform_after_;
    };

    class PropertyChangeUndoEntry : public UndoEntry {
    public:
        PropertyChangeUndoEntry(std::string property_path,
                                std::any before,
                                std::any after,
                                std::function<void(const std::any&)> applier);

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string name() const override { return label_; }
        [[nodiscard]] UndoMetadata metadata() const override;
        [[nodiscard]] size_t estimatedBytes() const override { return estimated_bytes_; }
        bool tryMerge(const UndoEntry& incoming) override;

    private:
        std::string property_path_;
        std::string label_;
        std::any before_;
        std::any after_;
        std::function<void(const std::any&)> applier_;
        size_t estimated_bytes_ = 0;
        std::chrono::steady_clock::time_point updated_at_;
    };

    enum class SceneGraphCaptureMode : uint8_t {
        FULL,
        METADATA_ONLY,
    };

    struct SceneGraphCaptureOptions {
        SceneGraphCaptureMode mode = SceneGraphCaptureMode::FULL;
        bool include_selected_nodes = true;
        bool include_scene_context = true;
    };

    struct SceneGraphCameraSnapshot {
        lfs::core::Tensor R;
        lfs::core::Tensor T;
        lfs::core::Tensor radial_distortion;
        lfs::core::Tensor tangential_distortion;
        lfs::core::CameraModelType camera_model_type = lfs::core::CameraModelType::PINHOLE;
        std::string image_name;
        std::filesystem::path image_path;
        std::filesystem::path mask_path;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        int camera_width = 0;
        int camera_height = 0;
        int image_width = 0;
        int image_height = 0;
        int uid = -1;
        int camera_id = 0;
    };

    struct SceneGraphNodeSnapshot {
        std::string name;
        std::string parent_name;
        lfs::core::NodeType type = lfs::core::NodeType::SPLAT;
        glm::mat4 local_transform{1.0f};
        bool visible = true;
        bool locked = false;
        bool training_enabled = true;
        size_t gaussian_count = 0;
        glm::vec3 centroid{0.0f};
        std::optional<std::filesystem::path> source_path;
        std::unique_ptr<lfs::core::SplatData> model;
        std::shared_ptr<lfs::core::PointCloud> point_cloud;
        std::shared_ptr<lfs::core::MeshData> mesh;
        std::unique_ptr<lfs::core::CropBoxData> cropbox;
        std::unique_ptr<lfs::core::EllipsoidData> ellipsoid;
        std::unique_ptr<lfs::core::KeyframeData> keyframe;
        std::optional<SceneGraphCameraSnapshot> camera;
        std::vector<SceneGraphNodeSnapshot> children;
    };

    struct SceneGraphContextSnapshot {
        int content_type = 0;
        std::filesystem::path dataset_path;
        std::string training_model_node_name;
    };

    struct SceneGraphStateSnapshot {
        std::vector<SceneGraphNodeSnapshot> roots;
        std::optional<std::vector<std::string>> selected_node_names;
        std::optional<SceneGraphContextSnapshot> context;
    };

    struct SceneGraphNodeMetadataSnapshot {
        std::string name;
        std::string parent_name;
        glm::mat4 local_transform{1.0f};
        bool visible = true;
        bool locked = false;
        bool training_enabled = true;
        std::optional<std::filesystem::path> source_path;
    };

    struct SceneGraphNodeMetadataDiff {
        SceneGraphNodeMetadataSnapshot before;
        SceneGraphNodeMetadataSnapshot after;
    };

    class SceneGraphMetadataEntry : public UndoEntry {
    public:
        static std::vector<SceneGraphNodeMetadataSnapshot> captureNodes(const SceneManager& scene,
                                                                        const std::vector<std::string>& node_names);

        SceneGraphMetadataEntry(SceneManager& scene,
                                std::string name,
                                std::vector<SceneGraphNodeMetadataDiff> diffs);

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string name() const override { return name_; }
        [[nodiscard]] UndoMetadata metadata() const override;
        [[nodiscard]] size_t estimatedBytes() const override;

    private:
        void apply(bool use_after_state);

        SceneManager& scene_;
        std::string name_;
        std::vector<SceneGraphNodeMetadataDiff> diffs_;
    };

    class SceneGraphPatchEntry : public UndoEntry {
    public:
        static SceneGraphStateSnapshot captureState(const SceneManager& scene,
                                                   const std::vector<std::string>& root_names,
                                                   SceneGraphCaptureOptions options = {});

        SceneGraphPatchEntry(SceneManager& scene,
                            std::string name,
                            SceneGraphStateSnapshot before,
                            SceneGraphStateSnapshot after);

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string name() const override { return name_; }
        [[nodiscard]] UndoMetadata metadata() const override;
        [[nodiscard]] size_t estimatedBytes() const override;

    private:
        void applyState(const SceneGraphStateSnapshot& desired,
                        const SceneGraphStateSnapshot& current);

        SceneManager& scene_;
        std::string name_;
        SceneGraphStateSnapshot before_;
        SceneGraphStateSnapshot after_;
    };

} // namespace lfs::vis::op
