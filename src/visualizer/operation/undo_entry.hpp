/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "rendering/dirty_flags.hpp"
#include <any>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis {
    class RenderingManager;
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis::op {

    struct LFS_VIS_API UndoMemoryBreakdown {
        size_t cpu_bytes = 0;
        size_t gpu_bytes = 0;

        [[nodiscard]] size_t totalBytes() const { return cpu_bytes + gpu_bytes; }

        UndoMemoryBreakdown& operator+=(const UndoMemoryBreakdown& other) {
            cpu_bytes += other.cpu_bytes;
            gpu_bytes += other.gpu_bytes;
            return *this;
        }
    };

    struct LFS_VIS_API UndoMetadata {
        std::string id;
        std::string label;
        std::string source = "system";
        std::string scope = "general";
    };

    class LFS_VIS_API HistoryCorruptionError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
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
        [[nodiscard]] virtual UndoMemoryBreakdown memoryBreakdown() const {
            return UndoMemoryBreakdown{
                .cpu_bytes = estimatedBytes(),
                .gpu_bytes = 0,
            };
        }
        virtual bool tryMerge(const UndoEntry& incoming) {
            (void)incoming;
            return false;
        }
        virtual void offloadToCPU() {}
        virtual void restoreToPreferredDevice() {}
        [[nodiscard]] virtual DirtyMask dirtyFlags() const { return DirtyFlag::ALL; }
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

    enum class TensorSwapStorageMode : uint8_t {
        NONE = 0,
        DENSE = 1,
        SPARSE = 2,
    };

    struct TensorSwapStorage {
        TensorSwapStorageMode mode = TensorSwapStorageMode::NONE;
        lfs::core::Tensor indices;
        lfs::core::Tensor stored_values;
        size_t total_size = 0;
        lfs::core::Device device = lfs::core::Device::CUDA;
        lfs::core::DataType dtype = lfs::core::DataType::UInt8;
        bool before_present = false;
        bool after_present = false;

        [[nodiscard]] bool hasChanges() const {
            switch (mode) {
            case TensorSwapStorageMode::DENSE:
                return stored_values.is_valid() || before_present != after_present;
            case TensorSwapStorageMode::SPARSE:
                return (indices.is_valid() && indices.numel() > 0) || before_present != after_present;
            case TensorSwapStorageMode::NONE:
            default:
                return false;
            }
        }

        [[nodiscard]] size_t estimatedBytes() const {
            size_t total = 0;
            if (indices.is_valid()) {
                total += indices.bytes();
            }
            if (stored_values.is_valid()) {
                total += stored_values.bytes();
            }
            return total;
        }

        [[nodiscard]] UndoMemoryBreakdown memoryBreakdown() const;
        void offloadToCPU();
        void restoreToDevice();
    };

    struct TensorPresenceSnapshot {
        std::shared_ptr<lfs::core::Tensor> tensor;
        size_t total_size = 0;
        lfs::core::Device device = lfs::core::Device::CUDA;
        bool present = false;
    };

    class LFS_VIS_API SceneSnapshot : public UndoEntry {
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
        [[nodiscard]] UndoMemoryBreakdown memoryBreakdown() const override;
        void offloadToCPU() override;
        void restoreToPreferredDevice() override;
        [[nodiscard]] DirtyMask dirtyFlags() const override;

    private:
        SceneManager& scene_;
        std::string name_;

        lfs::core::Scene::SelectionStateSnapshot selection_before_;
        lfs::core::Scene::SelectionStateMetadata selection_after_metadata_;
        TensorSwapStorage selection_mask_storage_;

        std::unordered_map<std::string, glm::mat4> transforms_before_;
        std::unordered_map<std::string, glm::mat4> transforms_after_;

        std::unordered_map<std::string, TensorPresenceSnapshot> deleted_masks_before_;
        std::unordered_map<std::string, TensorSwapStorage> deleted_mask_storage_;
        std::optional<TensorPresenceSnapshot> combined_deleted_before_;
        TensorSwapStorage combined_deleted_storage_;

        ModifiesFlag captured_ = ModifiesFlag::NONE;

        void captureDeletedMasks(std::unordered_map<std::string, TensorPresenceSnapshot>& target);
        void compactSelection();
        void compactTopology();
        void applySelection(bool undo_direction);
        void applyTopology(bool undo_direction);

    public:
        [[nodiscard]] size_t estimatedBytes() const override;
    };

    LFS_VIS_API bool pushSceneSnapshotIfChanged(std::unique_ptr<SceneSnapshot> snapshot);

    class LFS_VIS_API TensorUndoEntry : public UndoEntry {
    public:
        using TensorAccessor = std::function<lfs::core::Tensor*()>;

        TensorUndoEntry(std::string name,
                        UndoMetadata metadata,
                        std::string target_name,
                        lfs::core::Tensor before,
                        TensorAccessor accessor);

        void captureAfter();
        [[nodiscard]] bool hasChanges() const;

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string name() const override { return name_; }
        [[nodiscard]] UndoMetadata metadata() const override { return metadata_; }
        [[nodiscard]] size_t estimatedBytes() const override { return storage_.estimatedBytes(); }
        [[nodiscard]] UndoMemoryBreakdown memoryBreakdown() const override { return storage_.memoryBreakdown(); }
        void offloadToCPU() override { storage_.offloadToCPU(); }
        void restoreToPreferredDevice() override { storage_.restoreToDevice(); }
        [[nodiscard]] DirtyMask dirtyFlags() const override;

    private:
        void apply();

        std::string name_;
        UndoMetadata metadata_;
        std::string target_name_;
        TensorAccessor accessor_;
        lfs::core::Tensor before_;
        TensorSwapStorage storage_;
        lfs::core::TensorShape tensor_shape_;
        size_t element_count_ = 0;
        lfs::core::DataType dtype_ = lfs::core::DataType::Float32;
        bool captured_after_ = false;
    };

    class LFS_VIS_API CropBoxUndoEntry : public UndoEntry {
    public:
        CropBoxUndoEntry(SceneManager& scene,
                         RenderingManager* rendering_manager,
                         std::string node_name,
                         lfs::core::CropBoxData before,
                         glm::mat4 transform_before,
                         bool show_before,
                         bool use_before);

        void undo() override;
        void redo() override;
        [[nodiscard]] bool hasChanges() const;
        [[nodiscard]] std::string name() const override { return "cropbox.transform"; }
        [[nodiscard]] UndoMetadata metadata() const override;
        [[nodiscard]] size_t estimatedBytes() const override { return sizeof(*this) + node_name_.size(); }
        [[nodiscard]] DirtyMask dirtyFlags() const override;

    private:
        void captureAfter();

        SceneManager& scene_;
        RenderingManager* rendering_manager_ = nullptr;
        std::string node_name_;
        lfs::core::CropBoxData before_;
        lfs::core::CropBoxData after_;
        glm::mat4 transform_before_;
        glm::mat4 transform_after_;
        bool show_before_ = false;
        bool use_before_ = false;
        bool show_after_ = false;
        bool use_after_ = false;
    };

    class LFS_VIS_API EllipsoidUndoEntry : public UndoEntry {
    public:
        EllipsoidUndoEntry(SceneManager& scene,
                           RenderingManager* rendering_manager,
                           std::string node_name,
                           lfs::core::EllipsoidData before,
                           glm::mat4 transform_before,
                           bool show_before,
                           bool use_before);

        void undo() override;
        void redo() override;
        [[nodiscard]] bool hasChanges() const;
        [[nodiscard]] std::string name() const override { return "ellipsoid.transform"; }
        [[nodiscard]] UndoMetadata metadata() const override;
        [[nodiscard]] size_t estimatedBytes() const override { return sizeof(*this) + node_name_.size(); }
        [[nodiscard]] DirtyMask dirtyFlags() const override;

    private:
        void captureAfter();

        SceneManager& scene_;
        RenderingManager* rendering_manager_ = nullptr;
        std::string node_name_;
        lfs::core::EllipsoidData before_;
        lfs::core::EllipsoidData after_;
        glm::mat4 transform_before_;
        glm::mat4 transform_after_;
        bool show_before_ = false;
        bool use_before_ = false;
        bool show_after_ = false;
        bool use_after_ = false;
    };

    class LFS_VIS_API PropertyChangeUndoEntry : public UndoEntry {
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
        [[nodiscard]] DirtyMask dirtyFlags() const override;

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
        lfs::core::Device device = lfs::core::Device::CUDA;
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

    struct LFS_VIS_API SceneGraphNodeSnapshot {
        SceneGraphNodeSnapshot();
        SceneGraphNodeSnapshot(const SceneGraphNodeSnapshot& other);
        SceneGraphNodeSnapshot(SceneGraphNodeSnapshot&& other) noexcept;
        SceneGraphNodeSnapshot& operator=(const SceneGraphNodeSnapshot& other);
        SceneGraphNodeSnapshot& operator=(SceneGraphNodeSnapshot&& other) noexcept;
        ~SceneGraphNodeSnapshot();

        std::string name;
        std::string parent_name;
        lfs::core::NodeType type = lfs::core::NodeType::SPLAT;
        glm::mat4 local_transform{1.0f};
        bool visible = true;
        bool locked = false;
        bool training_enabled = true;
        size_t gaussian_count = 0;
        glm::vec3 centroid{0.0f};
        lfs::core::Device payload_device = lfs::core::Device::CUDA;
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

    struct LFS_VIS_API SceneGraphStateSnapshot {
        std::vector<SceneGraphNodeSnapshot> roots;
        std::optional<std::vector<std::string>> selected_node_names;
        std::optional<SceneGraphContextSnapshot> context;
    };

    struct LFS_VIS_API SceneGraphNodeMetadataSnapshot {
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

    class LFS_VIS_API SceneGraphMetadataEntry : public UndoEntry {
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
        bool tryMerge(const UndoEntry& incoming) override;
        [[nodiscard]] DirtyMask dirtyFlags() const override;

    private:
        void apply(bool use_after_state);

        SceneManager& scene_;
        std::string name_;
        std::vector<SceneGraphNodeMetadataDiff> diffs_;
        std::chrono::steady_clock::time_point updated_at_;
    };

    class LFS_VIS_API SceneGraphPatchEntry : public UndoEntry {
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
        [[nodiscard]] UndoMemoryBreakdown memoryBreakdown() const override;
        void offloadToCPU() override;
        void restoreToPreferredDevice() override;
        [[nodiscard]] DirtyMask dirtyFlags() const override;

    private:
        void applyState(const SceneGraphStateSnapshot& desired,
                        const SceneGraphStateSnapshot& current);

        SceneManager& scene_;
        std::string name_;
        SceneGraphStateSnapshot before_;
        SceneGraphStateSnapshot after_;
    };

} // namespace lfs::vis::op
