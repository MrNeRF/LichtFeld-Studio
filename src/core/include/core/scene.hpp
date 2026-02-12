/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/animatable_property.hpp"
#include "core/camera.hpp"
#include "core/export.hpp"
#include "core/mesh_data.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include <atomic>
#include <cassert>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::core {

    using NodeId = int32_t;
    constexpr NodeId NULL_NODE = -1;

    enum class NodeType : uint8_t {
        SPLAT,          // Contains gaussian splat data
        POINTCLOUD,     // Contains point cloud (pre-training, can be cropped)
        GROUP,          // Empty transform node for organization
        CROPBOX,        // Crop box visualization (child of SPLAT, POINTCLOUD, or DATASET)
        ELLIPSOID,      // Ellipsoid selection (child of SPLAT, POINTCLOUD, or DATASET)
        DATASET,        // Root node for training dataset (contains cameras + model)
        CAMERA_GROUP,   // Container for camera nodes (e.g., "Training", "Validation")
        CAMERA,         // Individual camera from dataset (may have mask_path)
        IMAGE_GROUP,    // Container for image nodes
        IMAGE,          // Individual image file reference (not loaded, just path)
        MESH,           // Triangle mesh (imported via Assimp, processed via OpenMesh)
        KEYFRAME_GROUP, // Container for keyframe nodes (camera animation)
        KEYFRAME        // Individual camera animation keyframe
    };

    struct CropBoxData {
        glm::vec3 min{-1.0f, -1.0f, -1.0f};
        glm::vec3 max{1.0f, 1.0f, 1.0f};
        bool inverse = false; // Invert crop (keep outside instead of inside)
        bool enabled = false; // Whether to use for filtering gaussians
        glm::vec3 color{1.0f, 1.0f, 0.0f};
        float line_width = 2.0f;
        float flash_intensity = 0.0f;
    };

    struct EllipsoidData {
        glm::vec3 radii{1.0f, 1.0f, 1.0f};
        bool inverse = false;
        bool enabled = false;
        glm::vec3 color{1.0f, 1.0f, 0.0f};
        float line_width = 2.0f;
        float flash_intensity = 0.0f;
    };

    inline constexpr float DEFAULT_KEYFRAME_FOCAL_MM = 35.0f;

    struct KeyframeData {
        size_t keyframe_index = 0;
        float time = 0.0f;
        glm::vec3 position{0.0f};
        glm::quat rotation{1, 0, 0, 0};
        float focal_length_mm = DEFAULT_KEYFRAME_FOCAL_MM;
        uint8_t easing = 0; // 0=LINEAR, 1=EASE_IN, 2=EASE_OUT, 3=EASE_IN_OUT
    };

    struct SelectionGroup {
        uint8_t id = 0; // 1-255, 0 means unselected
        std::string name;
        glm::vec3 color{1.0f, 0.0f, 0.0f};
        size_t count = 0;    // Number of selected Gaussians
        bool locked = false; // If true, painting with other groups won't overwrite
    };

    class Scene;

    class LFS_CORE_API SceneNode {
    public:
        SceneNode() = default;
        explicit SceneNode(Scene* scene);

        void initObservables(Scene* scene);

        NodeId id = NULL_NODE;
        NodeId parent_id = NULL_NODE;
        std::vector<NodeId> children;
        NodeType type = NodeType::SPLAT;
        std::string name;

        std::unique_ptr<lfs::core::SplatData> model;
        std::shared_ptr<lfs::core::PointCloud> point_cloud;
        std::shared_ptr<lfs::core::MeshData> mesh;
        std::unique_ptr<CropBoxData> cropbox;
        std::unique_ptr<EllipsoidData> ellipsoid;
        std::unique_ptr<KeyframeData> keyframe;
        size_t gaussian_count = 0;
        glm::vec3 centroid{0.0f};

        std::shared_ptr<lfs::core::Camera> camera;
        int camera_uid = -1;
        bool training_enabled = true;

        std::string image_path;
        std::string mask_path;

        mutable glm::mat4 world_transform{1.0f};
        mutable bool transform_dirty = true;

        lfs::core::prop::AnimatableProperty<glm::mat4> local_transform{glm::mat4{1.0f}};
        lfs::core::prop::AnimatableProperty<bool> visible{true};
        lfs::core::prop::AnimatableProperty<bool> locked{false};

        [[nodiscard]] const glm::mat4& transform() const { return local_transform.get(); }

    private:
        Scene* scene_ = nullptr;
    };

    class LFS_CORE_API Scene {
    public:
        using Node = SceneNode;

        enum class MutationType : uint32_t {
            NODE_ADDED = 1 << 0,
            NODE_REMOVED = 1 << 1,
            NODE_RENAMED = 1 << 2,
            NODE_REPARENTED = 1 << 3,
            TRANSFORM_CHANGED = 1 << 4,
            VISIBILITY_CHANGED = 1 << 5,
            MODEL_CHANGED = 1 << 6,
            SELECTION_CHANGED = 1 << 7,
            CLEARED = 1 << 8,
        };

        void notifyMutation(MutationType type);

        class LFS_CORE_API Transaction {
        public:
            explicit Transaction(Scene& scene);
            ~Transaction();
            Transaction(const Transaction&) = delete;
            Transaction& operator=(const Transaction&) = delete;

        private:
            Scene& scene_;
        };

        Scene();
        ~Scene() = default;

        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        Scene(Scene&&) = default;
        Scene& operator=(Scene&&) = default;

        void addNode(const std::string& name, std::unique_ptr<lfs::core::SplatData> model);
        void removeNode(const std::string& name, bool keep_children = false);
        void replaceNodeModel(const std::string& name, std::unique_ptr<lfs::core::SplatData> model);
        void setNodeVisibility(const std::string& name, bool visible);
        void setNodeTransform(const std::string& name, const glm::mat4& transform);
        glm::mat4 getNodeTransform(const std::string& name) const;
        bool renameNode(const std::string& old_name, const std::string& new_name);
        void clear();
        std::pair<std::string, std::string> cycleVisibilityWithNames();

        NodeId addGroup(const std::string& name, NodeId parent = NULL_NODE);
        NodeId addSplat(const std::string& name, std::unique_ptr<lfs::core::SplatData> model, NodeId parent = NULL_NODE);
        NodeId addPointCloud(const std::string& name, std::shared_ptr<lfs::core::PointCloud> point_cloud, NodeId parent = NULL_NODE);
        NodeId addMesh(const std::string& name, std::shared_ptr<lfs::core::MeshData> mesh_data, NodeId parent = NULL_NODE);
        NodeId addCropBox(const std::string& name, NodeId parent_id);
        NodeId addEllipsoid(const std::string& name, NodeId parent_id);
        NodeId addDataset(const std::string& name);
        NodeId addCameraGroup(const std::string& name, NodeId parent, size_t camera_count);
        NodeId addCamera(const std::string& name, NodeId parent, std::shared_ptr<lfs::core::Camera> camera);
        NodeId addKeyframeGroup(const std::string& name, NodeId parent = NULL_NODE);
        NodeId addKeyframe(const std::string& name, NodeId parent, std::unique_ptr<KeyframeData> data);
        void removeKeyframeNodes();
        void reparent(NodeId node, NodeId new_parent);
        [[nodiscard]] std::string duplicateNode(const std::string& name);
        [[nodiscard]] std::string mergeGroup(const std::string& group_name);
        [[nodiscard]] const glm::mat4& getWorldTransform(NodeId node) const;
        [[nodiscard]] std::vector<NodeId> getRootNodes() const;
        [[nodiscard]] SceneNode* getNodeById(NodeId id);
        [[nodiscard]] const SceneNode* getNodeById(NodeId id) const;

        [[nodiscard]] bool isNodeEffectivelyVisible(NodeId id) const;
        [[nodiscard]] glm::vec3 getNodeBoundsCenter(NodeId id) const;
        [[nodiscard]] bool getNodeBounds(NodeId id, glm::vec3& out_min, glm::vec3& out_max) const;

        [[nodiscard]] NodeId getCropBoxForSplat(NodeId splat_id) const;
        [[nodiscard]] NodeId getOrCreateCropBoxForSplat(NodeId splat_id);
        [[nodiscard]] CropBoxData* getCropBoxData(NodeId cropbox_id);
        [[nodiscard]] const CropBoxData* getCropBoxData(NodeId cropbox_id) const;
        void setCropBoxData(NodeId cropbox_id, const CropBoxData& data);

        struct RenderableCropBox {
            NodeId node_id = NULL_NODE;
            NodeId parent_splat_id = NULL_NODE;
            const CropBoxData* data = nullptr;
            glm::mat4 world_transform{1.0f};
            glm::mat4 local_transform{1.0f};
        };
        [[nodiscard]] std::vector<RenderableCropBox> getVisibleCropBoxes() const;

        [[nodiscard]] NodeId getEllipsoidForSplat(NodeId splat_id) const;
        [[nodiscard]] NodeId getOrCreateEllipsoidForSplat(NodeId splat_id);
        [[nodiscard]] EllipsoidData* getEllipsoidData(NodeId ellipsoid_id);
        [[nodiscard]] const EllipsoidData* getEllipsoidData(NodeId ellipsoid_id) const;
        void setEllipsoidData(NodeId ellipsoid_id, const EllipsoidData& data);

        struct RenderableEllipsoid {
            NodeId node_id = NULL_NODE;
            NodeId parent_splat_id = NULL_NODE;
            const EllipsoidData* data = nullptr;
            glm::mat4 world_transform{1.0f};
            glm::mat4 local_transform{1.0f};
        };
        [[nodiscard]] std::vector<RenderableEllipsoid> getVisibleEllipsoids() const;

        const lfs::core::SplatData* getCombinedModel() const;
        size_t consolidateNodeModels();
        [[nodiscard]] bool isConsolidated() const { return consolidated_; }
        [[nodiscard]] std::vector<bool> getNodeVisibilityMask() const;

        [[nodiscard]] std::unique_ptr<lfs::core::SplatData> createMergedModelWithTransforms() const;
        [[nodiscard]] static std::unique_ptr<lfs::core::SplatData> mergeSplatsWithTransforms(
            const std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>>& splats);

        [[nodiscard]] const lfs::core::PointCloud* getVisiblePointCloud() const;

        struct VisibleMesh {
            const lfs::core::MeshData* mesh;
            glm::mat4 transform;
            NodeId node_id = NULL_NODE;
            bool is_selected = false;
        };
        [[nodiscard]] std::vector<VisibleMesh> getVisibleMeshes() const;
        [[nodiscard]] bool hasVisibleMeshes() const;

        std::vector<glm::mat4> getVisibleNodeTransforms() const;
        std::shared_ptr<lfs::core::Tensor> getTransformIndices() const;
        [[nodiscard]] int getVisibleNodeIndex(const std::string& name) const;
        [[nodiscard]] int getVisibleNodeIndex(NodeId node_id) const;

        [[nodiscard]] std::vector<bool> getSelectedNodeMask(const std::string& selected_node_name) const;
        [[nodiscard]] std::vector<bool> getSelectedNodeMask(const std::vector<std::string>& selected_node_names) const;

        std::shared_ptr<lfs::core::Tensor> getSelectionMask() const;
        void setSelection(const std::vector<size_t>& selected_indices);
        void setSelectionMask(std::shared_ptr<lfs::core::Tensor> mask);
        void clearSelection();
        bool hasSelection() const;

        uint8_t addSelectionGroup(const std::string& name, const glm::vec3& color);
        void removeSelectionGroup(uint8_t id);
        void renameSelectionGroup(uint8_t id, const std::string& name);
        void setSelectionGroupColor(uint8_t id, const glm::vec3& color);
        void setSelectionGroupLocked(uint8_t id, bool locked);
        [[nodiscard]] bool isSelectionGroupLocked(uint8_t id) const;
        void setActiveSelectionGroup(uint8_t id) { active_selection_group_ = id; }
        [[nodiscard]] uint8_t getActiveSelectionGroup() const { return active_selection_group_; }
        [[nodiscard]] const std::vector<SelectionGroup>& getSelectionGroups() const { return selection_groups_; }
        [[nodiscard]] const SelectionGroup* getSelectionGroup(uint8_t id) const;
        void updateSelectionGroupCounts();
        void clearSelectionGroup(uint8_t id);
        void resetSelectionState();

        void setInitialPointCloud(std::shared_ptr<lfs::core::PointCloud> point_cloud);
        void setSceneCenter(lfs::core::Tensor scene_center);
        void setImagesHaveAlpha(bool have_alpha) { images_have_alpha_ = have_alpha; }

        [[nodiscard]] std::shared_ptr<lfs::core::PointCloud> getInitialPointCloud() const { return initial_point_cloud_; }
        [[nodiscard]] const lfs::core::Tensor& getSceneCenter() const { return scene_center_; }
        [[nodiscard]] bool imagesHaveAlpha() const { return images_have_alpha_; }

        [[nodiscard]] bool hasTrainingData() const;

        [[nodiscard]] std::shared_ptr<const lfs::core::Camera> getCameraByUid(int uid) const;
        [[nodiscard]] std::vector<std::shared_ptr<lfs::core::Camera>> getAllCameras() const;
        [[nodiscard]] std::vector<std::shared_ptr<lfs::core::Camera>> getActiveCameras() const;
        [[nodiscard]] size_t getActiveCameraCount() const;
        void setCameraTrainingEnabled(const std::string& name, bool enabled);

        [[nodiscard]] std::unordered_set<int> getTrainingDisabledCameraUids() const;

        [[nodiscard]] lfs::core::SplatData* getTrainingModel();
        [[nodiscard]] const lfs::core::SplatData* getTrainingModel() const;

        void setTrainingModelNode(const std::string& name);
        [[nodiscard]] const std::string& getTrainingModelNodeName() const { return training_model_node_; }

        void setTrainingModel(std::unique_ptr<lfs::core::SplatData> splat_data, const std::string& name);

        size_t getNodeCount() const { return nodes_.size(); }
        size_t getTotalGaussianCount() const;
        std::vector<const SceneNode*> getNodes() const;
        const SceneNode* getNode(const std::string& name) const;
        SceneNode* getMutableNode(const std::string& name);
        [[nodiscard]] NodeId getNodeIdByName(const std::string& name) const;
        bool hasNodes() const { return !nodes_.empty(); }

        std::vector<const SceneNode*> getVisibleNodes() const;
        [[nodiscard]] std::vector<std::shared_ptr<const lfs::core::Camera>> getVisibleCameras() const;

        void pinForExport() const { ++export_pin_count_; }
        void unpinForExport() const {
            assert(export_pin_count_.load(std::memory_order_acquire) > 0);
            --export_pin_count_;
        }

        void invalidateCache() {
            model_cache_valid_ = false;
            transform_cache_valid_ = false;
        }
        void invalidateTransformCache() { transform_cache_valid_ = false; }
        void markDirty() { invalidateCache(); }
        void markTransformDirty(NodeId node);

        size_t applyDeleted();

    private:
        std::vector<std::unique_ptr<SceneNode>> nodes_;
        std::unordered_map<NodeId, size_t> id_to_index_;
        std::unordered_map<std::string, NodeId> name_to_id_;
        NodeId next_node_id_ = 0;

        uint32_t pending_mutations_ = 0;
        int transaction_depth_ = 0;
        void flushMutations();
        mutable std::atomic<int> export_pin_count_{0};
        mutable std::unique_ptr<lfs::core::SplatData> cached_combined_;
        mutable std::shared_ptr<lfs::core::Tensor> cached_transform_indices_;
        mutable bool model_cache_valid_ = false;
        mutable const lfs::core::SplatData* single_node_model_ = nullptr;

        mutable std::vector<glm::mat4> cached_transforms_;
        mutable bool transform_cache_valid_ = false;
        mutable bool consolidated_ = false;
        mutable std::vector<NodeId> consolidated_node_ids_;

        mutable std::shared_ptr<lfs::core::Tensor> selection_mask_;
        mutable bool has_selection_ = false;

        std::vector<SelectionGroup> selection_groups_;
        uint8_t active_selection_group_ = 1;
        uint8_t next_group_id_ = 1;

        void rebuildCacheIfNeeded() const;
        void rebuildModelCacheIfNeeded() const;
        void rebuildTransformCacheIfNeeded() const;
        void updateWorldTransform(const SceneNode& node) const;
        void removeNodeInternal(const std::string& name, bool keep_children, bool force);
        void setNodeVisibilityById(NodeId id, bool visible);

        SelectionGroup* findGroup(uint8_t id);
        const SelectionGroup* findGroup(uint8_t id) const;

        std::shared_ptr<lfs::core::PointCloud> initial_point_cloud_;
        lfs::core::Tensor scene_center_;
        bool images_have_alpha_ = false;
        std::string training_model_node_;
    };

} // namespace lfs::core
