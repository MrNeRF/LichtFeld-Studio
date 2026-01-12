/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/observable.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Forward declarations for training data
namespace lfs::training {
    class CameraDataset;
}
namespace lfs::core {
    struct PointCloud;
    class Camera;
} // namespace lfs::core

namespace lfs::vis {

    // Node identifier (-1 = invalid/root)
    using NodeId = int32_t;
    constexpr NodeId NULL_NODE = -1;

    // Node types
    enum class NodeType : uint8_t {
        SPLAT,        // Contains gaussian splat data
        POINTCLOUD,   // Contains point cloud (pre-training, can be cropped)
        GROUP,        // Empty transform node for organization
        CROPBOX,      // Crop box visualization (child of SPLAT or POINTCLOUD)
        DATASET,      // Root node for training dataset (contains cameras + model)
        CAMERA_GROUP, // Container for camera nodes (e.g., "Training", "Validation")
        CAMERA,       // Individual camera from dataset (may have mask_path)
        IMAGE_GROUP,  // Container for image nodes
        IMAGE         // Individual image file reference (not loaded, just path)
    };

    // Crop box data for CROPBOX nodes
    struct CropBoxData {
        glm::vec3 min{-1.0f, -1.0f, -1.0f};
        glm::vec3 max{1.0f, 1.0f, 1.0f};
        bool inverse = false; // Invert crop (keep outside instead of inside)
        bool enabled = false; // Whether to use for filtering gaussians
        glm::vec3 color{1.0f, 1.0f, 0.0f};
        float line_width = 2.0f;
        float flash_intensity = 0.0f;
    };

    // Selection group with ID, name, and color
    struct SelectionGroup {
        uint8_t id = 0; // 1-255, 0 means unselected
        std::string name;
        glm::vec3 color{1.0f, 0.0f, 0.0f};
        size_t count = 0;    // Number of selected Gaussians
        bool locked = false; // If true, painting with other groups won't overwrite
    };

    class Scene; // Forward declaration

    // Scene graph node with Observable properties
    // Changes to observable properties automatically invalidate the scene cache
    class SceneNode {
    public:
        SceneNode() = default;
        explicit SceneNode(Scene* scene);

        // Initialize observables with scene callback (called after node is added to scene)
        void initObservables(Scene* scene);

        // Non-observable identity
        NodeId id = NULL_NODE;
        NodeId parent_id = NULL_NODE;
        std::vector<NodeId> children;
        NodeType type = NodeType::SPLAT;
        std::string name;

        // Data (changes require manual cache invalidation via scene)
        std::unique_ptr<lfs::core::SplatData> model;        // For SPLAT nodes
        std::shared_ptr<lfs::core::PointCloud> point_cloud; // For POINTCLOUD nodes
        std::unique_ptr<CropBoxData> cropbox;
        size_t gaussian_count = 0; // For SPLAT: num gaussians, for POINTCLOUD: num points
        glm::vec3 centroid{0.0f};

        // Camera data (for CAMERA nodes)
        int camera_index = -1; // Index into CameraDataset
        int camera_uid = -1;   // Camera unique identifier (for GoToCamView)

        // Image data (for IMAGE and CAMERA nodes) - just the filename, not loaded
        std::string image_path; // Path to image file

        // Mask data (for MASK and CAMERA nodes) - path to attention mask file
        std::string mask_path; // Path to mask file

        // Cached world transform (mutable for lazy evaluation)
        mutable glm::mat4 world_transform{1.0f};
        mutable bool transform_dirty = true;

        // Observable properties - changes auto-invalidate scene cache
        lfs::core::Observable<glm::mat4> local_transform{glm::mat4{1.0f}, nullptr};
        lfs::core::Observable<bool> visible{true, nullptr};
        lfs::core::Observable<bool> locked{false, nullptr};

        // Legacy accessor
        [[nodiscard]] const glm::mat4& transform() const { return local_transform.get(); }

    private:
        Scene* scene_ = nullptr;
    };

    class Scene {
    public:
        // Alias for backwards compatibility
        using Node = SceneNode;

        Scene();
        ~Scene() = default;

        // Delete copy operations
        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        // Allow move operations
        Scene(Scene&&) = default;
        Scene& operator=(Scene&&) = default;

        // Node management (by name - legacy API)
        void addNode(const std::string& name, std::unique_ptr<lfs::core::SplatData> model);
        void removeNode(const std::string& name, bool keep_children = false);
        void replaceNodeModel(const std::string& name, std::unique_ptr<lfs::core::SplatData> model);
        void setNodeVisibility(const std::string& name, bool visible);
        void setNodeTransform(const std::string& name, const glm::mat4& transform);
        glm::mat4 getNodeTransform(const std::string& name) const;
        bool renameNode(const std::string& old_name, const std::string& new_name);
        void clear();
        std::pair<std::string, std::string> cycleVisibilityWithNames();

        // Scene graph operations
        NodeId addGroup(const std::string& name, NodeId parent = NULL_NODE);
        NodeId addSplat(const std::string& name, std::unique_ptr<lfs::core::SplatData> model, NodeId parent = NULL_NODE);
        NodeId addPointCloud(const std::string& name, std::shared_ptr<lfs::core::PointCloud> point_cloud, NodeId parent = NULL_NODE);
        NodeId addCropBox(const std::string& name, NodeId parent_node); // Parent can be SPLAT or POINTCLOUD
        NodeId addDataset(const std::string& name);                     // Root node for training dataset
        NodeId addCameraGroup(const std::string& name, NodeId parent, size_t camera_count);
        NodeId addCamera(const std::string& name, NodeId parent, int camera_index, int camera_uid,
                         const std::string& image_path = "", const std::string& mask_path = "");
        void reparent(NodeId node, NodeId new_parent);
        // Duplicate a node (and all children recursively for groups)
        // Returns new node name (original name with "_copy" or "_copy_N" suffix)
        [[nodiscard]] std::string duplicateNode(const std::string& name);

        // Merge all child SPLATs of a group into a single SPLAT node
        // Applies world transforms, removes original children, replaces group with merged SPLAT
        // Returns name of merged node, or empty string on failure
        [[nodiscard]] std::string mergeGroup(const std::string& group_name);
        [[nodiscard]] const glm::mat4& getWorldTransform(NodeId node) const;
        [[nodiscard]] std::vector<NodeId> getRootNodes() const;
        [[nodiscard]] Node* getNodeById(NodeId id);
        [[nodiscard]] const Node* getNodeById(NodeId id) const;

        // Check if node is effectively visible (considers parent hierarchy)
        [[nodiscard]] bool isNodeEffectivelyVisible(NodeId id) const;

        // Get bounding box center for a node (for groups: includes all descendants)
        [[nodiscard]] glm::vec3 getNodeBoundsCenter(NodeId id) const;
        [[nodiscard]] bool getNodeBounds(NodeId id, glm::vec3& out_min, glm::vec3& out_max) const;

        // Cropbox operations
        [[nodiscard]] NodeId getCropBoxForSplat(NodeId splat_id) const;
        [[nodiscard]] NodeId getOrCreateCropBoxForSplat(NodeId splat_id);
        [[nodiscard]] CropBoxData* getCropBoxData(NodeId cropbox_id);
        [[nodiscard]] const CropBoxData* getCropBoxData(NodeId cropbox_id) const;
        void setCropBoxData(NodeId cropbox_id, const CropBoxData& data);

        // Renderable cropbox info for rendering
        struct RenderableCropBox {
            NodeId node_id = NULL_NODE;
            NodeId parent_splat_id = NULL_NODE;
            const CropBoxData* data = nullptr;
            glm::mat4 world_transform{1.0f};
            glm::mat4 local_transform{1.0f}; // Cropbox's local transform (relative to parent)
        };
        [[nodiscard]] std::vector<RenderableCropBox> getVisibleCropBoxes() const;

        // Get combined model for rendering (transforms NOT baked, applied at render time)
        const lfs::core::SplatData* getCombinedModel() const;
        size_t consolidateNodeModels();

        // Consolidation state
        [[nodiscard]] bool isConsolidated() const { return consolidated_; }
        [[nodiscard]] std::vector<bool> getNodeVisibilityMask() const;

        // Create merged model with transforms baked in (for saving)
        [[nodiscard]] std::unique_ptr<lfs::core::SplatData> createMergedModelWithTransforms() const;

        // Merge splats with transforms baked in (shared implementation)
        [[nodiscard]] static std::unique_ptr<lfs::core::SplatData> mergeSplatsWithTransforms(
            const std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>>& splats);

        // Get visible point cloud for rendering (before training starts)
        // Returns first visible POINTCLOUD node's data, or nullptr
        [[nodiscard]] const lfs::core::PointCloud* getVisiblePointCloud() const;

        // Get transforms for visible nodes (for kernel-based transform)
        std::vector<glm::mat4> getVisibleNodeTransforms() const;

        // Get per-Gaussian transform indices tensor (for kernel-based transform)
        // Returns nullptr if no transforms needed (single node with identity transform)
        std::shared_ptr<lfs::core::Tensor> getTransformIndices() const;

        // Get node index in combined model (-1 if not found or not visible)
        [[nodiscard]] int getVisibleNodeIndex(const std::string& name) const;

        // Get mask of selected visible SPLAT nodes for desaturation
        // When a group is selected, all descendant SPLAT nodes are marked as selected
        // Returns vector of bools, one per visible SPLAT node (same order as transforms)
        [[nodiscard]] std::vector<bool> getSelectedNodeMask(const std::string& selected_node_name) const;
        [[nodiscard]] std::vector<bool> getSelectedNodeMask(const std::vector<std::string>& selected_node_names) const;

        // Selection mask for highlighting selected Gaussians
        // Returns nullptr if no selection (all zeros = no selection)
        std::shared_ptr<lfs::core::Tensor> getSelectionMask() const;

        // Set selection for Gaussians (indices into combined model)
        void setSelection(const std::vector<size_t>& selected_indices);

        // Set selection mask directly from GPU tensor (for GPU-based brush selection)
        void setSelectionMask(std::shared_ptr<lfs::core::Tensor> mask);

        // Clear all selection
        void clearSelection();

        // Check if any Gaussians are selected
        bool hasSelection() const;

        // Selection groups management
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
        void resetSelectionState(); // Full reset: clear mask, remove all groups, create default

        // ========== Training Data Storage ==========
        // Scene owns training data (cameras + initial point cloud)
        // This allows unified handling in both headless and GUI modes

        void setTrainCameras(std::shared_ptr<lfs::training::CameraDataset> dataset);
        void setValCameras(std::shared_ptr<lfs::training::CameraDataset> dataset);
        void setInitialPointCloud(std::shared_ptr<lfs::core::PointCloud> point_cloud);
        void setSceneCenter(lfs::core::Tensor scene_center);

        [[nodiscard]] std::shared_ptr<lfs::training::CameraDataset> getTrainCameras() const { return train_cameras_; }
        [[nodiscard]] std::shared_ptr<lfs::training::CameraDataset> getValCameras() const { return val_cameras_; }
        [[nodiscard]] std::shared_ptr<lfs::core::PointCloud> getInitialPointCloud() const { return initial_point_cloud_; }
        [[nodiscard]] const lfs::core::Tensor& getSceneCenter() const { return scene_center_; }

        [[nodiscard]] bool hasTrainingData() const { return train_cameras_ != nullptr; }

        // Camera access helpers (delegates to CameraDataset)
        [[nodiscard]] std::shared_ptr<const lfs::core::Camera> getCameraByUid(int uid) const;
        [[nodiscard]] std::vector<std::shared_ptr<const lfs::core::Camera>> getAllCameras() const;

        // Get the primary training model node (for Trainer to operate on)
        // Returns nullptr if no training model exists
        [[nodiscard]] lfs::core::SplatData* getTrainingModel();
        [[nodiscard]] const lfs::core::SplatData* getTrainingModel() const;

        // Set which node is the training model (by name)
        void setTrainingModelNode(const std::string& name);
        [[nodiscard]] const std::string& getTrainingModelNodeName() const { return training_model_node_; }

        // Create training model from SplatData (used for checkpoint loading)
        void setTrainingModel(std::unique_ptr<lfs::core::SplatData> splat_data, const std::string& name);

        // Direct queries
        size_t getNodeCount() const { return nodes_.size(); }
        size_t getTotalGaussianCount() const;
        std::vector<const Node*> getNodes() const;
        const Node* getNode(const std::string& name) const;
        Node* getMutableNode(const std::string& name);
        bool hasNodes() const { return !nodes_.empty(); }

        // Get visible nodes for split view
        std::vector<const Node*> getVisibleNodes() const;

        // Get visible camera indices (for frustum rendering)
        // Returns set of camera_index values for CAMERA nodes that are visible
        [[nodiscard]] std::unordered_set<int> getVisibleCameraIndices() const;

        // Mark scene data as changed (e.g., after modifying a node's deleted mask)
        // Also called by SceneNode Observable properties when they change
        void invalidateCache() {
            model_cache_valid_ = false;
            transform_cache_valid_ = false;
        }
        void invalidateTransformCache() { transform_cache_valid_ = false; }
        void markDirty() { invalidateCache(); }
        void markTransformDirty(NodeId node);

        // Permanently remove soft-deleted gaussians from all nodes
        // Returns total number of gaussians removed
        size_t applyDeleted();

    private:
        std::vector<std::unique_ptr<Node>> nodes_;       // unique_ptr for stable addresses (Observable callbacks capture 'this')
        std::unordered_map<NodeId, size_t> id_to_index_; // NodeId -> index in nodes_
        NodeId next_node_id_ = 0;

        // Caching for combined model (rebuilt when models/visibility change)
        mutable std::unique_ptr<lfs::core::SplatData> cached_combined_;
        mutable std::shared_ptr<lfs::core::Tensor> cached_transform_indices_;
        mutable bool model_cache_valid_ = false;
        mutable const lfs::core::SplatData* single_node_model_ = nullptr;

        // Transform cache (rebuilt when transforms change, much cheaper)
        mutable std::vector<glm::mat4> cached_transforms_;
        mutable bool transform_cache_valid_ = false;
        mutable bool consolidated_ = false;
        mutable std::vector<NodeId> consolidated_node_ids_;

        // Selection mask: UInt8 [N], value = group ID (0=unselected, 1-255=group ID)
        mutable std::shared_ptr<lfs::core::Tensor> selection_mask_;
        mutable bool has_selection_ = false;

        // Selection groups (ID 0 is reserved for "unselected")
        std::vector<SelectionGroup> selection_groups_;
        uint8_t active_selection_group_ = 1; // Default to group 1
        uint8_t next_group_id_ = 1;

        void rebuildCacheIfNeeded() const;
        void rebuildModelCacheIfNeeded() const;
        void rebuildTransformCacheIfNeeded() const;
        void updateWorldTransform(const Node& node) const;
        void removeNodeInternal(const std::string& name, bool keep_children, bool force);
        void setNodeVisibilityById(NodeId id, bool visible);

        // Helper to find group by ID
        SelectionGroup* findGroup(uint8_t id);
        const SelectionGroup* findGroup(uint8_t id) const;

        // Training data storage
        std::shared_ptr<lfs::training::CameraDataset> train_cameras_;
        std::shared_ptr<lfs::training::CameraDataset> val_cameras_;
        std::shared_ptr<lfs::core::PointCloud> initial_point_cloud_;
        lfs::core::Tensor scene_center_;  // Scene center (mean of camera positions)
        std::string training_model_node_; // Name of the node being trained
    };

} // namespace lfs::vis