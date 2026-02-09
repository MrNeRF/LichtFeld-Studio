/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/animatable_property.hpp"
#include "core/camera.hpp"
#include "core/export.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include <atomic>
#include <cassert>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::core {

    // Node identifier (-1 = invalid/root)
    using NodeId = int32_t;
    constexpr NodeId NULL_NODE = -1;

    // Node types
    enum class NodeType : uint8_t {
        SPLAT,        // Contains gaussian splat data
        POINTCLOUD,   // Contains point cloud (pre-training, can be cropped)
        GROUP,        // Empty transform node for organization
        CROPBOX,      // Crop box visualization (child of SPLAT, POINTCLOUD, or DATASET)
        ELLIPSOID,    // Ellipsoid selection (child of SPLAT, POINTCLOUD, or DATASET)
        DATASET,      // Root node for training dataset (contains cameras + model)
        CAMERA_GROUP, // Container for camera nodes (e.g., "Training", "Validation")
        CAMERA,       // Individual camera from dataset (may have mask_path)
        IMAGE_GROUP,  // Container for image nodes
        IMAGE         // Individual image file reference (not loaded, just path)
    };

    // Crop box data for CROPBOX nodes (parent_id references associated splat)
    struct CropBoxData {
        glm::vec3 min{-1.0f, -1.0f, -1.0f};
        glm::vec3 max{1.0f, 1.0f, 1.0f};
        bool inverse = false; // Invert crop (keep outside instead of inside)
        bool enabled = false; // Whether to use for filtering gaussians
        glm::vec3 color{1.0f, 1.0f, 0.0f};
        float line_width = 2.0f;
        float flash_intensity = 0.0f;
    };

    // Ellipsoid data for ELLIPSOID nodes (parent_id references associated splat)
    struct EllipsoidData {
        glm::vec3 radii{1.0f, 1.0f, 1.0f};
        bool inverse = false;
        bool enabled = false;
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
    class LFS_CORE_API SceneNode {
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
        std::unique_ptr<EllipsoidData> ellipsoid;
        size_t gaussian_count = 0; // For SPLAT: num gaussians, for POINTCLOUD: num points
        glm::vec3 centroid{0.0f};

        // Camera data (for CAMERA nodes) - node owns the Camera object directly
        std::shared_ptr<lfs::core::Camera> camera;
        int camera_uid = -1; // Camera unique identifier (for GoToCamView)
        bool training_enabled = true;

        // Image data (for IMAGE and CAMERA nodes) - just the filename, not loaded
        std::string image_path; // Path to image file

        // Mask data (for MASK and CAMERA nodes) - path to attention mask file
        std::string mask_path; // Path to mask file

        // Cached world transform (mutable for lazy evaluation)
        mutable glm::mat4 world_transform{1.0f};
        mutable bool transform_dirty = true;

        // Animatable properties - changes auto-invalidate scene cache and generate undo
        lfs::core::prop::AnimatableProperty<glm::mat4> local_transform{glm::mat4{1.0f}};
        lfs::core::prop::AnimatableProperty<bool> visible{true};
        lfs::core::prop::AnimatableProperty<bool> locked{false};

        // Legacy accessor
        [[nodiscard]] const glm::mat4& transform() const { return local_transform.get(); }

    private:
        Scene* scene_ = nullptr;
    };

    class LFS_CORE_API Scene {
    public:
        // Alias for backwards compatibility
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

        class Transaction {
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
        NodeId addCropBox(const std::string& name, NodeId parent_id);   // Child of splat node
        NodeId addEllipsoid(const std::string& name, NodeId parent_id); // Child of splat node
        NodeId addDataset(const std::string& name);                     // Root node for training dataset
        NodeId addCameraGroup(const std::string& name, NodeId parent, size_t camera_count);
        NodeId addCamera(const std::string& name, NodeId parent, std::shared_ptr<lfs::core::Camera> camera);
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

        // Ellipsoid operations
        [[nodiscard]] NodeId getEllipsoidForSplat(NodeId splat_id) const;
        [[nodiscard]] NodeId getOrCreateEllipsoidForSplat(NodeId splat_id);
        [[nodiscard]] EllipsoidData* getEllipsoidData(NodeId ellipsoid_id);
        [[nodiscard]] const EllipsoidData* getEllipsoidData(NodeId ellipsoid_id) const;
        void setEllipsoidData(NodeId ellipsoid_id, const EllipsoidData& data);

        // Renderable ellipsoid info for rendering
        struct RenderableEllipsoid {
            NodeId node_id = NULL_NODE;
            NodeId parent_splat_id = NULL_NODE;
            const EllipsoidData* data = nullptr;
            glm::mat4 world_transform{1.0f};
            glm::mat4 local_transform{1.0f};
        };
        [[nodiscard]] std::vector<RenderableEllipsoid> getVisibleEllipsoids() const;

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
        [[nodiscard]] int getVisibleNodeIndex(NodeId node_id) const;

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
        void setInitialPointCloud(std::shared_ptr<lfs::core::PointCloud> point_cloud);
        void setSceneCenter(lfs::core::Tensor scene_center);
        void setImagesHaveAlpha(bool have_alpha) { images_have_alpha_ = have_alpha; }

        [[nodiscard]] std::shared_ptr<lfs::core::PointCloud> getInitialPointCloud() const { return initial_point_cloud_; }
        [[nodiscard]] const lfs::core::Tensor& getSceneCenter() const { return scene_center_; }
        [[nodiscard]] bool imagesHaveAlpha() const { return images_have_alpha_; }

        [[nodiscard]] bool hasTrainingData() const;

        // Camera access (iterates CAMERA nodes)
        [[nodiscard]] std::shared_ptr<const lfs::core::Camera> getCameraByUid(int uid) const;
        [[nodiscard]] std::vector<std::shared_ptr<lfs::core::Camera>> getAllCameras() const;
        [[nodiscard]] std::vector<std::shared_ptr<lfs::core::Camera>> getActiveCameras() const;
        [[nodiscard]] size_t getActiveCameraCount() const;
        void setCameraTrainingEnabled(const std::string& name, bool enabled);

        // Get UIDs of cameras with training disabled (for frustum rendering dimming)
        [[nodiscard]] std::unordered_set<int> getTrainingDisabledCameraUids() const;

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

        // Get visible cameras (for frustum rendering)
        // Returns Camera objects from visible CAMERA nodes
        [[nodiscard]] std::vector<std::shared_ptr<const lfs::core::Camera>> getVisibleCameras() const;

        // Mark scene data as changed (e.g., after modifying a node's deleted mask)
        // Also called by SceneNode Observable properties when they change
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

        // Permanently remove soft-deleted gaussians from all nodes
        // Returns total number of gaussians removed
        size_t applyDeleted();

    private:
        std::vector<std::unique_ptr<Node>> nodes_;       // unique_ptr for stable addresses (Observable callbacks capture 'this')
        std::unordered_map<NodeId, size_t> id_to_index_; // NodeId -> index in nodes_
        NodeId next_node_id_ = 0;

        uint32_t pending_mutations_ = 0;
        int transaction_depth_ = 0;
        void flushMutations();

        // Export pin prevents cache invalidation while video/file export holds a pointer
        mutable std::atomic<int> export_pin_count_{0};

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
        std::shared_ptr<lfs::core::PointCloud> initial_point_cloud_;
        lfs::core::Tensor scene_center_;
        bool images_have_alpha_ = false;
        std::string training_model_node_;
    };

} // namespace lfs::core
