/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/events.hpp"
#include "py_prop.hpp"
#include "py_splat_data.hpp"
#include "py_tensor.hpp"
#include "visualizer/scene/scene.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>

namespace nb = nanobind;

namespace lfs::python {

    // Forward declarations
    class PyScene;
    class PyCameraDataset;

    // Selection group view (read-only)
    struct PySelectionGroup {
        uint8_t id;
        std::string name;
        std::tuple<float, float, float> color;
        size_t count;
        bool locked;
    };

    class PyCropBox {
    public:
        explicit PyCropBox(vis::CropBoxData* data) : data_(data),
                                                     prop_(data, "crop_box") {
            assert(data_ != nullptr);
        }

        [[nodiscard]] const std::string& property_group() const { return prop_.group_id(); }
        [[nodiscard]] nb::object get(const std::string& name) const { return prop_.getattr(name); }
        void set(const std::string& name, nb::object value) { prop_.setattr(name, value); }
        [[nodiscard]] nb::dict prop_info(const std::string& name) const {
            return prop_.prop_info(name);
        }
        [[nodiscard]] nb::object prop_getattr(const std::string& name) const {
            return prop_.getattr(name);
        }
        void prop_setattr(const std::string& name, nb::object value) { set(name, value); }
        [[nodiscard]] bool has_prop(const std::string& name) const {
            return core::prop::PropertyRegistry::instance().get_property(prop_.group_id(), name).has_value();
        }
        [[nodiscard]] nb::list python_dir() const {
            nb::list result;
            result.append("prop_info");
            result.append("get");
            result.append("set");
            const nb::list props = prop_.dir();
            for (size_t i = 0; i < nb::len(props); ++i) {
                result.append(props[i]);
            }
            return result;
        }
        [[nodiscard]] vis::CropBoxData* data() { return data_; }
        [[nodiscard]] const vis::CropBoxData* data() const { return data_; }

    private:
        vis::CropBoxData* data_;
        PyProp<vis::CropBoxData> prop_;
    };

    class PyEllipsoid {
    public:
        explicit PyEllipsoid(vis::EllipsoidData* data) : data_(data),
                                                         prop_(data, "ellipsoid") {
            assert(data_ != nullptr);
        }

        [[nodiscard]] const std::string& property_group() const { return prop_.group_id(); }
        [[nodiscard]] nb::object get(const std::string& name) const { return prop_.getattr(name); }
        void set(const std::string& name, nb::object value) { prop_.setattr(name, value); }
        [[nodiscard]] nb::dict prop_info(const std::string& name) const {
            return prop_.prop_info(name);
        }
        [[nodiscard]] nb::object prop_getattr(const std::string& name) const {
            return prop_.getattr(name);
        }
        void prop_setattr(const std::string& name, nb::object value) { set(name, value); }
        [[nodiscard]] bool has_prop(const std::string& name) const {
            return core::prop::PropertyRegistry::instance().get_property(prop_.group_id(), name).has_value();
        }
        [[nodiscard]] nb::list python_dir() const {
            nb::list result;
            result.append("prop_info");
            result.append("get");
            result.append("set");
            const nb::list props = prop_.dir();
            for (size_t i = 0; i < nb::len(props); ++i) {
                result.append(props[i]);
            }
            return result;
        }
        [[nodiscard]] vis::EllipsoidData* data() { return data_; }
        [[nodiscard]] const vis::EllipsoidData* data() const { return data_; }

    private:
        vis::EllipsoidData* data_;
        PyProp<vis::EllipsoidData> prop_;
    };

    // PointCloud wrapper
    class PyPointCloud {
    public:
        explicit PyPointCloud(core::PointCloud* pc, bool owns = false,
                              vis::SceneNode* node = nullptr, vis::Scene* scene = nullptr)
            : pc_(pc),
              owns_(owns),
              node_(node),
              scene_(scene) {
            assert(pc_ != nullptr);
        }

        PyTensor means() const { return PyTensor(pc_->means, false); }
        PyTensor colors() const { return PyTensor(pc_->colors, false); }

        std::optional<PyTensor> normals() const {
            if (!pc_->normals.is_valid())
                return std::nullopt;
            return PyTensor(pc_->normals, false);
        }
        std::optional<PyTensor> sh0() const {
            if (!pc_->sh0.is_valid())
                return std::nullopt;
            return PyTensor(pc_->sh0, false);
        }
        std::optional<PyTensor> shN() const {
            if (!pc_->shN.is_valid())
                return std::nullopt;
            return PyTensor(pc_->shN, false);
        }
        std::optional<PyTensor> opacity() const {
            if (!pc_->opacity.is_valid())
                return std::nullopt;
            return PyTensor(pc_->opacity, false);
        }
        std::optional<PyTensor> scaling() const {
            if (!pc_->scaling.is_valid())
                return std::nullopt;
            return PyTensor(pc_->scaling, false);
        }
        std::optional<PyTensor> rotation() const {
            if (!pc_->rotation.is_valid())
                return std::nullopt;
            return PyTensor(pc_->rotation, false);
        }

        int64_t size() const { return pc_->size(); }
        bool is_gaussian() const { return pc_->is_gaussian(); }
        std::vector<std::string> attribute_names() const { return pc_->attribute_names; }

        void normalize_colors() { pc_->normalize_colors(); }

        // Filter points by boolean mask - keeps points where mask is True
        int64_t filter(const PyTensor& keep_mask);

        // Filter points by index tensor - keeps points at specified indices
        int64_t filter_indices(const PyTensor& indices);

        // Replace point cloud data (for live updates)
        void set_data(const PyTensor& points, const PyTensor& colors);
        void set_colors(const PyTensor& colors);
        void set_means(const PyTensor& points);

        core::PointCloud* data() { return pc_; }
        const core::PointCloud* data() const { return pc_; }

    private:
        core::PointCloud* pc_;
        bool owns_;
        vis::SceneNode* node_ = nullptr;
        vis::Scene* scene_ = nullptr;
    };

    // Scene node wrapper
    class PySceneNode {
    public:
        PySceneNode(vis::SceneNode* node, vis::Scene* scene)
            : node_(node),
              scene_(scene),
              prop_(node, "scene_node") {
            assert(node_ != nullptr);
            assert(scene_ != nullptr);
        }

        // Identity (read-only, not in prop system)
        int32_t id() const { return node_->id; }
        int32_t parent_id() const { return node_->parent_id; }
        std::vector<int32_t> children() const { return node_->children; }
        vis::NodeType type() const { return node_->type; }

        // Transform (special matrix conversion)
        void set_local_transform(nb::ndarray<float, nb::shape<4, 4>> transform);
        nb::tuple world_transform() const;

        // Metadata (read-only)
        size_t gaussian_count() const { return node_->gaussian_count; }
        std::tuple<float, float, float> centroid() const {
            return {node_->centroid.x, node_->centroid.y, node_->centroid.z};
        }

        // Data accessors
        std::optional<PySplatData> splat_data();
        std::optional<PyPointCloud> point_cloud();
        std::optional<PyCropBox> cropbox();
        std::optional<PyEllipsoid> ellipsoid();

        // Camera node specific
        int camera_uid() const { return node_->camera_uid; }
        std::string image_path() const { return node_->image_path; }
        std::string mask_path() const { return node_->mask_path; }
        bool has_camera() const { return node_->camera != nullptr; }
        bool has_mask() const {
            if (!node_->camera)
                return false;
            return node_->camera->has_mask();
        }
        std::optional<PyTensor> load_mask(int resize_factor = 1, int max_width = 3840,
                                          bool invert = false, float threshold = 0.5f) {
            if (!node_->camera || !node_->camera->has_mask())
                return std::nullopt;
            return PyTensor(node_->camera->load_and_get_mask(resize_factor, max_width, invert, threshold), true);
        }
        std::optional<PyTensor> camera_R() const {
            if (!node_->camera)
                return std::nullopt;
            return PyTensor(node_->camera->R(), false);
        }
        std::optional<PyTensor> camera_T() const {
            if (!node_->camera)
                return std::nullopt;
            return PyTensor(node_->camera->T(), false);
        }
        std::optional<float> camera_focal_x() const {
            if (!node_->camera)
                return std::nullopt;
            return node_->camera->focal_x();
        }
        std::optional<float> camera_focal_y() const {
            if (!node_->camera)
                return std::nullopt;
            return node_->camera->focal_y();
        }
        std::optional<int> camera_width() const {
            if (!node_->camera)
                return std::nullopt;
            return node_->camera->camera_width();
        }
        std::optional<int> camera_height() const {
            if (!node_->camera)
                return std::nullopt;
            return node_->camera->camera_height();
        }

        // Property group interface for generic prop()
        const std::string& property_group() const { return prop_.group_id(); }
        nb::object get(const std::string& name) const { return prop_.getattr(name); }
        void set(const std::string& name, nb::object value) { prop_.setattr(name, value); }

        // Property introspection
        nb::dict prop_info(const std::string& name) const { return prop_.prop_info(name); }

        // Descriptor protocol support
        nb::object prop_getattr(const std::string& name) const { return prop_.getattr(name); }
        void prop_setattr(const std::string& name, nb::object value) { set(name, value); }

        bool has_prop(const std::string& name) const {
            return core::prop::PropertyRegistry::instance().get_property(
                                                               prop_.group_id(), name)
                .has_value();
        }

        nb::list python_dir() const {
            nb::list result;
            for (const char* attr : {"id", "parent_id", "children", "type",
                                     "world_transform", "set_local_transform",
                                     "gaussian_count", "centroid",
                                     "splat_data", "point_cloud", "cropbox", "ellipsoid",
                                     "camera_uid", "image_path", "mask_path", "has_camera",
                                     "has_mask", "load_mask",
                                     "camera_R", "camera_T", "camera_focal_x", "camera_focal_y",
                                     "camera_width", "camera_height",
                                     "prop_info"}) {
                result.append(attr);
            }
            nb::list props = prop_.dir();
            for (size_t i = 0; i < nb::len(props); ++i) {
                result.append(props[i]);
            }
            return result;
        }

    private:
        vis::SceneNode* node_;
        vis::Scene* scene_;
        PyProp<vis::SceneNode> prop_;
    };

    // Node collection for scene.nodes property
    class PyNodeCollection {
    public:
        explicit PyNodeCollection(vis::Scene* scene) : scene_(scene) {
            assert(scene_ != nullptr);
        }

        size_t size() const { return scene_->getNodeCount(); }

        PySceneNode getitem(int64_t index) const {
            auto nodes = scene_->getNodes();
            if (index < 0) {
                index += static_cast<int64_t>(nodes.size());
            }
            if (index < 0 || static_cast<size_t>(index) >= nodes.size()) {
                throw nb::index_error("Node index out of range");
            }
            return PySceneNode(const_cast<vis::SceneNode*>(nodes[static_cast<size_t>(index)]),
                               scene_);
        }

        std::vector<PySceneNode> items() const {
            std::vector<PySceneNode> result;
            for (const auto* node : scene_->getNodes()) {
                result.emplace_back(const_cast<vis::SceneNode*>(node), scene_);
            }
            return result;
        }

        class Iterator {
        public:
            Iterator(vis::Scene* scene)
                : scene_(scene),
                  index_(0),
                  nodes_(scene->getNodes()) {}

            PySceneNode next() {
                if (index_ >= nodes_.size()) {
                    throw nb::stop_iteration();
                }
                return PySceneNode(const_cast<vis::SceneNode*>(nodes_[index_++]), scene_);
            }

        private:
            vis::Scene* scene_;
            size_t index_;
            std::vector<const vis::SceneNode*> nodes_;
        };

        Iterator iter() const { return Iterator(scene_); }

    private:
        vis::Scene* scene_;
    };

    // Main scene wrapper
    class PyScene {
    public:
        explicit PyScene(vis::Scene* scene);

        // Thread-safe validity checking
        bool is_valid() const;
        uint64_t generation() const;

        // Node CRUD
        int32_t add_group(const std::string& name, int32_t parent = vis::NULL_NODE);
        int32_t add_splat(const std::string& name,
                          const PyTensor& means,
                          const PyTensor& sh0,
                          const PyTensor& shN,
                          const PyTensor& scaling,
                          const PyTensor& rotation,
                          const PyTensor& opacity,
                          int sh_degree,
                          float scene_scale,
                          int32_t parent = vis::NULL_NODE);
        int32_t add_point_cloud(const std::string& name,
                                const PyTensor& points,
                                const PyTensor& colors,
                                int32_t parent = vis::NULL_NODE);
        int32_t add_camera_group(const std::string& name, int32_t parent, size_t camera_count);
        int32_t add_camera(const std::string& name,
                           int32_t parent,
                           const PyTensor& R,
                           const PyTensor& T,
                           float focal_x,
                           float focal_y,
                           int width,
                           int height,
                           const std::string& image_path = "",
                           int uid = -1);
        void remove_node(const std::string& name, bool keep_children = false);
        bool rename_node(const std::string& old_name, const std::string& new_name);
        void clear() { scene_->clear(); }

        // Hierarchy
        void reparent(int32_t node_id, int32_t new_parent_id);
        std::vector<int32_t> root_nodes() const { return scene_->getRootNodes(); }

        // Queries
        std::optional<PySceneNode> get_node_by_id(int32_t id);
        std::optional<PySceneNode> get_node(const std::string& name);
        std::vector<PySceneNode> get_nodes();
        std::vector<PySceneNode> get_visible_nodes();
        bool is_node_effectively_visible(int32_t id) const {
            return scene_->isNodeEffectivelyVisible(id);
        }

        // Transforms
        nb::tuple get_world_transform(int32_t node_id) const;
        void set_node_transform(const std::string& name, nb::ndarray<float, nb::shape<4, 4>> transform);
        void set_node_transform_tensor(const std::string& name, const PyTensor& transform);

        // Combined model
        std::optional<PySplatData> combined_model();
        std::optional<PySplatData> training_model();
        void set_training_model_node(const std::string& name) {
            scene_->setTrainingModelNode(name);
        }
        std::string training_model_node_name() const {
            return scene_->getTrainingModelNodeName();
        }

        // Bounds
        std::optional<std::tuple<std::tuple<float, float, float>, std::tuple<float, float, float>>>
        get_node_bounds(int32_t id) const;
        std::tuple<float, float, float> get_node_bounds_center(int32_t id) const;

        // CropBox operations
        int32_t get_cropbox_for_splat(int32_t splat_id) const {
            return scene_->getCropBoxForSplat(splat_id);
        }
        int32_t get_or_create_cropbox_for_splat(int32_t splat_id) {
            return scene_->getOrCreateCropBoxForSplat(splat_id);
        }
        std::optional<PyCropBox> get_cropbox_data(int32_t cropbox_id);
        void set_cropbox_data(int32_t cropbox_id, const PyCropBox& data);

        // Selection (auto-invalidate + redraw for UI update)
        std::optional<PyTensor> selection_mask() const;
        void set_selection(const std::vector<size_t>& indices) {
            scene_->setSelection(indices);
            scene_->invalidateCache();
            core::events::state::SceneChanged{}.emit();
        }
        void set_selection_mask(const PyTensor& mask) {
            scene_->setSelectionMask(std::make_shared<core::Tensor>(mask.tensor()));
            scene_->invalidateCache();
            core::events::state::SceneChanged{}.emit();
        }
        void clear_selection() {
            scene_->clearSelection();
            scene_->invalidateCache();
            core::events::state::SceneChanged{}.emit();
        }
        bool has_selection() const { return scene_->hasSelection(); }

        // Selection groups
        uint8_t add_selection_group(const std::string& name, std::tuple<float, float, float> color);
        void remove_selection_group(uint8_t id) { scene_->removeSelectionGroup(id); }
        void rename_selection_group(uint8_t id, const std::string& name) {
            scene_->renameSelectionGroup(id, name);
        }
        void set_selection_group_color(uint8_t id, std::tuple<float, float, float> color);
        void set_selection_group_locked(uint8_t id, bool locked) {
            scene_->setSelectionGroupLocked(id, locked);
        }
        bool is_selection_group_locked(uint8_t id) const {
            return scene_->isSelectionGroupLocked(id);
        }
        void set_active_selection_group(uint8_t id) {
            scene_->setActiveSelectionGroup(id);
        }
        uint8_t active_selection_group() const {
            return scene_->getActiveSelectionGroup();
        }
        std::vector<PySelectionGroup> selection_groups() const;
        void update_selection_group_counts() { scene_->updateSelectionGroupCounts(); }
        void clear_selection_group(uint8_t id) {
            scene_->clearSelectionGroup(id);
            scene_->invalidateCache();
            core::events::state::SceneChanged{}.emit();
        }
        void reset_selection_state() {
            scene_->resetSelectionState();
            scene_->invalidateCache();
            core::events::state::SceneChanged{}.emit();
        }

        // Camera training control
        void set_camera_training_enabled(const std::string& name, bool enabled) {
            scene_->setCameraTrainingEnabled(name, enabled);
        }
        size_t active_camera_count() const { return scene_->getActiveCameraCount(); }
        std::vector<PySceneNode> get_active_cameras();

        // Training data
        bool has_training_data() const { return scene_->hasTrainingData(); }
        PyTensor scene_center() const;

        // Counts
        size_t node_count() const { return scene_->getNodeCount(); }
        size_t total_gaussian_count() const { return scene_->getTotalGaussianCount(); }
        bool has_nodes() const { return scene_->hasNodes(); }

        // Operations
        size_t apply_deleted() { return scene_->applyDeleted(); }
        void invalidate_cache() { scene_->invalidateCache(); }
        void notify_changed() {
            scene_->invalidateCache();
            core::events::state::SceneChanged{}.emit();
        }
        std::string duplicate_node(const std::string& name) {
            return scene_->duplicateNode(name);
        }
        std::string merge_group(const std::string& group_name) {
            return scene_->mergeGroup(group_name);
        }

        // Collection property for scene.nodes
        PyNodeCollection nodes() { return PyNodeCollection(scene_); }

        // Access underlying scene (for internal use)
        vis::Scene* scene() { return scene_; }
        const vis::Scene* scene() const { return scene_; }

    private:
        vis::Scene* scene_;
        uint64_t generation_;
    };

    // Register scene classes with nanobind module
    void register_scene(nb::module_& m);

} // namespace lfs::python
