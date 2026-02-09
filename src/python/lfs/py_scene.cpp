/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_scene.hpp"
#include "core/camera.hpp"
#include "core/events.hpp"
#include "core/path_utils.hpp"
#include "core/property_registry.hpp"
#include "python/python_runtime.hpp"
#include <nanobind/ndarray.h>

namespace lfs::python {

    namespace {
        void register_scene_node_properties() {
            using namespace lfs::core::prop;
            PropertyGroupBuilder<core::SceneNode>("scene_node", "Scene Node")
                .string_prop(&core::SceneNode::name, "name", "Name", "", "Node name")
                .animatable_prop(&core::SceneNode::local_transform, "local_transform", "Transform",
                                 glm::mat4(1.0f), "Local transform matrix")
                .animatable_prop(&core::SceneNode::visible, "visible", "Visible", true,
                                 "Node visibility")
                .animatable_prop(&core::SceneNode::locked, "locked", "Locked", false,
                                 "Lock node from editing")
                .bool_prop(&core::SceneNode::training_enabled, "training_enabled", "Training Enabled",
                           true, "Include camera in training dataset")
                .build();
        }

        void register_cropbox_properties() {
            using namespace lfs::core::prop;
            PropertyGroupBuilder<core::CropBoxData>("crop_box", "Crop Box")
                .vec3_prop(&core::CropBoxData::min, "min", "Min", glm::vec3(-1.0f),
                           "Minimum corner of crop box")
                .vec3_prop(&core::CropBoxData::max, "max", "Max", glm::vec3(1.0f),
                           "Maximum corner of crop box")
                .bool_prop(&core::CropBoxData::inverse, "inverse", "Invert", false,
                           "Invert crop logic")
                .bool_prop(&core::CropBoxData::enabled, "enabled", "Enabled", false,
                           "Enable crop filtering")
                .color3_prop(&core::CropBoxData::color, "color", "Color", glm::vec3(1.0f, 1.0f, 0.0f),
                             "Visualization color")
                .float_prop(&core::CropBoxData::line_width, "line_width", "Line Width", 2.0f, 0.1f,
                            10.0f, "Border line width")
                .float_prop(&core::CropBoxData::flash_intensity, "flash_intensity", "Flash", 0.0f,
                            0.0f, 1.0f, "Flash effect intensity")
                .build();
        }

        void register_ellipsoid_properties() {
            using namespace lfs::core::prop;
            PropertyGroupBuilder<core::EllipsoidData>("ellipsoid", "Ellipsoid")
                .vec3_prop(&core::EllipsoidData::radii, "radii", "Radii", glm::vec3(1.0f),
                           "Ellipsoid radii")
                .bool_prop(&core::EllipsoidData::inverse, "inverse", "Invert", false,
                           "Invert selection logic")
                .bool_prop(&core::EllipsoidData::enabled, "enabled", "Enabled", false,
                           "Enable selection filtering")
                .color3_prop(&core::EllipsoidData::color, "color", "Color", glm::vec3(1.0f, 1.0f, 0.0f),
                             "Visualization color")
                .float_prop(&core::EllipsoidData::line_width, "line_width", "Line Width", 2.0f, 0.1f,
                            10.0f, "Border line width")
                .float_prop(&core::EllipsoidData::flash_intensity, "flash_intensity", "Flash", 0.0f,
                            0.0f, 1.0f, "Flash effect intensity")
                .build();
        }
    } // namespace

    // Helper to convert glm::mat4 to nb::tuple (row-major for NumPy compatibility)
    static nb::tuple mat4_to_tuple(const glm::mat4& m) {
        nb::list rows;
        for (int i = 0; i < 4; ++i) {
            nb::list row;
            for (int j = 0; j < 4; ++j) {
                row.append(m[j][i]); // glm is column-major, so m[col][row]
            }
            rows.append(nb::tuple(row));
        }
        return nb::tuple(rows);
    }

    // Helper to convert ndarray to glm::mat4
    static glm::mat4 ndarray_to_mat4(nb::ndarray<float, nb::shape<4, 4>> arr) {
        glm::mat4 m;
        auto view = arr.view();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[j][i] = view(i, j); // glm is column-major
            }
        }
        return m;
    }

    // PySceneNode implementation
    void PySceneNode::set_local_transform(nb::ndarray<float, nb::shape<4, 4>> transform) {
        node_->local_transform = ndarray_to_mat4(transform);
    }

    nb::tuple PySceneNode::world_transform() const {
        return mat4_to_tuple(scene_->getWorldTransform(node_->id));
    }

    std::optional<PySplatData> PySceneNode::splat_data() {
        if (node_->type != core::NodeType::SPLAT || !node_->model) {
            return std::nullopt;
        }
        return PySplatData(node_->model.get());
    }

    std::optional<PyPointCloud> PySceneNode::point_cloud() {
        if (node_->type != core::NodeType::POINTCLOUD || !node_->point_cloud) {
            return std::nullopt;
        }
        return PyPointCloud(node_->point_cloud.get(), false, node_, scene_);
    }

    // PyPointCloud filter implementation - uses tensor[mask] row selection
    int64_t PyPointCloud::filter(const PyTensor& keep_mask) {
        const auto& mask = keep_mask.tensor();
        assert(mask.dtype() == core::DataType::Bool && "Mask must be boolean");
        assert(mask.shape().rank() == 1 && "Mask must be 1D");
        assert(static_cast<size_t>(mask.shape()[0]) == static_cast<size_t>(pc_->size()) && "Mask size must match point count");

        // Ensure mask is on same device as data
        const auto device = pc_->means.device();
        const auto mask_dev = mask.device() == device ? mask : mask.to(device);

        const int64_t old_size = pc_->size();
        pc_->means = pc_->means.is_valid() ? pc_->means[mask_dev] : pc_->means;
        pc_->colors = pc_->colors.is_valid() ? pc_->colors[mask_dev] : pc_->colors;
        pc_->normals = pc_->normals.is_valid() ? pc_->normals[mask_dev] : pc_->normals;
        pc_->sh0 = pc_->sh0.is_valid() ? pc_->sh0[mask_dev] : pc_->sh0;
        pc_->shN = pc_->shN.is_valid() ? pc_->shN[mask_dev] : pc_->shN;
        pc_->opacity = pc_->opacity.is_valid() ? pc_->opacity[mask_dev] : pc_->opacity;
        pc_->scaling = pc_->scaling.is_valid() ? pc_->scaling[mask_dev] : pc_->scaling;
        pc_->rotation = pc_->rotation.is_valid() ? pc_->rotation[mask_dev] : pc_->rotation;

        return old_size - pc_->size();
    }

    int64_t PyPointCloud::filter_indices(const PyTensor& indices) {
        const auto& idx = indices.tensor();
        assert(idx.shape().rank() == 1 && "Indices must be 1D");

        const auto device = pc_->means.device();
        const auto idx_dev = idx.device() == device ? idx : idx.to(device);

        const int64_t old_size = pc_->size();
        pc_->means = pc_->means.is_valid() ? pc_->means[idx_dev] : pc_->means;
        pc_->colors = pc_->colors.is_valid() ? pc_->colors[idx_dev] : pc_->colors;
        pc_->normals = pc_->normals.is_valid() ? pc_->normals[idx_dev] : pc_->normals;
        pc_->sh0 = pc_->sh0.is_valid() ? pc_->sh0[idx_dev] : pc_->sh0;
        pc_->shN = pc_->shN.is_valid() ? pc_->shN[idx_dev] : pc_->shN;
        pc_->opacity = pc_->opacity.is_valid() ? pc_->opacity[idx_dev] : pc_->opacity;
        pc_->scaling = pc_->scaling.is_valid() ? pc_->scaling[idx_dev] : pc_->scaling;
        pc_->rotation = pc_->rotation.is_valid() ? pc_->rotation[idx_dev] : pc_->rotation;

        return old_size - pc_->size();
    }

    void PyPointCloud::set_data(const PyTensor& points, const PyTensor& colors) {
        const auto& pts = points.tensor();
        const auto& cols = colors.tensor();
        assert(pts.shape().rank() == 2 && pts.shape()[1] == 3);
        assert(cols.shape().rank() == 2 && cols.shape()[1] == 3);
        assert(pts.shape()[0] == cols.shape()[0]);

        pc_->means = pts.to(core::Device::CUDA);
        pc_->colors = cols.to(core::Device::CUDA);

        const int64_t n = pc_->size();
        if (node_) {
            node_->gaussian_count = n;
            if (n > 0) {
                auto centroid = pc_->means.mean(0).cpu();
                auto acc = centroid.accessor<float, 1>();
                node_->centroid = glm::vec3(acc(0), acc(1), acc(2));
            }
        }
        if (scene_) {
            scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        }
    }

    void PyPointCloud::set_colors(const PyTensor& colors) {
        const auto& cols = colors.tensor();
        assert(cols.shape().rank() == 2 && cols.shape()[1] == 3);
        assert(cols.shape()[0] == pc_->size());
        pc_->colors = cols.to(core::Device::CUDA);
        if (scene_) {
            scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        }
    }

    void PyPointCloud::set_means(const PyTensor& points) {
        const auto& pts = points.tensor();
        assert(pts.shape().rank() == 2 && pts.shape()[1] == 3);
        assert(pts.shape()[0] == pc_->size());
        pc_->means = pts.to(core::Device::CUDA);
        if (node_ && pc_->size() > 0) {
            auto centroid = pc_->means.mean(0).cpu();
            auto acc = centroid.accessor<float, 1>();
            node_->centroid = glm::vec3(acc(0), acc(1), acc(2));
        }
        if (scene_) {
            scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        }
    }

    std::optional<PyCropBox> PySceneNode::cropbox() {
        if (node_->type != core::NodeType::CROPBOX || !node_->cropbox) {
            return std::nullopt;
        }
        return PyCropBox(node_->cropbox.get());
    }

    std::optional<PyEllipsoid> PySceneNode::ellipsoid() {
        if (node_->type != core::NodeType::ELLIPSOID || !node_->ellipsoid) {
            return std::nullopt;
        }
        return PyEllipsoid(node_->ellipsoid.get());
    }

    // PyScene implementation
    PyScene::PyScene(core::Scene* scene)
        : scene_(scene),
          generation_(get_scene_generation()) {
        assert(scene_ != nullptr);
    }

    bool PyScene::is_valid() const {
        return scene_ == get_application_scene() && generation_ == get_scene_generation();
    }

    uint64_t PyScene::generation() const {
        return generation_;
    }

    int32_t PyScene::add_group(const std::string& name, int32_t parent) {
        return scene_->addGroup(name, parent);
    }

    int32_t PyScene::add_splat(const std::string& name,
                               const PyTensor& means,
                               const PyTensor& sh0,
                               const PyTensor& shN,
                               const PyTensor& scaling,
                               const PyTensor& rotation,
                               const PyTensor& opacity,
                               const int sh_degree,
                               const float scene_scale,
                               const int32_t parent) {
        auto splat = std::make_unique<core::SplatData>(
            sh_degree,
            means.tensor().clone(),
            sh0.tensor().clone(),
            shN.tensor().clone(),
            scaling.tensor().clone(),
            rotation.tensor().clone(),
            opacity.tensor().clone(),
            scene_scale);

        const size_t gaussian_count = splat->size();
        const int32_t node_id = scene_->addSplat(name, std::move(splat), parent);

        lfs::core::events::state::PLYAdded{
            .name = name,
            .node_gaussians = gaussian_count,
            .total_gaussians = scene_->getTotalGaussianCount(),
            .is_visible = true,
            .parent_name = "",
            .is_group = false,
            .node_type = 0}
            .emit();

        return node_id;
    }

    int32_t PyScene::add_point_cloud(const std::string& name,
                                     const PyTensor& points,
                                     const PyTensor& colors,
                                     const int32_t parent) {
        const auto& pts = points.tensor();
        const auto& cols = colors.tensor();
        assert(pts.shape().rank() == 2 && pts.shape()[1] == 3);
        assert(cols.shape().rank() == 2 && cols.shape()[1] == 3);
        assert(pts.shape()[0] == cols.shape()[0]);

        auto pc = std::make_shared<core::PointCloud>(pts.clone(), cols.clone());
        return scene_->addPointCloud(name, std::move(pc), parent);
    }

    int32_t PyScene::add_camera_group(const std::string& name, const int32_t parent, const size_t camera_count) {
        return scene_->addCameraGroup(name, parent, camera_count);
    }

    int32_t PyScene::add_camera(const std::string& name,
                                const int32_t parent,
                                const PyTensor& R,
                                const PyTensor& T,
                                float focal_x,
                                float focal_y,
                                int width,
                                int height,
                                const std::string& image_path,
                                int uid) {
        const auto& R_tensor = R.tensor();
        const auto& T_tensor = T.tensor();
        assert(R_tensor.ndim() == 2 && R_tensor.size(0) == 3 && R_tensor.size(1) == 3);
        assert(T_tensor.ndim() == 2 && T_tensor.size(0) == 3 && T_tensor.size(1) == 1);

        auto camera = std::make_shared<lfs::core::Camera>(
            R_tensor.clone(),
            T_tensor.clone(),
            focal_x, focal_y,
            static_cast<float>(width) / 2.0f, static_cast<float>(height) / 2.0f,
            lfs::core::Tensor{},
            lfs::core::Tensor{},
            lfs::core::CameraModelType::PINHOLE,
            name,
            lfs::core::utf8_to_path(image_path),
            std::filesystem::path{},
            width, height,
            uid);

        return scene_->addCamera(name, parent, std::move(camera));
    }

    void PyScene::remove_node(const std::string& name, bool keep_children) {
        scene_->removeNode(name, keep_children);
    }

    bool PyScene::rename_node(const std::string& old_name, const std::string& new_name) {
        return scene_->renameNode(old_name, new_name);
    }

    void PyScene::reparent(int32_t node_id, int32_t new_parent_id) {
        scene_->reparent(node_id, new_parent_id);
    }

    std::optional<PySceneNode> PyScene::get_node_by_id(int32_t id) {
        auto* node = scene_->getNodeById(id);
        if (!node)
            return std::nullopt;
        return PySceneNode(node, scene_);
    }

    std::optional<PySceneNode> PyScene::get_node(const std::string& name) {
        auto* node = scene_->getMutableNode(name);
        if (!node)
            return std::nullopt;
        return PySceneNode(node, scene_);
    }

    std::vector<PySceneNode> PyScene::get_nodes() {
        std::vector<PySceneNode> result;
        for (const auto* node : scene_->getNodes()) {
            result.emplace_back(const_cast<core::SceneNode*>(node), scene_);
        }
        return result;
    }

    std::vector<PySceneNode> PyScene::get_visible_nodes() {
        std::vector<PySceneNode> result;
        for (const auto* node : scene_->getVisibleNodes()) {
            result.emplace_back(const_cast<core::SceneNode*>(node), scene_);
        }
        return result;
    }

    std::vector<PySceneNode> PyScene::get_active_cameras() {
        std::vector<PySceneNode> result;
        for (const auto* node : scene_->getNodes()) {
            if (node->type == core::NodeType::CAMERA && node->camera && node->training_enabled) {
                result.emplace_back(const_cast<core::SceneNode*>(node), scene_);
            }
        }
        return result;
    }

    nb::tuple PyScene::get_world_transform(int32_t node_id) const {
        return mat4_to_tuple(scene_->getWorldTransform(node_id));
    }

    void PyScene::set_node_transform(const std::string& name, nb::ndarray<float, nb::shape<4, 4>> transform) {
        scene_->setNodeTransform(name, ndarray_to_mat4(transform));
    }

    void PyScene::set_node_transform_tensor(const std::string& name, const PyTensor& transform) {
        const auto& t = transform.tensor();
        assert(t.ndim() == 2 && t.size(0) == 4 && t.size(1) == 4);
        auto cpu_t = t.device() == core::Device::CUDA ? t.cpu() : t;
        auto contiguous = cpu_t.contiguous();
        const float* data = contiguous.ptr<float>();
        glm::mat4 m;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[j][i] = data[i * 4 + j];
            }
        }
        scene_->setNodeTransform(name, m);
    }

    std::optional<PySplatData> PyScene::combined_model() {
        auto* model = const_cast<core::SplatData*>(scene_->getCombinedModel());
        if (!model)
            return std::nullopt;
        return PySplatData(model);
    }

    std::optional<PySplatData> PyScene::training_model() {
        auto* model = scene_->getTrainingModel();
        if (!model)
            return std::nullopt;
        return PySplatData(model);
    }

    std::optional<std::tuple<std::tuple<float, float, float>, std::tuple<float, float, float>>>
    PyScene::get_node_bounds(int32_t id) const {
        glm::vec3 min_bound, max_bound;
        if (!scene_->getNodeBounds(id, min_bound, max_bound)) {
            return std::nullopt;
        }
        return std::make_tuple(
            std::make_tuple(min_bound.x, min_bound.y, min_bound.z),
            std::make_tuple(max_bound.x, max_bound.y, max_bound.z));
    }

    std::tuple<float, float, float> PyScene::get_node_bounds_center(int32_t id) const {
        auto center = scene_->getNodeBoundsCenter(id);
        return {center.x, center.y, center.z};
    }

    std::optional<PyCropBox> PyScene::get_cropbox_data(int32_t cropbox_id) {
        auto* data = scene_->getCropBoxData(cropbox_id);
        if (!data)
            return std::nullopt;
        return PyCropBox(data);
    }

    void PyScene::set_cropbox_data(const int32_t cropbox_id, const PyCropBox& data) {
        auto* scene_data = scene_->getCropBoxData(cropbox_id);
        if (scene_data) {
            *scene_data = *data.data();
        }
    }

    std::optional<PyTensor> PyScene::selection_mask() const {
        auto mask = scene_->getSelectionMask();
        if (!mask || !mask->is_valid())
            return std::nullopt;
        return PyTensor(*mask, false);
    }

    uint8_t PyScene::add_selection_group(const std::string& name, std::tuple<float, float, float> color) {
        return scene_->addSelectionGroup(name, {std::get<0>(color), std::get<1>(color), std::get<2>(color)});
    }

    void PyScene::set_selection_group_color(uint8_t id, std::tuple<float, float, float> color) {
        scene_->setSelectionGroupColor(id, {std::get<0>(color), std::get<1>(color), std::get<2>(color)});
    }

    std::vector<PySelectionGroup> PyScene::selection_groups() const {
        std::vector<PySelectionGroup> result;
        for (const auto& group : scene_->getSelectionGroups()) {
            result.push_back({group.id,
                              group.name,
                              {group.color.x, group.color.y, group.color.z},
                              group.count,
                              group.locked});
        }
        return result;
    }

    PyTensor PyScene::scene_center() const {
        return PyTensor(scene_->getSceneCenter(), false);
    }

    void register_scene(nb::module_& m) {
        register_scene_node_properties();
        register_cropbox_properties();
        register_ellipsoid_properties();

        // NodeType enum
        nb::enum_<core::NodeType>(m, "NodeType")
            .value("SPLAT", core::NodeType::SPLAT)
            .value("POINTCLOUD", core::NodeType::POINTCLOUD)
            .value("GROUP", core::NodeType::GROUP)
            .value("CROPBOX", core::NodeType::CROPBOX)
            .value("ELLIPSOID", core::NodeType::ELLIPSOID)
            .value("DATASET", core::NodeType::DATASET)
            .value("CAMERA_GROUP", core::NodeType::CAMERA_GROUP)
            .value("CAMERA", core::NodeType::CAMERA)
            .value("IMAGE_GROUP", core::NodeType::IMAGE_GROUP)
            .value("IMAGE", core::NodeType::IMAGE);

        // SelectionGroup struct
        nb::class_<PySelectionGroup>(m, "SelectionGroup")
            .def_ro("id", &PySelectionGroup::id, "Group identifier")
            .def_ro("name", &PySelectionGroup::name, "Group display name")
            .def_ro("color", &PySelectionGroup::color, "Group color as (r, g, b) tuple")
            .def_ro("count", &PySelectionGroup::count, "Number of selected Gaussians in this group")
            .def_ro("locked", &PySelectionGroup::locked, "Whether the group is locked from editing");

        nb::class_<PyCropBox>(m, "CropBox")
            .def_prop_ro("__property_group__", &PyCropBox::property_group, "Property group name for introspection")
            .def("get", &PyCropBox::get, nb::arg("name"), "Get property value by name")
            .def("set", &PyCropBox::set, nb::arg("name"), nb::arg("value"), "Set property value by name")
            .def("prop_info", &PyCropBox::prop_info, nb::arg("name"), "Get metadata for a property")
            .def(
                "__getattr__",
                [](PyCropBox& self, const std::string& name) -> nb::object {
                    if (!self.has_prop(name)) {
                        throw nb::attribute_error(("CropBox has no attribute '" + name + "'").c_str());
                    }
                    return self.prop_getattr(name);
                })
            .def(
                "__setattr__",
                [](PyCropBox& self, const std::string& name, nb::object value) {
                    if (!self.has_prop(name)) {
                        throw nb::attribute_error(("Cannot set attribute '" + name + "'").c_str());
                    }
                    self.prop_setattr(name, value);
                })
            .def("__dir__", &PyCropBox::python_dir, "List available attributes");

        nb::class_<PyEllipsoid>(m, "Ellipsoid")
            .def_prop_ro("__property_group__", &PyEllipsoid::property_group, "Property group name for introspection")
            .def("get", &PyEllipsoid::get, nb::arg("name"), "Get property value by name")
            .def("set", &PyEllipsoid::set, nb::arg("name"), nb::arg("value"), "Set property value by name")
            .def("prop_info", &PyEllipsoid::prop_info, nb::arg("name"), "Get metadata for a property")
            .def(
                "__getattr__",
                [](PyEllipsoid& self, const std::string& name) -> nb::object {
                    if (!self.has_prop(name)) {
                        throw nb::attribute_error(("Ellipsoid has no attribute '" + name + "'").c_str());
                    }
                    return self.prop_getattr(name);
                })
            .def(
                "__setattr__",
                [](PyEllipsoid& self, const std::string& name, nb::object value) {
                    if (!self.has_prop(name)) {
                        throw nb::attribute_error(("Cannot set attribute '" + name + "'").c_str());
                    }
                    self.prop_setattr(name, value);
                })
            .def("__dir__", &PyEllipsoid::python_dir, "List available attributes");

        // PointCloud class
        nb::class_<PyPointCloud>(m, "PointCloud")
            .def_prop_ro("means", &PyPointCloud::means, "Position tensor [N, 3]")
            .def_prop_ro("colors", &PyPointCloud::colors, "Color tensor [N, 3]")
            .def_prop_ro("normals", &PyPointCloud::normals, "Normal tensor [N, 3]")
            .def_prop_ro("sh0", &PyPointCloud::sh0, "Base SH coefficients [N, 1, 3]")
            .def_prop_ro("shN", &PyPointCloud::shN, "Higher-order SH coefficients [N, K, 3]")
            .def_prop_ro("opacity", &PyPointCloud::opacity, "Opacity tensor [N, 1]")
            .def_prop_ro("scaling", &PyPointCloud::scaling, "Scaling tensor [N, 3]")
            .def_prop_ro("rotation", &PyPointCloud::rotation, "Rotation quaternion tensor [N, 4]")
            .def_prop_ro("size", &PyPointCloud::size, "Number of points")
            .def("is_gaussian", &PyPointCloud::is_gaussian, "Check if point cloud has Gaussian attributes")
            .def_prop_ro("attribute_names", &PyPointCloud::attribute_names, "List of valid attribute names")
            .def("normalize_colors", &PyPointCloud::normalize_colors, "Normalize color values to [0, 1] range")
            .def("filter", &PyPointCloud::filter, nb::arg("keep_mask"),
                 "Filter points by boolean mask, returns number of points removed")
            .def("filter_indices", &PyPointCloud::filter_indices, nb::arg("indices"),
                 "Keep only points at specified indices, returns number of points removed")
            .def("set_data", &PyPointCloud::set_data, nb::arg("points"), nb::arg("colors"),
                 "Replace point cloud data with new points and colors tensors")
            .def("set_colors", &PyPointCloud::set_colors, nb::arg("colors"),
                 "Update colors without re-uploading positions [N, 3]")
            .def("set_means", &PyPointCloud::set_means, nb::arg("points"),
                 "Update positions without re-uploading colors [N, 3]");

        // SceneNode class
        nb::class_<PySceneNode>(m, "SceneNode")
            // Property group interface for generic prop()
            .def_prop_ro("__property_group__", &PySceneNode::property_group, "Property group name for introspection")
            .def("get", &PySceneNode::get, nb::arg("name"), "Get property value by name")
            .def("set", &PySceneNode::set, nb::arg("name"), nb::arg("value"), "Set property value by name")
            // Identity (read-only)
            .def_prop_ro("id", &PySceneNode::id, "Unique node identifier")
            .def_prop_ro("parent_id", &PySceneNode::parent_id, "Parent node identifier (-1 for root)")
            .def_prop_ro("children", &PySceneNode::children, "List of child node IDs")
            .def_prop_ro("type", &PySceneNode::type, "Node type (SPLAT, GROUP, CAMERA, etc.)")
            // Transform (special conversion to tuple/ndarray)
            .def_prop_ro("world_transform", &PySceneNode::world_transform, "World-space transform as 4x4 row-major tuple")
            .def("set_local_transform", &PySceneNode::set_local_transform, "Set local transform from a [4, 4] ndarray")
            // Metadata (read-only)
            .def_prop_ro("gaussian_count", &PySceneNode::gaussian_count, "Number of Gaussians owned by this node")
            .def_prop_ro("centroid", &PySceneNode::centroid, "Centroid position (x, y, z)")
            // Data accessors
            .def("splat_data", &PySceneNode::splat_data, "Get SplatData for SPLAT nodes (None otherwise)")
            .def("point_cloud", &PySceneNode::point_cloud, "Get PointCloud for POINTCLOUD nodes (None otherwise)")
            .def("cropbox", &PySceneNode::cropbox, "Get CropBox for CROPBOX nodes (None otherwise)")
            .def("ellipsoid", &PySceneNode::ellipsoid, "Get Ellipsoid for ELLIPSOID nodes (None otherwise)")
            // Camera specific (read-only)
            .def_prop_ro("camera_uid", &PySceneNode::camera_uid, "Camera unique identifier")
            .def_prop_ro("image_path", &PySceneNode::image_path, "Path to the camera image file")
            .def_prop_ro("mask_path", &PySceneNode::mask_path, "Path to the camera mask file")
            .def_prop_ro("has_camera", &PySceneNode::has_camera, "Whether this node has camera data")
            .def_prop_ro("has_mask", &PySceneNode::has_mask, "Whether this camera node has a mask file")
            .def("load_mask", &PySceneNode::load_mask,
                 nb::arg("resize_factor") = 1, nb::arg("max_width") = 3840,
                 nb::arg("invert") = false, nb::arg("threshold") = 0.5f,
                 "Load mask as tensor [1, H, W] on CUDA (None if not a camera node or no mask)")
            .def_prop_ro("camera_R", &PySceneNode::camera_R, "Camera rotation matrix [3, 3]")
            .def_prop_ro("camera_T", &PySceneNode::camera_T, "Camera translation vector [3, 1]")
            .def_prop_ro("camera_focal_x", &PySceneNode::camera_focal_x, "Camera focal length in pixels (x)")
            .def_prop_ro("camera_focal_y", &PySceneNode::camera_focal_y, "Camera focal length in pixels (y)")
            .def_prop_ro("camera_width", &PySceneNode::camera_width, "Camera image width in pixels")
            .def_prop_ro("camera_height", &PySceneNode::camera_height, "Camera image height in pixels")
            // Property introspection
            .def("prop_info", &PySceneNode::prop_info, nb::arg("name"),
                 "Get metadata for a property")
            // Descriptor protocol: node.visible, node.name, node.locked, node.local_transform
            .def("__getattr__", [](PySceneNode& self, const std::string& name) -> nb::object {
                if (!self.has_prop(name)) {
                    throw nb::attribute_error(
                        ("SceneNode has no attribute '" + name + "'").c_str());
                }
                return self.prop_getattr(name);
            })
            .def("__setattr__", [](PySceneNode& self, const std::string& name, nb::object value) {
                if (!self.has_prop(name)) {
                    throw nb::attribute_error(
                        ("Cannot set attribute '" + name + "'").c_str());
                }
                self.prop_setattr(name, value);
            })
            .def("__dir__", &PySceneNode::python_dir);

        // NodeCollection iterator
        nb::class_<PyNodeCollection::Iterator>(m, "NodeCollectionIterator")
            .def("__next__", &PyNodeCollection::Iterator::next, "Advance to the next node");

        // NodeCollection for scene.nodes
        nb::class_<PyNodeCollection>(m, "NodeCollection")
            .def("__len__", &PyNodeCollection::size, "Return the number of nodes")
            .def("__getitem__", &PyNodeCollection::getitem, nb::arg("index"), "Get node by index")
            .def("__iter__", &PyNodeCollection::iter, "Return an iterator over all nodes");

        // Scene class
        nb::class_<PyScene>(m, "Scene")
            // Thread-safe validity checking
            .def("is_valid", &PyScene::is_valid,
                 "Check if scene reference is still valid (thread-safe)")
            .def_prop_ro("generation", &PyScene::generation,
                         "Generation counter when scene was acquired")
            // Node CRUD
            .def("add_group", &PyScene::add_group,
                 nb::arg("name"), nb::arg("parent") = core::NULL_NODE,
                 "Add an empty group node, returns node ID")
            .def("add_splat", &PyScene::add_splat,
                 nb::arg("name"),
                 nb::arg("means"),
                 nb::arg("sh0"),
                 nb::arg("shN"),
                 nb::arg("scaling"),
                 nb::arg("rotation"),
                 nb::arg("opacity"),
                 nb::arg("sh_degree") = 0,
                 nb::arg("scene_scale") = 1.0f,
                 nb::arg("parent") = core::NULL_NODE,
                 R"doc(Add a new splat node from tensor data.

Args:
    name: Node name in scene graph
    means: Position tensor [N, 3]
    sh0: Base SH color [N, 1, 3]
    shN: Higher SH coefficients [N, K, 3] or empty
    scaling: Log-scale factors [N, 3]
    rotation: Quaternions [N, 4] (wxyz)
    opacity: Logit opacity [N, 1]
    sh_degree: SH degree (0 for RGB only)
    scene_scale: Scene scale factor
    parent: Parent node ID (-1 for root)

Returns:
    Node ID of created splat
)doc")
            .def("add_point_cloud", &PyScene::add_point_cloud,
                 nb::arg("name"),
                 nb::arg("points"),
                 nb::arg("colors"),
                 nb::arg("parent") = core::NULL_NODE,
                 "Add a point cloud node from tensor data [N,3] positions and colors")
            .def("add_camera_group", &PyScene::add_camera_group,
                 nb::arg("name"),
                 nb::arg("parent"),
                 nb::arg("camera_count"),
                 "Add a camera group node")
            .def("add_camera", &PyScene::add_camera,
                 nb::arg("name"),
                 nb::arg("parent"),
                 nb::arg("R"),
                 nb::arg("T"),
                 nb::arg("focal_x"),
                 nb::arg("focal_y"),
                 nb::arg("width"),
                 nb::arg("height"),
                 nb::arg("image_path") = "",
                 nb::arg("uid") = -1,
                 R"doc(Add a camera node with intrinsic and extrinsic parameters.

Args:
    name: Camera node name
    parent: Parent node ID (typically a camera group)
    R: Rotation matrix [3,3] (world-to-camera)
    T: Translation vector [3,1]
    focal_x: Focal length in pixels (x)
    focal_y: Focal length in pixels (y)
    width: Image width in pixels
    height: Image height in pixels
    image_path: Optional path to camera image
    uid: Optional unique identifier (-1 for auto-assigned)

Returns:
    Node ID of created camera
)doc")
            .def("remove_node", &PyScene::remove_node,
                 nb::arg("name"), nb::arg("keep_children") = false,
                 "Remove a node by name, optionally keeping its children")
            .def("rename_node", &PyScene::rename_node,
                 nb::arg("old_name"), nb::arg("new_name"),
                 "Rename a node, returns true on success")
            .def("clear", &PyScene::clear,
                 "Remove all nodes from the scene")
            // Hierarchy
            .def("reparent", &PyScene::reparent,
                 nb::arg("node_id"), nb::arg("new_parent_id"),
                 "Move a node under a new parent")
            .def("root_nodes", &PyScene::root_nodes,
                 "Get all root-level nodes")
            // Queries
            .def("get_node_by_id", &PyScene::get_node_by_id, nb::arg("id"),
                 "Find a node by its integer ID (None if not found)")
            .def("get_node", &PyScene::get_node, nb::arg("name"),
                 "Find a node by name (None if not found)")
            .def(
                "get_nodes", [](PyScene& self, std::optional<core::NodeType> type) -> std::vector<PySceneNode> {
                    auto all = self.get_nodes();
                    if (!type)
                        return all;
                    std::vector<PySceneNode> filtered;
                    for (auto& node : all) {
                        if (node.type() == *type)
                            filtered.push_back(std::move(node));
                    }
                    return filtered;
                },
                nb::arg("type") = nb::none(), "Get nodes, optionally filtered by NodeType")
            .def("get_visible_nodes", &PyScene::get_visible_nodes, "Get all visible nodes in the scene")
            .def("is_node_effectively_visible", &PyScene::is_node_effectively_visible, nb::arg("id"), "Check if a node is visible considering parent visibility")
            // Transforms
            .def("get_world_transform", &PyScene::get_world_transform, nb::arg("node_id"), "Get world-space transform as 4x4 row-major tuple")
            .def("set_node_transform", &PyScene::set_node_transform, nb::arg("name"), nb::arg("transform"), "Set node local transform from a [4, 4] ndarray")
            .def("set_node_transform", &PyScene::set_node_transform_tensor, nb::arg("name"), nb::arg("transform"), "Set node local transform from a [4, 4] Tensor")
            // Combined/training model
            .def("combined_model", &PyScene::combined_model, "Get the merged SplatData for all visible splats (None if empty)")
            .def("training_model", &PyScene::training_model, "Get the SplatData used for training (None if unavailable)")
            .def("set_training_model_node", &PyScene::set_training_model_node, nb::arg("name"), "Set which node provides the training model")
            .def_prop_ro("training_model_node_name", &PyScene::training_model_node_name, "Name of the node providing the training model")
            // Bounds (by id)
            .def("get_node_bounds", &PyScene::get_node_bounds, nb::arg("id"), "Get axis-aligned bounding box as ((min_x, min_y, min_z), (max_x, max_y, max_z))")
            .def("get_node_bounds_center", &PyScene::get_node_bounds_center, nb::arg("id"), "Get center of the node bounding box as (x, y, z)")
            // Bounds (by name)
            .def(
                "get_node_bounds", [](PyScene& self, const std::string& name) {
                    auto node = self.get_node(name);
                    if (!node)
                        return decltype(self.get_node_bounds(0)){std::nullopt};
                    return self.get_node_bounds(node->id()); }, nb::arg("name"), "Get axis-aligned bounding box by node name")
            .def(
                "get_node_bounds_center", [](PyScene& self, const std::string& name) {
                    auto node = self.get_node(name);
                    if (!node)
                        throw std::runtime_error("Node not found: " + name);
                    return self.get_node_bounds_center(node->id()); }, nb::arg("name"), "Get center of the node bounding box by name")
            // CropBox
            .def("get_cropbox_for_splat", &PyScene::get_cropbox_for_splat, nb::arg("splat_id"), "Get the crop box node ID associated with a splat (-1 if none)")
            .def("get_or_create_cropbox_for_splat", &PyScene::get_or_create_cropbox_for_splat, nb::arg("splat_id"), "Get or create a crop box for a splat, returns cropbox node ID")
            .def("get_cropbox_data", &PyScene::get_cropbox_data, nb::arg("cropbox_id"), "Get CropBox data for a cropbox node (None if invalid)")
            .def("set_cropbox_data", &PyScene::set_cropbox_data, nb::arg("cropbox_id"), nb::arg("data"), "Set CropBox data for a cropbox node")
            // Selection
            .def_prop_ro("selection_mask", &PyScene::selection_mask, "Current selection mask tensor [N] uint8 (None if no selection)")
            .def("set_selection", &PyScene::set_selection, nb::arg("indices"), "Set selection from index tensor")
            .def("set_selection_mask", &PyScene::set_selection_mask, nb::arg("mask"), "Set selection from boolean mask tensor [N]")
            .def("clear_selection", &PyScene::clear_selection, "Clear all selected Gaussians")
            .def("has_selection", &PyScene::has_selection, "Check if any Gaussians are selected")
            // Selection groups
            .def("add_selection_group", &PyScene::add_selection_group, nb::arg("name"), nb::arg("color"), "Add a named selection group with (r, g, b) color, returns group ID")
            .def("remove_selection_group", &PyScene::remove_selection_group, nb::arg("id"), "Remove a selection group by ID")
            .def("rename_selection_group", &PyScene::rename_selection_group, nb::arg("id"), nb::arg("name"), "Rename a selection group")
            .def("set_selection_group_color", &PyScene::set_selection_group_color, nb::arg("id"), nb::arg("color"), "Set a selection group color as (r, g, b) tuple")
            .def("set_selection_group_locked", &PyScene::set_selection_group_locked, nb::arg("id"), nb::arg("locked"), "Lock or unlock a selection group")
            .def("is_selection_group_locked", &PyScene::is_selection_group_locked, nb::arg("id"), "Check if a selection group is locked")
            .def_prop_rw("active_selection_group", &PyScene::active_selection_group, &PyScene::set_active_selection_group, "Currently active selection group ID")
            .def("selection_groups", &PyScene::selection_groups, "Get all selection groups")
            .def("update_selection_group_counts", &PyScene::update_selection_group_counts, "Recompute selection counts for all groups")
            .def("clear_selection_group", &PyScene::clear_selection_group, nb::arg("id"), "Clear all selections in a group")
            .def("reset_selection_state", &PyScene::reset_selection_state, "Reset all selection state to defaults")
            // Camera training control
            .def("set_camera_training_enabled", &PyScene::set_camera_training_enabled, nb::arg("name"), nb::arg("enabled"), "Enable or disable a camera for training by name")
            .def_prop_ro("active_camera_count", &PyScene::active_camera_count, "Number of cameras enabled for training")
            .def("get_active_cameras", &PyScene::get_active_cameras, "Get camera nodes enabled for training")
            // Training data
            .def("has_training_data", &PyScene::has_training_data, "Check if training dataset is loaded")
            .def_prop_ro("scene_center", &PyScene::scene_center, "Scene center position as a [3] tensor")
            // Counts
            .def_prop_ro("node_count", &PyScene::node_count, "Total number of nodes in the scene")
            .def_prop_ro("total_gaussian_count", &PyScene::total_gaussian_count, "Total number of Gaussians across all nodes")
            .def("has_nodes", &PyScene::has_nodes, "Check if the scene contains any nodes")
            // Operations
            .def("apply_deleted", &PyScene::apply_deleted, "Permanently remove soft-deleted Gaussians from all nodes")
            .def("invalidate_cache", &PyScene::invalidate_cache, "Invalidate the combined model cache")
            .def("notify_changed", &PyScene::notify_changed, "Notify the renderer that scene data has changed")
            .def("duplicate_node", &PyScene::duplicate_node, nb::arg("name"), "Duplicate a node by name, returns new node ID")
            .def("merge_group", &PyScene::merge_group, nb::arg("group_name"), "Merge all splats in a group into a single node, returns merged node ID")
            .def_prop_ro("nodes", &PyScene::nodes, "Iterable collection of all scene nodes");
    }

} // namespace lfs::python
