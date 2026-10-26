/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_scene.hpp"
#include "core/camera.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/property_registry.hpp"
#include "io/loader.hpp"
#include "py_error.hpp"
#include "py_viewer_dispatch.hpp"
#include "python/python_runtime.hpp"
#include "rendering/selection_ops.hpp"
#include "visualizer/core/training_manager.hpp"
#include "visualizer/core/training_state.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/input/input_controller.hpp"
#include "visualizer/operation/undo_entry.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/rendering/graphics_external_tensor.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/type_ptr.hpp>
#include <limits>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace lfs::python {

    namespace {
        template <typename F>
        auto point_cloud_access(core::Scene* scene, F&& access) -> std::invoke_result_t<F> {
            using T = std::invoke_result_t<F>;
            if (scene)
                reject_viewer_work_in_node_execute();
            auto* viewer = scene ? get_visualizer() : nullptr;
            if (!viewer || viewer->isOnViewerThread())
                return access();
            const core::TaskContext context{
                .name = "python.point_cloud.access",
                .domain = lfs::ErrorDomain::Python,
                .operation_id = lfs::OperationId::generate(),
                .site = LFS_SOURCE_SITE_CURRENT(),
            };
            const auto cancelled = lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Cancelled,
                .domain = lfs::ErrorDomain::Python,
                .severity = lfs::Severity::Warning,
                .detail = "Viewer is shutting down",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
            nb::gil_scoped_release release;
            return unwrap(vis::post_guarded_and_wait<T>(
                *viewer, context,
                [&access]() -> lfs::Result<T> {
                    if constexpr (std::is_void_v<T>) {
                        access();
                        return {};
                    } else {
                        return access();
                    }
                },
                cancelled));
        }

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

        void register_crop_polygon_properties() {
            using namespace lfs::core::prop;
            PropertyGroupBuilder<core::CropPolygonData>("crop_polygon", "Crop Polygon")
                .float_prop(&core::CropPolygonData::half_depth, "half_depth", "Half Depth", 1.0f,
                            core::crop_polygon::kMinHalfDepth, 1.0e6f,
                            "Half the extrusion depth along the local z axis")
                .bool_prop(&core::CropPolygonData::inverse, "inverse", "Invert", false,
                           "Invert crop logic")
                .bool_prop(&core::CropPolygonData::enabled, "enabled", "Enabled", true,
                           "Enable crop filtering")
                .color3_prop(&core::CropPolygonData::color, "color", "Color", glm::vec3(1.0f, 1.0f, 0.0f),
                             "Visualization color")
                .float_prop(&core::CropPolygonData::line_width, "line_width", "Line Width", 2.0f, 0.1f,
                            10.0f, "Border line width")
                .float_prop(&core::CropPolygonData::flash_intensity, "flash_intensity", "Flash", 0.0f,
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
    static glm::mat4 ndarray_to_mat4(nb::ndarray<float, nb::device::cpu, nb::shape<4, 4>> arr) {
        glm::mat4 m;
        auto view = arr.view();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[j][i] = view(i, j); // glm is column-major
            }
        }
        return m;
    }

    void apply_node_transform_with_undo(const std::string& name,
                                        const glm::mat4& transform,
                                        core::Scene* scene) {
        if (auto* const scene_manager = get_scene_manager()) {
            if (auto result = vis::cap::setTransformMatrix(*scene_manager, {name}, transform,
                                                           "python.scene.set_node_transform");
                !result) {
                LOG_WARN("PyScene::set_node_transform fell back to direct update for '{}': {}",
                         name, result.error());
                scene_manager->setNodeTransform(name, transform);
            }
            return;
        }

        if (scene)
            scene->setNodeTransform(name, transform);
    }

    void ensure_scene_manager_content_type(vis::SceneManager& scene_manager,
                                           const vis::SceneManager::ContentType content_type) {
        if (scene_manager.getContentType() == vis::SceneManager::ContentType::Empty) {
            scene_manager.changeContentType(content_type);
            set_application_scene(&scene_manager.getScene());
        }
    }

    template <typename Mutator>
    void apply_selection_state_with_undo(vis::SceneManager& scene_manager,
                                         const std::string& undo_label,
                                         Mutator&& mutator) {
        auto snapshot = std::make_unique<vis::op::SceneSnapshot>(scene_manager, undo_label);
        snapshot->captureSelection();
        mutator(scene_manager.getScene());
        snapshot->captureAfter();
        vis::op::pushSceneSnapshotIfChanged(std::move(snapshot));
    }

    [[nodiscard]] vis::op::SceneGraphCaptureOptions scene_graph_history_options(
        const bool include_selected_nodes = false,
        const bool include_scene_context = true) {
        return vis::op::SceneGraphCaptureOptions{
            .mode = vis::op::SceneGraphCaptureMode::FULL,
            .include_selected_nodes = include_selected_nodes,
            .include_scene_context = include_scene_context,
            .payload_uuids = std::vector<core::Uuid>{},
        };
    }

    void push_scene_graph_metadata_history(
        vis::SceneManager& scene_manager,
        std::string label,
        const std::vector<vis::op::SceneGraphNodeMetadataSnapshot>& before,
        const std::vector<vis::op::SceneGraphNodeMetadataSnapshot>& after) {
        std::vector<vis::op::SceneGraphNodeMetadataDiff> diffs;
        const size_t count = std::min(before.size(), after.size());
        diffs.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            diffs.push_back(vis::op::SceneGraphNodeMetadataDiff{
                .before = before[i],
                .after = after[i],
            });
        }
        if (!diffs.empty()) {
            vis::op::undoHistory().push(
                std::make_unique<vis::op::SceneGraphMetadataEntry>(scene_manager, std::move(label), std::move(diffs)));
        }
    }

    // PySceneNode implementation
    void PySceneNode::set_local_transform(
        nb::ndarray<float, nb::device::cpu, nb::shape<4, 4>> transform) {
        apply_node_transform_with_undo(node().name, ndarray_to_mat4(transform), scene_);
    }

    nb::tuple PySceneNode::local_transform() const {
        return mat4_to_tuple(node().local_transform.get());
    }

    nb::tuple PySceneNode::world_transform() const {
        return mat4_to_tuple(scene_->getWorldTransform(node().id));
    }

    std::optional<PySplatData> PySceneNode::splat_data() {
        auto& live = node();
        if (live.type != core::NodeType::SPLAT || !live.model) {
            return std::nullopt;
        }
        return PySplatData(live.model.get(), scene_);
    }

    std::optional<PyPointCloud> PySceneNode::point_cloud() {
        return point_cloud_access(scene_, [&]() -> std::optional<PyPointCloud> {
            auto& live = node();
            if (live.type != core::NodeType::POINTCLOUD || !live.point_cloud)
                return std::nullopt;
            return PyPointCloud(&live, scene_);
        });
    }

    std::optional<PyMeshInfo> PySceneNode::mesh() {
        auto& live = node();
        if (live.type != core::NodeType::MESH || !live.mesh) {
            return std::nullopt;
        }
        return PyMeshInfo(live.mesh);
    }

    PyPointCloud::PyPointCloud(core::SceneNode* node, core::Scene* scene)
        : node_uuid_(node->uuid), scene_(scene) {
        assert(node->point_cloud != nullptr);
        assert(scene_ != nullptr);
        const auto epoch = scene->pointCloudUpdateEpoch();
        update_target_ = {scene, epoch->load(), node->uuid, node->point_cloud_revision,
                          node->point_cloud_revision->load(std::memory_order_acquire), 0, epoch};
    }

    std::shared_ptr<vis::PointCloudUpdateTicket> PyPointCloud::set_data_async(
        nb::object points, nb::object colors,
        std::optional<std::tuple<float, float, float>> centroid, const std::string& queue_policy) {
        reject_viewer_work_in_node_execute();
        if (queue_policy != "latest")
            throw nb::value_error("queue_policy must be 'latest'");
        auto* viewer = get_visualizer();
        auto manager = viewer ? viewer->pointCloudUpdates() : nullptr;
        if (!manager || !update_target_.revision)
            throw std::runtime_error("Async updates require an initialized viewer and a scene-owned point cloud");
        if (update_target_.scene_generation != update_target_.scene_epoch->load(std::memory_order_acquire) || update_target_.scene != get_application_scene())
            throw std::runtime_error("Scene reference is no longer valid");
        struct Owners {
            nb::object points, colors;
            nb::ndarray<nb::numpy, nb::device::cpu> points_array, colors_array;
        };
        auto owners = std::shared_ptr<Owners>(new Owners{std::move(points), std::move(colors), {}, {}}, [](Owners* p) {
            nb::gil_scoped_acquire gil;
            delete p;
        });
        const auto source = [](nb::object object, nb::ndarray<nb::numpy, nb::device::cpu>& array, const bool colors) {
            if (nb::isinstance<PyTensor>(object))
                return nb::cast<const PyTensor&>(object).tensor();
            if (!nb::try_cast(object, array, false))
                throw nb::type_error("Expected a LichtFeld Tensor or a contiguous CPU NumPy array");
            if (array.ndim() != 2 || array.shape(1) != 3 ||
                (array.shape(0) > 0 && (array.stride(1) != 1 || (array.shape(0) > 1 && array.stride(0) != 3))))
                throw nb::value_error("Point-cloud arrays must be C-contiguous [N, 3]");
            const auto dtype = array.dtype();
            if (dtype != nb::dtype<float>() && (!colors || dtype != nb::dtype<uint8_t>()))
                throw nb::value_error(colors ? "Colors must have dtype float32 or uint8" : "Positions must have dtype float32");
            return core::Tensor::from_blob(array.data(), {array.shape(0), size_t{3}}, core::Device::CPU,
                                           dtype == nb::dtype<float>() ? core::DataType::Float32 : core::DataType::UInt8);
        };
        vis::PointCloudUpdateInput input;
        input.points = source(owners->points, owners->points_array, false);
        input.colors = source(owners->colors, owners->colors_array, true);
        input.source_owners = owners;
        if (centroid)
            input.centroid = glm::vec3(std::get<0>(*centroid), std::get<1>(*centroid), std::get<2>(*centroid));
        auto target = update_target_;
        target.expected_revision = target.revision->load(std::memory_order_acquire);
        nb::gil_scoped_release gil;
        return manager->submit(std::move(target), std::move(input));
    }

    core::SceneNode& PyPointCloud::resolve_node() const {
        // A scene-backed handle follows node UUID identity, never a freed node pointer.
        if (scene_ && get_application_scene() == scene_) {
            if (auto* node = scene_->getNodeByUuid(node_uuid_);
                node && node->type == core::NodeType::POINTCLOUD && node->point_cloud) {
                return *node;
            }
        }
        throw std::runtime_error("PointCloud source node has been deleted or is no longer available");
    }

    std::shared_ptr<core::PointCloud> PyPointCloud::retain_data() const {
        return point_cloud_access(scene_, [&] {
            return owner_ ? owner_ : resolve_node().point_cloud;
        });
    }

    PyTensor PyPointCloud::means() const {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            return PyTensor(pc->means, false);
        });
    }

    PyTensor PyPointCloud::colors() const {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            return PyTensor(pc->colors, false);
        });
    }

    std::optional<PyTensor> PyPointCloud::normals() const {
        return point_cloud_access(scene_, [&]() -> std::optional<PyTensor> {
            const auto pc = retain_data();
            if (!pc->normals.is_valid())
                return std::nullopt;
            return PyTensor(pc->normals, false);
        });
    }

    std::optional<PyTensor> PyPointCloud::sh0() const {
        return point_cloud_access(scene_, [&]() -> std::optional<PyTensor> {
            const auto pc = retain_data();
            if (!pc->sh0.is_valid())
                return std::nullopt;
            return PyTensor(pc->sh0, false);
        });
    }

    std::optional<PyTensor> PyPointCloud::shN() const {
        return point_cloud_access(scene_, [&]() -> std::optional<PyTensor> {
            const auto pc = retain_data();
            if (!pc->shN.is_valid())
                return std::nullopt;
            return PyTensor(pc->shN, false);
        });
    }

    std::optional<PyTensor> PyPointCloud::opacity() const {
        return point_cloud_access(scene_, [&]() -> std::optional<PyTensor> {
            const auto pc = retain_data();
            if (!pc->opacity.is_valid())
                return std::nullopt;
            return PyTensor(pc->opacity, false);
        });
    }

    std::optional<PyTensor> PyPointCloud::scaling() const {
        return point_cloud_access(scene_, [&]() -> std::optional<PyTensor> {
            const auto pc = retain_data();
            if (!pc->scaling.is_valid())
                return std::nullopt;
            return PyTensor(pc->scaling, false);
        });
    }

    std::optional<PyTensor> PyPointCloud::rotation() const {
        return point_cloud_access(scene_, [&]() -> std::optional<PyTensor> {
            const auto pc = retain_data();
            if (!pc->rotation.is_valid())
                return std::nullopt;
            return PyTensor(pc->rotation, false);
        });
    }

    int64_t PyPointCloud::size() const {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            return pc->size();
        });
    }

    bool PyPointCloud::is_gaussian() const {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            return pc->is_gaussian();
        });
    }

    std::vector<std::string> PyPointCloud::attribute_names() const {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            return pc->attribute_names;
        });
    }

    void PyPointCloud::normalize_colors() {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            return pc->normalize_colors();
        });
    }

    int64_t PyPointCloud::filter(const PyTensor& keep_mask) {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            if (scene_)
                resolve_node().point_cloud_revision->fetch_add(1, std::memory_order_release);
            const auto& mask = keep_mask.tensor();
            assert(mask.dtype() == core::DataType::Bool && "Mask must be boolean");
            assert(mask.shape().rank() == 1 && "Mask must be 1D");
            assert(static_cast<size_t>(mask.shape()[0]) == static_cast<size_t>(pc->size()) && "Mask size must match point count");

            // Ensure mask is on same device as data
            const auto device = pc->means.device();
            const auto mask_dev = mask.device() == device ? mask : mask.to(device);

            const int64_t old_size = pc->size();
            pc->means = pc->means.is_valid() ? pc->means[mask_dev] : pc->means;
            pc->colors = pc->colors.is_valid() ? pc->colors[mask_dev] : pc->colors;
            pc->normals = pc->normals.is_valid() ? pc->normals[mask_dev] : pc->normals;
            pc->sh0 = pc->sh0.is_valid() ? pc->sh0[mask_dev] : pc->sh0;
            pc->shN = pc->shN.is_valid() ? pc->shN[mask_dev] : pc->shN;
            pc->opacity = pc->opacity.is_valid() ? pc->opacity[mask_dev] : pc->opacity;
            pc->scaling = pc->scaling.is_valid() ? pc->scaling[mask_dev] : pc->scaling;
            pc->rotation = pc->rotation.is_valid() ? pc->rotation[mask_dev] : pc->rotation;

            if (scene_) {
                scene_->setPointCloudModified(true);
                scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
            }
            return old_size - pc->size();
        });
    }

    int64_t PyPointCloud::filter_indices(const PyTensor& indices) {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            if (scene_)
                resolve_node().point_cloud_revision->fetch_add(1, std::memory_order_release);
            const auto& idx = indices.tensor();
            assert(idx.shape().rank() == 1 && "Indices must be 1D");

            const auto device = pc->means.device();
            const auto idx_dev = idx.device() == device ? idx : idx.to(device);

            const int64_t old_size = pc->size();
            pc->means = pc->means.is_valid() ? pc->means[idx_dev] : pc->means;
            pc->colors = pc->colors.is_valid() ? pc->colors[idx_dev] : pc->colors;
            pc->normals = pc->normals.is_valid() ? pc->normals[idx_dev] : pc->normals;
            pc->sh0 = pc->sh0.is_valid() ? pc->sh0[idx_dev] : pc->sh0;
            pc->shN = pc->shN.is_valid() ? pc->shN[idx_dev] : pc->shN;
            pc->opacity = pc->opacity.is_valid() ? pc->opacity[idx_dev] : pc->opacity;
            pc->scaling = pc->scaling.is_valid() ? pc->scaling[idx_dev] : pc->scaling;
            pc->rotation = pc->rotation.is_valid() ? pc->rotation[idx_dev] : pc->rotation;

            if (scene_) {
                scene_->setPointCloudModified(true);
                scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
            }
            return old_size - pc->size();
        });
    }

    void PyPointCloud::set_data(const PyTensor& points, const PyTensor& colors) {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            if (scene_)
                resolve_node().point_cloud_revision->fetch_add(1, std::memory_order_release);
            auto* const node = scene_ ? &resolve_node() : nullptr;
            const auto& pts = points.tensor();
            const auto& cols = colors.tensor();
            assert(pts.shape().rank() == 2 && pts.shape()[1] == 3);
            assert(cols.shape().rank() == 2 && cols.shape()[1] == 3);
            assert(pts.shape()[0] == cols.shape()[0]);

            pc->means = pts.to(core::Device::GPU);
            pc->colors = cols.to(core::Device::GPU);

            const int64_t n = pc->size();
            if (node) {
                node->gaussian_count.store(static_cast<size_t>(n), std::memory_order_release);
                if (n > 0) {
                    auto centroid = pc->means.mean(0).cpu();
                    auto acc = centroid.accessor<float, 1>();
                    node->centroid = glm::vec3(acc(0), acc(1), acc(2));
                }
            }
            if (scene_) {
                scene_->setPointCloudModified(true);
                scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
            }
        });
    }

    void PyPointCloud::set_colors(const PyTensor& colors) {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            if (scene_)
                resolve_node().point_cloud_revision->fetch_add(1, std::memory_order_release);
            const auto& cols = colors.tensor();
            assert(cols.shape().rank() == 2 && cols.shape()[1] == 3);
            assert(cols.shape()[0] == pc->size());
            pc->colors = cols.to(pc->means.device());
            if (scene_) {
                scene_->setPointCloudModified(true);
                scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
            }
        });
    }

    void PyPointCloud::set_means(const PyTensor& points) {
        return point_cloud_access(scene_, [&] {
            const auto pc = retain_data();
            if (scene_)
                resolve_node().point_cloud_revision->fetch_add(1, std::memory_order_release);
            auto* const node = scene_ ? &resolve_node() : nullptr;
            const auto& pts = points.tensor();
            assert(pts.shape().rank() == 2 && pts.shape()[1] == 3);
            assert(pts.shape()[0] == pc->size());
            pc->means = pts.to(pc->colors.device());
            if (node && pc->size() > 0) {
                auto centroid = pc->means.mean(0).cpu();
                auto acc = centroid.accessor<float, 1>();
                node->centroid = glm::vec3(acc(0), acc(1), acc(2));
            }
            if (scene_) {
                scene_->setPointCloudModified(true);
                scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
            }
        });
    }

    std::optional<PyCropBox> PySceneNode::cropbox() {
        auto& live = node();
        if (live.type != core::NodeType::CROPBOX || !live.cropbox) {
            return std::nullopt;
        }
        return PyCropBox(live.cropbox.get());
    }

    std::optional<PyEllipsoid> PySceneNode::ellipsoid() {
        auto& live = node();
        if (live.type != core::NodeType::ELLIPSOID || !live.ellipsoid) {
            return std::nullopt;
        }
        return PyEllipsoid(live.ellipsoid.get());
    }

    std::optional<PyCropPolygon> PySceneNode::crop_polygon() {
        auto& live = node();
        if (live.type != core::NodeType::CROP_POLYGON || !live.crop_polygon) {
            return std::nullopt;
        }
        return PyCropPolygon(live.crop_polygon.get());
    }

    std::optional<PyKeyframeData> PySceneNode::keyframe_data() {
        auto& live = node();
        if (live.type != core::NodeType::KEYFRAME || !live.keyframe) {
            return std::nullopt;
        }
        const auto& kf = *live.keyframe;
        return PyKeyframeData{
            .keyframe_index = kf.keyframe_index,
            .time = kf.time,
            .position = {kf.position.x, kf.position.y, kf.position.z},
            .rotation = {kf.rotation.w, kf.rotation.x, kf.rotation.y, kf.rotation.z},
            .focal_length_mm = kf.focal_length_mm,
            .easing = static_cast<int>(kf.easing)};
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

    uint64_t PyScene::render_generation() const {
        if (!is_valid()) {
            throw std::runtime_error("Scene reference is no longer valid");
        }
        return scene_->renderGeneration();
    }

    int32_t PyScene::add_group(const std::string& name, int32_t parent) {
        if (auto* const scene_manager = get_scene_manager()) {
            std::string parent_name;
            if (parent != core::NULL_NODE) {
                const auto* parent_node = scene_->getNodeById(parent);
                if (!parent_node) {
                    return core::NULL_NODE;
                }
                parent_name = parent_node->name;
            }

            const auto created_name = scene_manager->addGroupNode(name, parent_name);
            if (created_name.empty()) {
                return core::NULL_NODE;
            }
            return scene_->getNodeIdByName(created_name);
        }
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
                               const int32_t parent,
                               const bool transient) {
        const auto require_float32 = [](const char* field, const PyTensor& tensor) {
            const auto dtype = tensor.tensor().dtype();
            if (dtype != core::DataType::Float32) {
                throw std::runtime_error(std::string("add_splat: ") + field +
                                         " must be float32, got " + core::dtype_name(dtype));
            }
        };
        require_float32("means", means);
        require_float32("sh0", sh0);
        require_float32("shN", shN);
        require_float32("scaling", scaling);
        require_float32("rotation", rotation);
        require_float32("opacity", opacity);

        // A transient node is not part of the document: it is never saved, and
        // adding it leaves nothing to undo.
        std::optional<vis::op::SceneGraphStateSnapshot> history_before;
        if (auto* const scene_manager = get_scene_manager(); scene_manager && !transient) {
            history_before = vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {name});
        }

        auto splat = std::make_unique<core::SplatData>(
            sh_degree,
            means.tensor().clone(),
            sh0.tensor().clone(),
            shN.tensor().clone(),
            scaling.tensor().clone(),
            rotation.tensor().clone(),
            opacity.tensor().clone(),
            scene_scale);

        if (auto allocator = vis::makeViewerSplatTensorAllocator()) {
            if (auto migrated = io::migrateSplatTensorsToAllocator(*splat, allocator); !migrated) {
                throw std::runtime_error("Failed to prepare splat tensors for rendering: " +
                                         migrated.error().format());
            }
            scene_->setCombinedModelAllocator(std::move(allocator));
        }

        const size_t gaussian_count = splat->size();
        const int32_t node_id = scene_->addSplat(name, std::move(splat), parent);
        if (node_id == core::NULL_NODE) {
            return core::NULL_NODE;
        }
        auto* const added = scene_->getNodeById(node_id);
        assert(added);
        const std::string added_name = added ? added->name : name;
        if (added) {
            added->transient = transient;
        }

        lfs::core::events::state::PLYAdded{
            .name = added_name,
            .uuid = added->uuid,
            .node_gaussians = gaussian_count,
            .total_gaussians = scene_->getTotalGaussianCount(),
            .is_visible = true,
            .parent_name = "",
            .is_group = false,
            .node_type = 0}
            .emit();

        if (auto* const scene_manager = get_scene_manager()) {
            ensure_scene_manager_content_type(*scene_manager, vis::SceneManager::ContentType::SplatFiles);
            if (history_before) {
                vis::op::undoHistory().push(std::make_unique<vis::op::SceneGraphPatchEntry>(
                    *scene_manager,
                    "Add Splat",
                    std::move(*history_before),
                    vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {added_name})));
            }
        }

        return node_id;
    }

    int32_t PyScene::add_point_cloud(const std::string& name,
                                     const PyTensor& points,
                                     const PyTensor& colors,
                                     const int32_t parent) {
        std::optional<vis::op::SceneGraphStateSnapshot> history_before;
        if (auto* const scene_manager = get_scene_manager()) {
            history_before = vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {name});
        }

        const auto& pts = points.tensor();
        const auto& cols = colors.tensor();
        assert(pts.shape().rank() == 2 && pts.shape()[1] == 3);
        assert(cols.shape().rank() == 2 && cols.shape()[1] == 3);
        assert(pts.shape()[0] == cols.shape()[0]);
        if (cols.dtype() != core::DataType::UInt8 && cols.dtype() != core::DataType::Float32)
            throw nb::value_error("colors must have dtype uint8 or float32");

        auto pc = std::make_shared<core::PointCloud>(pts.to(core::Device::GPU), cols.to(core::Device::GPU));
        const int32_t node_id = scene_->addPointCloud(name, std::move(pc), parent);
        if (node_id == core::NULL_NODE) {
            return core::NULL_NODE;
        }
        const auto* const added = scene_->getNodeById(node_id);
        const std::string added_name = added ? added->name : name;

        if (auto* const scene_manager = get_scene_manager()) {
            ensure_scene_manager_content_type(*scene_manager, vis::SceneManager::ContentType::SplatFiles);
            vis::op::undoHistory().push(std::make_unique<vis::op::SceneGraphPatchEntry>(
                *scene_manager,
                "Add Point Cloud",
                std::move(*history_before),
                vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {added_name})));
        }

        return node_id;
    }

    int32_t PyScene::add_mesh(const std::string& name,
                              const PyTensor& vertices,
                              const PyTensor& indices,
                              std::optional<PyTensor> colors,
                              std::optional<PyTensor> normals,
                              const int32_t parent) {
        std::optional<vis::op::SceneGraphStateSnapshot> history_before;
        if (auto* const scene_manager = get_scene_manager()) {
            history_before = vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {name});
        }

        const auto& verts = vertices.tensor();
        const auto& idx = indices.tensor();
        assert(verts.shape().rank() == 2 && verts.shape()[1] == 3);
        assert(idx.shape().rank() == 2 && idx.shape()[1] == 3);

        auto mesh = std::make_shared<core::MeshData>(
            verts.to(core::DataType::Float32).to(core::Device::CPU),
            idx.to(core::DataType::Int32).to(core::Device::CPU));

        if (colors && colors->tensor().is_valid()) {
            const auto& c = colors->tensor();
            if (c.ndim() != 2 || c.size(0) != verts.size(0) || c.size(1) != 4)
                throw nb::value_error("colors must have shape [N, 4] matching vertices");
            mesh->colors = c.to(core::DataType::Float32).to(core::Device::CPU);
        }

        if (normals && normals->tensor().is_valid()) {
            const auto& n = normals->tensor();
            assert(n.shape().rank() == 2 && n.shape()[1] == 3 && n.shape()[0] == verts.shape()[0]);
            mesh->normals = n.to(core::DataType::Float32).to(core::Device::CPU);
        } else {
            mesh->compute_normals();
        }

        const int32_t node_id = scene_->addMesh(name, std::move(mesh), parent);
        if (node_id == core::NULL_NODE) {
            return core::NULL_NODE;
        }

        if (auto* const scene_manager = get_scene_manager()) {
            ensure_scene_manager_content_type(*scene_manager, vis::SceneManager::ContentType::SplatFiles);
            const auto* const added = scene_->getNodeById(node_id);
            const std::string added_name = added ? added->name : name;
            vis::op::undoHistory().push(std::make_unique<vis::op::SceneGraphPatchEntry>(
                *scene_manager,
                "Add Mesh",
                std::move(*history_before),
                vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {added_name})));
        }

        return node_id;
    }

    int32_t PyScene::add_camera_group(const std::string& name, const int32_t parent, const size_t camera_count) {
        std::optional<vis::op::SceneGraphStateSnapshot> history_before;
        if (auto* const scene_manager = get_scene_manager()) {
            history_before = vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {name});
        }

        const int32_t node_id = scene_->addCameraGroup(name, parent, camera_count);
        if (node_id == core::NULL_NODE) {
            return core::NULL_NODE;
        }
        const auto* const added = scene_->getNodeById(node_id);
        const std::string added_name = added ? added->name : name;

        if (auto* const scene_manager = get_scene_manager()) {
            ensure_scene_manager_content_type(*scene_manager, vis::SceneManager::ContentType::Dataset);
            vis::op::undoHistory().push(std::make_unique<vis::op::SceneGraphPatchEntry>(
                *scene_manager,
                "Add Camera Group",
                std::move(*history_before),
                vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {added_name})));
        }
        return node_id;
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
                                int uid,
                                std::optional<PyTensor> mask) {
        std::optional<vis::op::SceneGraphStateSnapshot> history_before;
        if (auto* const scene_manager = get_scene_manager()) {
            history_before = vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {name});
        }

        const auto& R_tensor = R.tensor();
        const auto& T_tensor = T.tensor();
        if (!R_tensor.is_valid() || R_tensor.dtype() != core::DataType::Float32 || R_tensor.ndim() != 2 ||
            R_tensor.size(0) != 3 || R_tensor.size(1) != 3)
            throw nb::value_error("R must be a float32 tensor of shape [3, 3]");
        if (!T_tensor.is_valid() || T_tensor.dtype() != core::DataType::Float32 || T_tensor.numel() != 3 ||
            !(T_tensor.ndim() == 1 || (T_tensor.ndim() == 2 && T_tensor.size(1) == 1)))
            throw nb::value_error("T must be a float32 tensor of shape [3] or [3, 1]");
        const auto all_finite = [](const core::Tensor& tensor) {
            const auto cpu = tensor.cpu().contiguous();
            const auto* values = cpu.ptr<float>();
            return std::all_of(values, values + cpu.numel(), [](const float value) { return std::isfinite(value); });
        };
        if (!all_finite(R_tensor))
            throw nb::value_error("R must contain finite values");
        if (!all_finite(T_tensor))
            throw nb::value_error("T must contain finite values");
        if (!std::isfinite(focal_x) || focal_x <= 0.0f || !std::isfinite(focal_y) || focal_y <= 0.0f)
            throw nb::value_error("focal_x and focal_y must be finite and positive");
        if (width <= 0 || height <= 0)
            throw nb::value_error("width and height must be positive");

        auto T_flat = T_tensor.ndim() == 2 ? T_tensor.reshape({3}) : T_tensor;

        // A camera without a uid is skipped by uid-keyed features such as the frustum overlay: give it the next free one.
        if (uid < 0) {
            uid = 0;
            for (const auto& existing : *scene_->getAllCamerasCached()) {
                if (!existing)
                    continue;
                if (existing->uid() == std::numeric_limits<int>::max())
                    throw nb::value_error("Cannot auto-assign a camera uid above INT_MAX");
                uid = std::max(uid, existing->uid() + 1);
            }
        }

        auto camera = std::make_shared<lfs::core::Camera>(
            R_tensor.clone(),
            T_flat.clone(),
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

        // Attach an in-memory mask if the caller passed one (direct-scene
        // plugins use this to bypass the on-disk masks/ workflow).
        if (mask.has_value() && mask->tensor().is_valid()) {
            camera->set_mask_tensor(mask->tensor().clone());
        }

        const int32_t node_id = scene_->addCamera(name, parent, std::move(camera));
        if (node_id == core::NULL_NODE) {
            return core::NULL_NODE;
        }

        if (auto* const scene_manager = get_scene_manager()) {
            const auto* added = scene_->getNodeById(node_id);
            const std::string added_name = added ? added->name : name;
            ensure_scene_manager_content_type(*scene_manager, vis::SceneManager::ContentType::Dataset);
            vis::op::undoHistory().push(std::make_unique<vis::op::SceneGraphPatchEntry>(
                *scene_manager,
                "Add Camera",
                std::move(*history_before),
                vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {added_name})));
        }
        return node_id;
    }

    void PyScene::remove_node(const std::string& name, bool keep_children) {
        point_cloud_access(scene_, [&] {
            if (auto* const scene_manager = get_scene_manager()) {
                if (auto result = scene_manager->removePLYWithResult(name, keep_children); !result) {
                    throw std::runtime_error(result.error());
                }
                return;
            }
            scene_->removeNode(name, keep_children);
        });
    }

    bool PyScene::rename_node(const std::string& old_name, const std::string& new_name) {
        if (auto* const scene_manager = get_scene_manager()) {
            return scene_manager->renamePLY(old_name, new_name);
        }
        return scene_->renameNode(old_name, new_name);
    }

    void PyScene::clear() {
        if (get_visualizer()) {
            nb::gil_scoped_release release;
            unwrap(clear_application_scene());
            return;
        }
        if (auto* const scene_manager = get_scene_manager()) {
            if (scene_manager->clear()) {
                return;
            }

            if (auto* const trainer_manager = lfs::python::get_trainer_manager();
                trainer_manager &&
                scene_manager->getContentType() == lfs::vis::SceneManager::ContentType::Dataset &&
                !trainer_manager->canPerform(lfs::vis::TrainingAction::ClearScene)) {
                throw std::runtime_error(
                    std::string(trainer_manager->getActionBlockedReason(lfs::vis::TrainingAction::ClearScene)));
            }

            throw std::runtime_error("Scene clear request was rejected");
            return;
        }
        scene_->clear();
    }

    bool PyScene::reparent(int32_t node_id, int32_t new_parent_id) {
        if (auto* const scene_manager = get_scene_manager()) {
            return scene_manager->reparentNode(node_id, new_parent_id);
        }
        return scene_->reparent(node_id, new_parent_id);
    }

    std::optional<PySceneNode> PyScene::get_node_by_id(int32_t id) {
        auto* node = scene_->getNodeById(id);
        if (!node)
            return std::nullopt;
        return PySceneNode(node, scene_);
    }

    std::optional<PySceneNode> PyScene::get_node_by_uuid(const std::string& uuid) {
        const auto parsed = core::Uuid::from_string(uuid);
        if (!parsed)
            return std::nullopt;
        auto* node = scene_->getNodeByUuid(*parsed);
        if (!node)
            return std::nullopt;
        return PySceneNode(node, scene_);
    }

    std::optional<PySceneNode> PyScene::get_node(const std::string& name) {
        // Use the const lookup + const_cast so we don't blow the combined-model
        // cache on every read. Actual mutations go through SceneNode property
        // setters whose callbacks trigger Scene::notifyMutation, which still
        // invalidates the cache — same pattern as get_nodes/get_visible_nodes.
        const auto* node = scene_->getNode(name);
        if (!node)
            return std::nullopt;
        return PySceneNode(const_cast<core::SceneNode*>(node), scene_);
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

    std::vector<PySceneSplatSnapshot> PyScene::snapshot_visible_splats() {
        std::vector<PySceneSplatSnapshot> result;
        for (auto& snapshot : scene_->snapshotVisibleSplats()) {
            result.emplace_back(std::move(snapshot));
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

    namespace {
        bool camera_needs_distortion_handling(const core::Camera& cam) {
            return cam.radial_distortion().numel() != 0 ||
                   cam.tangential_distortion().numel() != 0 ||
                   cam.camera_model_type() != core::CameraModelType::PINHOLE;
        }
    } // namespace

    bool PyScene::has_active_distorted_cameras() const {
        if (!scene_) {
            return false;
        }
        for (const auto& cam : scene_->getActiveCameras()) {
            if (cam && camera_needs_distortion_handling(*cam)) {
                return true;
            }
        }
        return false;
    }

    bool PyScene::can_undistort_active_cameras() const {
        if (!scene_) {
            return false;
        }
        bool has_any = false;
        for (const auto& cam : scene_->getActiveCameras()) {
            if (!cam) {
                continue;
            }
            if (camera_needs_distortion_handling(*cam)) {
                has_any = true;
                if (!cam->has_distortion()) {
                    return false;
                }
            }
        }
        return has_any;
    }

    void PyScene::set_camera_training_enabled(const std::string& name, bool enabled) {
        if (auto* const scene_manager = get_scene_manager()) {
            const auto* node = scene_->getNode(name);
            if (!node || node->training_enabled == enabled) {
                return;
            }
            const auto before = vis::op::SceneGraphMetadataEntry::captureNodes(*scene_manager, {name});
            scene_->setCameraTrainingEnabled(name, enabled);
            push_scene_graph_metadata_history(
                *scene_manager,
                "Set Camera Training",
                before,
                vis::op::SceneGraphMetadataEntry::captureNodes(*scene_manager, {name}));
            return;
        }
        scene_->setCameraTrainingEnabled(name, enabled);
    }

    nb::tuple PyScene::get_world_transform(int32_t node_id) const {
        return mat4_to_tuple(scene_->getWorldTransform(node_id));
    }

    void PyScene::set_node_transform(
        const std::string& name,
        nb::ndarray<float, nb::device::cpu, nb::shape<4, 4>> transform) {
        apply_node_transform_with_undo(name, ndarray_to_mat4(transform), scene_);
    }

    void PyScene::set_node_transform_tensor(const std::string& name, const PyTensor& transform) {
        const auto& t = transform.tensor();
        assert(t.ndim() == 2 && t.size(0) == 4 && t.size(1) == 4);
        auto cpu_t = t.device() == core::Device::GPU ? t.cpu() : t;
        auto contiguous = cpu_t.contiguous();
        const float* data = contiguous.ptr<float>();
        glm::mat4 m;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[j][i] = data[i * 4 + j];
            }
        }
        apply_node_transform_with_undo(name, m, scene_);
    }

    std::optional<PySplatData> PyScene::combined_model() {
        auto* model = const_cast<core::SplatData*>(scene_->getCombinedModel());
        if (!model)
            return std::nullopt;
        return PySplatData(model);
    }

    void PyScene::apply_crop_filter(PyTensor& mask) {
        if (mask.tensor().numel() == 0)
            return;
        const auto boxes = scene_->getRenderableCropBoxes();
        const auto ellipsoids = scene_->getRenderableEllipsoids();
        const auto polygons = scene_->getRenderableCropPolygons();
        const auto enabled = [](const auto& volume) {
            return volume.data && volume.data->enabled && volume.parent_node_index >= 0;
        };
        if (!std::any_of(boxes.begin(), boxes.end(), enabled) &&
            !std::any_of(ellipsoids.begin(), ellipsoids.end(), enabled) &&
            !std::any_of(polygons.begin(), polygons.end(), enabled)) {
            return;
        }

        const auto* model = scene_->getCombinedModel();
        auto& selection = mask.tensor();
        if (!model || selection.ndim() != 1 || selection.size(0) != model->size() ||
            selection.dtype() != core::DataType::Bool || !selection.is_contiguous() ||
            selection.device() != core::Device::GPU) {
            throw std::invalid_argument("Crop filtering requires a contiguous GPU bool mask matching the combined model");
        }
        const auto backend = core::gpu_backend_of(selection);
        if (!backend || core::gpu_backend_of(model->means()) != backend)
            throw std::invalid_argument("Crop mask must use the combined model backend");
        const core::GpuBackendScope backend_scope(*backend);
        const auto& means = model->means();
        const auto transforms = scene_->getVisibleNodeTransforms();
        std::vector<float> transform_values;
        transform_values.reserve(transforms.size() * 16);
        for (const auto& transform : transforms) {
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    transform_values.push_back(transform[col][row]);
                }
            }
        }
        const auto model_transforms = core::Tensor::from_vector(
            transform_values, {transforms.size(), 4, 4}, core::Device::GPU);
        // The single-model kernel already defaults to slot zero. Avoid creating
        // a full-size index tensor for scenes that do not otherwise need one.
        const auto indices = transforms.size() > 1 ? scene_->getTransformIndices() : nullptr;
        const auto upload_transform = [](const glm::mat4& world) {
            const auto inverse = glm::inverse(world);
            const auto* ptr = glm::value_ptr(inverse);
            return core::Tensor::from_vector(std::vector<float>(ptr, ptr + 16), {4, 4}, core::Device::GPU);
        };
        const auto upload_vec = [](const glm::vec3& value) {
            return core::Tensor::from_vector({value.x, value.y, value.z}, {3}, core::Device::GPU);
        };
        for (const auto& box : boxes) {
            if (!enabled(box))
                continue;
            const auto transform = upload_transform(box.world_transform);
            const auto min = upload_vec(box.data->min);
            const auto max = upload_vec(box.data->max);
            rendering::filter_selection_by_crop(selection, means, &transform, &min, &max, box.data->inverse,
                                                nullptr, nullptr, false, &model_transforms, indices.get(),
                                                box.parent_node_index);
        }
        for (const auto& ellipsoid : ellipsoids) {
            if (!enabled(ellipsoid))
                continue;
            const auto transform = upload_transform(ellipsoid.world_transform);
            // Match the renderer's treatment of degenerate ellipsoid axes.
            const auto radii = upload_vec(glm::max(glm::abs(ellipsoid.data->radii), glm::vec3(1e-8f)));
            rendering::filter_selection_by_crop(selection, means, nullptr, nullptr, nullptr, false,
                                                &transform, &radii, ellipsoid.data->inverse, &model_transforms,
                                                indices.get(), ellipsoid.parent_node_index);
        }
        for (const auto& polygon : polygons) {
            if (!enabled(polygon))
                continue;
            const auto transform = upload_transform(polygon.world_transform);
            std::vector<float> outline;
            outline.reserve(polygon.data->vertices.size() * 2);
            for (const glm::vec2& vertex : polygon.data->vertices)
                outline.insert(outline.end(), {vertex.x, vertex.y});
            const auto vertices =
                core::Tensor::from_vector(outline, {polygon.data->vertices.size(), size_t{2}}, core::Device::GPU);
            const rendering::SelectionCropPolygon filter{
                .transform = &transform,
                .vertices = &vertices,
                .half_depth = polygon.data->half_depth,
                .inverse = polygon.data->inverse,
            };
            rendering::filter_selection_by_crop(selection, means, nullptr, nullptr, nullptr, false,
                                                nullptr, nullptr, false, &model_transforms, indices.get(),
                                                polygon.parent_node_index, &filter);
        }
    }

    std::optional<PySplatData> PyScene::training_model() {
        auto* model = scene_->getTrainingModel();
        if (!model)
            return std::nullopt;
        return PySplatData(model);
    }

    void PyScene::set_training_model_node(const std::string& name) {
        // Binding a transient preview leaves nothing to undo, like adding it.
        // A preview is for display only: it must not take the place of a
        // saved training model, which training continues from and saves.
        if (const auto* const target = scene_->getNode(name); target && target->transient) {
            if (const auto* const bound = scene_->getNodeByUuid(scene_->getTrainingModelNodeUuid());
                bound && !bound->transient) {
                throw std::runtime_error("set_training_model_node: transient node '" + name +
                                         "' cannot replace the training model '" + bound->name + "'");
            }
            scene_->setTrainingModelNode(name);
            return;
        }
        if (auto* const scene_manager = get_scene_manager()) {
            const auto options = scene_graph_history_options(false, true);
            auto before = vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {}, options);
            scene_->setTrainingModelNode(name);
            vis::op::undoHistory().push(std::make_unique<vis::op::SceneGraphPatchEntry>(
                *scene_manager,
                "Set Training Model",
                std::move(before),
                vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {}, options)));
            return;
        }
        scene_->setTrainingModelNode(name);
    }

    bool PyScene::set_node_transient(const std::string& name, const bool transient) {
        auto* const node = scene_->getMutableNode(name);
        if (!node) {
            return false;
        }
        if (node->transient == transient) {
            return true;
        }
        // A bound transient model is only a preview and a bound saved one is
        // what training continues from, so neither may cross over while bound.
        const auto* const bound = scene_->getNodeByUuid(scene_->getTrainingModelNodeUuid());
        for (const auto* holder = bound; holder; holder = scene_->getNodeById(holder->parent_id)) {
            if (holder == node && (holder == bound || (transient && !bound->transient))) {
                throw std::runtime_error("set_node_transient: '" + name + "' holds the training model '" +
                                         bound->name + "'");
            }
        }
        node->transient = transient;
        // The node leaves or joins the document, so the project has changes to save.
        scene_->notifyMutation(transient ? core::Scene::MutationType::NODE_REMOVED
                                         : core::Scene::MutationType::NODE_ADDED);
        return true;
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

    int32_t PyScene::get_or_create_cropbox_for_splat(const int32_t splat_id) {
        if (auto* const scene_manager = get_scene_manager()) {
            const core::NodeId existing_cropbox = scene_->getCropBoxForSplat(splat_id);
            if (auto result = vis::cap::ensureCropBox(*scene_manager, get_rendering_manager(), splat_id);
                result) {
                return *result;
            }

            auto before = vis::op::SceneGraphPatchEntry::captureState(*scene_manager, {});
            const core::NodeId cropbox_id = scene_->getOrCreateCropBoxForSplat(splat_id);
            if (cropbox_id != core::NULL_NODE && existing_cropbox == core::NULL_NODE) {
                const auto* cropbox_node = scene_->getNodeById(cropbox_id);
                const auto root_names = cropbox_node ? std::vector<std::string>{cropbox_node->name}
                                                     : std::vector<std::string>{};
                vis::op::undoHistory().push(std::make_unique<vis::op::SceneGraphPatchEntry>(
                    *scene_manager,
                    "Add Crop Box",
                    std::move(before),
                    vis::op::SceneGraphPatchEntry::captureState(*scene_manager, root_names)));
            }
            return cropbox_id;
        }
        return scene_->getOrCreateCropBoxForSplat(splat_id);
    }

    void PyScene::set_cropbox_data(const int32_t cropbox_id, const PyCropBox& data) {
        if (auto* const scene_manager = get_scene_manager()) {
            vis::cap::CropBoxUpdate update;
            update.min_bounds = data.data()->min;
            update.max_bounds = data.data()->max;
            update.has_inverse = true;
            update.inverse = data.data()->inverse;
            update.has_enabled = true;
            update.enabled = data.data()->enabled;
            if (auto result = vis::cap::updateCropBox(*scene_manager,
                                                      scene_manager->getRenderingManager(),
                                                      cropbox_id,
                                                      update);
                !result) {
                LOG_WARN("PyScene::set_cropbox_data fell back to direct update for {}: {}",
                         cropbox_id, result.error());
            } else {
                return;
            }
        }

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
        uint8_t created_id = 0;
        if (auto* const scene_manager = get_scene_manager()) {
            apply_selection_state_with_undo(
                *scene_manager, "selection_group.add",
                [&](core::Scene& scene) {
                    created_id = scene.addSelectionGroup(
                        name, {std::get<0>(color), std::get<1>(color), std::get<2>(color)});
                });
            return created_id;
        }
        return scene_->addSelectionGroup(name, {std::get<0>(color), std::get<1>(color), std::get<2>(color)});
    }

    void PyScene::remove_selection_group(const uint8_t id) {
        if (auto* const scene_manager = get_scene_manager()) {
            apply_selection_state_with_undo(*scene_manager, "selection_group.remove",
                                            [id](core::Scene& scene) { scene.removeSelectionGroup(id); });
            return;
        }
        scene_->removeSelectionGroup(id);
    }

    void PyScene::rename_selection_group(const uint8_t id, const std::string& name) {
        if (auto* const scene_manager = get_scene_manager()) {
            apply_selection_state_with_undo(*scene_manager, "selection_group.rename",
                                            [id, &name](core::Scene& scene) { scene.renameSelectionGroup(id, name); });
            return;
        }
        scene_->renameSelectionGroup(id, name);
    }

    void PyScene::set_selection(const std::vector<size_t>& indices) {
        if (auto* const scene_manager = get_scene_manager()) {
            std::vector<uint8_t> mask(scene_->getTotalGaussianCount(), 0);
            for (const auto index : indices) {
                if (index < mask.size()) {
                    mask[index] = 1;
                }
            }
            (void)scene_manager->applySelectionMask(mask);
            return;
        }
        scene_->setSelection(indices);
    }

    void PyScene::set_selection_mask(const PyTensor& mask) {
        if (auto* const scene_manager = get_scene_manager()) {
            (void)scene_manager->applySelectionMask(mask.tensor());
            return;
        }
        scene_->setSelectionMask(std::make_shared<core::Tensor>(mask.tensor()));
    }

    void PyScene::preview_selection_mask(const PyTensor& mask) {
        if (auto* const scene_manager = get_scene_manager()) {
            (void)scene_manager->previewSelectionMask(mask.tensor());
            return;
        }
        scene_->setSelectionMask(std::make_shared<core::Tensor>(mask.tensor()));
    }

    void PyScene::commit_selection_preview() {
        if (auto* const scene_manager = get_scene_manager()) {
            scene_manager->commitSelectionPreview();
        }
    }

    void PyScene::cancel_selection_preview() {
        if (auto* const scene_manager = get_scene_manager()) {
            scene_manager->cancelSelectionPreview();
        }
    }

    void PyScene::clear_selection() {
        if (auto* const scene_manager = get_scene_manager()) {
            scene_manager->deselectAllGaussians();
            return;
        }
        scene_->clearSelection();
    }

    void PyScene::set_selection_group_color(uint8_t id, std::tuple<float, float, float> color) {
        if (auto* const scene_manager = get_scene_manager()) {
            apply_selection_state_with_undo(
                *scene_manager, "selection_group.set_color",
                [id, &color](core::Scene& scene) {
                    scene.setSelectionGroupColor(id, {std::get<0>(color), std::get<1>(color), std::get<2>(color)});
                });
            return;
        }
        scene_->setSelectionGroupColor(id, {std::get<0>(color), std::get<1>(color), std::get<2>(color)});
    }

    void PyScene::set_selection_group_locked(const uint8_t id, const bool locked) {
        if (auto* const scene_manager = get_scene_manager()) {
            apply_selection_state_with_undo(
                *scene_manager, "selection_group.set_locked",
                [id, locked](core::Scene& scene) { scene.setSelectionGroupLocked(id, locked); });
            return;
        }
        scene_->setSelectionGroupLocked(id, locked);
    }

    void PyScene::set_active_selection_group(const uint8_t id) {
        if (auto* const scene_manager = get_scene_manager()) {
            apply_selection_state_with_undo(
                *scene_manager, "selection_group.set_active",
                [id](core::Scene& scene) { scene.setActiveSelectionGroup(id); });
            return;
        }
        scene_->setActiveSelectionGroup(id);
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

    void PyScene::clear_selection_group(const uint8_t id) {
        if (auto* const scene_manager = get_scene_manager()) {
            apply_selection_state_with_undo(*scene_manager, "selection_group.clear",
                                            [id](core::Scene& scene) { scene.clearSelectionGroup(id); });
            return;
        }
        scene_->clearSelectionGroup(id);
    }

    void PyScene::reset_selection_state() {
        if (auto* const scene_manager = get_scene_manager()) {
            apply_selection_state_with_undo(*scene_manager, "selection.reset_state",
                                            [](core::Scene& scene) { scene.resetSelectionState(); });
            return;
        }
        scene_->resetSelectionState();
    }

    PyTensor PyScene::scene_center() const {
        return PyTensor(scene_->getSceneCenter(), false);
    }

    std::string PyScene::duplicate_node(const std::string& name) {
        if (auto* const scene_manager = get_scene_manager()) {
            return scene_manager->duplicateNodeTree(name);
        }
        return scene_->duplicateNode(name);
    }

    std::string PyScene::merge_group(const std::string& group_name) {
        if (auto* const scene_manager = get_scene_manager()) {
            return scene_manager->mergeGroupNode(group_name);
        }
        return scene_->mergeGroup(group_name);
    }

    void register_scene(nb::module_& m) {
        const auto collision_tasks = []() -> vis::gui::AsyncTaskManager& {
            auto* viewer = dynamic_cast<vis::VisualizerImpl*>(get_visualizer());
            if (!viewer || !viewer->getGuiManager())
                throw std::runtime_error("Collision authoring requires the GUI");
            return viewer->getGuiManager()->asyncTasks();
        };
        m.def("generate_collision", [collision_tasks](float size, float opacity, bool automatic, bool cleanup, float max_area) {
            auto* manager = get_scene_manager();
            point_cloud_access(manager ? &manager->getScene() : nullptr, [&] {
                collision_tasks().startCollision({.voxel_size = size, .opacity = opacity, .max_gaussian_area = max_area,
                                                  .auto_resolution = automatic, .cleanup = cleanup});
            }); }, nb::arg("voxel_size") = .05f, nb::arg("opacity") = .1f, nb::arg("auto_resolution") = false, nb::arg("cleanup") = true, nb::arg("max_gaussian_area") = 0.f);
        m.def("export_collision", [collision_tasks](const std::string& uuid, const std::string& path) {
            const auto parsed = core::Uuid::from_string(uuid);
            if (!parsed)
                throw std::invalid_argument("Invalid collision target UUID");
            auto* manager = get_scene_manager();
            point_cloud_access(manager ? &manager->getScene() : nullptr, [&] {
                collision_tasks().startCollisionExport(*parsed, core::utf8_to_path(path));
            }); }, nb::arg("uuid"), nb::arg("path"));
        m.def("cancel_collision", [collision_tasks] {
            auto* manager = get_scene_manager();
            point_cloud_access(manager ? &manager->getScene() : nullptr, [&] { collision_tasks().cancelCollision(); });
        });
        m.def("get_collision_state", [collision_tasks] {
            struct State {
                std::optional<vis::JobSnapshot> job;
                std::string uuid, name, source, walk_error;
                float resolved_size = 0;
                core::CollisionSettings settings;
                core::WalkSettings walk_settings;
                bool colliders = false, walking = false;
            };
            auto* manager = get_scene_manager();
            const auto state = point_cloud_access(manager ? &manager->getScene() : nullptr, [&] {
                State state;
                state.job = collision_tasks().collisionJob();
                state.resolved_size = collision_tasks().collisionResolvedSize();
                state.settings = collision_tasks().collisionGenerationSettings();
                if (manager) {
                    const auto& scene = manager->getScene();
                    for (const auto* node : scene.getNodes())
                        state.colliders |= node && bool(node->collision);
                    const auto* target = scene.getNode(manager->getSelectedNodeName());
                    if (target && target->type == core::NodeType::SPLAT && manager->getSelectedNodeIds().size() == 1 && (target->model || scene.isConsolidated()))
                        state.source = target->name;
                    if (target && target->type == core::NodeType::COLLISION && target->collision) {
                        state.uuid = target->uuid.to_string();
                        state.name = target->name;
                    }
                }
                if (const auto* input = vis::InputController::instance()) {
                    state.walking = input->cameraNavigationMode() == vis::InputController::CameraNavigationMode::Walk;
                    state.walk_error = input->walk_error;
                    state.walk_settings = input->walk_settings;
                }
                return state;
            });
            nb::dict result;
            result["active"] = state.job && state.job->running();
            result["progress"] = state.job ? state.job->progress : 0.f;
            result["stage"] = state.job ? state.job->stage : "";
            result["error"] = state.job ? state.job->error : "";
            result["walk_error"] = state.walk_error;
            result["has_collision"] = state.colliders;
            result["walking"] = state.walking;
            result["walk_settings"] = nb::make_tuple(state.walk_settings.height, state.walk_settings.radius,
                                                     state.walk_settings.step, state.walk_settings.speed);
            result["target_uuid"] = state.uuid;
            result["target_name"] = state.name;
            result["source_name"] = state.source;
            result["resolved_voxel_size"] = state.resolved_size;
            result["generation_settings"] = nb::make_tuple(state.settings.voxel_size, state.settings.opacity,
                                                           state.settings.auto_resolution, state.settings.cleanup, state.settings.max_gaussian_area);
            return result;
        });
        m.def("set_walk_settings", [](float height, float radius, float step, float speed) {
            const core::WalkSettings settings{height, radius, step, speed};
            settings.validate();
            auto* manager = get_scene_manager();
            point_cloud_access(manager ? &manager->getScene() : nullptr, [&] {
                auto* input = vis::InputController::instance();
                if (!input)
                    throw std::runtime_error("No input controller");
                input->setWalkSettings(settings);
            }); }, nb::arg("height"), nb::arg("radius"), nb::arg("step"), nb::arg("speed"));
        nb::class_<PySceneSplatSnapshot>(m, "SceneSplatSnapshot")
            .def_prop_ro("transform", [](const PySceneSplatSnapshot& value) {
                return mat4_to_tuple(value.snapshot().world_transform);
            })
            .def_prop_ro("sh_degree", [](const PySceneSplatSnapshot& value) {
                return value.snapshot().active_sh_degree;
            })
            .def("splat_data", [](const PySceneSplatSnapshot& value) {
                std::shared_ptr<core::SplatData> data;
                {
                    nb::gil_scoped_release release;
                    data = value.snapshot().materialize();
                }
                return PySplatData(std::move(data)); }, "Materialize this owned node's local geometry. May run on an export worker.");
        register_scene_node_properties();
        register_cropbox_properties();
        register_ellipsoid_properties();
        register_crop_polygon_properties();

        // NodeType enum
        nb::enum_<core::NodeType>(m, "NodeType")
            .value("SPLAT", core::NodeType::SPLAT)
            .value("POINTCLOUD", core::NodeType::POINTCLOUD)
            .value("GROUP", core::NodeType::GROUP)
            .value("PLY_SEQUENCE", core::NodeType::PLY_SEQUENCE)
            .value("CROPBOX", core::NodeType::CROPBOX)
            .value("ELLIPSOID", core::NodeType::ELLIPSOID)
            .value("DATASET", core::NodeType::DATASET)
            .value("CAMERA_GROUP", core::NodeType::CAMERA_GROUP)
            .value("CAMERA", core::NodeType::CAMERA)
            .value("IMAGE_GROUP", core::NodeType::IMAGE_GROUP)
            .value("IMAGE", core::NodeType::IMAGE)
            .value("MESH", core::NodeType::MESH)
            .value("COLLISION", core::NodeType::COLLISION)
            .value("KEYFRAME_GROUP", core::NodeType::KEYFRAME_GROUP)
            .value("KEYFRAME", core::NodeType::KEYFRAME)
            .value("CROP_POLYGON", core::NodeType::CROP_POLYGON);

        nb::class_<PyMeshInfo>(m, "MeshInfo")
            .def_prop_ro("vertex_count", &PyMeshInfo::vertex_count)
            .def_prop_ro("face_count", &PyMeshInfo::face_count)
            .def_prop_ro("has_normals", &PyMeshInfo::has_normals)
            .def_prop_ro("has_texcoords", &PyMeshInfo::has_texcoords);

        nb::class_<PyKeyframeData>(m, "KeyframeData")
            .def_ro("keyframe_index", &PyKeyframeData::keyframe_index)
            .def_ro("time", &PyKeyframeData::time)
            .def_ro("position", &PyKeyframeData::position)
            .def_ro("rotation", &PyKeyframeData::rotation)
            .def_ro("focal_length_mm", &PyKeyframeData::focal_length_mm)
            .def_ro("easing", &PyKeyframeData::easing);

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

        nb::class_<vis::PointCloudUpdateTicket>(m, "PointCloudUpdateTicket")
            .def_prop_ro("state", &vis::PointCloudUpdateTicket::state)
            .def_prop_ro("error", &vis::PointCloudUpdateTicket::error)
            .def_prop_ro("inputs_released", &vis::PointCloudUpdateTicket::inputsReleased)
            .def("cancel", &vis::PointCloudUpdateTicket::cancel,
                 "Cancel before publication. In-flight source memory remains retained until inputs_released.");

        nb::class_<PyCropPolygon>(m, "CropPolygon")
            .def_prop_ro("__property_group__", &PyCropPolygon::property_group, "Property group name for introspection")
            .def_prop_ro("vertices", &PyCropPolygon::vertices, "Outline vertices (x, y) in the node's local XY plane")
            .def("get", &PyCropPolygon::get, nb::arg("name"), "Get property value by name")
            .def("set", &PyCropPolygon::set, nb::arg("name"), nb::arg("value"), "Set property value by name")
            .def("prop_info", &PyCropPolygon::prop_info, nb::arg("name"), "Get metadata for a property")
            .def(
                "__getattr__",
                [](PyCropPolygon& self, const std::string& name) -> nb::object {
                    if (!self.has_prop(name)) {
                        throw nb::attribute_error(("CropPolygon has no attribute '" + name + "'").c_str());
                    }
                    return self.prop_getattr(name);
                })
            .def(
                "__setattr__",
                [](PyCropPolygon& self, const std::string& name, nb::object value) {
                    if (!self.has_prop(name)) {
                        throw nb::attribute_error(("Cannot set attribute '" + name + "'").c_str());
                    }
                    self.prop_setattr(name, value);
                })
            .def("__dir__", &PyCropPolygon::python_dir, "List available attributes");

        // PointCloud class
        nb::class_<PyPointCloud>(m, "PointCloud",
                                 "Point-cloud data owned by a load result or referenced by scene node identity.\n\n"
                                 "Loaded data remains usable after its LoadResult is released. Scene-backed\n"
                                 "handles raise RuntimeError while their source node is absent; undo restoring\n"
                                 "the same node makes the handle usable again.")
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
            .def("set_data_async", &PyPointCloud::set_data_async,
                 nb::arg("points"), nb::arg("colors"), nb::kw_only(),
                 nb::arg("centroid") = nb::none(), nb::arg("queue_policy") = "latest",
                 "Queue an independent snapshot off the viewer thread. Keep inputs immutable until inputs_released; poll state for publication.")
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
            .def_prop_ro("uuid", &PySceneNode::uuid, "Durable node UUID")
            .def_prop_ro("parent_id", &PySceneNode::parent_id, "Parent node identifier (-1 for root)")
            .def_prop_ro("children", &PySceneNode::children, "List of child node IDs")
            .def_prop_ro("type", &PySceneNode::type, "Node type (SPLAT, GROUP, CAMERA, etc.)")
            // Transform (special conversion to tuple/ndarray)
            .def_prop_ro("local_transform", &PySceneNode::local_transform, "Local transform as 4x4 row-major tuple")
            .def_prop_ro("world_transform", &PySceneNode::world_transform, "World-space transform as 4x4 row-major tuple")
            .def("set_local_transform", &PySceneNode::set_local_transform, "Set local transform from a [4, 4] ndarray")
            // Metadata (read-only)
            .def_prop_ro("gaussian_count", &PySceneNode::gaussian_count, "Number of Gaussians owned by this node")
            .def_prop_ro("centroid", &PySceneNode::centroid, "Centroid position (x, y, z)")
            // Data accessors
            .def("splat_data", &PySceneNode::splat_data, "Get SplatData for SPLAT nodes (None otherwise)")
            .def("point_cloud", &PySceneNode::point_cloud, "Get PointCloud for POINTCLOUD nodes (None otherwise)")
            .def("mesh", &PySceneNode::mesh, "Get MeshInfo for MESH nodes (None otherwise)")
            .def("cropbox", &PySceneNode::cropbox, "Get CropBox for CROPBOX nodes (None otherwise)")
            .def("ellipsoid", &PySceneNode::ellipsoid, "Get Ellipsoid for ELLIPSOID nodes (None otherwise)")
            .def("crop_polygon", &PySceneNode::crop_polygon, "Get CropPolygon for CROP_POLYGON nodes (None otherwise)")
            .def("keyframe_data", &PySceneNode::keyframe_data, "Get KeyframeData for KEYFRAME nodes (None otherwise)")
            // Camera specific (read-only)
            .def_prop_ro("camera_uid", &PySceneNode::camera_uid, "Camera unique identifier")
            .def_prop_ro("image_path", &PySceneNode::image_path, "Path to the camera image file")
            .def_prop_ro("mask_path", &PySceneNode::mask_path, "Path to the camera mask file")
            .def_prop_ro("depth_path", &PySceneNode::depth_path, "Path to the camera depth map file")
            .def_prop_ro("has_camera", &PySceneNode::has_camera, "Whether this node has camera data")
            .def_prop_ro("has_mask", &PySceneNode::has_mask, "Whether this camera node has a mask file")
            .def_prop_ro("has_depth", &PySceneNode::has_depth, "Whether this camera node has a depth map file")
            .def_prop_ro("has_distortion", &PySceneNode::has_distortion, "Whether this camera node has lens distortion or non-pinhole camera model")
            .def("load_mask", &PySceneNode::load_mask,
                 nb::arg("resize_factor") = 1, nb::arg("max_width") = 0,
                 nb::arg("invert") = false, nb::arg("threshold") = 0.5f,
                 "Load mask as tensor [1, H, W] on CUDA (None if not a camera node or no mask)")
            .def("load_depth", &PySceneNode::load_depth,
                 nb::arg("resize_factor") = 1, nb::arg("max_width") = 0,
                 "Load depth map as tensor [H, W] on CUDA (None if not a camera node or no depth map)")
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
            .def_prop_ro("render_generation", &PyScene::render_generation,
                         "Scene content revision, excluding Gaussian selection changes")
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
                 nb::arg("transient") = false,
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
    transient: Show the node but leave it out of project saves and undo history

Returns:
    Node ID of created splat
)doc")
            .def("add_point_cloud", &PyScene::add_point_cloud,
                 nb::arg("name"),
                 nb::arg("points"),
                 nb::arg("colors"),
                 nb::arg("parent") = core::NULL_NODE,
                 "Add a point cloud node from [N,3] positions and uint8 or float32 colors; other color dtypes raise ValueError")
            .def("add_mesh", &PyScene::add_mesh,
                 nb::arg("name"),
                 nb::arg("vertices"),
                 nb::arg("indices"),
                 nb::arg("colors") = nb::none(),
                 nb::arg("normals") = nb::none(),
                 nb::arg("parent") = core::NULL_NODE,
                 "Add a mesh node from [V,3] vertices, [F,3] face indices, optional [V,4] colors and [V,3] normals; invalid color shapes raise ValueError")
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
                 nb::arg("mask") = nb::none(),
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
    mask: Optional in-memory mask tensor (H, W) or (1, H, W) at the image
        resolution. Bypasses the on-disk masks/ workflow — useful for
        direct-scene plugins that want to attach per-frame masks without
        writing files. Set the session's ``mask_mode`` to ``Ignore`` or
        ``Segment`` for it to take effect during training.

Returns:
    Node ID of created camera
)doc")
            .def("remove_node", &PyScene::remove_node,
                 nb::arg("name"), nb::arg("keep_children") = false,
                 "Remove a node by name, optionally keeping its children. Raises RuntimeError if the GUI scene manager rejects removal.")
            .def("rename_node", &PyScene::rename_node,
                 nb::arg("old_name"), nb::arg("new_name"),
                 "Rename a node, returns true on success")
            .def("clear", &PyScene::clear,
                 "Remove all nodes from the scene")
            // Hierarchy
            .def("reparent", &PyScene::reparent,
                 nb::arg("node_id"), nb::arg("new_parent_id"),
                 "Move a node under a new parent, returns true on success")
            .def("root_nodes", &PyScene::root_nodes,
                 "Get all root-level nodes")
            // Queries
            .def("get_node_by_id", &PyScene::get_node_by_id, nb::arg("id"),
                 "Find a node by its integer ID (None if not found)")
            .def("get_node_by_uuid", &PyScene::get_node_by_uuid, nb::arg("uuid"),
                 "Find a node by its durable UUID (None if invalid or not found)")
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
            .def("snapshot_visible_splats", &PyScene::snapshot_visible_splats, "Copy visible splats and world transforms at a UI safe point. Returned data owns its storage and supports worker-side export after scene edits or deletion.")
            .def("is_node_effectively_visible", &PyScene::is_node_effectively_visible, nb::arg("id"), "Check if a node is visible considering parent visibility")
            // Transforms
            .def("get_world_transform", &PyScene::get_world_transform, nb::arg("node_id"), "Get world-space transform as 4x4 row-major tuple")
            .def("set_node_transform", &PyScene::set_node_transform_tensor, nb::arg("name"), nb::arg("transform"), "Set node local transform from a [4, 4] Tensor")
            .def("set_node_transform", &PyScene::set_node_transform, nb::arg("name"), nb::arg("transform"), "Set node local transform from a [4, 4] ndarray")
            // Combined/training model
            .def("combined_model", &PyScene::combined_model, "Get the merged SplatData for all visible splats (None if empty)")
            .def("apply_crop_filter", &PyScene::apply_crop_filter, nb::arg("mask"), "Filter a combined-model GPU bool mask in place by enabled render crop boxes, ellipsoids and polygons. Scene selection is unchanged.")
            .def("training_model", &PyScene::training_model, "Get the SplatData used for training (None if unavailable)")
            .def("set_training_model_node", &PyScene::set_training_model_node, nb::arg("name"), "Set which node provides the training model. A transient node may not replace a saved training model (RuntimeError)")
            .def_prop_ro("training_model_node_name", &PyScene::training_model_node_name, "Name of the node providing the training model")
            .def("set_node_transient", &PyScene::set_node_transient, nb::arg("name"), nb::arg("transient"), "Show a node in the viewer but leave it and everything under it out of project saves; False if there is no such node. Raises RuntimeError if the node holds the training model")
            // Bounds (by id)
            .def("get_node_bounds", &PyScene::get_node_bounds, nb::arg("id"), "Get axis-aligned bounding box as ((min_x, min_y, min_z), (max_x, max_y, max_z))")
            .def("get_node_bounds_center", &PyScene::get_node_bounds_center, nb::arg("id"), "Get center of the node bounding box as (x, y, z)")
            // Bounds (by name)
            .def("get_node_bounds", [](PyScene& self, const std::string& name) {
                    auto node = self.get_node(name);
                    if (!node)
                        return decltype(self.get_node_bounds(0)){std::nullopt};
                    return self.get_node_bounds(node->id()); }, nb::arg("name"), "Get axis-aligned bounding box by node name")
            .def("get_node_bounds_center", [](PyScene& self, const std::string& name) {
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
            .def("preview_selection_mask", &PyScene::preview_selection_mask, nb::arg("mask"), "Preview a selection mask without pushing an undo step")
            .def("commit_selection_preview", &PyScene::commit_selection_preview, "Commit a transient selection update as one undo step")
            .def("cancel_selection_preview", &PyScene::cancel_selection_preview, "Cancel a transient selection update and restore the original selection")
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
            .def("has_active_distorted_cameras", &PyScene::has_active_distorted_cameras, "Check if any active camera enabled for training has distortion or non-pinhole model")
            .def("can_undistort_active_cameras", &PyScene::can_undistort_active_cameras, "Check if all active cameras needing distortion handling can be undistorted")
            // Training data
            .def("has_training_data", &PyScene::has_training_data, "Check if training dataset is loaded")
            .def_prop_rw("is_point_cloud_modified", &PyScene::is_point_cloud_modified, &PyScene::set_point_cloud_modified, "Whether the point cloud has been modified since loading")
            .def_prop_ro("scene_center", &PyScene::scene_center, "Scene center position as a [3] tensor")
            // Counts
            .def_prop_ro("node_count", &PyScene::node_count, "Total number of nodes in the scene")
            .def_prop_ro("total_gaussian_count", &PyScene::total_gaussian_count, "Total number of Gaussians across all nodes")
            .def("has_nodes", &PyScene::has_nodes, "Check if the scene contains any nodes")
            // Operations
            .def("apply_deleted", &PyScene::apply_deleted, "Permanently remove soft-deleted Gaussians from all nodes")
            .def("invalidate_cache", &PyScene::invalidate_cache, "Invalidate the combined model cache")
            .def("notify_changed", &PyScene::notify_changed, "Notify the renderer that scene data has changed")
            .def("duplicate_node", &PyScene::duplicate_node, nb::arg("name"), "Duplicate a node by name, returns the new node name")
            .def("merge_group", &PyScene::merge_group, nb::arg("group_name"), "Merge all splats in a group into a single node, returns merged node ID")
            .def_prop_ro("nodes", &PyScene::nodes, "Iterable collection of all scene nodes");
    }

} // namespace lfs::python
