/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/collision_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "gui/ui_context.hpp"
#include "input/input_controller.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene/scene_manager.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControl.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <glm/gtc/quaternion.hpp>
#include <stdexcept>

namespace lfs::vis::gui {
    namespace {
        std::shared_ptr<core::MeshData> makeMesh(core::CollisionMesh data, std::stop_token stop = {}) {
            if (data.vertices.empty() || data.indices.empty())
                throw std::runtime_error("Collision generation produced an empty surface");
            auto mesh = std::make_shared<core::MeshData>();
            auto vertices = data.vertices.size() / 3, faces = data.indices.size() / 3;
            mesh->vertices = core::Tensor::from_vector(data.vertices, {vertices, 3}, core::Device::CPU);
            mesh->indices = core::Tensor::from_vector(data.indices, {faces, 3}, core::Device::CPU);
            // A simple parallel-safe indexed normal pass avoids OpenMesh topology
            // construction and its large adjacency/conversion allocations.
            mesh->normals = core::Tensor::zeros({vertices, 3}, core::Device::CPU);
            auto* normals = mesh->normals.ptr<float>();
            for (size_t f = 0; f < data.indices.size(); f += 3) {
                if ((f & 4095) == 0 && stop.stop_requested())
                    throw std::runtime_error("Collision surface cancelled");
                const auto a = data.indices[f], b = data.indices[f + 1], c = data.indices[f + 2];
                auto point = [&](int i) { return glm::vec3(data.vertices[i * 3], data.vertices[i * 3 + 1], data.vertices[i * 3 + 2]); };
                auto n = glm::cross(point(b) - point(a), point(c) - point(a));
                for (int i : {a, b, c})
                    for (int k = 0; k < 3; ++k)
                        normals[i * 3 + k] += n[k];
            }
            for (size_t i = 0; i < vertices; ++i) {
                if ((i & 4095) == 0 && stop.stop_requested())
                    throw std::runtime_error("Collision surface cancelled");
                glm::vec3 n(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
                if (auto length = glm::length(n); length > 0)
                    n /= length;
                for (int k = 0; k < 3; ++k)
                    normals[i * 3 + k] = n[k];
            }
            // Uniform collision color belongs to one material, not V copies.
            core::Material material;
            material.base_color = {0.1f, 0.65f, 0.8f, 1.0f};
            mesh->materials.push_back(material);
            return mesh;
        }
        std::vector<core::CollisionGaussian> extract(
            const std::vector<core::Scene::SplatGeometrySnapshot>& snapshots, glm::mat4 world_to_grid,
            std::stop_token stop) {
            std::vector<core::CollisionGaussian> result;
            for (const auto& snapshot : snapshots) {
                if (stop.stop_requested())
                    throw std::runtime_error("Collision generation cancelled");
                const auto& means = snapshot.means;
                const auto& scaling = snapshot.scaling;
                const auto& rotation = snapshot.rotation;
                const auto& opacity = snapshot.opacity;
                const auto& deleted = snapshot.deleted;
                const auto* m = means.ptr<float>();
                const auto* s = scaling.ptr<float>();
                const auto* r = rotation.ptr<float>();
                const auto* o = opacity.ptr<float>();
                const auto transform = world_to_grid * snapshot.world_transform;
                for (size_t i = 0; i < snapshot.size; ++i) {
                    if ((i & 4095) == 0 && stop.stop_requested())
                        throw std::runtime_error("Collision generation cancelled");
                    if (deleted.is_valid() && deleted.ptr<bool>()[i])
                        continue;
                    if (result.size() >= 10'000'000)
                        throw std::runtime_error("Collision source exceeds 10 million Gaussians; hide or isolate source nodes");
                    auto q = glm::normalize(glm::quat(r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]));
                    auto axes = glm::mat3_cast(q);
                    for (int a = 0; a < 3; ++a)
                        axes[a] *= std::exp(s[i * 3 + a]);
                    result.push_back({glm::vec3(transform * glm::vec4(m[i * 3], m[i * 3 + 1], m[i * 3 + 2], 1)),
                                      glm::mat3(transform) * axes, 1.0f / (1.0f + std::exp(-o[i]))});
                }
            }
            return result;
        }
    } // namespace
    CollisionPanel::CollisionPanel(RmlUIManager* manager)
        : host_(manager, "native.collision", "rmlui/collision.rml") {}
    CollisionPanel::~CollisionPanel() {
        worker_.request_stop();
        if (worker_.joinable())
            worker_.join();
        if (document_) {
            document_->RemoveEventListener("click", this);
            document_->RemoveEventListener("change", this);
        }
    }
    void CollisionPanel::reloadRmlResources() {
        if (document_) {
            document_->RemoveEventListener("click", this);
            document_->RemoveEventListener("change", this);
        }
        document_ = nullptr;
        host_.reloadDocument();
    }
    std::string CollisionPanel::value(const char* id) const {
        auto* control = dynamic_cast<Rml::ElementFormControl*>(document_->GetElementById(id));
        if (!control)
            throw std::runtime_error(std::string("Missing collision control: ") + id);
        return control->GetValue();
    }
    float CollisionPanel::number(const char* id) const {
        auto text = value(id);
        float result = 0;
        auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
        if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(result))
            throw std::invalid_argument(std::string("Invalid number: ") + id);
        return result;
    }
    bool CollisionPanel::checked(const char* id) const {
        return document_->GetElementById(id)->HasAttribute("checked");
    }
    void CollisionPanel::setValue(const char* id, std::string v) {
        if (auto* control = dynamic_cast<Rml::ElementFormControl*>(document_->GetElementById(id)))
            control->SetValue(v);
    }
    void CollisionPanel::status(std::string text) {
        auto* el = document_->GetElementById("collision-status");
        auto encoded = Rml::StringUtilities::EncodeRml(text);
        if (el->GetInnerRML() != encoded) {
            el->SetInnerRML(encoded);
            host_.markContentDirty();
        }
    }
    void CollisionPanel::writeInputs() {
        updating_ = true;
        auto set = [&](const char* id, float n) { setValue(id, std::format("{:.6g}", n)); };
        auto check = [&](const char* id, bool on) {
            auto* el = document_->GetElementById(id);
            if (on)
                el->SetAttribute("checked", "checked");
            else
                el->RemoveAttribute("checked");
        };
        setValue("units-per-meter", std::format("{:.6g}", settings_.units_per_meter));
        setValue("detail", std::to_string(settings_.voxel_size / settings_.units_per_meter));
        setValue("overlay-opacity", std::to_string(display_.opacity));
        setValue("cutaway", std::to_string(display_.slice));
        for (auto [id, n] : {std::pair{"overlay-opacity-value", display_.opacity}, {"cutaway-value", display_.slice}})
            document_->GetElementById(id)->SetInnerRML(std::format("{:.0f}%", n * 100));
        setValue("surface-stride", std::to_string(display_.surface_stride));
        setValue("shape", std::to_string(int(display_.surface_ready ? display_.surface_shape : settings_.shape)));
        check("cluster", settings_.cluster);
        set("cluster-resolution", settings_.cluster_resolution);
        set("cluster-opacity", settings_.cluster_opacity);
        set("cluster-min", settings_.cluster_min_contribution);
        set("voxel-size", settings_.voxel_size);
        set("opacity", settings_.opacity);
        check("external-fill", settings_.external_fill);
        set("external-size", settings_.external_size);
        check("floor-fill", settings_.floor_fill);
        set("floor-size", settings_.floor_size);
        check("floor-below-seed", settings_.floor_below_seed);
        set("carve-height", settings_.carve_height);
        set("carve-radius", settings_.carve_radius);
        set("seed-x", world_seed_.x);
        set("seed-y", world_seed_.y);
        set("seed-z", world_seed_.z);
        if (auto* input = InputController::instance()) {
            const auto& w = input->walkSettings();
            const auto ratio = settings_.units_per_meter / w.units_per_meter;
            set("walk-height", w.height * ratio);
            set("walk-radius", w.radius * ratio);
            set("walk-speed", w.speed * ratio);
            set("walk-step", w.step_height * ratio);
        }
        updating_ = false;
        updateSummary();
        host_.markContentDirty();
    }
    core::CollisionSettings CollisionPanel::readInputs() const {
        auto s = settings_;
        s.units_per_meter = number("units-per-meter");
        s.cluster = checked("cluster");
        s.cluster_resolution = number("cluster-resolution");
        s.cluster_opacity = number("cluster-opacity");
        s.cluster_min_contribution = number("cluster-min");
        s.voxel_size = number("voxel-size");
        s.opacity = number("opacity");
        s.external_fill = checked("external-fill");
        s.external_size = number("external-size");
        s.floor_fill = checked("floor-fill");
        s.floor_size = number("floor-size");
        s.floor_below_seed = checked("floor-below-seed");
        s.carve_height = number("carve-height");
        s.carve_radius = number("carve-radius");
        s.shape = value("shape") == "1" ? core::CollisionShape::Faces : core::CollisionShape::Smooth;
        s.seed = {number("seed-x"), number("seed-y"), number("seed-z")};
        s.validate();
        return s;
    }
    void CollisionPanel::loadSettings(const core::SceneNode& node) {
        target_ = node.uuid;
        settings_ = node.collision->settings;
        display_ = node.collision_display;
        world_seed_ = glm::vec3(scene_->getWorldTransform(node.id) * glm::vec4(settings_.seed, 1));
        writeInputs();
        document_->GetElementById("collision-target")->SetInnerRML(Rml::StringUtilities::EncodeRml(node.name));
        if (counted_volume_.lock() != node.collision) {
            occupied_count_ = node.collision->occupiedCount();
            counted_volume_ = node.collision;
        }
        status(std::format("{} occupied voxels; {}. {} {}",
                           occupied_count_,
                           display_.surface_ready ? std::format("{} surface triangles", node.mesh ? node.mesh->face_count() : 0)
                                                  : "voxel preview ready — no triangle mesh needed for Walk",
                           last_stats_, node.collision->warning));
    }
    bool CollisionPanel::prepare(const PanelDrawContext& ctx) {
        scene_ = ctx.scene;
        training_ = ctx.is_training;
        if (!scene_ || !host_.ensureDocumentLoaded())
            return false;
        if (!document_) {
            settings_.floor_below_seed = true;
            document_ = host_.getDocument();
            document_->AddEventListener("click", this);
            document_->AddEventListener("change", this);
            if (auto* title = document_->GetElementById("title-text"))
                title->SetInnerRML(Rml::StringUtilities::EncodeRml(LOC("collision.title")));
            if (auto* input = InputController::instance(); target_.is_nil() && input)
                world_seed_ = lfs::rendering::visualizerWorldPointFromDataWorld(input->cameraPosition());
            writeInputs();
        }
        adoptResult();
        if (!busy_ && !target_.is_nil()) {
            auto* node = scene_->getNodeByUuid(target_);
            if (!node) {
                target_ = {};
                last_stats_.clear();
                document_->GetElementById("collision-target")->SetInnerRML(Rml::StringUtilities::EncodeRml(LOC("collision.new_node")));
                status(LOC("collision.initial_status"));
            } else if (node->collision && counted_volume_.lock() != node->collision) {
                // A project reload can restore the same UUID with a different
                // payload and display state. UUID equality is not a cache key.
                last_stats_.clear();
                loadSettings(*node);
            }
        }
        if (auto* manager = services().sceneOrNull(); manager && !busy_) {
            auto ids = manager->getSelectedNodeIds();
            if (ids.size() == 1)
                if (auto* node = scene_->getNodeById(ids[0]); node && node->collision && node->uuid != target_) {
                    last_stats_.clear();
                    loadSettings(*node);
                }
        }
        if (busy_) {
            std::lock_guard lock(mutex_);
            status(progress_);
            document_->GetElementById("generation-progress")->SetAttribute("value", progress_value_ * 100);
            document_->GetElementById("progress-time")->SetInnerRML(LOCF("collision.progress_time", progress_step_ + 1, progress_total_, std::chrono::duration<double>(std::chrono::steady_clock::now() - job_started_).count()));
            host_.markContentDirty();
        }
        document_->GetElementById("job-progress")->SetProperty("display", busy_ ? "block" : "none");
        for (const auto* id : {"generate", "generate-new", "cancel"}) {
            auto* control = document_->GetElementById(id);
            const bool disabled = std::string_view(id) == "cancel" ? !busy_ : bool(busy_);
            if (disabled != control->HasAttribute("disabled")) {
                if (disabled)
                    control->SetAttribute("disabled", "disabled");
                else
                    control->RemoveAttribute("disabled");
            }
        }
        if (auto* input = InputController::instance(); input && !input->walkError().empty() && !busy_)
            status(input->walkError());
        return true;
    }
    void CollisionPanel::start(bool new_node) {
        if (busy_)
            throw std::runtime_error("Collision generation is already running");
        if (training_)
            throw std::runtime_error("Stop training before taking a collision snapshot");
        if (!scene_)
            throw std::runtime_error("No scene");
        if (scene_->getVisibleGaussianCount() > 10'000'000)
            throw std::runtime_error("Collision source exceeds 10 million Gaussians; isolate source nodes first");
        settings_ = readInputs();
        world_seed_ = settings_.seed;
        auto* input = InputController::instance();
        if (!input)
            throw std::runtime_error("No camera controller");
        auto frame = glm::mat4(navigationFrame(input->navigationUpAxis()));
        auto grid_to_world = lfs::rendering::VISUALIZER_TO_DATA_WORLD_AXES_4 * frame;
        auto world_to_grid = glm::inverse(grid_to_world);
        auto s = settings_;
        s.seed = glm::vec3(world_to_grid * glm::vec4(world_seed_, 1));
        // The scene helper clones owned storage at the GUI safe point. Workers
        // never retain live node/model pointers while selections or nodes change.
        auto snapshots = scene_->snapshotVisibleSplatGeometry();
        if (snapshots.empty())
            throw std::runtime_error("No visible loaded splats to voxelize");
        std::vector<core::Uuid> sources;
        for (const auto* node : scene_->getNodes())
            if (node->model && scene_->isNodeEffectivelyVisible(node->id))
                sources.push_back(node->uuid);
        core::Uuid target = new_node ? core::Uuid{} : target_;
        auto display = display_;
        display.preview = true;
        display.surface_ready = false;
        if (!target.is_nil())
            if (auto* node = scene_->getNodeByUuid(target); !node || node->locked.get())
                throw std::runtime_error("Collision target is missing or locked");
        const uint64_t scene_generation = scene_->renderGeneration();
        const uint64_t graph_epoch = scene_->graphEpoch();
        {
            std::lock_guard lock(mutex_);
            result_.reset();
            error_.clear();
            progress_ = "Taking source snapshot";
        }
        done_ = false;
        busy_ = true;
        job_started_ = std::chrono::steady_clock::now();
        progress_value_ = 0;
        progress_step_ = 0;
        progress_total_ = 3 + s.cluster + s.external_fill + s.floor_fill + (s.carve_height > 0);
        worker_ = std::jthread([this, snapshots = std::move(snapshots), sources = std::move(sources),
                                s, grid_to_world, world_to_grid, target, display, scene_generation, graph_epoch](std::stop_token stop) mutable {
            try {
                auto report = [&](std::string text) {
                    std::lock_guard lock(mutex_);
                    int phase = 0;
                    float fraction = 0;
                    if (text.starts_with("Clustering")) {
                        phase = 1;
                        if (text.starts_with("Clustering coarse Voxelizing regions: "))
                            fraction = 0.35f * float(std::stoi(text.substr(text.find(": ") + 2))) / 8;
                        else if (text.starts_with("Clustering connected"))
                            fraction = 0.4f;
                        else if (text.starts_with("Clustering splats: ")) {
                            const auto slash = text.find('/');
                            fraction = 0.5f + 0.5f * float(std::stoull(text.substr(text.find(": ") + 2))) / std::stoull(text.substr(slash + 1));
                        } else if (text.starts_with("Clustering complete"))
                            fraction = 1;
                    } else if (text.starts_with("Voxelizing")) {
                        phase = 1 + s.cluster;
                        if (text.starts_with("Voxelizing regions: ")) {
                            auto start = text.find(": ") + 2;
                            fraction = float(std::stoi(text.substr(start))) / 8;
                        }
                    } else if (text.starts_with("Sealing"))
                        phase = 2 + s.cluster;
                    else if (text.starts_with("Filling"))
                        phase = 2 + s.cluster + s.external_fill;
                    else if (text.starts_with("Flooding"))
                        phase = 2 + s.cluster + s.external_fill + s.floor_fill;
                    else if (text.starts_with("Finalizing"))
                        phase = progress_total_ - 1;
                    progress_step_ = phase;
                    progress_value_ = std::max(progress_value_, (phase + fraction) / progress_total_);
                    progress_ = std::move(text);
                };
                report("Reading Gaussian snapshot");
                auto gaussians = extract(snapshots, world_to_grid, stop);
                snapshots.clear(); // Release snapshot buffers before allocating occupancy.
                const auto begin = std::chrono::steady_clock::now();
                auto volume = std::make_shared<core::CollisionVolume>(
                    core::generateCollision(gaussians, s, stop, report));
                const auto voxelized = std::chrono::steady_clock::now();
                volume->show_voxels = true;
                // Scene graph uses a tiny bounds carrier; preview and Walk read
                // occupancy directly. Build a physical surface only on request.
                auto mesh = makeMesh(volume->boundsMesh());
                const auto stats = std::format("{:.2f}s generation; {:.1f} MiB occupancy",
                                               std::chrono::duration<double>(voxelized - begin).count(), volume->tree.size() * 4.0 / (1024 * 1024));
                LOG_INFO("Collision generation: source={}, voxel_size={}, opacity={}, occupied={}, tree_nodes={}, generation_ms={}, mesh_ms={}",
                         gaussians.size(), s.voxel_size, s.opacity, volume->occupiedCount(), volume->tree.size(),
                         std::chrono::duration_cast<std::chrono::milliseconds>(voxelized - begin).count(),
                         std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - voxelized).count());
                if (stop.stop_requested())
                    throw std::runtime_error("Collision generation cancelled");
                std::lock_guard lock(mutex_);
                result_ = Result{std::move(volume), std::move(mesh), grid_to_world, sources, target, scene_generation, graph_epoch, display, stats};
            } catch (const std::exception& error) {
                std::lock_guard lock(mutex_);
                error_ = error.what();
            }
            done_.store(true, std::memory_order_release);
            python::request_redraw();
        });
    }
    void CollisionPanel::adoptResult() {
        if (!done_.exchange(false, std::memory_order_acq_rel))
            return;
        if (worker_.joinable())
            worker_.join();
        busy_ = false;
        std::lock_guard lock(mutex_);
        if (!error_.empty()) {
            LOG_ERROR("Collision generation: {}", error_);
            status(error_);
            return;
        }
        if (!result_)
            return;
        if (scene_->renderGeneration() != result_->scene_generation || scene_->graphEpoch() != result_->graph_epoch) {
            status("Collision result discarded: scene changed during generation. Regenerate from the current scene.");
            result_.reset();
            return;
        }
        for (auto uuid : result_->sources) {
            if (!scene_->getNodeByUuid(uuid)) {
                status("Collision result discarded: source was deleted or the project changed");
                result_.reset();
                return;
            }
        }
        auto* node = result_->target.is_nil() ? nullptr : scene_->getNodeByUuid(result_->target);
        if (!result_->target.is_nil() && !node) {
            status("Collision result discarded: target node was deleted");
            result_.reset();
            return;
        }
        const auto centroid = result_->mesh->vertices.mean({0}, false).to_vector();
        if (!node) {
            auto id = scene_->addMesh("Collision", result_->mesh);
            node = scene_->getNodeById(id);
            if (!node)
                throw std::runtime_error("Could not create collision node");
        } else {
            if (node->locked.get()) {
                status("Collision result discarded: target is locked");
                result_.reset();
                return;
            }
            node->mesh = result_->mesh;
        }
        node->gaussian_count.store(static_cast<size_t>(node->mesh->vertex_count()), std::memory_order_release);
        node->centroid = {centroid[0], centroid[1], centroid[2]};
        auto parent = node->parent_id == core::NULL_NODE ? glm::mat4(1) : scene_->getWorldTransform(node->parent_id);
        node->local_transform = glm::inverse(parent) * result_->world_transform;
        scene_->markTransformDirty(node->id);
        node->collision = result_->volume;
        node->collision_display = result_->display;
        node->collision_display.opacity = display_.opacity;
        node->collision_display.slice = display_.slice;
        last_stats_ = result_->stats;
        node->visible = true;
        scene_->markPayloadDiverged(node->id);
        scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        target_ = node->uuid;
        if (auto* manager = services().sceneOrNull())
            manager->selectNode(node->id);
        loadSettings(*node);
        result_.reset();
    }
    void CollisionPanel::resetSurfaceSettings() {
        const auto previous = readInputs();
        settings_ = {};
        settings_.units_per_meter = previous.units_per_meter;
        settings_.applyPreset(core::CollisionPreset::Outdoor);
        settings_.seed = previous.seed;
        settings_.voxel_size = previous.voxel_size;
        settings_.floor_below_seed = true;
        world_seed_ = settings_.seed;
        writeInputs();
    }
    void CollisionPanel::updateSummary() {
        const float units = number("units-per-meter");
        const float edge = number("voxel-size");
        document_->GetElementById("scale-summary")->SetInnerRML(Rml::StringUtilities::EncodeRml(LOCF("collision.scale_summary", edge / units * 100, edge, units)));
        std::string pipeline = LOC("collision.voxelization");
        for (const auto& [enabled, key] : {std::pair{checked("cluster"), "collision.cluster_filter"},
                                           {checked("external-fill"), "collision.external"},
                                           {checked("floor-fill"), "collision.floor"},
                                           {number("carve-height") > 0, "collision.capsule_carve"}})
            if (enabled)
                pipeline += " · " + std::string(LOC(key));
        document_->GetElementById("pipeline-summary")->SetInnerRML(Rml::StringUtilities::EncodeRml(LOCF("collision.pipeline", pipeline)));
        host_.markContentDirty();
    }
    void CollisionPanel::updateDisplay() {
        display_.opacity = std::clamp(number("overlay-opacity"), 0.0f, 1.0f);
        display_.slice = std::clamp(number("cutaway"), 0.0f, 1.0f);
        for (auto [id, n] : {std::pair{"overlay-opacity-value", display_.opacity}, {"cutaway-value", display_.slice}})
            document_->GetElementById(id)->SetInnerRML(std::format("{:.0f}%", n * 100));
        if (auto* node = scene_->getNodeByUuid(target_); !busy_ && node && node->collision && !node->locked.get()) {
            node->collision_display.opacity = display_.opacity;
            node->collision_display.slice = display_.slice;
            scene_->notifyMutation(core::Scene::MutationType::DISPLAY_CHANGED);
        }
        host_.markContentDirty();
    }
    void CollisionPanel::buildSurface() {
        if (busy_)
            throw std::runtime_error("Wait for generation to finish");
        auto* node = scene_->getNodeByUuid(target_);
        if (!node || !node->collision || node->locked.get())
            throw std::runtime_error("Select an unlocked collision node");
        auto volume = node->collision;
        const auto shape = readInputs().shape;
        auto display = node->collision_display;
        display.preview = false;
        display.surface_ready = true;
        display.surface_shape = shape;
        display.surface_stride = std::stoi(value("surface-stride"));
        auto transform = scene_->getWorldTransform(node->id);
        const auto generation = scene_->renderGeneration(), epoch = scene_->graphEpoch();
        const auto target = node->uuid;
        {
            std::lock_guard lock(mutex_);
            error_.clear();
            progress_ = "Building optional surface (Walk uses full-resolution voxels)";
        }
        busy_ = true;
        done_ = false;
        job_started_ = std::chrono::steady_clock::now();
        progress_value_ = 0;
        progress_step_ = 0;
        progress_total_ = 3;
        worker_ = std::jthread([this, volume, display, shape, transform, generation, epoch, target](std::stop_token stop) {
            try {
                const auto begin = std::chrono::steady_clock::now();
                auto report = [&](std::string text) {
                    std::lock_guard lock(mutex_);
                    progress_ = std::move(text);
                    progress_step_ = 1;
                    progress_value_ = 1.0f / 3;
                };
                report("Preparing surface resolution");
                auto surface = display.surface_stride == 1 ? volume->mesh(shape, stop, report) : volume->coarsened(display.surface_stride, stop).mesh(shape, stop, report);
                {
                    std::lock_guard lock(mutex_);
                    progress_ = "Computing surface normals";
                    progress_step_ = 2;
                    progress_value_ = 2.0f / 3;
                }
                auto mesh = makeMesh(std::move(surface), stop);
                auto stats = std::format("{:.2f}s surface; {:.3g} scene-unit surface cells; original voxels retained for Walk",
                                         std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(),
                                         volume->settings.voxel_size * display.surface_stride);
                std::lock_guard lock(mutex_);
                result_ = Result{volume, mesh, transform, {}, target, generation, epoch, display, stats};
            } catch (const std::exception& error) {
                std::lock_guard lock(mutex_);
                error_ = error.what();
            }
            done_.store(true, std::memory_order_release);
            python::request_redraw();
        });
    }
    void CollisionPanel::ProcessEvent(Rml::Event& event) {
        auto* el = event.GetTargetElement();
        while (el && el->GetId().empty())
            el = el->GetParentNode();
        if (!el)
            return;
        auto id = el->GetId();
        if (updating_)
            return;
        try {
            if (event.GetType() == "change") {
                if (id == "overlay-opacity" || id == "cutaway")
                    updateDisplay();
                else if (id == "detail") {
                    setValue("voxel-size", std::format("{:.6g}", number("detail") * number("units-per-meter")));
                } else if (id == "units-per-meter") {
                    const float units = number("units-per-meter");
                    if (!(units > 0))
                        throw std::invalid_argument("Scene units per metre must be positive");
                    const auto ratio = units / settings_.units_per_meter;
                    for (const auto* field : {"voxel-size", "cluster-resolution", "external-size", "floor-size", "carve-height", "carve-radius", "walk-height", "walk-radius", "walk-speed", "walk-step"})
                        setValue(field, std::format("{:.6g}", number(field) * ratio));
                    settings_.units_per_meter = units;
                }
                updateSummary();
                return;
            }
            if (id == "advanced-toggle") {
                advanced_ = !advanced_;
                document_->GetElementById("advanced")->SetProperty("display", advanced_ ? "block" : "none");
                host_.markContentDirty();
                return;
            }

            if (id == "generate")
                start(false);
            else if (id == "generate-new")
                start(true);
            else if (id == "cancel") {
                worker_.request_stop();
                status("Cancelling collision generation");
            } else if (id == "reset-surface") {
                resetSurfaceSettings();
            } else if (id == "camera-seed") {
                if (auto* input = InputController::instance()) {
                    auto p = lfs::rendering::visualizerWorldPointFromDataWorld(input->cameraPosition());
                    setValue("seed-x", std::format("{:.6g}", p.x));
                    setValue("seed-y", std::format("{:.6g}", p.y));
                    setValue("seed-z", std::format("{:.6g}", p.z));
                }
            } else if (id == "walk" || id == "exit-walk") {
                if (auto* input = InputController::instance()) {
                    if (id == "exit-walk")
                        input->setCameraNavigationMode(InputController::CameraNavigationMode::FPV);
                    else {
                        auto s = input->walkSettings();
                        s.height = number("walk-height");
                        s.units_per_meter = number("units-per-meter");
                        s.gravity = 9.81f * s.units_per_meter;
                        s.radius = number("walk-radius");
                        s.speed = number("walk-speed");
                        s.step_height = number("walk-step");
                        input->setWalkSettings(s);
                        input->setCameraNavigationMode(InputController::CameraNavigationMode::Walk);
                        status(input->walkError().empty() ? "Walk: WASD, drag to look, Shift to run, Esc to exit" : input->walkError());
                    }
                }
            } else if (id == "edit-splats") {
                if (auto* input = InputController::instance())
                    input->setCameraNavigationMode(InputController::CameraNavigationMode::FPV);
                std::vector<core::NodeId> ids;
                for (const auto* node : scene_->getNodes())
                    if (node->model && scene_->isNodeEffectivelyVisible(node->id))
                        ids.push_back(node->id);
                if (auto* manager = services().sceneOrNull())
                    manager->selectNodesById(ids);
                if (auto* node = scene_->getNodeByUuid(target_); node && !node->locked.get())
                    node->visible = false;
                status("Collision hidden for source editing. Use native selection, Delete, crop and transforms (undo supported), then Regenerate.");
            } else if (id == "show-voxels" || id == "show-mesh") {
                if (busy_)
                    throw std::runtime_error("Wait for generation before changing visualization");
                auto* node = scene_->getNodeByUuid(target_);
                if (!node || !node->collision)
                    throw std::runtime_error("Select a collision node first");
                if (node->locked.get())
                    throw std::runtime_error("Collision target is locked");
                if (id == "show-mesh" && (!node->collision_display.surface_ready ||
                                          node->collision_display.surface_stride != std::stoi(value("surface-stride")) ||
                                          readInputs().shape != node->collision_display.surface_shape)) {
                    buildSurface();
                } else {
                    node->collision_display.preview = id == "show-voxels";
                    display_ = node->collision_display;
                    scene_->notifyMutation(core::Scene::MutationType::DISPLAY_CHANGED);
                    loadSettings(*node);
                }
            }
        } catch (const std::exception& error) {
            LOG_ERROR("Collision authoring: {}", error.what());
            status(error.what());
        }
    }
    void CollisionPanel::draw(const PanelDrawContext& ctx) {
        if (prepare(ctx))
            host_.draw(ctx);
    }
    void CollisionPanel::preload(const PanelDrawContext& ctx) { prepare(ctx); }
    PanelDirectRenderResult CollisionPanel::renderDirect(const PanelDirectRenderRequest& r, const PanelDrawContext& ctx) {
        host_.setFloating(r.space == PanelSpace::Floating);
        host_.setInput(r.input);
        host_.setInputClipY(r.clip_y_min, r.clip_y_max);
        host_.setForcedHeight(r.forced_height);
        if (prepare(ctx)) {
            if (r.mode == PanelDirectRenderMode::Preload)
                host_.prepareDirect(r.width, r.height);
            else if (r.mode != PanelDirectRenderMode::Measure)
                host_.drawDirect(r.x, r.y, r.width, r.height);
        }
        host_.setInput(nullptr);
        host_.setInputClipY(-1, -1);
        host_.setForcedHeight(0);
        return {.handled = true, .height = host_.getContentHeight()};
    }
} // namespace lfs::vis::gui
