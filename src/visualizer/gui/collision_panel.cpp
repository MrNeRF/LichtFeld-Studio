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
        std::shared_ptr<core::MeshData> makeMesh(core::CollisionMesh data) {
            if (data.vertices.empty() || data.indices.empty())
                throw std::runtime_error("Collision generation produced an empty surface");
            auto mesh = std::make_shared<core::MeshData>();
            auto vertices = data.vertices.size() / 3, faces = data.indices.size() / 3;
            mesh->vertices = core::Tensor::from_vector(data.vertices, {vertices, 3}, core::Device::CPU);
            mesh->indices = core::Tensor::from_vector(data.indices, {faces, 3}, core::Device::CPU);
            mesh->compute_normals();
            std::vector<float> colors(vertices * 4);
            for (size_t i = 0; i < vertices; ++i) {
                colors[i * 4] = 0.1f;
                colors[i * 4 + 1] = 0.65f;
                colors[i * 4 + 2] = 0.8f;
                colors[i * 4 + 3] = 1;
            }
            mesh->colors = core::Tensor::from_vector(colors, {vertices, 4}, core::Device::CPU);
            return mesh;
        }
        std::vector<core::CollisionGaussian> extract(
            const std::vector<core::Scene::SplatSnapshot>& snapshots, glm::mat4 world_to_grid,
            std::stop_token stop) {
            std::vector<core::CollisionGaussian> result;
            for (const auto& snapshot : snapshots) {
                if (stop.stop_requested())
                    throw std::runtime_error("Collision generation cancelled");
                auto model = snapshot.materialize();
                auto means = model->means().to(core::Device::CPU).contiguous();
                auto scaling = model->get_scaling().to(core::Device::CPU).contiguous();
                auto rotation = model->get_rotation().to(core::Device::CPU).contiguous();
                auto opacity = model->get_opacity().to(core::Device::CPU).contiguous();
                auto deleted = model->has_deleted_mask() ? model->deleted().to(core::Device::CPU).contiguous() : core::Tensor{};
                auto* m = means.ptr<float>();
                auto* s = scaling.ptr<float>();
                auto* r = rotation.ptr<float>();
                auto* o = opacity.ptr<float>();
                const auto transform = world_to_grid * snapshot.world_transform;
                for (size_t i = 0; i < size_t(model->size()); ++i) {
                    if ((i & 4095) == 0 && stop.stop_requested())
                        throw std::runtime_error("Collision generation cancelled");
                    if (deleted.is_valid() && deleted.ptr<bool>()[i])
                        continue;
                    if (result.size() >= 10'000'000)
                        throw std::runtime_error("Collision source exceeds 10 million Gaussians; hide or isolate source nodes");
                    auto q = glm::quat(r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]);
                    auto axes = glm::mat3_cast(q);
                    for (int a = 0; a < 3; ++a)
                        axes[a] *= s[i * 3 + a];
                    result.push_back({glm::vec3(transform * glm::vec4(m[i * 3], m[i * 3 + 1], m[i * 3 + 2], 1)),
                                      glm::mat3(transform) * axes, o[i]});
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
        if (document_)
            document_->RemoveEventListener("click", this);
    }
    void CollisionPanel::reloadRmlResources() {
        if (document_)
            document_->RemoveEventListener("click", this);
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
        auto set = [&](const char* id, float n) { setValue(id, std::format("{:.6g}", n)); };
        auto check = [&](const char* id, bool on) {
            auto* el = document_->GetElementById(id);
            if (on)
                el->SetAttribute("checked", "checked");
            else
                el->RemoveAttribute("checked");
        };
        setValue("preset", std::to_string(int(settings_.preset)));
        setValue("shape", std::to_string(int(settings_.shape)));
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
        set("carve-height", settings_.carve_height);
        set("carve-radius", settings_.carve_radius);
        set("seed-x", world_seed_.x);
        set("seed-y", world_seed_.y);
        set("seed-z", world_seed_.z);
        if (auto* input = InputController::instance()) {
            const auto& w = input->walkSettings();
            set("walk-height", w.height);
            set("walk-radius", w.radius);
            set("walk-speed", w.speed);
            set("walk-step", w.step_height);
        }
        host_.markContentDirty();
    }
    core::CollisionSettings CollisionPanel::readInputs() const {
        auto s = settings_;
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
        world_seed_ = glm::vec3(scene_->getWorldTransform(node.id) * glm::vec4(settings_.seed, 1));
        writeInputs();
        document_->GetElementById("collision-target")->SetInnerRML(Rml::StringUtilities::EncodeRml(node.name));
        status(std::format("{} occupied voxels; {} triangles. {}",
                           node.collision->occupiedCount(),
                           node.mesh ? node.mesh->face_count() : 0, node.collision->warning));
    }
    bool CollisionPanel::prepare(const PanelDrawContext& ctx) {
        scene_ = ctx.scene;
        training_ = ctx.is_training;
        if (!scene_ || !host_.ensureDocumentLoaded())
            return false;
        if (!document_) {
            document_ = host_.getDocument();
            document_->AddEventListener("click", this);
            if (auto* title = document_->GetElementById("title-text"))
                title->SetInnerRML(Rml::StringUtilities::EncodeRml(LOC("collision.title")));
            if (auto* input = InputController::instance(); target_.is_nil() && input)
                world_seed_ = lfs::rendering::visualizerWorldPointFromDataWorld(input->cameraPosition());
            writeInputs();
        }
        adoptResult();
        if (!busy_ && !target_.is_nil() && !scene_->getNodeByUuid(target_)) {
            target_ = {};
            document_->GetElementById("collision-target")->SetInnerRML(Rml::StringUtilities::EncodeRml(LOC("collision.new_node")));
            status(LOC("collision.initial_status"));
        }
        if (auto* manager = services().sceneOrNull(); manager && !busy_) {
            auto ids = manager->getSelectedNodeIds();
            if (ids.size() == 1)
                if (auto* node = scene_->getNodeById(ids[0]); node && node->collision && node->uuid != target_)
                    loadSettings(*node);
        }
        if (busy_) {
            std::lock_guard lock(mutex_);
            status(progress_);
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
        auto snapshots = scene_->snapshotVisibleSplats();
        if (snapshots.empty())
            throw std::runtime_error("No visible loaded splats to voxelize");
        std::vector<core::Uuid> sources;
        for (const auto* node : scene_->getNodes())
            if (node->model && scene_->isNodeEffectivelyVisible(node->id))
                sources.push_back(node->uuid);
        core::Uuid target = new_node ? core::Uuid{} : target_;
        bool show_voxels = true;
        if (!target.is_nil()) {
            if (auto* node = scene_->getNodeByUuid(target); !node || node->locked.get())
                throw std::runtime_error("Collision target is missing or locked");
            else if (node->collision)
                show_voxels = node->collision->show_voxels;
        }
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
        worker_ = std::jthread([this, snapshots = std::move(snapshots), sources = std::move(sources),
                                s, grid_to_world, world_to_grid, target, show_voxels, scene_generation, graph_epoch](std::stop_token stop) {
            try {
                auto report = [&](std::string text) {
                    std::lock_guard lock(mutex_);
                    progress_ = std::move(text);
                };
                report("Reading Gaussian snapshot");
                auto gaussians = extract(snapshots, world_to_grid, stop);
                const auto begin = std::chrono::steady_clock::now();
                auto volume = std::make_shared<core::CollisionVolume>(
                    core::generateCollision(gaussians, s, stop, report));
                const auto voxelized = std::chrono::steady_clock::now();
                volume->show_voxels = show_voxels;
                auto mesh = makeMesh(volume->mesh(s.shape, stop));
                LOG_INFO("Collision generation: source={}, voxel_size={}, opacity={}, occupied={}, tree_nodes={}, generation_ms={}, mesh_ms={}",
                         gaussians.size(), s.voxel_size, s.opacity, volume->occupiedCount(), volume->tree.size(),
                         std::chrono::duration_cast<std::chrono::milliseconds>(voxelized - begin).count(),
                         std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - voxelized).count());
                if (stop.stop_requested())
                    throw std::runtime_error("Collision generation cancelled");
                std::lock_guard lock(mutex_);
                result_ = Result{std::move(volume), std::move(mesh), grid_to_world, sources, target, scene_generation, graph_epoch};
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
        node->visible = true;
        scene_->markPayloadDiverged(node->id);
        scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        target_ = node->uuid;
        if (auto* manager = services().sceneOrNull())
            manager->selectNode(node->id);
        loadSettings(*node);
        result_.reset();
    }
    void CollisionPanel::ProcessEvent(Rml::Event& event) {
        auto* el = event.GetTargetElement();
        while (el && el->GetId().empty())
            el = el->GetParentNode();
        if (!el)
            return;
        auto id = el->GetId();
        try {
            if (id == "generate")
                start(false);
            else if (id == "generate-new")
                start(true);
            else if (id == "cancel") {
                worker_.request_stop();
                status("Cancelling collision generation");
            } else if (id == "apply-preset") {
                settings_ = readInputs();
                auto preset = value("preset");
                settings_.applyPreset(preset == "0" ? core::CollisionPreset::Indoor : preset == "2" ? core::CollisionPreset::Object
                                                                                                    : core::CollisionPreset::Outdoor);
                world_seed_ = settings_.seed;
                writeInputs();
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
                auto volume = std::make_shared<core::CollisionVolume>(*node->collision);
                volume->show_voxels = id == "show-voxels";
                node->collision = volume;
                scene_->markPayloadDiverged(node->id);
                scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
                loadSettings(*node);
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
