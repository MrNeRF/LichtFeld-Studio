/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/collision_panel.hpp"
#include "core/collision_source.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/number_format.hpp"
#include "core/services.hpp"
#include "gui/collision_job_status.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/ui_context.hpp"
#include "input/input_controller.hpp"
#include "internal/resource_paths.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene/scene_manager.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControl.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <SDL3/SDL_clipboard.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <glm/gtc/quaternion.hpp>
#include <stdexcept>
#include <utility>

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
            for (const auto* type : {"mousedown", "mousemove", "mouseup", "keydown", "blur"})
                document_->RemoveEventListener(type, this, true);
        }
    }
    void CollisionPanel::reloadRmlResources() {
        if (document_) {
            document_->RemoveEventListener("click", this);
            document_->RemoveEventListener("change", this);
            for (const auto* type : {"mousedown", "mousemove", "mouseup", "keydown", "blur"})
                document_->RemoveEventListener(type, this, true);
        }
        drag_scrub_.clear();
        edit_scrub_.clear();
        document_ = nullptr;
        if (!busy_) {
            job_outcome_.clear();
            error_report_.clear();
        }
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
        LOG_DEBUG("Collision: {}", text);
    }
    void CollisionPanel::clearError() {
        error_report_.clear();
        document_->GetElementById("collision-error")->SetProperty("display", "none");
    }
    void CollisionPanel::finishJob(std::string outcome) {
        job_outcome_ = std::move(outcome);
        job_graph_epoch_ = scene_ ? scene_->graphEpoch() : 0;
        if (job_outcome_ == "completed")
            progress_value_ = 1;
        document_->GetElementById("progress-stage")->SetInnerRML(Rml::StringUtilities::EncodeRml(job_title_ + ": " + job_outcome_));
        updateProgressBar();
        host_.markContentDirty();
    }
    void CollisionPanel::updateProgressBar() {
        const float percent = std::clamp(progress_value_, 0.0f, 1.0f) * 100;
        document_->GetElementById("generation-progress")->SetAttribute("aria-valuenow", percent);
        document_->GetElementById("generation-progress-fill")->SetProperty("width", std::format("{:.2f}%", percent));
        document_->GetElementById("progress-percent")->SetInnerRML(std::format("{:.0f}%", percent));
    }
    void CollisionPanel::showError(std::string message, bool failed_job) {
        const auto info = collisionFailureInfo(message);
        if (failed_job)
            finishJob(info.cancelled ? "cancelled" : "failed");
        status(info.title);
        document_->GetElementById("collision-scroll")->SetScrollTop(0);
        auto set = [&](const char* id, const std::string& text) {
            document_->GetElementById(id)->SetInnerRML(Rml::StringUtilities::EncodeRml(text));
        };
        const auto context = failed_job
                                 ? std::format("{} · Failed step: {} · {:.1f}s elapsed. {}", job_title_, collisionTaskLabel(progress_),
                                               std::chrono::duration<double>(std::chrono::steady_clock::now() - job_started_).count(), job_context_)
                                 : "The requested action was not applied.";
        set("error-advice", info.advice);
        set("error-context", context);
        set("error-details", message);
        error_report_ = info.title + "\n" + info.advice + "\n" + context + "\n" + message;
        document_->GetElementById("collision-error")->SetProperty("display", "block");
        host_.markContentDirty();
    }
    void CollisionPanel::syncScrubs() {
        for (const auto* id : {"voxel-size", "opacity"}) {
            if (edit_scrub_ == id)
                continue;
            auto* field = document_->GetElementById(std::string(id) + "-scrub");
            field->SetAttribute("data-max", std::max(field->GetAttribute<float>("data-max", 1), number(id)));
            const float n = number(id), lo = field->GetAttribute<float>("data-min", 0), hi = std::max(field->GetAttribute<float>("data-max", 1), n);
            document_->GetElementById(std::string(id) + "-fill")->SetProperty("width", std::format("{:.2f}%", hi > lo ? std::clamp((n - lo) / (hi - lo), 0.0f, 1.0f) * 100 : 0));
            const auto caption = std::format("{:.4g}{}", n, field->GetAttribute<Rml::String>("data-suffix", ""));
            auto* display = document_->GetElementById(std::string(id) + "-display");
            display->SetInnerRML(Rml::StringUtilities::EncodeRml(caption));
            field->SetAttribute("aria-valuenow", n);
        }
        host_.markContentDirty();
    }
    void CollisionPanel::applyControlChange(const std::string& id) {
        if (id == "units-per-meter") {
            const float units = number("units-per-meter");
            if (!(units > 0))
                throw std::invalid_argument("Scene units per metre must be positive");
            const auto ratio = units / settings_.units_per_meter;
            if (auto* input = InputController::instance()) {
                auto w = input->walkSettings();
                w.height *= ratio;
                w.radius *= ratio;
                w.speed *= ratio;
                w.step_height *= ratio;
                w.units_per_meter = units;
                w.gravity = 9.81f * units;
                input->setWalkSettings(w);
            }
            for (const auto* field : {"cluster-resolution", "external-size", "floor-size", "carve-height", "carve-radius"})
                setValue(field, std::format("{:.6g}", number(field) * ratio));
            settings_.units_per_meter = units;
        } else if (id == "voxel-size" || id == "opacity") {
            (void)readInputs();
            if (id == "voxel-size")
                document_->GetElementById("auto-resolution")->RemoveAttribute("checked");
        }
        syncScrubs();
        syncGenerationControls();
    }
    void CollisionPanel::commitScrub(const std::string& id, float n) {
        const auto before = value(id.c_str());
        updating_ = true;
        setValue(id.c_str(), std::format("{:.6g}", n));
        updating_ = false;
        try {
            applyControlChange(id);
        } catch (...) {
            setValue(id.c_str(), before);
            syncScrubs();
            throw;
        }
    }
    bool CollisionPanel::scrubEvent(Rml::Event& event) {
        auto* field = event.GetTargetElement();
        while (field && !field->HasAttribute("data-scrub"))
            field = field->GetParentNode();
        const auto type = event.GetType();
        auto beginEdit = [&](const std::string& id) {
            edit_scrub_ = id;
            scrub_before_ = number(id.c_str());
            auto* box = document_->GetElementById(id + "-scrub");
            box->SetClass("is-editing", true);
            auto* input = dynamic_cast<Rml::ElementFormControlInput*>(document_->GetElementById(id));
            input->Focus();
            input->Select();
            host_.markContentDirty();
        };
        auto endEdit = [&](bool commit, bool focus) {
            const auto id = edit_scrub_;
            const float n = commit ? number(id.c_str()) : scrub_before_;
            edit_scrub_.clear();
            document_->GetElementById(id + "-scrub")->SetClass("is-editing", false);
            setValue(id.c_str(), std::format("{:.6g}", scrub_before_));
            if (commit)
                commitScrub(id, n);
            else
                syncScrubs();
            if (focus)
                document_->Focus();
        };
        if (!edit_scrub_.empty() && event.GetTargetElement() == document_->GetElementById(edit_scrub_)) {
            if (type == "blur" || type == "change") {
                endEdit(true, type == "change");
                return true;
            }
            if (type == "keydown") {
                const auto key = event.GetParameter<int>("key_identifier", 0);
                if (key == Rml::Input::KI_RETURN || key == Rml::Input::KI_NUMPADENTER || key == Rml::Input::KI_ESCAPE) {
                    endEdit(key != Rml::Input::KI_ESCAPE, true);
                    event.StopPropagation();
                    return true;
                }
            }
            return false;
        }
        if (!drag_scrub_.empty() && type == "mousemove") {
            const float x = event.GetParameter<float>("mouse_x", scrub_last_x_);
            if (!scrub_dragged_ && std::abs(x - scrub_start_x_) < 4)
                return true;
            auto* box = document_->GetElementById(drag_scrub_ + "-scrub");
            scrub_dragged_ = true;
            box->SetClass("is-dragging", true);
            const float factor = event.GetParameter<int>("shift_key", 0) || event.GetParameter<int>("ctrl_key", 0) ? 10.0f : 1.0f;
            const float step = box->GetAttribute<float>("data-step", .01f);
            const auto lo = box->GetAttribute<float>("data-min", 0), hi = box->GetAttribute<float>("data-max", 1);
            const auto width = std::max(box->GetBox().GetSize(Rml::BoxArea::Border).x, 1.0f);
            scrub_value_ += (x - scrub_last_x_) * (hi - lo) / width * factor;
            scrub_last_x_ = x;
            scrub_value_ = std::clamp(scrub_value_, box->GetAttribute<float>("data-min", 0), std::max(box->GetAttribute<float>("data-max", 1), number(drag_scrub_.c_str())));
            const auto snapped = std::clamp(std::round(scrub_value_ / step) * step,
                                            box->GetAttribute<float>("data-min", 0),
                                            std::max(box->GetAttribute<float>("data-max", 1), number(drag_scrub_.c_str())));
            commitScrub(drag_scrub_, snapped);
            event.StopPropagation();
            return true;
        }
        if (!drag_scrub_.empty() && type == "mouseup" && event.GetParameter<int>("button", 0) == 0) {
            const auto id = std::exchange(drag_scrub_, {});
            document_->GetElementById(id + "-scrub")->SetClass("is-dragging", false);
            if (!scrub_dragged_)
                beginEdit(id);
            else
                document_->Focus();
            event.StopPropagation();
            return true;
        }
        if (field && edit_scrub_.empty()) {
            const auto id = field->GetAttribute<Rml::String>("data-scrub", "");
            if (type == "mousedown" && event.GetParameter<int>("button", 0) == 0) {
                drag_scrub_ = id;
                scrub_dragged_ = false;
                scrub_start_x_ = scrub_last_x_ = event.GetParameter<float>("mouse_x", 0);
                scrub_value_ = scrub_before_ = number(id.c_str());
                field->Focus();
                event.StopPropagation();
                return true;
            }
            if (type == "keydown") {
                const auto key = event.GetParameter<int>("key_identifier", 0);
                if (key == Rml::Input::KI_RETURN || key == Rml::Input::KI_SPACE) {
                    beginEdit(id);
                    event.StopPropagation();
                    return true;
                }
                if (key == Rml::Input::KI_LEFT || key == Rml::Input::KI_RIGHT || key == Rml::Input::KI_UP || key == Rml::Input::KI_DOWN) {
                    const float sign = key == Rml::Input::KI_LEFT || key == Rml::Input::KI_DOWN ? -1 : 1;
                    commitScrub(id, std::clamp(number(id.c_str()) + sign * field->GetAttribute<float>("data-step", .01f), field->GetAttribute<float>("data-min", 0), std::max(field->GetAttribute<float>("data-max", 1), number(id.c_str()))));
                    event.StopPropagation();
                    return true;
                }
            }
        }
        return false;
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
        setValue("surface-stride", std::to_string(display_.surface_stride));
        setValue("shape", std::to_string(int(display_.surface_ready ? display_.surface_shape : settings_.shape)));
        setValue("filter-mode", settings_.adaptive_filter ? (settings_.cluster ? "2" : "1") : (settings_.cluster ? "3" : "0"));
        check("auto-resolution", settings_.auto_resolution);
        set("cluster-resolution", settings_.cluster_resolution);
        set("cluster-opacity", settings_.cluster_opacity);
        set("cluster-min", settings_.cluster_min_contribution);
        set("max-area", settings_.max_gaussian_area / (settings_.units_per_meter * settings_.units_per_meter));
        set("voxel-size", settings_.voxel_size / settings_.units_per_meter);
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
        updating_ = false;
        syncScrubs();
        syncGenerationControls();
        host_.markContentDirty();
    }
    core::CollisionSettings CollisionPanel::readInputs() const {
        auto s = settings_;
        s.units_per_meter = number("units-per-meter");
        s.auto_resolution = checked("auto-resolution");
        const auto filter = value("filter-mode");
        s.cluster = filter == "2" || filter == "3";
        s.adaptive_filter = filter == "1" || filter == "2";
        s.cluster_resolution = number("cluster-resolution");
        s.cluster_opacity = number("cluster-opacity");
        s.cluster_min_contribution = number("cluster-min");
        s.max_gaussian_area = number("max-area") * s.units_per_meter * s.units_per_meter;
        s.voxel_size = number("voxel-size") * s.units_per_meter;
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
        loaded_volume_ = node.collision;
    }
    bool CollisionPanel::prepare(const PanelDrawContext& ctx) {
        scene_ = ctx.scene;
        training_ = ctx.is_training;
        if (!scene_ || !host_.ensureDocumentLoaded())
            return false;
        if (!document_) {
            settings_.floor_below_seed = true;
            if (target_.is_nil())
                settings_.auto_resolution = true;
            document_ = host_.getDocument();
            document_->GetElementById("collision-icon")->SetAttribute("src", rml_theme::pathToRmlImageSource(lfs::vis::getAssetPath("icon/scene/collision.png")));
            document_->AddEventListener("click", this);
            document_->AddEventListener("change", this);
            for (const auto* type : {"mousedown", "mousemove", "mouseup", "keydown", "blur"})
                document_->AddEventListener(type, this, true);
            if (auto* title = document_->GetElementById("title-text"))
                title->SetInnerRML(Rml::StringUtilities::EncodeRml(LOC("collision.title")));
            if (auto* input = InputController::instance(); target_.is_nil() && input)
                world_seed_ = lfs::rendering::visualizerWorldPointFromDataWorld(input->cameraPosition());
            writeInputs();
        }
        adoptResult();
        document_->SetClass("collision-targeted", !target_.is_nil());
        if (!busy_ && !job_outcome_.empty() && job_graph_epoch_ != scene_->graphEpoch()) {
            job_outcome_.clear();
            clearError();
        }
        if (!busy_ && !target_.is_nil()) {
            auto* node = scene_->getNodeByUuid(target_);
            if (!node) {
                target_ = {};
                status(LOC("collision.initial_status"));
            } else if (node->collision && loaded_volume_.lock() != node->collision) {
                // A project reload can restore the same UUID with a different
                // payload and display state. UUID equality is not a cache key.
                loadSettings(*node);
            }
        }
        if (auto* manager = services().sceneOrNull(); manager && !busy_) {
            auto ids = manager->getSelectedNodeIds();
            if (ids.size() == 1)
                if (auto* node = scene_->getNodeById(ids[0]); node && node->collision && !node->collision_display.surface_ready && node->uuid != target_) {
                    loadSettings(*node);
                }
        }
        if (!busy_)
            if (const auto* node = scene_->getNodeByUuid(target_); node && node->collision)
                display_ = node->collision_display;
        const auto now = std::chrono::steady_clock::now();
        if (busy_ && now >= next_progress_ui_) {
            std::lock_guard lock(mutex_);
            status(job_title_ + "…");
            document_->GetElementById("progress-stage")->SetInnerRML(Rml::StringUtilities::EncodeRml(collisionTaskLabel(progress_)));
            updateProgressBar();
            next_progress_ui_ = now + std::chrono::milliseconds(100);
            host_.markContentDirty();
        }
        document_->GetElementById("job-progress")->SetProperty("display", job_outcome_.empty() ? "none" : "block");
        document_->GetElementById("job-progress")->SetClass("is-failed", job_outcome_ == "failed");
        document_->GetElementById("job-progress")->SetClass("is-running", bool(busy_));
        for (const auto* id : {"generate", "generate-new", "cancel"}) {
            auto* control = document_->GetElementById(id);
            const bool disabled = std::string_view(id) == "cancel" ? !busy_ || worker_.get_stop_token().stop_requested() : bool(busy_);
            if (disabled != control->HasAttribute("disabled")) {
                if (disabled)
                    control->SetAttribute("disabled", "disabled");
                else
                    control->RemoveAttribute("disabled");
            }
        }
        return true;
    }
    void CollisionPanel::start(bool new_node) {
        if (busy_)
            throw std::runtime_error("Collision generation is already running");
        if (training_)
            throw std::runtime_error("Stop training before taking a collision snapshot");
        if (!scene_)
            throw std::runtime_error("No scene");
        if (scene_->getVisibleGaussianCount() > core::CollisionGaussian::max_count)
            throw std::runtime_error("Collision source exceeds 32 million Gaussians; isolate source nodes first");
        settings_ = readInputs();
        clearError();
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
        document_->GetElementById("collision-scroll")->SetScrollTop(0);
        job_started_ = std::chrono::steady_clock::now();
        progress_updated_ = job_started_;
        next_progress_ui_ = {};
        job_outcome_ = "running";
        job_graph_epoch_ = scene_->graphEpoch();
        job_title_ = LOC("collision.generating");
        job_context_ = std::format("{} visible source splats; cell size {:.4g} scene units; opacity {:.4g}; connected filter {}.",
                                   scene_->getVisibleGaussianCount(), s.voxel_size, s.opacity, s.cluster ? "on" : "off");
        if (s.auto_resolution)
            job_context_ += " Automatic resolution enabled; displayed cell size is the previous/minimum value.";
        progress_value_ = 0;
        progress_step_ = 0;
        progress_total_ = 3 + (s.cluster || s.adaptive_filter) + s.external_fill + s.floor_fill + (s.carve_height > 0);
        worker_ = std::jthread([this, snapshots = std::move(snapshots), sources = std::move(sources),
                                s, grid_to_world, world_to_grid, target, display, scene_generation, graph_epoch](std::stop_token stop) mutable {
            try {
                auto report = [&](std::string text) {
                    std::lock_guard lock(mutex_);
                    progress_updated_ = std::chrono::steady_clock::now();
                    if (text.starts_with("Voxelizing Auto cell size:"))
                        job_context_ += " " + text.substr(11);
                    int phase = 0;
                    float fraction = 0;
                    if (text.starts_with("Clustering")) {
                        phase = 1;
                        if (text.starts_with("Clustering coarse Voxelizing GPU: "))
                            fraction = 0.35f * float(std::stoi(text.substr(text.find(": ") + 2))) / 100;
                        else if (text.starts_with("Clustering coarse Voxelizing regions: "))
                            fraction = 0.35f * float(std::stoi(text.substr(text.find(": ") + 2))) / 8;
                        else if (text.starts_with("Clustering connected"))
                            fraction = 0.4f;
                        else if (text.starts_with("Clustering splats: ")) {
                            const auto slash = text.find('/');
                            fraction = 0.5f + 0.5f * float(std::stoull(text.substr(text.find(": ") + 2))) / std::stoull(text.substr(slash + 1));
                        } else if (text.starts_with("Clustering source connectivity: ")) {
                            const auto slash = text.find('/');
                            fraction = 0.3f + 0.6f * float(std::stoull(text.substr(text.find(": ") + 2))) / std::max(1ull, std::stoull(text.substr(slash + 1)));
                        } else if (text.starts_with("Clustering cleanup:"))
                            fraction = s.cluster ? 0.3f : 1.f;
                        else if (text.starts_with("Clustering complete"))
                            fraction = 1;
                    } else if (text.starts_with("Voxelizing")) {
                        phase = 1 + (s.cluster || s.adaptive_filter);
                        if (text.starts_with("Voxelizing GPU: "))
                            fraction = float(std::stoi(text.substr(text.find(": ") + 2))) / 100;
                        else if (text.starts_with("Voxelizing regions: ")) {
                            auto start = text.find(": ") + 2;
                            fraction = float(std::stoi(text.substr(start))) / 8;
                        }
                    } else if (text.starts_with("Sealing"))
                        phase = 2 + (s.cluster || s.adaptive_filter);
                    else if (text.starts_with("Filling"))
                        phase = 2 + (s.cluster || s.adaptive_filter) + s.external_fill;
                    else if (text.starts_with("Flooding"))
                        phase = 2 + (s.cluster || s.adaptive_filter) + s.external_fill + s.floor_fill;
                    else if (text.starts_with("Finalizing"))
                        phase = progress_total_ - 1;
                    progress_step_ = phase;
                    progress_value_ = std::max(progress_value_, (phase + fraction) / progress_total_);
                    progress_ = stop.stop_requested() ? "Cancelling — waiting for the current batch to finish" : std::move(text);
                };
                report("Reading Gaussian snapshot");
                auto gaussians = core::extractCollisionSource(snapshots, world_to_grid, stop);
                snapshots.clear(); // Release snapshot buffers before allocating occupancy.
                const auto begin = std::chrono::steady_clock::now();
                auto volume = std::make_shared<core::CollisionVolume>(
                    core::generateCollision(gaussians, s, stop, report, core::voxelizeCollisionGpu));
                const auto voxelized = std::chrono::steady_clock::now();
                volume->show_voxels = true;
                report("Finalizing viewport preview");
                // Scene graph uses a tiny bounds carrier; preview and Walk read
                // occupancy directly. Build a physical surface only on request.
                auto mesh = makeMesh(volume->boundsMesh());
                const auto stats = std::format("{:.2f}s generation; {:.1f} MiB occupancy",
                                               std::chrono::duration<double>(voxelized - begin).count(), volume->tree.size() * 4.0 / (1024 * 1024));
                LOG_INFO("Collision generation: source={}, voxel_size={}, opacity={}, occupied={}, tree_nodes={}, generation_ms={}, mesh_ms={}",
                         gaussians.size(), volume->settings.voxel_size, s.opacity, volume->occupiedCount(), volume->tree.size(),
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
            showError(error_, true);
            return;
        }
        if (!result_)
            return;
        if (scene_->renderGeneration() != result_->scene_generation || scene_->graphEpoch() != result_->graph_epoch) {
            finishJob("discarded");
            status("Collision result discarded: scene changed during generation. Regenerate from the current scene.");
            result_.reset();
            return;
        }
        for (auto uuid : result_->sources) {
            if (!scene_->getNodeByUuid(uuid)) {
                finishJob("discarded");
                status("Collision result discarded: source was deleted or the project changed");
                result_.reset();
                return;
            }
        }
        auto* node = result_->target.is_nil() ? nullptr : scene_->getNodeByUuid(result_->target);
        if (!result_->target.is_nil() && !node) {
            finishJob("discarded");
            status("Collision result discarded: target node was deleted");
            result_.reset();
            return;
        }
        if (result_->surface_only) {
            if (!node || node->locked.get()) {
                finishJob("discarded");
                status("Surface result discarded: source collision is missing or locked");
                result_.reset();
                return;
            }
            const auto parent_id = node->parent_id;
            const auto id = scene_->addMesh(node->name + " surface", result_->mesh, parent_id);
            auto* surface = scene_->getNodeById(id);
            if (!surface)
                throw std::runtime_error("Could not create collision surface node");
            const auto parent = parent_id == core::NULL_NODE ? glm::mat4(1) : scene_->getWorldTransform(parent_id);
            surface->local_transform = glm::inverse(parent) * result_->world_transform;
            surface->collision = result_->volume;
            surface->collision_display = result_->display;
            scene_->markTransformDirty(id);
            scene_->markPayloadDiverged(id);
            node->collision_display.surface_stride = result_->display.surface_stride;
            node->collision_display.surface_shape = result_->display.surface_shape;
            scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
            if (auto* manager = services().sceneOrNull())
                manager->selectNode(id);
            finishJob("completed");
            status(surface->name + ": " + result_->stats);
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
                finishJob("discarded");
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
        node->visible = true;
        scene_->markPayloadDiverged(node->id);
        scene_->notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        target_ = node->uuid;
        if (auto* manager = services().sceneOrNull())
            manager->selectNode(node->id);
        loadSettings(*node);
        finishJob("completed");
        result_.reset();
    }
    void CollisionPanel::resetSurfaceSettings() {
        const auto previous = readInputs();
        settings_ = {};
        settings_.units_per_meter = previous.units_per_meter;
        settings_.applyPreset(core::CollisionPreset::Outdoor);
        settings_.seed = previous.seed;
        settings_.voxel_size = previous.voxel_size;
        settings_.auto_resolution = previous.auto_resolution;
        settings_.floor_below_seed = true;
        world_seed_ = settings_.seed;
        writeInputs();
    }
    void CollisionPanel::syncGenerationControls() {
        const auto filter = value("filter-mode");
        document_->SetClass("collision-auto-resolution", checked("auto-resolution"));
        document_->SetClass("collision-strict", filter == "3");
        document_->SetClass("collision-cleanup", filter == "1" || filter == "2");
        document_->SetClass("collision-seeded", filter == "2" || filter == "3");
        host_.markContentDirty();
    }
    void CollisionPanel::buildSurface() {
        if (busy_)
            throw std::runtime_error("Wait for generation to finish");
        auto* node = scene_->getNodeByUuid(target_);
        if (!node || !node->collision || node->locked.get())
            throw std::runtime_error("Select an unlocked collision node");
        auto volume = node->collision;
        clearError();
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
        document_->GetElementById("collision-scroll")->SetScrollTop(0);
        done_ = false;
        job_started_ = std::chrono::steady_clock::now();
        progress_updated_ = job_started_;
        next_progress_ui_ = {};
        job_outcome_ = "running";
        job_graph_epoch_ = scene_->graphEpoch();
        job_title_ = LOC("collision.building_surface");
        job_context_ = std::format("Surface resolution {}×; source collision cell {:.4g} scene units.", display.surface_stride, volume->settings.voxel_size);
        progress_value_ = 0;
        progress_step_ = 0;
        progress_total_ = 3;
        worker_ = std::jthread([this, volume, display, shape, transform, generation, epoch, target](std::stop_token stop) {
            try {
                const auto begin = std::chrono::steady_clock::now();
                auto report = [&](std::string text) {
                    std::lock_guard lock(mutex_);
                    progress_updated_ = std::chrono::steady_clock::now();
                    progress_ = std::move(text);
                    progress_step_ = 1;
                    progress_value_ = 1.0f / 3;
                };
                report("Preparing surface resolution");
                auto surface = display.surface_stride == 1 ? volume->mesh(shape, stop, report) : volume->coarsened(display.surface_stride, stop).mesh(shape, stop, report);
                {
                    std::lock_guard lock(mutex_);
                    progress_ = "Computing surface normals";
                    progress_updated_ = std::chrono::steady_clock::now();
                    progress_step_ = 2;
                    progress_value_ = 2.0f / 3;
                }
                auto mesh = makeMesh(std::move(surface), stop);
                auto stats = std::format("{:.2f}s surface; {:.3g} scene-unit surface cells; original voxels retained for Walk",
                                         std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(),
                                         volume->settings.voxel_size * display.surface_stride);
                std::lock_guard lock(mutex_);
                result_ = Result{volume, mesh, transform, {}, target, generation, epoch, display, stats, true};
            } catch (const std::exception& error) {
                std::lock_guard lock(mutex_);
                error_ = error.what();
            }
            done_.store(true, std::memory_order_release);
            python::request_redraw();
        });
    }
    void CollisionPanel::ProcessEvent(Rml::Event& event) {
        if (updating_)
            return;
        try {
            if (scrubEvent(event))
                return;
        } catch (const std::exception& error) {
            showError(error.what());
            return;
        }
        if (event.GetType() != "click" && event.GetType() != "change")
            return;
        auto* el = event.GetTargetElement();
        while (el && el->GetId().empty())
            el = el->GetParentNode();
        if (!el)
            return;
        auto id = el->GetId();
        if (updating_)
            return;
        try {
            if (event.GetType() == "click" && el->HasAttribute("data-collision-section")) {
                const auto section = el->GetAttribute<Rml::String>("data-collision-section", "");
                if (auto* content = document_->GetElementById(section)) {
                    const bool expanded = content->IsClassSet("collapsed");
                    content->SetClass("collapsed", !expanded);
                    el->SetClass("is-expanded", expanded);
                    el->SetAttribute("aria-expanded", expanded ? "true" : "false");
                    host_.markContentDirty();
                }
                return;
            }
            if (id == "copy-error") {
                SDL_SetClipboardText(error_report_.c_str());
                return;
            }
            if (event.GetType() == "change") {
                applyControlChange(id);
                return;
            }
            if (id == "generate")
                start(false);
            else if (id == "generate-new")
                start(true);
            else if (id == "cancel") {
                worker_.request_stop();
                std::lock_guard lock(mutex_);
                progress_ = "Cancelling — waiting for the current batch to finish";
                progress_updated_ = std::chrono::steady_clock::now();
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
            } else if (id == "show-voxels" || id == "show-mesh") {
                if (busy_)
                    throw std::runtime_error("Wait for generation before changing visualization");
                auto* node = scene_->getNodeByUuid(target_);
                if (!node || !node->collision)
                    throw std::runtime_error("Select a collision node first");
                if (node->locked.get())
                    throw std::runtime_error("Collision target is locked");
                if (id == "show-mesh") {
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
            showError(error.what());
        }
    }
    void CollisionPanel::draw(const PanelDrawContext& ctx) {
        if (prepare(ctx)) {
            host_.draw(ctx);
        }
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
            else if (r.mode != PanelDirectRenderMode::Measure) {
                host_.drawDirect(r.x, r.y, r.width, r.height);
            }
        }
        host_.setInput(nullptr);
        host_.setInputClipY(-1, -1);
        host_.setForcedHeight(0);
        return {.handled = true, .height = host_.getContentHeight()};
    }
} // namespace lfs::vis::gui
