/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "modifier_evaluation_worker.hpp"

#include "core/assert.hpp"
#include "core/logger.hpp"
#include "core/nodes/events.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "visualizer/core/training_manager.hpp"
#include "visualizer/nodes/camera_nodes.hpp"
#include "visualizer/nodes/node_animation.hpp"

#include <algorithm>
#include <format>
#include <ranges>

namespace lfs::vis {

    ModifierEvaluationRequest ModifierManager::captureRequest() const {
        ModifierEvaluationRequest request;
        request.generation = output_generation_;
        request.requested_at = std::chrono::steady_clock::now();
        request.source_generation = source_generation_;
        request.seconds = animationTime();
        const auto* controller = sequencer();
        request.frames_per_second = export_time_ ? export_fps_ : controller ? controller->framesPerSecond()
                                                                            : 24.0f;
        request.inputs_ready = std::make_shared<core::TensorCompletion>();
        const auto include = [&](const core::Tensor& tensor) {
            if (tensor.is_valid())
                request.inputs_ready->include(tensor);
        };
        for (const auto& [id, tree] : trees_) {
            auto animated = *tree;
            applyNodeAnimation(animated, controller ? controller->timeline().animationClip() : nullptr, request.seconds);
            request.trees[id] = animated.to_json();
        }
        const auto& scene = scene_manager_->getScene();
        const bool dataset = scene_manager_->getContentType() == SceneManager::ContentType::Dataset;
        const core::Uuid training_model = scene.getTrainingModelNodeUuid();
        request.cameras = captureNodeCameras(scene);
        for (const auto* node : scene.getNodes()) {
            if (!node || (!node->model && !node->point_cloud && !node->mesh))
                continue;
            ModifierObjectSnapshot object;
            object.uuid = node->uuid;
            object.name = node->name;
            object.type = node->type;
            object.world = scene.getWorldTransform(node->id);
            if (node->model) {
                object.splats = std::make_shared<core::SplatData>(node->model->readOnlySnapshot());
                include(object.splats->means_raw());
                include(object.splats->sh0_raw());
                include(object.splats->shN_raw());
                include(object.splats->shN_value_bounds());
                include(object.splats->scaling_raw());
                include(object.splats->rotation_raw());
                include(object.splats->opacity_raw());
                include(object.splats->deleted());
            }
            if (node->point_cloud) {
                object.points = std::make_shared<core::PointCloud>(*node->point_cloud);
                include(object.points->means);
                include(object.points->colors);
                include(object.points->normals);
            }
            if (node->mesh) {
                object.mesh = node->mesh->readOnlySnapshot();
                include(object.mesh->vertices);
                include(object.mesh->indices);
                include(object.mesh->normals);
                include(object.mesh->tangents);
                include(object.mesh->colors);
                include(object.mesh->texcoords);
            }
            if (const auto* modifiers = stack(node->uuid)) {
                object.stack = *modifiers;
                if ((!dataset || node->uuid == training_model) &&
                    (!animation_only_request_ || timeDependent(node->uuid)))
                    request.targets.push_back(node->uuid);
            }
            request.objects.push_back(std::move(object));
        }
        return request;
    }

    void ModifierManager::installReady() {
        auto ready = worker_->takeReady();
        if (!ready)
            return;
        if (ready->generation != output_generation_) {
            worker_->retire(std::move(*ready));
            return;
        }
        LFS_ASSERT_MSG(std::this_thread::get_id() == viewer_thread_,
                       std::format("Modifier results must be installed on the viewer thread (current={}, viewer={})",
                                   std::hash<std::thread::id>{}(std::this_thread::get_id()),
                                   std::hash<std::thread::id>{}(viewer_thread_)));
        LFS_ASSERT_MSG(!ready->ready || ready->ready->ready(),
                       std::format("Published modifier result must already be fence-complete (generation={})",
                                   ready->generation));
        // The worker already waited for this marker. This never waits for an
        // evaluation still running, or for unrelated rendering/device work.
        if (ready->ready)
            ready->ready->wait();
        auto& scene = scene_manager_->getScene();
        ModifierWorkerResult retired;
        for (auto& [uuid, result] : ready->hosts) {
            const auto* node = scene.getNodeByUuid(uuid);
            if (!node)
                continue;
            auto& state = runtime_[uuid];
            retired.hosts.emplace(uuid, ModifierHostResult{
                                            .uuid = uuid,
                                            .evaluation = std::move(state.evaluation),
                                            .splats = node->evaluated_model,
                                            .points = node->evaluated_point_cloud,
                                            .mesh = node->evaluated_mesh,
                                            .previews = std::move(state.previews)});
            state.evaluation = std::move(result.evaluation);
            state.previews = std::move(result.previews);
            if (!result.enabled && state.evaluation.ok)
                scene.clearNodeEvaluatedPayload(node->id);
            else if (state.evaluation.ok &&
                     (node->evaluated_model != result.splats || node->evaluated_point_cloud != result.points ||
                      node->evaluated_mesh != result.mesh))
                scene.setNodeEvaluatedPayload(node->id, std::move(result.splats),
                                              std::move(result.points), std::move(result.mesh));
            // Failed and pending requests leave the last successful payload visible.
            lfs::nodes::EvaluationEvent{.phase = state.evaluation.ok ? "completed" : "failed", .generation = ready->generation, .target = uuid.to_string(), .ok = state.evaluation.ok}.emit();
        }
        ++result_generation_;
        ++installed_count_;
        last_installed_generation_ = ready->generation;
        if (measure_canvas_ && evaluation_latency_ms_.size() < 4096)
            evaluation_latency_ms_.push_back(std::chrono::duration<double, std::milli>(
                                                 std::chrono::steady_clock::now() - ready->requested_at)
                                                 .count());
        last_scene_generation_ = scene.renderGeneration();
        worker_->retire(std::move(retired));
        worker_->retire(std::move(*ready));
    }

    void ModifierManager::tick() {
        updateAnimationTime();
        auto& scene = scene_manager_->getScene();
        const auto content = scene_manager_->getContentType();
        if (content != SceneManager::ContentType::SplatFiles && content != SceneManager::ContentType::Dataset) {
            if (worker_->progress().busy)
                worker_->invalidate(++output_generation_);
            clearEvaluatedPayloads();
            last_scene_generation_ = scene.renderGeneration();
            training_suspended_ = false;
            return;
        }
        const auto* trainer_manager = scene_manager_->getTrainerManager();
        const bool training_running = content == SceneManager::ContentType::Dataset &&
                                      trainer_manager && trainer_manager->isRunning();
        if (training_running) {
            if (!training_suspended_) {
                training_suspended_ = true;
                ++generation_;
                worker_->invalidate(++output_generation_);
                clearEvaluatedPayloads();
            }
            // Training publishes a new stored payload generation every iteration.
            // Consume it without scheduling modifier work.
            last_scene_generation_ = scene.renderGeneration();
            return;
        }
        if (training_suspended_) {
            training_suspended_ = false;
            ++generation_;
            ++source_generation_;
            markDirty();
            last_scene_generation_ = scene.renderGeneration();
        }
        if (last_scene_generation_ != scene.renderGeneration()) {
            ++source_generation_;
            markDirty();
            last_scene_generation_ = scene.renderGeneration();
        }
        installReady();
        const auto progress = worker_->progress();
        if (progress.busy && !progress.node.empty() &&
            (progress_event_generation_ != progress.generation || progress_event_node_ != progress.node)) {
            progress_event_generation_ = progress.generation;
            progress_event_node_ = progress.node;
            lfs::nodes::EvaluationEvent{.phase = "progress", .generation = progress.generation, .target = progress.host.to_string(), .node = progress.node, .label = progress.label, .completed = progress.completed, .total = progress.total}.emit();
        }
        const bool dirty = std::ranges::any_of(runtime_, [](const auto& entry) { return entry.second.dirty; });
        if (!dirty || requested_generation_ == output_generation_)
            return;
        auto request = captureRequest();
        requested_generation_ = request.generation;
        for (auto& [_, state] : runtime_)
            state.dirty = false;
        worker_->submit(std::move(request));
        lfs::nodes::EvaluationEvent{.phase = "started", .generation = requested_generation_}.emit();
    }

    ModifierEvaluation ModifierManager::evaluate(const core::Uuid& node_uuid) {
        tick();
        if (requested_generation_ == output_generation_ && worker_->progress().busy) {
            // Explicit synchronous Python/API request only; canvas code never calls this.
            worker_->wait(requested_generation_);
            installReady();
        }
        const auto* result = lastResult(node_uuid);
        return result ? *result : ModifierEvaluation{};
    }

    ModifierHostResult ModifierManager::evaluateForApply(const core::Uuid& node_uuid, const size_t last_modifier) {
        // Apply is an explicit synchronous bake, but its tensor work still runs
        // on the same worker queue, never on the viewer/render queue.
        worker_->invalidate(++output_generation_);
        auto request = captureRequest();
        request.targets = {node_uuid};
        request.bake = true;
        const auto object = std::ranges::find(request.objects, node_uuid, &ModifierObjectSnapshot::uuid);
        if (object == request.objects.end())
            return {};
        object->stack.modifiers.resize(std::min(last_modifier + 1, object->stack.modifiers.size()));
        worker_->submit(std::move(request));
        worker_->wait(output_generation_);
        auto result = worker_->takeReady();
        if (result) {
            const auto host = result->hosts.find(node_uuid);
            if (host != result->hosts.end())
                return std::move(host->second);
        }
        return ModifierHostResult{.uuid = node_uuid, .evaluation = {.ok = false, .errors = {{"Graph", "Modifier evaluation was cancelled"}}}};
    }

    const ModifierEvaluation* ModifierManager::lastResult(const core::Uuid& node_uuid) const {
        const auto found = runtime_.find(node_uuid);
        return found == runtime_.end() ? nullptr : &found->second.evaluation;
    }

    std::optional<lfs::nodes::Geometry> ModifierManager::evaluated(const core::Uuid& node_uuid) const {
        const auto* result = lastResult(node_uuid);
        return result && result->ok ? std::optional{result->geometry} : std::nullopt;
    }

    std::optional<core::Tensor> ModifierManager::selectionPreview(const core::Uuid& node_uuid,
                                                                  const std::string_view modifier_uuid,
                                                                  const std::string_view node_name) {
        const auto found = runtime_.find(node_uuid);
        if (found == runtime_.end())
            return std::nullopt;
        const auto preview = found->second.previews.find(std::string(modifier_uuid) + "/" + std::string(node_name));
        return preview == found->second.previews.end() ? std::nullopt : std::optional{preview->second};
    }

    ModifierWorkerProgress ModifierManager::progress() const {
        return worker_->progress();
    }

    void ModifierManager::recordCanvasFrame(const double milliseconds) {
        if (measure_canvas_)
            canvas_work_ms_ = canvas_work_ms_.value_or(0.0) + milliseconds;
    }

    nlohmann::json ModifierManager::performance(const bool reset) {
        auto result = worker_->performance(reset);
        result["canvas_frame_ms"] = canvas_frame_ms_;
        result["viewer_frame_ms"] = viewer_frame_ms_;
        result["busy_viewport_frame_ms"] = busy_viewport_frame_ms_;
        result["idle_viewport_frame_ms"] = idle_viewport_frame_ms_;
        result["installed"] = installed_count_;
        result["evaluation_latency_ms"] = evaluation_latency_ms_;
        result["result_generation"] = result_generation_;
        result["busy"] = worker_->progress().busy;
        if (reset) {
            installed_count_ = 0;
            evaluation_latency_ms_.clear();
            canvas_frame_ms_.clear();
            canvas_work_ms_.reset();
            viewer_frame_ms_.clear();
            busy_viewport_frame_ms_.clear();
            idle_viewport_frame_ms_.clear();
            measure_canvas_ = true;
        }
        return result;
    }

    void ModifierManager::recordViewerFrame(const double milliseconds, const bool rendered_viewport) {
        if (!measure_canvas_)
            return;
        if (rendered_viewport && milliseconds > 16.0) {
            const auto current = worker_->progress();
            LOG_INFO("Node editor profiling: viewport {:.2f} ms, busy={}, node={}, result={}",
                     milliseconds, current.busy, current.label, result_generation_);
        }
        if (canvas_work_ms_) {
            if (canvas_frame_ms_.size() < 4096)
                canvas_frame_ms_.push_back(*canvas_work_ms_);
            canvas_work_ms_.reset();
        }
        if (viewer_frame_ms_.size() < 4096)
            viewer_frame_ms_.push_back(milliseconds);
        if (rendered_viewport) {
            auto& frames = worker_->progress().busy ? busy_viewport_frame_ms_ : idle_viewport_frame_ms_;
            if (frames.size() < 4096)
                frames.push_back(milliseconds);
        }
    }

} // namespace lfs::vis
