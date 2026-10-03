/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "modifier_evaluation_worker.hpp"

#include "core/logger.hpp"
#include "core/services.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"
#include "window/window_manager.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <ranges>
#include <unordered_set>

namespace lfs::vis {
    namespace {
        using namespace lfs::nodes;

        core::Tensor copied(const core::Tensor& tensor) {
            return tensor.is_valid() ? tensor.clone() : core::Tensor{};
        }

        std::shared_ptr<core::MeshData> copyMesh(const core::MeshData& source) {
            auto mesh = std::make_shared<core::MeshData>();
            mesh->vertices = copied(source.vertices);
            mesh->indices = copied(source.indices);
            mesh->normals = copied(source.normals);
            mesh->tangents = copied(source.tangents);
            mesh->texcoords = copied(source.texcoords);
            mesh->colors = copied(source.colors);
            mesh->materials = source.materials;
            mesh->submeshes = source.submeshes;
            mesh->texture_images = source.texture_images;
            return mesh;
        }

        core::Tensor matrix_tensor(const glm::mat3& matrix, core::Device device) {
            std::vector<float> values;
            values.reserve(9);
            for (int column = 0; column < 3; ++column)
                for (int row = 0; row < 3; ++row)
                    values.push_back(matrix[column][row]);
            return core::Tensor::from_vector(values, {3, 3}, core::Device::CPU).to(device);
        }

        core::Tensor vector_tensor(const glm::vec3 value, core::Device device) {
            return core::Tensor::from_vector({value.x, value.y, value.z}, {1, 3}, core::Device::CPU)
                .to(device);
        }

        Geometry transform_geometry(Geometry geometry, const glm::mat4& matrix) {
            if (geometry.splats) {
                auto attributes = geometry.splats->attributes;
                auto data = lfs::nodes::splat_data_from_geometry(geometry);
                core::transform(*data, matrix);
                geometry.splats = lfs::nodes::geometry_from_splat_data(*data).splats;
                geometry.splats->attributes = std::move(attributes);
            }
            const auto transform_positions = [&](const core::Tensor& positions) {
                return positions.matmul(matrix_tensor(glm::mat3(matrix), positions.device())) +
                       vector_tensor(glm::vec3(matrix[3]), positions.device());
            };
            if (geometry.points)
                geometry.points->positions = transform_positions(geometry.points->positions);
            if (geometry.mesh && geometry.mesh->mesh) {
                const auto& source = *geometry.mesh->mesh;
                auto result = std::make_shared<core::MeshData>();
                result->vertices = transform_positions(source.vertices);
                result->indices = source.indices;
                result->normals = source.normals;
                result->tangents = source.tangents;
                result->texcoords = source.texcoords;
                result->colors = source.colors;
                result->materials = source.materials;
                result->submeshes = source.submeshes;
                result->texture_images = source.texture_images;
                geometry.mesh = lfs::nodes::MeshComponent{std::move(result), geometry.mesh->textures};
            }
            return geometry;
        }

        void preparePayload(const ModifierObjectSnapshot& node, ModifierHostResult& result) {
            std::shared_ptr<core::SplatData> splats;
            std::shared_ptr<core::PointCloud> points;
            std::shared_ptr<core::MeshData> mesh;
            if (node.type == core::NodeType::SPLAT) {
                Geometry output = result.evaluation.geometry;
                auto backend = output.splats
                                   ? core::gpu_backend_of(output.splats->means)
                                   : std::optional<core::GpuBackend>{};
                if (!backend && node.splats)
                    backend = core::gpu_backend_of(node.splats->means_raw());
                if (!backend) {
                    const auto configured = core::configured_gpu_backend();
                    if (core::gpu_backend_available(configured))
                        backend = configured;
                    else
                        for (const auto candidate : core::kGpuBackends)
                            if (core::gpu_backend_available(candidate)) {
                                backend = candidate;
                                break;
                            }
                }
                if (!output.splats) {
                    if (backend) {
                        const core::GpuBackendScope backend_scope(*backend);
                        splats = std::make_shared<core::SplatData>(
                            0,
                            core::Tensor::zeros({0, 3}, core::Device::GPU),
                            core::Tensor::zeros({0, 1, 3}, core::Device::GPU),
                            core::Tensor::zeros({0}, core::Device::GPU),
                            core::Tensor::zeros({0, 3}, core::Device::GPU),
                            core::Tensor::zeros({0, 4}, core::Device::GPU),
                            core::Tensor::zeros({0, 1}, core::Device::GPU),
                            1.0f,
                            core::SplatData::ShNLayout::Swizzled);
                    } else {
                        splats = std::make_shared<core::SplatData>(
                            0,
                            core::Tensor::zeros({0, 3}, core::Device::CPU),
                            core::Tensor::zeros({0, 1, 3}, core::Device::CPU),
                            core::Tensor::zeros({0}, core::Device::CPU),
                            core::Tensor::zeros({0, 3}, core::Device::CPU),
                            core::Tensor::zeros({0, 4}, core::Device::CPU),
                            core::Tensor::zeros({0, 1}, core::Device::CPU),
                            1.0f,
                            core::SplatData::ShNLayout::Swizzled);
                    }
                } else {
                    auto& component = *output.splats;
                    if (backend) {
                        const core::GpuBackendScope backend_scope(*backend);
                        if (component.means.device() == core::Device::CPU) {
                            component.means = component.means.to(core::Device::GPU);
                            component.sh0 = component.sh0.to(core::Device::GPU);
                            component.shN = component.shN.to(core::Device::GPU);
                            component.scaling = component.scaling.to(core::Device::GPU);
                            component.rotation = component.rotation.to(core::Device::GPU);
                            component.opacity = component.opacity.to(core::Device::GPU);
                        }
                        splats = std::shared_ptr<core::SplatData>(
                            lfs::nodes::splat_data_from_geometry(output).release());
                    } else {
                        splats = std::shared_ptr<core::SplatData>(
                            lfs::nodes::splat_data_from_geometry(output).release());
                    }
                }
            } else if (node.type == core::NodeType::POINTCLOUD && result.evaluation.geometry.points) {
                points = std::make_shared<core::PointCloud>(
                    lfs::nodes::point_cloud_from_geometry(result.evaluation.geometry));
            } else if (node.type == core::NodeType::MESH && result.evaluation.geometry.mesh) {
                mesh = std::const_pointer_cast<core::MeshData>(result.evaluation.geometry.mesh->mesh);
            }
            result.splats = std::move(splats);
            result.points = std::move(points);
            result.mesh = std::move(mesh);
            // Published buffers must not alias the worker cache. In particular,
            // Metal's storage readiness includes readers: reusing displayed
            // storage as an evaluation input would stall the renderer on it.
            if (result.splats)
                result.splats = std::make_shared<core::SplatData>(result.splats->clone());
            if (result.points) {
                result.points->means = copied(result.points->means);
                result.points->colors = copied(result.points->colors);
                result.points->normals = copied(result.points->normals);
            }
            if (result.mesh)
                result.mesh = copyMesh(*result.mesh);
        }

        Geometry storedGeometry(const ModifierObjectSnapshot& object) {
            if (object.splats)
                return geometry_from_splat_data(object.splats->clone());
            if (object.points) {
                auto points = *object.points;
                points.means = copied(points.means);
                points.colors = copied(points.colors);
                points.normals = copied(points.normals);
                return Geometry{std::nullopt, PointsComponent{points.means, points.colors, {}}, std::nullopt};
            }
            if (object.mesh)
                return geometry_from_mesh(object.mesh);
            return {};
        }

        std::optional<core::GpuBackend> backendOf(const ModifierEvaluationRequest& request) {
            for (const auto& object : request.objects) {
                const core::Tensor* tensor = object.splats   ? &object.splats->means_raw()
                                             : object.points ? &object.points->means
                                             : object.mesh   ? &object.mesh->vertices
                                                             : nullptr;
                if (tensor)
                    if (const auto backend = core::gpu_backend_of(*tensor))
                        return backend;
            }
            const auto backend = core::default_gpu_backend();
            return core::gpu_backend_available(backend) ? std::optional{backend} : std::nullopt;
        }

        std::optional<FieldContext> fieldContext(const Geometry& geometry, const std::uint64_t identity) {
            FieldContext context;
            context.identity = identity;
            if (geometry.splats) {
                context.domain = Domain::Splat;
                context.splats = &*geometry.splats;
            } else if (geometry.points) {
                context.domain = Domain::Point;
                context.points = &*geometry.points;
            } else if (geometry.mesh) {
                context.domain = Domain::Vertex;
                context.mesh = &*geometry.mesh;
            } else {
                return std::nullopt;
            }
            return context;
        }

        void selectionPreviews(const NodeTree& tree, const EvalCache& cache,
                               const std::string& modifier, ModifierHostResult& result,
                               const std::function<bool()>& cancelled) {
            const auto context = fieldContext(result.evaluation.geometry, result.evaluation.geometry.splats
                                                                              ? result.evaluation.geometry.splats->means.debug_id()
                                                                              : 0);
            if (!context)
                return;
            FieldMemo memo;
            std::unordered_map<std::string, std::pair<core::Tensor, double>> masks;
            for (const auto& node : tree.nodes) {
                if (cancelled())
                    return;
                const auto type = tree.registry().find(node.type_id);
                if (!type)
                    continue;
                std::string source = node.name;
                std::string socket;
                const auto output = std::ranges::find(type->outputs, "Selection", &SocketDecl::identifier);
                if (output != type->outputs.end()) {
                    socket = output->identifier;
                } else {
                    const auto input = std::ranges::find(type->inputs, "Selection", &SocketDecl::identifier);
                    if (input == type->inputs.end())
                        continue;
                    const auto link = std::ranges::find_if(tree.links, [&](const Link& item) {
                        return item.to_node == node.name && item.to_socket == input->identifier;
                    });
                    if (link == tree.links.end())
                        continue;
                    source = link->from_node;
                    socket = link->from_socket;
                }
                const auto cached = cache.nodes.find(source);
                if (cached == cache.nodes.end())
                    continue;
                const auto value = cached->second.outputs.find(socket);
                if (value == cached->second.outputs.end())
                    continue;
                const std::string key = source + "/" + socket;
                auto found = masks.find(key);
                if (found == masks.end()) {
                    const auto* field = value->second.get_if<Field>();
                    auto mask = (field ? convert_field(*field, FLOAT_SOCKET)
                                       : constant_field(value->second, FLOAT_SOCKET))
                                    .evaluate(*context, memo)
                                    .ge(0.5f);
                    const auto count = mask.numel();
                    const double share = count ? static_cast<double>(mask.count_nonzero()) / count : 0.0;
                    found = masks.emplace(key, std::pair{std::move(mask), share}).first;
                }
                result.previews[modifier + "/" + node.name] = found->second.first;
                if (auto status = result.evaluation.nodes.find(modifier + "/" + node.name);
                    status != result.evaluation.nodes.end())
                    status->second.selected_share = found->second.second;
            }
        }

        class SnapshotHost final : public EvalHost {
        public:
            SnapshotHost(const ModifierEvaluationRequest& request, const NodeTypeRegistry& registry,
                         std::unordered_map<std::string, EvalCache>& caches,
                         std::unordered_map<core::Uuid, Geometry>& sources,
                         const std::unordered_map<core::Uuid, ModifierHostResult>& previous,
                         const EvalControl& control,
                         std::function<void(const core::Uuid&, const std::string&, const NodeTree&)> stack_started)
                : request_(request), registry_(registry), caches_(caches), sources_(sources), previous_(previous), control_(control), stack_started_(std::move(stack_started)) {}

            std::uint64_t generation() const override { return request_.generation; }

            std::optional<Geometry> object_geometry(const std::string_view name, const TransformSpace space) override {
                const auto target = std::ranges::find(request_.objects, name, &ModifierObjectSnapshot::name);
                if (target == request_.objects.end())
                    throw NodeError(std::format("Object '{}' does not exist", name));
                auto& result = run(*target);
                if (!result.evaluation.ok)
                    throw NodeError("Object Info target modifier evaluation failed");
                auto geometry = result.evaluation.geometry;
                if (space == TransformSpace::Relative)
                    geometry = transform_geometry(std::move(geometry), glm::inverse(current_world_) * target->world);
                return geometry;
            }

            ModifierHostResult& run(const ModifierObjectSnapshot& object) {
                if (active_.contains(object.uuid))
                    throw NodeError("Object Info dependency cycle");
                if (auto found = results.find(object.uuid); found != results.end())
                    return found->second;
                active_.insert(object.uuid);
                const auto previous_world = current_world_;
                current_world_ = object.world;
                auto& result = results[object.uuid];
                result.uuid = object.uuid;
                const auto start = std::chrono::steady_clock::now();
                Geometry geometry = sources_.at(object.uuid);
                std::uint64_t input_generation = request_.source_generation;
                for (const auto& modifier : object.stack.modifiers) {
                    if (control_.cancelled())
                        break;
                    if (!modifier.enabled || !modifier.show_viewport)
                        continue;
                    result.enabled = true;
                    const auto source = request_.trees.find(modifier.tree_uuid);
                    if (source == request_.trees.end()) {
                        result.evaluation.ok = false;
                        result.evaluation.errors[modifier.name] = "Referenced node graph is missing";
                        break;
                    }
                    auto tree = NodeTree::from_json(source->second, registry_);
                    for (const auto& [name, selection] : modifier.stored_selections)
                        if (auto* node = tree.find_node(name))
                            node->properties.update(selection);
                    stack_started_(object.uuid, modifier.uuid, tree);
                    auto& cache = caches_[modifier.uuid];
                    std::unordered_set<std::string_view> node_names;
                    for (const auto& node : tree.nodes)
                        node_names.insert(node.name);
                    std::erase_if(cache.nodes, [&](const auto& entry) { return !node_names.contains(entry.first); });
                    auto evaluated = lfs::nodes::evaluate(tree,
                                                          {.geometry = geometry, .interface_overrides = modifier.input_overrides, .geometry_generation = input_generation}, this, &cache, control_);
                    result.evaluation.time_ms.insert(evaluated.time_ms.begin(), evaluated.time_ms.end());
                    for (auto& [name, status] : evaluated.nodes)
                        result.evaluation.nodes[modifier.uuid + "/" + name] = std::move(status);
                    if (evaluated.cancelled)
                        break;
                    if (!evaluated.ok) {
                        result.evaluation.ok = false;
                        for (const auto& [name, error] : evaluated.errors) {
                            result.evaluation.errors[modifier.name + "/" + name] = error;
                            auto& status = result.evaluation.nodes[modifier.uuid + "/" + name];
                            if (const auto time = evaluated.time_ms.find(name); time != evaluated.time_ms.end())
                                status.time_ms = time->second;
                        }
                        break;
                    }
                    geometry = std::move(evaluated.geometry);
                    if (const auto output = cache.nodes.find(tree.output_node().name); output != cache.nodes.end())
                        input_generation = output->second.key;
                }
                result.evaluation.geometry = std::move(geometry);
                result.output_key = input_generation;
                const auto previous = previous_.find(object.uuid);
                const bool cached = previous != previous_.end() && previous->second.enabled == result.enabled &&
                                    previous->second.output_key == result.output_key &&
                                    (!request_.bake || previous->second.splats || previous->second.points || previous->second.mesh) &&
                                    std::ranges::all_of(result.evaluation.nodes, [](const auto& entry) { return entry.second.cached; });
                if (!control_.cancelled() && result.evaluation.ok) {
                    if (cached) {
                        result.splats = previous->second.splats;
                        result.points = previous->second.points;
                        result.mesh = previous->second.mesh;
                        result.previews = previous->second.previews;
                        result.evaluation.unchanged = true;
                        for (auto& [name, status] : result.evaluation.nodes)
                            if (const auto old = previous->second.evaluation.nodes.find(name); old != previous->second.evaluation.nodes.end())
                                status.selected_share = old->second.selected_share;
                    } else {
                        for (const auto& modifier : object.stack.modifiers) {
                            if (!modifier.enabled || !modifier.show_viewport)
                                continue;
                            const auto source = request_.trees.find(modifier.tree_uuid);
                            if (source != request_.trees.end())
                                selectionPreviews(NodeTree::from_json(source->second, registry_), caches_[modifier.uuid],
                                                  modifier.uuid, result, control_.cancelled);
                        }
                        if ((result.enabled || request_.bake) && !control_.cancelled())
                            preparePayload(object, result);
                    }
                }
                result.evaluation.total_time_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                active_.erase(object.uuid);
                current_world_ = previous_world;
                return result;
            }

            std::unordered_map<core::Uuid, ModifierHostResult> results;

        private:
            const ModifierEvaluationRequest& request_;
            const NodeTypeRegistry& registry_;
            std::unordered_map<std::string, EvalCache>& caches_;
            std::unordered_map<core::Uuid, Geometry>& sources_;
            const std::unordered_map<core::Uuid, ModifierHostResult>& previous_;
            const EvalControl& control_;
            std::function<void(const core::Uuid&, const std::string&, const NodeTree&)> stack_started_;
            std::unordered_set<core::Uuid> active_;
            glm::mat4 current_world_{1.0f};
        };
    } // namespace

    ModifierEvaluationWorker::ModifierEvaluationWorker(const lfs::nodes::NodeTypeRegistry& registry)
        : registry_(registry), thread_([this](std::stop_token stop) { run(stop); }) {}

    ModifierEvaluationWorker::~ModifierEvaluationWorker() {
        thread_.request_stop();
        generation_.fetch_add(1);
        changed_.notify_all();
    }

    void ModifierEvaluationWorker::invalidate(const std::uint64_t generation) {
        generation_.store(generation, std::memory_order_release);
        std::lock_guard lock(mutex_);
        pending_.reset();
    }

    void ModifierEvaluationWorker::submit(ModifierEvaluationRequest request) {
        std::lock_guard lock(mutex_);
        generation_.store(request.generation, std::memory_order_release);
        pending_ = std::move(request);
        ++requests_;
        changed_.notify_all();
    }

    std::optional<ModifierWorkerResult> ModifierEvaluationWorker::takeReady() {
        std::lock_guard lock(mutex_);
        if (!ready_)
            return std::nullopt;
        auto result = std::move(ready_);
        ready_.reset();
        return result;
    }

    void ModifierEvaluationWorker::retire(ModifierWorkerResult result) {
        std::lock_guard lock(mutex_);
        retired_.push_back(std::move(result));
        changed_.notify_all();
    }

    void ModifierEvaluationWorker::wait(const std::uint64_t generation) {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&] { return finished_generation_ >= generation; });
    }

    ModifierWorkerProgress ModifierEvaluationWorker::progress() const {
        std::lock_guard lock(mutex_);
        auto result = progress_;
        result.busy |= pending_.has_value();
        return result;
    }

    nlohmann::json ModifierEvaluationWorker::performance(const bool reset) {
        std::lock_guard lock(mutex_);
        nlohmann::json result{{"requests", requests_}, {"evaluations", evaluations_}, {"discarded", discarded_}, {"node_runs", node_runs_}};
        if (reset) {
            requests_ = 0;
            evaluations_ = 0;
            discarded_ = 0;
            node_runs_.clear();
        }
        return result;
    }

    void ModifierEvaluationWorker::run(const std::stop_token stop) {
        while (true) {
            ModifierEvaluationRequest request;
            std::vector<ModifierWorkerResult> retired;
            bool evaluate_request = false;
            {
                std::unique_lock lock(mutex_);
                changed_.wait(lock, stop, [&] { return pending_.has_value() || !retired_.empty(); });
                retired.swap(retired_);
                if (pending_ && !stop.stop_requested()) {
                    request = std::move(*pending_);
                    pending_.reset();
                    progress_ = {.busy = true, .generation = request.generation, .started_at = std::chrono::steady_clock::now()};
                    ++evaluations_;
                    evaluate_request = true;
                }
            }
            // Releasing published storage can wait for the renderer's consumer
            // queue on MoltenVK. Do not do this on the viewer or under mutex_.
            retired.clear();
            if (stop.stop_requested())
                return;
            if (!evaluate_request)
                continue;
            auto result = evaluate(request);
            std::optional<ModifierWorkerResult> superseded;
            {
                std::lock_guard lock(mutex_);
                finished_generation_ = request.generation;
                progress_.busy = false;
                if (result.cancelled || request.generation != generation_.load(std::memory_order_acquire))
                    ++discarded_;
                else {
                    superseded = std::move(ready_);
                    ready_ = std::move(result);
                }
                changed_.notify_all();
            }
            superseded.reset();
            if (auto* window = services().windowOrNull())
                window->wakeEventLoop();
        }
    }

    ModifierWorkerResult ModifierEvaluationWorker::evaluate(const ModifierEvaluationRequest& request) {
        ModifierWorkerResult result;
        result.generation = request.generation;
        result.requested_at = request.requested_at;
        const auto cancelled = [&] { return request.generation != generation_.load(std::memory_order_acquire); };
        const auto backend = backendOf(request);
        try {
            if (backend && (!queue_ || queue_->backend() != *backend))
                queue_ = std::make_unique<core::TensorWorkQueue>(*backend);
            std::optional<core::TensorWorkQueue::Scope> scope;
            if (backend)
                scope.emplace(*queue_);
            // Wait only for the captured producers, and only on this worker.
            if (request.inputs_ready)
                request.inputs_ready->wait();
            std::unordered_set<std::string_view> modifier_ids;
            for (const auto& object : request.objects)
                for (const auto& modifier : object.stack.modifiers)
                    modifier_ids.insert(modifier.uuid);
            std::erase_if(caches_, [&](const auto& entry) { return !modifier_ids.contains(entry.first); });
            // Keep the previous immutable captures alive until their replacement
            // uploads are resolved, including meshes unchanged by a scene edit.
            decltype(source_meshes_) previous_meshes;
            if (source_generation_ != request.source_generation) {
                previous_meshes = std::move(source_meshes_);
                sources_.clear();
                previous_hosts_.clear();
                source_generation_ = request.source_generation;
            }
            for (const auto& object : request.objects) {
                if (cancelled())
                    break;
                if (!sources_.contains(object.uuid)) {
                    auto geometry = source_devices_.convert(storedGeometry(object), backend ? core::Device::GPU : core::Device::CPU);
                    // Resolve cached textures by the source identity before
                    // separating already-GPU geometry from renderer storage.
                    if (object.mesh && object.mesh->vertices.device() == core::Device::GPU)
                        geometry.mesh->mesh = copyMesh(*geometry.mesh->mesh);
                    sources_[object.uuid] = std::move(geometry);
                    if (object.mesh)
                        source_meshes_[object.uuid] = object.mesh;
                }
            }
            // Finish the short source-copy phase before encoding any expensive
            // nodes, so the renderer's imported source buffers cannot inherit their
            // readiness dependency. All waits are on this worker's own marker.
            if (backend) {
                core::TensorFence copied_inputs(*backend);
                queue_->record(copied_inputs);
                copied_inputs.wait();
            }
            const lfs::nodes::EvalControl control{
                .cancelled = cancelled,
                .started = [&](const lfs::nodes::Node& node) {
                    const auto type = registry_.find(node.type_id);
                    std::lock_guard lock(mutex_);
                    progress_.node = node.name;
                    progress_.label = type ? type->label : node.type_id;
                    ++node_runs_[node.name]; },
                .finished = [&](const std::string& name, const lfs::nodes::NodeEvaluation& status) {
                    std::lock_guard lock(mutex_);
                    progress_.finished_nodes[name] = status;
                    progress_.pending_nodes.erase(name);
                    progress_.completed = progress_.finished_nodes.size();
                    progress_.node.clear(); }};
            SnapshotHost host(request, registry_, caches_, sources_, previous_hosts_, control,
                              [&](const core::Uuid& uuid, const std::string& modifier, const lfs::nodes::NodeTree& tree) {
                                  std::unordered_map<std::string, std::vector<std::string>> parents;
                                  for (const auto& link : tree.links)
                                      parents[link.to_node].push_back(link.from_node);
                                  std::unordered_set<std::string> pending;
                                  std::vector<std::string> work{tree.output_node().name};
                                  while (!work.empty()) {
                                      auto name = std::move(work.back());
                                      work.pop_back();
                                      if (pending.insert(name).second)
                                          for (const auto& parent : parents[name])
                                              work.push_back(parent);
                                  }
                                  std::lock_guard lock(mutex_);
                                  progress_.host = uuid;
                                  progress_.modifier = modifier;
                                  progress_.node.clear();
                                  progress_.finished_nodes.clear();
                                  progress_.completed = 0;
                                  progress_.total = pending.size();
                                  progress_.pending_nodes = std::move(pending);
                              });
            for (const auto& uuid : request.targets) {
                if (cancelled())
                    break;
                const auto object = std::ranges::find(request.objects, uuid, &ModifierObjectSnapshot::uuid);
                if (object != request.objects.end())
                    host.run(*object);
            }
            result.hosts = std::move(host.results);
            // Publish only after this request's own marker, including packing.
            if (backend) {
                result.ready = std::make_shared<core::TensorFence>(*backend);
                queue_->record(*result.ready);
                result.ready->wait();
            }
            if (!cancelled())
                for (const auto& [uuid, host_result] : result.hosts)
                    if (host_result.evaluation.ok)
                        previous_hosts_[uuid] = host_result;
        } catch (const std::exception& error) {
            // Worker boundary: preserve the previous payload and report a failed request.
            LOG_ERROR("Node modifier worker failed: {}", error.what());
            for (const auto& uuid : request.targets) {
                auto& host = result.hosts[uuid];
                host.uuid = uuid;
                host.evaluation.ok = false;
                host.evaluation.errors["Graph"] = error.what();
            }
        }
        result.cancelled = cancelled();
        return result;
    }
} // namespace lfs::vis
