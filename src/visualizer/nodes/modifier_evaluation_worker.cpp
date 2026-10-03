/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "modifier_evaluation_worker.hpp"

#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/services.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_vulkan_interop.hpp"
#include "io/loader.hpp"
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
                geometry.mesh = lfs::nodes::MeshComponent{std::move(result), geometry.mesh->textures,
                                                          geometry.mesh->attributes};
            }
            return geometry;
        }

        void preparePayload(const ModifierObjectSnapshot& node, ModifierHostResult& result,
                            const core::SplatTensorAllocator& allocator) {
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
            // Splats reach the renderer as its backend requires; moving them into renderer storage
            // also separates them from the worker cache.
            if (result.splats) {
                const auto backend = core::gpu_backend_of(result.splats->means_raw());
                switch (backend ? core::splat_publication(*backend) : core::SplatPublication::Copied) {
                case core::SplatPublication::RendererStorage:
                    if (allocator) {
                        if (auto moved = lfs::io::migrateSplatTensorsToAllocator(*result.splats, allocator, false); !moved)
                            throw std::runtime_error(moved.error().format());
                        break;
                    }
                    [[fallthrough]];
                case core::SplatPublication::Copied:
                    result.splats = std::make_shared<core::SplatData>(result.splats->clone());
                    break;
                case core::SplatPublication::Shared:
                    break;
                }
            }
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

        std::optional<FieldContext> fieldContext(const Geometry& geometry) {
            FieldContext context;
            if (geometry.splats) {
                context.domain = Domain::Splat;
                context.splats = &*geometry.splats;
                const auto& splats = *geometry.splats;
                context.identity = static_cast<std::uint64_t>(splats.means.debug_id()) ^
                                   (static_cast<std::uint64_t>(splats.sh0.debug_id()) << 8U) ^
                                   (static_cast<std::uint64_t>(splats.scaling.debug_id()) << 16U) ^
                                   (static_cast<std::uint64_t>(splats.opacity.debug_id()) << 24U) ^
                                   (static_cast<std::uint64_t>(splats.shN.debug_id()) << 32U);
            } else if (geometry.points) {
                context.domain = Domain::Point;
                context.points = &*geometry.points;
                context.identity = static_cast<std::uint64_t>(geometry.points->positions.debug_id()) ^
                                   (static_cast<std::uint64_t>(geometry.points->colors.debug_id()) << 32U);
            } else if (geometry.mesh) {
                context.domain = Domain::Vertex;
                context.mesh = &*geometry.mesh;
                context.identity = geometry.mesh->mesh ? geometry.mesh->mesh->id() : 0;
            } else {
                return std::nullopt;
            }
            return context;
        }

        // Attribute nodes replace tensors but keep element order, so tensor identity
        // would hide every preview behind them. Structural nodes change the count.
        bool sameElements(const FieldContext& left, const FieldContext& right) {
            return left.domain == right.domain && left.size() == right.size();
        }

        void selectionPreviews(const NodeTree& tree, const EvalCache& cache,
                               const std::string& modifier, ModifierHostResult& result,
                               const std::function<bool()>& cancelled,
                               const TreeResolver& resolver,
                               const std::string& name_space = {}) {
            const auto displayed = fieldContext(result.evaluation.geometry);
            if (!displayed)
                return;
            FieldMemo memo;
            std::unordered_map<std::string, std::pair<core::Tensor, double>> masks;
            for (const auto& node : tree.nodes) {
                if (cancelled())
                    return;
                std::string source = name_space + node.name;
                std::string socket;
                const auto outputs = effective_outputs(tree, node, resolver);
                const auto inputs = effective_inputs(tree, node, resolver);
                const auto output = std::ranges::find(outputs, "Selection", &SocketDecl::identifier);
                const CachedNodeOutput* consumer = nullptr;
                if (output != outputs.end() && output->type != GEOMETRY_SOCKET) {
                    socket = output->identifier;
                    const auto link = std::ranges::find_if(tree.links, [&](const Link& item) {
                        return item.from_node == node.name && item.from_socket == output->identifier;
                    });
                    if (link == tree.links.end())
                        continue;
                    const auto cached_consumer = cache.nodes.find(name_space + link->to_node);
                    if (cached_consumer == cache.nodes.end())
                        continue;
                    consumer = &cached_consumer->second;
                } else {
                    const auto input = std::ranges::find(inputs, "Selection", &SocketDecl::identifier);
                    if (input == inputs.end())
                        continue;
                    const auto link = std::ranges::find_if(tree.links, [&](const Link& item) {
                        return item.to_node == node.name && item.to_socket == input->identifier;
                    });
                    if (link == tree.links.end())
                        continue;
                    source = name_space + link->from_node;
                    socket = link->from_socket;
                    const auto cached_consumer = cache.nodes.find(name_space + node.name);
                    if (cached_consumer == cache.nodes.end())
                        continue;
                    consumer = &cached_consumer->second;
                }
                if (!consumer || !consumer->geometry_input)
                    continue;
                const auto context = fieldContext(*consumer->geometry_input);
                if (!context)
                    continue;
                const auto cached = cache.nodes.find(source);
                if (cached == cache.nodes.end())
                    continue;
                const auto value = cached->second.outputs.find(socket);
                if (value == cached->second.outputs.end())
                    continue;
                const std::string key = source + "/" + socket + "/" +
                                        std::to_string(static_cast<int>(context->domain)) + "/" +
                                        std::to_string(context->identity) + "/" +
                                        std::to_string(context->size());
                auto found = masks.find(key);
                if (found == masks.end()) {
                    try {
                        const auto* field = value->second.get_if<Field>();
                        auto mask = (field ? convert_field(*field, FLOAT_SOCKET)
                                           : constant_field(value->second, FLOAT_SOCKET))
                                        .evaluate(*context, memo)
                                        .ge(0.5f);
                        const auto count = mask.numel();
                        const double share = count ? static_cast<double>(mask.count_nonzero()) / count : 0.0;
                        found = masks.emplace(key, std::pair{std::move(mask), share}).first;
                    } catch (const std::exception&) {
                        // LFS-CENSUS-OK(empty-catch): optional selection diagnostics cannot fail evaluation.
                        // Selection statistics and viewport previews are ancillary. A field that
                        // cannot be evaluated in this context must not change the graph result.
                        continue;
                    }
                }
                if (auto status = result.evaluation.nodes.find(modifier + "/" + name_space + node.name);
                    status != result.evaluation.nodes.end())
                    status->second.selected_share = found->second.second;
                if (sameElements(*context, *displayed))
                    result.previews[modifier + "/" + name_space + node.name] = found->second.first;
            }
            for (const auto& node : tree.nodes) {
                if (node.type_id != "lfs.group")
                    continue;
                const auto property = node.properties.find("tree");
                const auto* nested = property != node.properties.end() && property->is_string()
                                         ? resolver(property->get_ref<const std::string&>())
                                         : nullptr;
                if (nested)
                    selectionPreviews(*nested, cache, modifier, result, cancelled, resolver,
                                      name_space + node.name + "/");
            }
        }

        class SnapshotHost final : public EvalHost {
        public:
            SnapshotHost(const ModifierEvaluationRequest& request, const NodeTypeRegistry& registry,
                         std::unordered_map<std::string, EvalCache>& caches,
                         std::function<const Geometry&(const ModifierObjectSnapshot&)> source,
                         const std::unordered_map<core::Uuid, ModifierHostResult>& previous,
                         const EvalControl& control,
                         std::function<void(const core::Uuid&, const std::string&, const NodeTree&)> stack_started)
                : request_(request), registry_(registry), caches_(caches), source_(std::move(source)), previous_(previous), control_(control), stack_started_(std::move(stack_started)) {}

            std::uint64_t generation() const override { return request_.generation; }
            std::span<const EvaluationCamera> cameras() const override { return request_.cameras; }
            glm::mat4 object_to_world() const override { return current_world_; }

            std::optional<Geometry> object_geometry(const std::string_view name, const TransformSpace space) override {
                const auto target = std::ranges::find(request_.objects, name, &ModifierObjectSnapshot::name);
                if (target == request_.objects.end())
                    throw NodeError(std::format("Object '{}' does not exist", name));
                auto& result = run(*target);
                if (!result.evaluation.ok) {
                    const auto cycle = std::ranges::find_if(result.evaluation.errors, [](const auto& error) {
                        return error.second.starts_with("Dependency cycle:");
                    });
                    if (cycle != result.evaluation.errors.end())
                        throw NodeError(cycle->second);
                    if (!result.evaluation.errors.empty())
                        throw NodeError(result.evaluation.errors.begin()->second);
                    throw NodeError("Object Info target modifier evaluation failed");
                }
                auto geometry = result.enabled ? result.evaluation.geometry : source_(*target);
                if (space == TransformSpace::Relative)
                    geometry = transform_geometry(std::move(geometry), glm::inverse(current_world_) * target->world);
                return geometry;
            }

            ModifierHostResult& run(const ModifierObjectSnapshot& object) {
                if (active_.contains(object.uuid)) {
                    const auto first = std::ranges::find(active_path_, object.uuid,
                                                         &ModifierObjectSnapshot::uuid);
                    std::string path;
                    for (auto current = first; current != active_path_.end(); ++current) {
                        if (!path.empty())
                            path += " → ";
                        path += current->name;
                    }
                    if (!path.empty())
                        path += " → ";
                    path += object.name;
                    throw NodeError("Dependency cycle: " + path);
                }
                if (auto found = results.find(object.uuid); found != results.end())
                    return found->second;
                active_.insert(object.uuid);
                active_path_.push_back(object);
                const auto previous_world = current_world_;
                current_world_ = object.world;
                auto& result = results[object.uuid];
                result.uuid = object.uuid;
                const auto start = std::chrono::steady_clock::now();
                // An object without visible modifiers displays its stored payload; it
                // needs no evaluation copy unless it is baked or read by Object Info.
                const bool active = std::ranges::any_of(object.stack.modifiers, [](const Modifier& modifier) {
                    return modifier.enabled && modifier.show_viewport;
                });
                Geometry geometry = active || request_.bake ? source_(object) : Geometry{};
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
                    std::erase_if(cache.nodes, [&](const auto& entry) {
                        const auto slash = entry.first.find('/');
                        return !node_names.contains(std::string_view(entry.first).substr(0, slash));
                    });
                    auto evaluated = lfs::nodes::evaluate(tree,
                                                          {.geometry = geometry,
                                                           .interface_overrides = modifier.input_overrides,
                                                           .geometry_generation = input_generation,
                                                           .tree_resolver = [this](const std::string_view uuid) {
                                                               return resolveTree(uuid);
                                                           },
                                                           .seconds = request_.seconds,
                                                           .frames_per_second = request_.frames_per_second},
                                                          this, &cache, control_);
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
                                                  modifier.uuid, result, control_.cancelled,
                                                  [this](const std::string_view uuid) { return resolveTree(uuid); });
                        }
                        if ((result.enabled || request_.bake) && !control_.cancelled())
                            preparePayload(object, result, request_.splat_allocator);
                    }
                }
                result.evaluation.total_time_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                active_.erase(object.uuid);
                active_path_.pop_back();
                current_world_ = previous_world;
                return result;
            }

            std::unordered_map<core::Uuid, ModifierHostResult> results;

        private:
            const NodeTree* resolveTree(const std::string_view uuid) {
                const std::string key(uuid);
                if (const auto found = resolved_trees_.find(key); found != resolved_trees_.end())
                    return found->second.get();
                const auto source = request_.trees.find(key);
                if (source == request_.trees.end())
                    return nullptr;
                auto value = std::make_unique<NodeTree>(NodeTree::from_json(source->second, registry_));
                const auto* result = value.get();
                resolved_trees_.emplace(key, std::move(value));
                return result;
            }

            const ModifierEvaluationRequest& request_;
            const NodeTypeRegistry& registry_;
            std::unordered_map<std::string, EvalCache>& caches_;
            std::function<const Geometry&(const ModifierObjectSnapshot&)> source_;
            const std::unordered_map<core::Uuid, ModifierHostResult>& previous_;
            const EvalControl& control_;
            std::function<void(const core::Uuid&, const std::string&, const NodeTree&)> stack_started_;
            std::unordered_set<core::Uuid> active_;
            std::vector<ModifierObjectSnapshot> active_path_;
            std::unordered_map<std::string, std::unique_ptr<NodeTree>> resolved_trees_;
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
        const auto attempt = [&] {
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
                    if (modifier.enabled && modifier.show_viewport)
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
            // Sources are copied only for objects this request evaluates or reads.
            std::unordered_set<core::Uuid> used_sources;
            const auto source = [&](const ModifierObjectSnapshot& object) -> const Geometry& {
                used_sources.insert(object.uuid);
                if (const auto found = sources_.find(object.uuid); found != sources_.end())
                    return found->second;
                auto geometry = source_devices_.convert(storedGeometry(object), backend ? core::Device::GPU : core::Device::CPU);
                // Resolve cached textures by the source identity before
                // separating already-GPU geometry from renderer storage.
                if (object.mesh && object.mesh->vertices.device() == core::Device::GPU)
                    geometry.mesh->mesh = copyMesh(*geometry.mesh->mesh);
                if (object.mesh)
                    source_meshes_[object.uuid] = object.mesh;
                // Finish the short source copy before encoding any expensive nodes,
                // so the renderer's imported source buffers cannot inherit their
                // readiness dependency. All waits are on this worker's own marker.
                if (backend) {
                    core::TensorFence copied_inputs(*backend);
                    queue_->record(copied_inputs);
                    copied_inputs.wait();
                }
                return sources_.emplace(object.uuid, std::move(geometry)).first->second;
            };
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
                    progress_.node.clear(); },
                .propagate_out_of_memory = true};
            SnapshotHost host(request, registry_, caches_, source, previous_hosts_, control,
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
                                  std::erase_if(pending, [&](const auto& name) {
                                      const auto* node = tree.find_node(name);
                                      return node && (node->type_id == "lfs.reroute" || node->type_id == "lfs.frame" || node->type_id == "lfs.note");
                                  });
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
            if (!cancelled()) {
                for (const auto& [uuid, host_result] : result.hosts)
                    if (host_result.evaluation.ok)
                        previous_hosts_[uuid] = host_result;
                // Drop copies no evaluation reads any more, e.g. after the last modifier
                // of an object is removed or hidden. A bake covers one object only.
                if (!request.bake) {
                    std::erase_if(sources_, [&](const auto& entry) { return !used_sources.contains(entry.first); });
                    std::erase_if(source_meshes_, [&](const auto& entry) { return !used_sources.contains(entry.first); });
                    if (source_meshes_.empty())
                        source_devices_.clear();
                    std::erase_if(previous_hosts_, [&](const auto& entry) { return !std::ranges::contains(request.targets, entry.first); });
                }
            }
        };
        try {
            try {
                attempt();
            } catch (const core::MemoryAllocationError& error) {
                // Cached intermediates and source copies only save time: drop them and retry once.
                LOG_WARN("Node modifier evaluation ran out of device memory, retrying without cached results: {}",
                         error.what());
                caches_.clear();
                sources_.clear();
                source_meshes_.clear();
                source_devices_.clear();
                previous_hosts_.clear();
                result.hosts.clear();
                core::Tensor::trim_memory_pool();
                attempt();
            }
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
