/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nodes/evaluator.hpp"
#include "core/memory_pressure.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"
#include "core/tensor_execution.hpp"

#include <algorithm>
#include <format>
#include <functional>
#include <unordered_set>

namespace lfs::nodes {
    namespace {

        void hash_combine(std::size_t& seed, std::size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        }

        std::size_t hash_value(const Value& value) {
            if (const auto* field = value.get_if<Field>())
                return std::hash<const void*>{}(field->identity());
            nlohmann::json json = value;
            return std::hash<std::string>{}(json.dump());
        }

        std::optional<core::GpuBackend> geometry_backend(const Geometry& geometry) {
            if (geometry.splats)
                return core::gpu_backend_of(geometry.splats->means);
            if (geometry.points)
                return core::gpu_backend_of(geometry.points->positions);
            if (geometry.mesh && geometry.mesh->mesh)
                return core::gpu_backend_of(geometry.mesh->mesh->vertices);
            return std::nullopt;
        }

        const SocketDecl* find_socket(const NodeTypeInfo& type, std::string_view identifier, bool output) {
            const auto& sockets = output ? type.outputs : type.inputs;
            const auto exact = std::ranges::find(sockets, identifier, &SocketDecl::identifier);
            if (exact != sockets.end())
                return &*exact;
            return nullptr;
        }

        Value prepare_input(Value value, const SocketDecl& declaration) {
            if (const auto* field = value.get_if<Field>(); field && !declaration.field) {
                if (field->context_dependent())
                    throw NodeError(
                        std::format("Input '{}' requires a single value, not a context-dependent field",
                                    declaration.identifier));
                PointsComponent point{core::Tensor::zeros({1, 3}, core::Device::CPU), {}, {}};
                FieldContext context{Domain::Point, nullptr, &point, nullptr, point.positions.debug_id()};
                FieldMemo memo;
                const auto evaluated =
                    field->evaluate(context, memo).to(core::DataType::Float32).reshape({-1});
                if (declaration.type == VECTOR_SOCKET || declaration.type == COLOUR_SOCKET)
                    value = glm::vec3(evaluated.slice(0, 0, 1).item<float>(),
                                      evaluated.slice(0, 1, 2).item<float>(),
                                      evaluated.slice(0, 2, 3).item<float>());
                else if (declaration.type == INT_SOCKET)
                    value = static_cast<std::int64_t>(evaluated.item<float>());
                else if (declaration.type == BOOL_SOCKET)
                    value = evaluated.item<float>() != 0;
                else
                    value = evaluated.item<float>();
            }
            const auto clamp = [&](double scalar) {
                if (declaration.min)
                    scalar = std::max(scalar, *declaration.min);
                if (declaration.max)
                    scalar = std::min(scalar, *declaration.max);
                return scalar;
            };
            if (auto* scalar = value.get_if<float>())
                *scalar = static_cast<float>(clamp(*scalar));
            else if (auto* integer = value.get_if<std::int64_t>())
                *integer = static_cast<std::int64_t>(clamp(static_cast<double>(*integer)));
            else if (auto* vector = value.get_if<glm::vec3>())
                for (int axis = 0; axis < 3; ++axis)
                    (*vector)[axis] = static_cast<float>(clamp((*vector)[axis]));
            return value;
        }

        // Follow fields across group interfaces without executing consumers. The
        // returned geometry producer is qualified relative to the modifier root.
        std::optional<std::pair<std::string, std::string>> preview_context_source(
            const NodeTree& root, const EvalInputs& inputs) {
            struct Scope {
                const NodeTree* tree;
                int parent;
                std::string instance;
                std::string prefix;
            };
            struct Output {
                int scope;
                std::string node;
                std::string socket;
            };
            std::vector<Scope> scopes{{&root, -1, {}, {}}};
            const auto enter = [&](const int parent, const Node& instance) {
                const auto* graph = inputs.tree_resolver
                                        ? inputs.tree_resolver(instance.properties.value("tree", std::string{}))
                                        : nullptr;
                if (!graph)
                    return -1;
                for (int ancestor = parent; ancestor >= 0; ancestor = scopes[ancestor].parent)
                    if (scopes[ancestor].tree->uuid == graph->uuid)
                        return -1;
                const auto prefix = scopes[parent].prefix + instance.name + "/";
                for (size_t i = 0; i < scopes.size(); ++i)
                    if (scopes[i].prefix == prefix)
                        return static_cast<int>(i);
                scopes.push_back({graph, parent, instance.name, prefix});
                return static_cast<int>(scopes.size() - 1);
            };
            int scope = 0;
            std::string path = inputs.requested_node;
            while (!scopes[scope].tree->find_node(path)) {
                const auto slash = path.find('/');
                if (slash == std::string::npos)
                    return std::nullopt;
                const auto* group = scopes[scope].tree->find_node(path.substr(0, slash));
                if (!group || group->type_id != "lfs.group")
                    return std::nullopt;
                scope = enter(scope, *group);
                if (scope < 0)
                    return std::nullopt;
                path.erase(0, slash + 1);
            }
            std::vector<Output> pending{{scope, path, inputs.requested_socket}};
            std::unordered_set<std::string> visited;
            for (size_t index = 0; index < pending.size(); ++index) {
                const auto output = pending[index];
                const auto current = scopes[output.scope];
                if (!visited.insert(current.prefix + output.node + "/" + output.socket).second)
                    continue;
                for (const auto& link : current.tree->links) {
                    if (link.from_node != output.node || link.from_socket != output.socket)
                        continue;
                    const auto* consumer = current.tree->find_node(link.to_node);
                    if (!consumer)
                        continue;
                    if (consumer->type_id == "lfs.group_output") {
                        if (current.parent >= 0)
                            pending.push_back({current.parent, current.instance, link.to_socket});
                        continue;
                    }
                    if (consumer->type_id == "lfs.group") {
                        const auto inner = enter(output.scope, *consumer);
                        if (inner >= 0)
                            pending.push_back({inner, scopes[inner].tree->input_node().name, link.to_socket});
                        continue;
                    }
                    for (const auto& input : effective_inputs(*current.tree, *consumer, inputs.tree_resolver)) {
                        if (input.type != GEOMETRY_SOCKET)
                            continue;
                        for (const auto& source : current.tree->links)
                            if (source.to_node == consumer->name && source.to_socket == input.identifier)
                                return std::pair{current.prefix + source.from_node, source.from_socket};
                    }
                    for (const auto& next : effective_outputs(*current.tree, *consumer, inputs.tree_resolver))
                        if (next.type != GEOMETRY_SOCKET)
                            pending.push_back({output.scope, consumer->name, next.identifier});
                }
            }
            return std::nullopt;
        }

    } // namespace

    namespace {
        thread_local const EvalControl* active_control = nullptr;

        class ActiveControl {
        public:
            explicit ActiveControl(const EvalControl& control) : previous_(active_control) {
                active_control = &control;
            }
            ~ActiveControl() { active_control = previous_; }
            ActiveControl(const ActiveControl&) = delete;
            ActiveControl& operator=(const ActiveControl&) = delete;

        private:
            const EvalControl* previous_;
        };
    } // namespace

    void throw_if_evaluation_cancelled() {
        if (active_control && active_control->cancelled && active_control->cancelled())
            throw EvaluationCancelled();
    }

    const Value& NodeContext::input(std::string_view identifier) const {
        static const Value empty;
        auto found = inputs_.find(std::string(identifier));
        if (found == inputs_.end() || found->second.empty())
            return empty;
        return found->second.front();
    }

    std::vector<Value> NodeContext::inputs(std::string_view identifier) const {
        const auto found = inputs_.find(std::string(identifier));
        return found == inputs_.end() ? std::vector<Value>{} : found->second;
    }

    Field NodeContext::field(std::string_view identifier, std::string_view type) const {
        const Value& value = input(identifier);
        if (const auto* field_value = value.get_if<Field>())
            return field_value->type_id() == type ? *field_value : convert_field(*field_value, type);
        return constant_field(value, type);
    }

    core::Tensor NodeContext::evaluate_field(std::string_view identifier, const FieldContext& context,
                                             std::string_view type) const {
        return field(identifier, type).evaluate(context, *memo_);
    }

    const nlohmann::json& NodeContext::properties() const {
        return node_->properties;
    }
    const Node& NodeContext::node() const {
        return *node_;
    }
    EvalHost* NodeContext::host() const noexcept {
        return host_;
    }

    void NodeContext::set_output(std::string identifier, Value value) {
        outputs_[std::move(identifier)] = std::move(value);
    }

    void NodeContext::record_selection(const FieldContext& context, const core::Tensor& mask) const {
        if (!selection_)
            selection_ = ConsumedSelection{context.identity, mask};
    }

    namespace {
        // A graph and every graph its groups reference, so an edit anywhere below a group changes its key.
        std::size_t group_graph_revision(const std::string& uuid, const TreeResolver& resolver) {
            std::size_t revision = 0;
            std::unordered_set<std::string> visited;
            std::vector<std::string> pending{uuid};
            while (!pending.empty()) {
                auto current = std::move(pending.back());
                pending.pop_back();
                if (!visited.insert(current).second)
                    continue;
                const NodeTree* graph = resolver(current);
                hash_combine(revision, std::hash<std::string>{}(graph ? graph->to_json().dump() : current));
                if (!graph)
                    continue;
                for (const auto& node : graph->nodes)
                    if (node.type_id == "lfs.group")
                        if (const auto found = node.properties.find("tree"); found != node.properties.end() && found->is_string())
                            pending.push_back(found->get<std::string>());
            }
            return revision;
        }
    } // namespace

    EvalResult evaluate(const NodeTree& tree, EvalInputs inputs, EvalHost* host, EvalCache* cache,
                        const EvalControl& control) {
        if (!inputs.requested_node.empty() && inputs.resolve_preview_context) {
            inputs.resolve_preview_context = false;
            auto result = evaluate(tree, inputs, host, cache, control);
            const auto output = result.output_values.find(inputs.requested_socket);
            if (result.ok && output != result.output_values.end() && !output->second.get_if<Geometry>()) {
                result.geometry = inputs.geometry;
                if (const auto source = preview_context_source(tree, inputs)) {
                    inputs.requested_node = source->first;
                    inputs.requested_socket = source->second;
                    const auto context = evaluate(tree, inputs, host, cache, control);
                    result.geometry = context.geometry;
                    hash_combine(result.output_key, context.output_key);
                    result.ok = context.ok;
                    result.cancelled |= context.cancelled;
                    result.errors.insert(context.errors.begin(), context.errors.end());
                    result.nodes.insert(context.nodes.begin(), context.nodes.end());
                    result.time_ms.insert(context.time_ms.begin(), context.time_ms.end());
                }
            }
            return result;
        }
        const auto backend = geometry_backend(inputs.geometry).value_or(core::TensorExecutionTarget::current().backend());
        core::GpuBackendScope execution_scope(backend);
        const ActiveControl active_control_scope(control);
        const auto device = inputs.device.value_or(evaluation_device());
        GeometryDeviceCache local_devices;
        auto& devices = cache ? cache->devices : local_devices;
        EvalResult result;
        result.geometry = inputs.geometry;
        if (const auto cycle = std::ranges::find(inputs.group_stack, tree.uuid, &std::pair<std::string, std::string>::first);
            cycle != inputs.group_stack.end()) {
            std::string message = "Group cycle: ";
            for (auto item = cycle; item != inputs.group_stack.end(); ++item) {
                if (!message.ends_with(": "))
                    message += " → ";
                message += item->second;
            }
            message += " → " + tree.name;
            result.ok = false;
            result.errors["Group Output"] = std::move(message);
            return result;
        }
        FieldMemo memo;
        std::unordered_map<std::string, CachedNodeOutput> transient;
        std::unordered_set<std::string> active;
        auto& output_cache = cache ? cache->nodes : transient;
        const auto report = [&](const std::string& name, const CachedNodeOutput& output, const bool cached) {
            if (const auto* node = tree.find_node(name); node && node->type_id == "lfs.reroute")
                return;
            NodeEvaluation status{.time_ms = output.time_ms, .cached = cached};
            for (const auto& [_, value] : output.outputs) {
                if (const auto* geometry = value.get_if<Geometry>()) {
                    if (geometry->splats)
                        status.element_count = geometry->splats->means.shape()[0];
                    else if (geometry->points)
                        status.element_count = geometry->points->positions.shape()[0];
                    else if (geometry->mesh && geometry->mesh->mesh)
                        status.element_count = geometry->mesh->mesh->vertices.shape()[0];
                    break;
                }
            }
            result.nodes[name] = status;
            if (control.finished)
                control.finished(name, status);
        };

        struct Evaluation {
            std::unordered_map<std::string, Value> outputs;
            std::size_t key = 0;
            bool ok = false;
        };
        std::unordered_map<std::string, Evaluation> completed;

        std::function<Evaluation(const Node&)> run = [&](const Node& node) -> Evaluation {
            if (result.cancelled || (control.cancelled && control.cancelled())) {
                result.cancelled = true;
                return {};
            }
            if (const auto found = completed.find(node.name); found != completed.end())
                return found->second;
            if (!active.insert(node.name).second)
                throw NodeError("Node graph contains a cycle");

            Evaluation evaluation;
            const auto type = tree.registry().find(node.type_id);
            if (!type) {
                result.errors[node.name] = std::format("Missing node type '{}'", node.type_id);
                result.ok = false;
                active.erase(node.name);
                completed[node.name] = evaluation;
                return evaluation;
            }

            std::size_t key = std::hash<std::string>{}(type->id);
            hash_combine(key, std::hash<int>{}(type->version));
            hash_combine(key, std::hash<const void*>{}(type.get()));
            hash_combine(key, std::hash<bool>{}(node.muted));
            if (node.type_id == "lfs.scene_time" || node.type_id == "lfs.group") {
                hash_combine(key, std::hash<float>{}(inputs.seconds));
                hash_combine(key, std::hash<float>{}(inputs.frames_per_second));
            }
            hash_combine(key, std::hash<std::string>{}(tree.uuid));
            if (type->uses_host) {
                hash_combine(key, host ? host->generation() : 0);
            }
            hash_combine(key, std::hash<std::string>{}(node.properties.dump()));
            if (node.type_id == "lfs.group" && inputs.tree_resolver) {
                const auto graph = node.properties.find("tree");
                if (graph != node.properties.end() && graph->is_string())
                    hash_combine(key, group_graph_revision(graph->get_ref<const std::string&>(), inputs.tree_resolver));
            }
            hash_combine(key, std::hash<std::uint64_t>{}(inputs.geometry_generation));
            hash_combine(key, static_cast<size_t>(device));
            hash_combine(key, static_cast<size_t>(backend));
            for (const auto& [identifier, value] : node.input_values) {
                hash_combine(key, std::hash<std::string>{}(identifier));
                hash_combine(key, hash_value(value));
            }
            if (node.type_id == "lfs.group_input")
                for (const auto& [identifier, value] : inputs.interface_overrides) {
                    hash_combine(key, std::hash<std::string>{}(identifier));
                    hash_combine(key, hash_value(value));
                }

            NodeContext context;
            context.node_ = &node;
            context.memo_ = &memo;
            context.host_ = host;
            context.seconds_ = inputs.seconds;
            context.frames_per_second_ = inputs.frames_per_second;
            std::optional<Geometry> geometry_input;
            bool upstream_ok = true;
            const auto input_declarations = effective_inputs(tree, node, inputs.tree_resolver);
            const auto output_declarations = effective_outputs(tree, node, inputs.tree_resolver);
            for (const auto& declaration : input_declarations) {
                std::vector<Value> resolved;
                for (const auto& link : tree.links) {
                    if (link.to_node != node.name || link.to_socket != declaration.identifier)
                        continue;
                    const Node* upstream_node = tree.find_node(link.from_node);
                    if (!upstream_node) {
                        upstream_ok = false;
                        continue;
                    }
                    Evaluation upstream = run(*upstream_node);
                    hash_combine(key, upstream.key);
                    hash_combine(key, std::hash<std::string>{}(link.from_node));
                    hash_combine(key, std::hash<std::string>{}(link.from_socket));
                    hash_combine(key, std::hash<std::string>{}(link.to_socket));
                    if (!upstream.ok) {
                        upstream_ok = false;
                        continue;
                    }
                    const auto upstream_outputs = effective_outputs(tree, *upstream_node,
                                                                    inputs.tree_resolver);
                    const auto output = std::ranges::find(upstream_outputs, link.from_socket,
                                                          &SocketDecl::identifier);
                    auto value = upstream.outputs.find(link.from_socket);
                    if (output == upstream_outputs.end() || value == upstream.outputs.end()) {
                        upstream_ok = false;
                        continue;
                    }
                    try {
                        resolved.push_back(convert_value(value->second, output->type, declaration.type));
                    } catch (const std::exception& exception) {
                        // LFS-CENSUS-OK(empty-catch): normalize a link conversion failure into EvalResult.
                        result.errors[node.name] = exception.what();
                        upstream_ok = false;
                    }
                }
                if (resolved.empty()) {
                    const auto own = node.input_values.find(declaration.identifier);
                    resolved.push_back(own == node.input_values.end() ? declaration.default_value
                                                                      : own->second);
                }
                context.inputs_[declaration.identifier] = std::move(resolved);
            }

            evaluation.key = key;
            if (result.cancelled)
                return evaluation;
            if (!upstream_ok) {
                if (!result.errors.contains(node.name))
                    result.errors[node.name] = "An upstream node failed";
                result.ok = false;
                active.erase(node.name);
                completed[node.name] = evaluation;
                return evaluation;
            }

            const std::string cache_name = inputs.cache_namespace + node.name;
            if (const auto found = output_cache.find(cache_name);
                found != output_cache.end() && found->second.key == key) {
                evaluation.outputs = found->second.outputs;
                evaluation.ok = true;
                if (node.type_id != "lfs.reroute")
                    result.time_ms[node.name] = found->second.time_ms;
                report(node.name, found->second, true);
                active.erase(node.name);
                completed[node.name] = evaluation;
                return evaluation;
            }

            const auto start = std::chrono::steady_clock::now();
            if (control.cancelled && control.cancelled()) {
                result.cancelled = true;
                return {};
            }
            if (control.started && node.type_id != "lfs.reroute")
                control.started(node);
            try {
                if (type->uses_host && !host && !node.muted)
                    throw NodeError("This node requires an evaluation host");
                for (const auto& property : type->properties) {
                    if (property.kind == PropertyKind::Enum) {
                        const auto found = node.properties.find(property.identifier);
                        const auto value = found == node.properties.end() ? property.default_value : *found;
                        if (!value.is_string() ||
                            std::ranges::find(property.items, value.get<std::string>()) ==
                                property.items.end())
                            throw NodeError(
                                std::format("Invalid value for property '{}'", property.identifier));
                    }
                }
                for (const auto& declaration : input_declarations)
                    for (auto& value : context.inputs_[declaration.identifier]) {
                        value = prepare_input(value, declaration);
                        if (auto* geometry = value.get_if<Geometry>()) {
                            *geometry = devices.convert(std::move(*geometry), device);
                            if (!geometry_input)
                                geometry_input = *geometry;
                        }
                    }
                if (node.type_id == "lfs.group_input") {
                    for (const auto& declaration : tree.group_interface.inputs) {
                        const auto override_value = inputs.interface_overrides.find(declaration.identifier);
                        if (override_value != inputs.interface_overrides.end())
                            context.outputs_[declaration.identifier] = override_value->second;
                        else if (declaration.type == GEOMETRY_SOCKET)
                            context.outputs_[declaration.identifier] = inputs.geometry;
                        else
                            context.outputs_[declaration.identifier] = declaration.default_value;
                    }
                } else if (node.type_id == "lfs.group_output") {
                    for (const auto& output : tree.group_interface.outputs)
                        context.outputs_[output.identifier] = context.input(output.identifier);
                } else if (node.type_id == "lfs.reroute") {
                    context.outputs_["Output"] = context.input("Input");
                } else if (node.type_id == "lfs.group") {
                    const auto graph = node.properties.find("tree");
                    const NodeTree* nested = graph != node.properties.end() && graph->is_string() &&
                                                     inputs.tree_resolver
                                                 ? inputs.tree_resolver(graph->get_ref<const std::string&>())
                                                 : nullptr;
                    if (!nested)
                        throw NodeError("Missing graph");
                    if (nested->tree_type != tree.tree_type)
                        throw NodeError("Group graph has a different tree type");
                    std::string cycle;
                    if (group_reference_would_cycle(tree, nested->uuid, inputs.tree_resolver,
                                                    &cycle))
                        throw NodeError("Group cycle: " + cycle);
                    EvalInputs nested_inputs;
                    nested_inputs.geometry = inputs.geometry;
                    nested_inputs.seconds = inputs.seconds;
                    nested_inputs.frames_per_second = inputs.frames_per_second;
                    // Interface geometry hashes only by type; the group's own key carries its upstream
                    // revisions, so nested caches follow upstream edits.
                    nested_inputs.geometry_generation = evaluation.key;
                    nested_inputs.device = inputs.device;
                    nested_inputs.tree_resolver = inputs.tree_resolver;
                    nested_inputs.group_stack = inputs.group_stack;
                    nested_inputs.group_stack.emplace_back(tree.uuid, tree.name);
                    nested_inputs.cache_namespace = inputs.cache_namespace + node.name + "/";
                    for (const auto& declaration : input_declarations)
                        nested_inputs.interface_overrides[declaration.identifier] =
                            context.input(declaration.identifier);
                    if (const auto geometry = std::ranges::find(input_declarations,
                                                                std::string(GEOMETRY_SOCKET),
                                                                &SocketDecl::type);
                        geometry != input_declarations.end())
                        if (const auto* value = context.input(geometry->identifier).get_if<Geometry>())
                            nested_inputs.geometry = *value;
                    auto nested_result = evaluate(*nested, std::move(nested_inputs), host, cache);
                    for (const auto& [inner, status] : nested_result.nodes)
                        result.nodes[node.name + "/" + inner] = status;
                    for (const auto& [inner, time] : nested_result.time_ms)
                        result.time_ms[node.name + "/" + inner] = time;
                    for (const auto& [inner, message] : nested_result.errors)
                        result.errors[node.name + "/" + inner] = message;
                    if (!nested_result.ok) {
                        if (nested_result.errors.empty())
                            throw NodeError("Group Output: Evaluation failed");
                        const auto& error = *nested_result.errors.begin();
                        throw NodeError(error.first + ": " + error.second);
                    }
                    for (const auto& output : output_declarations) {
                        const auto value = nested_result.output_values.find(output.identifier);
                        context.outputs_[output.identifier] =
                            value == nested_result.output_values.end() ? output.default_value : value->second;
                    }
                } else if (node.muted) {
                    for (const auto& output : output_declarations) {
                        const auto same_type =
                            std::ranges::find_if(input_declarations, [&](const SocketDecl& input) {
                                return input.type == output.type;
                            });
                        if (same_type != input_declarations.end())
                            context.outputs_[output.identifier] = context.input(same_type->identifier);
                        else
                            context.outputs_[output.identifier] = output.default_value;
                    }
                } else if (type->evaluate) {
                    type->evaluate(context);
                } else {
                    throw NodeError(std::format("Node type '{}' has no evaluator", type->id));
                }
                for (const auto& output : output_declarations) {
                    if (!context.outputs_.contains(output.identifier))
                        context.outputs_[output.identifier] = output.default_value;
                    if (auto* geometry = context.outputs_[output.identifier].get_if<Geometry>())
                        *geometry = devices.convert(std::move(*geometry), device);
                }
                evaluation.outputs = std::move(context.outputs_);
                evaluation.ok = true;
            } catch (const EvaluationCancelled&) {
                result.cancelled = true;
                evaluation.ok = false;
            } catch (const FieldNodeError& exception) {
                result.errors[exception.node()] = exception.what();
                result.ok = false;
                evaluation.ok = false;
            } catch (const core::MemoryAllocationError& exception) {
                if (control.propagate_out_of_memory)
                    throw;
                result.errors[node.name] = exception.what();
                result.ok = false;
                evaluation.ok = false;
            } catch (const std::exception& exception) {
                // LFS-CENSUS-OK(empty-catch): node exceptions are the user-facing per-node error channel.
                result.errors[node.name] = exception.what();
                result.ok = false;
                evaluation.ok = false;
            }
            if (control.synchronize_nodes) {
                core::TensorCompletion completion;
                completion.include_current_gpu();
                completion.wait();
            }
            const auto stop = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double, std::milli>(stop - start).count();
            if (node.type_id != "lfs.reroute")
                result.time_ms[node.name] = elapsed;
            if (evaluation.ok) {
                output_cache[cache_name] = CachedNodeOutput{key, evaluation.outputs, elapsed,
                                                            std::move(geometry_input), std::move(context.selection_)};
                report(node.name, output_cache.at(cache_name), false);
            }
            active.erase(node.name);
            completed[node.name] = evaluation;
            return evaluation;
        };

        try {
            const Node* requested = inputs.requested_node.empty() ? nullptr : tree.find_node(inputs.requested_node);
            if (!requested && inputs.requested_node.contains('/')) {
                const auto slash = inputs.requested_node.find('/');
                const auto* instance = tree.find_node(inputs.requested_node.substr(0, slash));
                const auto* nested = instance && instance->type_id == "lfs.group" && inputs.tree_resolver
                                         ? inputs.tree_resolver(instance->properties.value("tree", std::string{}))
                                         : nullptr;
                if (!nested)
                    throw NodeError("Preview group no longer exists");
                EvalInputs inner = inputs;
                inner.interface_overrides.clear();
                inner.requested_node = inputs.requested_node.substr(slash + 1);
                inner.cache_namespace += instance->name + "/";
                inner.group_stack.emplace_back(tree.uuid, tree.name);
                size_t generation = inputs.geometry_generation;
                for (const auto& input : effective_inputs(tree, *instance, inputs.tree_resolver)) {
                    const auto own = instance->input_values.find(input.identifier);
                    Value value = own == instance->input_values.end() ? input.default_value : own->second;
                    for (const auto& link : tree.links) {
                        if (link.to_node != instance->name || link.to_socket != input.identifier)
                            continue;
                        const auto* producer = tree.find_node(link.from_node);
                        if (!producer)
                            throw NodeError("Preview group input no longer exists");
                        const auto upstream = run(*producer);
                        if (!upstream.ok)
                            throw NodeError("Preview group input failed");
                        hash_combine(generation, upstream.key);
                        const auto declarations = effective_outputs(tree, *producer, inputs.tree_resolver);
                        const auto output = std::ranges::find(declarations, link.from_socket, &SocketDecl::identifier);
                        if (output == declarations.end())
                            throw NodeError("Preview group input socket no longer exists");
                        value = convert_value(upstream.outputs.at(link.from_socket), output->type, input.type);
                        break;
                    }
                    inner.interface_overrides[input.identifier] = value;
                    if (const auto* geometry = value.get_if<Geometry>())
                        inner.geometry = *geometry;
                }
                inner.geometry_generation = generation;
                auto evaluated = evaluate(*nested, std::move(inner), host, cache, control);
                result.geometry = std::move(evaluated.geometry);
                result.output_values = std::move(evaluated.output_values);
                result.output_key = evaluated.output_key;
                result.ok = evaluated.ok;
                result.cancelled = evaluated.cancelled;
                for (const auto& [name, status] : evaluated.nodes)
                    result.nodes[instance->name + "/" + name] = status;
                for (const auto& [name, time] : evaluated.time_ms)
                    result.time_ms[instance->name + "/" + name] = time;
                for (const auto& [name, error] : evaluated.errors)
                    result.errors[instance->name + "/" + name] = error;
                return result;
            }
            if (!inputs.requested_node.empty() && !requested)
                throw NodeError("Preview node no longer exists");
            const Node& output = requested ? *requested : tree.output_node();
            Evaluation evaluated = run(output);
            if (evaluated.ok) {
                result.output_values = evaluated.outputs;
                result.output_key = evaluated.key;
                auto geometry = evaluated.outputs.find(requested ? inputs.requested_socket : "Geometry");
                if (geometry != evaluated.outputs.end()) {
                    if (const auto* value = geometry->second.get_if<Geometry>())
                        result.geometry = *value;
                    else if (!requested) {
                        result.errors[output.name] = "Group Output did not produce geometry";
                        result.ok = false;
                    }
                } else if (requested) {
                    throw NodeError("Preview socket no longer exists");
                }
            }
        } catch (const EvaluationCancelled&) {
            result.cancelled = true;
        } catch (const core::MemoryAllocationError& exception) {
            if (control.propagate_out_of_memory)
                throw;
            result.errors["Group Output"] = exception.what();
            result.ok = false;
        } catch (const std::exception& exception) {
            // LFS-CENSUS-OK(empty-catch): normalize graph-boundary failures into EvalResult.
            result.errors["Group Output"] = exception.what();
            result.ok = false;
        }
        if (result.cancelled)
            result.ok = false;
        if (!result.ok)
            result.geometry = inputs.geometry;
        return result;
    }

} // namespace lfs::nodes
