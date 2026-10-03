/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nodes/evaluator.hpp"
#include "core/tensor_backend.hpp"
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

    } // namespace

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

    EvalResult evaluate(const NodeTree& tree, EvalInputs inputs, EvalHost* host, EvalCache* cache,
                        const EvalControl& control) {
        const auto backend = geometry_backend(inputs.geometry).value_or(core::TensorExecutionTarget::current().backend());
        core::GpuBackendScope execution_scope(backend);
        const auto device = inputs.device.value_or(evaluation_device());
        GeometryDeviceCache local_devices;
        auto& devices = cache ? cache->devices : local_devices;
        EvalResult result;
        result.geometry = inputs.geometry;
        FieldMemo memo;
        std::unordered_map<std::string, CachedNodeOutput> transient;
        std::unordered_set<std::string> active;
        auto& output_cache = cache ? cache->nodes : transient;
        const auto report = [&](const std::string& name, const CachedNodeOutput& output, const bool cached) {
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
            hash_combine(key, std::hash<std::string>{}(tree.uuid));
            if (type->uses_host) {
                hash_combine(key, host ? host->generation() : 0);
            }
            hash_combine(key, std::hash<std::string>{}(node.properties.dump()));
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
            bool upstream_ok = true;
            for (const auto& declaration : type->inputs) {
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
                    const auto upstream_type = tree.registry().find(upstream_node->type_id);
                    if (!upstream_type) {
                        upstream_ok = false;
                        continue;
                    }
                    const SocketDecl* output = find_socket(*upstream_type, link.from_socket, true);
                    std::optional<SocketDecl> interface_output;
                    if (!output && upstream_node->type_id == "lfs.group_input") {
                        const auto found = std::ranges::find(tree.interface.inputs, link.from_socket,
                                                             &InterfaceSocket::identifier);
                        if (found != tree.interface.inputs.end()) {
                            interface_output = SocketDecl{found->identifier, found->label, found->type,
                                                          found->default_value};
                            output = &*interface_output;
                        }
                    }
                    auto value = upstream.outputs.find(link.from_socket);
                    if (!output || value == upstream.outputs.end()) {
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

            if (const auto found = output_cache.find(node.name);
                found != output_cache.end() && found->second.key == key) {
                evaluation.outputs = found->second.outputs;
                evaluation.ok = true;
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
            if (control.started)
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
                for (const auto& declaration : type->inputs)
                    for (auto& value : context.inputs_[declaration.identifier]) {
                        value = prepare_input(value, declaration);
                        if (auto* geometry = value.get_if<Geometry>())
                            *geometry = devices.convert(std::move(*geometry), device);
                    }
                if (node.type_id == "lfs.group_input") {
                    for (const auto& declaration : tree.interface.inputs) {
                        const auto override_value = inputs.interface_overrides.find(declaration.identifier);
                        if (override_value != inputs.interface_overrides.end())
                            context.outputs_[declaration.identifier] = override_value->second;
                        else if (declaration.type == GEOMETRY_SOCKET)
                            context.outputs_[declaration.identifier] = inputs.geometry;
                        else
                            context.outputs_[declaration.identifier] = declaration.default_value;
                    }
                } else if (node.muted) {
                    for (const auto& output : type->outputs) {
                        const auto same_type =
                            std::ranges::find_if(type->inputs, [&](const SocketDecl& input) {
                                return input.type == output.type;
                            });
                        if (same_type != type->inputs.end())
                            context.outputs_[output.identifier] = context.input(same_type->identifier);
                        else
                            context.outputs_[output.identifier] = output.default_value;
                    }
                } else if (type->evaluate) {
                    type->evaluate(context);
                } else {
                    throw NodeError(std::format("Node type '{}' has no evaluator", type->id));
                }
                for (const auto& output : type->outputs) {
                    if (!context.outputs_.contains(output.identifier))
                        context.outputs_[output.identifier] = output.default_value;
                    if (auto* geometry = context.outputs_[output.identifier].get_if<Geometry>())
                        *geometry = devices.convert(std::move(*geometry), device);
                }
                evaluation.outputs = std::move(context.outputs_);
                evaluation.ok = true;
            } catch (const std::exception& exception) {
                // LFS-CENSUS-OK(empty-catch): node exceptions are the user-facing per-node error channel.
                result.errors[node.name] = exception.what();
                result.ok = false;
                evaluation.ok = false;
            }
            const auto stop = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double, std::milli>(stop - start).count();
            result.time_ms[node.name] = elapsed;
            if (evaluation.ok) {
                output_cache[node.name] = CachedNodeOutput{key, evaluation.outputs, elapsed};
                report(node.name, output_cache.at(node.name), false);
            }
            active.erase(node.name);
            completed[node.name] = evaluation;
            return evaluation;
        };

        try {
            const Node& output = tree.output_node();
            Evaluation evaluated = run(output);
            if (evaluated.ok) {
                auto geometry = evaluated.outputs.find("Geometry");
                if (geometry != evaluated.outputs.end()) {
                    if (const auto* value = geometry->second.get_if<Geometry>())
                        result.geometry = *value;
                    else {
                        result.errors[output.name] = "Group Output did not produce geometry";
                        result.ok = false;
                    }
                }
            }
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
