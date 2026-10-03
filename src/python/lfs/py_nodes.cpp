/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_nodes.hpp"

#include "core/logger.hpp"
#include "core/nodes/nodes.hpp"
#include "py_tensor.hpp"
#include "py_ui.hpp"
#include "py_viewer_dispatch.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/scene/scene_manager.hpp"

#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace nb = nanobind;

namespace lfs::python {
    namespace {
        using namespace lfs::nodes;

        struct StandaloneLibrary {
            NodeTypeRegistry registry;
            std::unordered_map<std::string, std::unique_ptr<NodeTree>> trees;
            std::unordered_map<std::string, EvalResult> evaluations;

            StandaloneLibrary() {
                register_builtin_nodes(registry);
            }
        };

        StandaloneLibrary& standalone() {
            static auto* value = new StandaloneLibrary;
            return *value;
        }

        std::string standalone_tree_name(std::string name, std::string_view except = {}) {
            if (name.empty())
                name = "Node Graph";
            std::unordered_set<std::string_view> names;
            for (const auto& [uuid, tree] : standalone().trees)
                if (uuid != except)
                    names.insert(tree->name);
            const auto base = name;
            for (int suffix = 2; names.contains(name); ++suffix)
                name = base + " " + std::to_string(suffix);
            return name;
        }

        vis::ModifierManager* live_manager() {
            auto* scene_manager = get_scene_manager();
            return scene_manager ? &scene_manager->modifierManager() : nullptr;
        }

        NodeTypeRegistry& active_registry() {
            if (auto* manager = live_manager())
                return manager->registry();
            return standalone().registry;
        }

        NodeTree* active_tree(const std::string& id) {
            if (auto* manager = live_manager())
                return manager->tree(id);
            const auto found = standalone().trees.find(id);
            return found == standalone().trees.end() ? nullptr : found->second.get();
        }

        nb::object value_to_python(const Value& value);

        Value python_to_value(const nb::handle value) {
            if (value.is_none())
                return {};
            if (nb::isinstance<nb::bool_>(value))
                return nb::cast<bool>(value);
            if (nb::isinstance<nb::int_>(value))
                return nb::cast<std::int64_t>(value);
            if (nb::isinstance<nb::float_>(value))
                return nb::cast<float>(value);
            if (nb::isinstance<nb::str>(value))
                return nb::cast<std::string>(value);
            if (nb::isinstance<nb::sequence>(value)) {
                const auto sequence = nb::cast<nb::sequence>(value);
                if (nb::len(sequence) == 3) {
                    return glm::vec3(nb::cast<float>(sequence[0]),
                                     nb::cast<float>(sequence[1]),
                                     nb::cast<float>(sequence[2]));
                }
                if (nb::len(sequence) == 4) {
                    return glm::vec4(nb::cast<float>(sequence[0]),
                                     nb::cast<float>(sequence[1]),
                                     nb::cast<float>(sequence[2]),
                                     nb::cast<float>(sequence[3]));
                }
            }
            throw nb::type_error("Unsupported node value");
        }

        nlohmann::json python_to_json(const nb::handle value) {
            if (value.is_none())
                return nullptr;
            if (nb::isinstance<nb::bool_>(value))
                return nb::cast<bool>(value);
            if (nb::isinstance<nb::int_>(value))
                return nb::cast<std::int64_t>(value);
            if (nb::isinstance<nb::float_>(value))
                return nb::cast<double>(value);
            if (nb::isinstance<nb::str>(value))
                return nb::cast<std::string>(value);
            if (nb::isinstance<nb::dict>(value)) {
                nlohmann::json result = nlohmann::json::object();
                for (const auto [key, item] : nb::cast<nb::dict>(value))
                    result[nb::cast<std::string>(key)] = python_to_json(item);
                return result;
            }
            if (nb::isinstance<nb::sequence>(value)) {
                nlohmann::json result = nlohmann::json::array();
                for (const auto item : nb::cast<nb::sequence>(value))
                    result.push_back(python_to_json(item));
                return result;
            }
            throw nb::type_error("Unsupported node property value");
        }

        nb::object json_to_python(const nlohmann::json& value) {
            if (value.is_null())
                return nb::none();
            if (value.is_boolean())
                return nb::bool_(value.get<bool>());
            if (value.is_number_integer())
                return nb::int_(value.get<std::int64_t>());
            if (value.is_number())
                return nb::float_(value.get<double>());
            if (value.is_string())
                return nb::str(value.get<std::string>().c_str());
            if (value.is_array()) {
                nb::list result;
                for (const auto& item : value)
                    result.append(json_to_python(item));
                return std::move(result);
            }
            nb::dict result;
            for (const auto& [key, item] : value.items())
                result[nb::str(key.c_str())] = json_to_python(item);
            return std::move(result);
        }

        struct PySplats {
            SplatsComponent value;

            PySplats replace(const nb::kwargs& kwargs) const {
                PySplats result = *this;
                const auto set = [&](const char* key, core::Tensor& target) {
                    if (kwargs.contains(key))
                        target = nb::cast<PyTensor>(kwargs[key]).tensor();
                };
                set("means", result.value.means);
                set("sh0", result.value.sh0);
                set("shN", result.value.shN);
                set("scaling", result.value.scaling);
                set("rotation", result.value.rotation);
                set("opacity", result.value.opacity);
                return result;
            }
        };

        struct PyPoints {
            PointsComponent value;

            PyPoints replace(const nb::kwargs& kwargs) const {
                PyPoints result = *this;
                if (kwargs.contains("positions"))
                    result.value.positions = nb::cast<PyTensor>(kwargs["positions"]).tensor();
                if (kwargs.contains("colors"))
                    result.value.colors = nb::cast<PyTensor>(kwargs["colors"]).tensor();
                return result;
            }
        };

        struct PyMesh {
            MeshComponent value;

            PyMesh replace(const nb::kwargs& kwargs) const {
                if (!value.mesh)
                    return *this;
                const auto& source = *value.mesh;
                auto result = std::make_shared<core::MeshData>();
                result->vertices = kwargs.contains("vertices")
                                       ? nb::cast<PyTensor>(kwargs["vertices"]).tensor()
                                       : source.vertices;
                result->indices = kwargs.contains("indices")
                                      ? nb::cast<PyTensor>(kwargs["indices"]).tensor()
                                      : source.indices;
                result->normals = source.normals;
                result->tangents = source.tangents;
                result->texcoords = source.texcoords;
                result->colors = source.colors;
                result->materials = source.materials;
                result->submeshes = source.submeshes;
                result->texture_images = source.texture_images;
                return PyMesh{MeshComponent{std::move(result)}};
            }
        };

        struct PyGeometry {
            Geometry value;

            PyGeometry replace(const nb::kwargs& kwargs) const {
                PyGeometry result = *this;
                if (kwargs.contains("splats")) {
                    result.value.splats = kwargs["splats"].is_none()
                                              ? std::optional<SplatsComponent>{}
                                              : nb::cast<PySplats>(kwargs["splats"]).value;
                }
                if (kwargs.contains("points")) {
                    result.value.points = kwargs["points"].is_none()
                                              ? std::optional<PointsComponent>{}
                                              : nb::cast<PyPoints>(kwargs["points"]).value;
                }
                if (kwargs.contains("mesh")) {
                    result.value.mesh = kwargs["mesh"].is_none()
                                            ? std::optional<MeshComponent>{}
                                            : nb::cast<PyMesh>(kwargs["mesh"]).value;
                }
                return result;
            }
        };

        nb::object value_to_python(const Value& value) {
            if (const auto* geometry = value.get_if<Geometry>())
                return nb::cast(PyGeometry{*geometry});
            if (const auto* number = value.get_if<float>())
                return nb::float_(*number);
            if (const auto* integer = value.get_if<std::int64_t>())
                return nb::int_(*integer);
            if (const auto* boolean = value.get_if<bool>())
                return nb::bool_(*boolean);
            if (const auto* vector = value.get_if<glm::vec3>())
                return nb::make_tuple(vector->x, vector->y, vector->z);
            if (const auto* color = value.get_if<glm::vec4>())
                return nb::make_tuple(color->x, color->y, color->z, color->w);
            if (const auto* text = value.get_if<std::string>())
                return nb::str(text->c_str());
            return nb::none();
        }

        struct PyTree {
            std::string uuid;
        };

        struct PyNode {
            std::string tree_uuid;
            std::string name;
        };

        NodeTree& require_tree(const PyTree& handle) {
            auto* tree = active_tree(handle.uuid);
            if (!tree)
                throw nb::key_error("Node tree no longer exists");
            return *tree;
        }

        Node& require_node(const PyNode& handle) {
            auto* tree = active_tree(handle.tree_uuid);
            auto* node = tree ? tree->find_node(handle.name) : nullptr;
            if (!node)
                throw nb::key_error("Node no longer exists");
            return *node;
        }

        void record_tree_mutation(NodeTree& tree, nlohmann::json before,
                                  std::string merge_key = {}) {
            if (auto* manager = live_manager())
                manager->recordTreeEdit(tree.uuid, std::move(before), std::move(merge_key));
            else if (tree.name != before.value("name", ""))
                tree.name = standalone_tree_name(tree.name, tree.uuid);
        }

        nlohmann::json modifier_stack_json(const core::Uuid& node_uuid) {
            const auto* manager = live_manager();
            const auto* stack = manager ? manager->stack(node_uuid) : nullptr;
            return stack ? nlohmann::json(stack->modifiers) : nlohmann::json::array();
        }

        struct PyInputDecl {
            std::string identifier;
            std::string type;
            Value default_value;
            std::optional<double> min;
            std::optional<double> max;
            bool field = false;
        };

        struct PyOutputDecl {
            std::string identifier;
            std::string type;
        };

        struct PyPropertyDecl {
            std::string identifier;
            std::string type;
            nlohmann::json default_value;
            std::vector<std::string> items;
        };

        struct PyNodeBase {};

        using SafeClass = std::shared_ptr<nb::object>;
        std::mutex& python_types_mutex() {
            static auto* value = new std::mutex;
            return *value;
        }

        std::unordered_map<std::string, SafeClass>& python_types() {
            static auto* value = new std::unordered_map<std::string, SafeClass>;
            return *value;
        }

        std::unordered_map<std::string, std::string>& python_type_modules() {
            static auto* value = new std::unordered_map<std::string, std::string>;
            return *value;
        }

        SafeClass safe_class(nb::object type) {
            return {new nb::object(std::move(type)), [](nb::object* object) {
                        if (!Py_IsInitialized())
                            return;
                        if (PyGILState_Check()) {
                            delete object;
                        } else {
                            nb::gil_scoped_acquire gil;
                            delete object;
                        }
                    }};
        }

        struct PyNodeContext {
            NodeContext* context = nullptr;

            nb::object input(const std::string& name) const {
                return value_to_python(context->input(name));
            }

            PyTensor field(const std::string& name, const PyGeometry& geometry) const {
                FieldContext field_context;
                if (geometry.value.splats) {
                    field_context.domain = Domain::Splat;
                    field_context.splats = &*geometry.value.splats;
                } else if (geometry.value.points) {
                    field_context.domain = Domain::Point;
                    field_context.points = &*geometry.value.points;
                } else if (geometry.value.mesh) {
                    field_context.domain = Domain::Vertex;
                    field_context.mesh = &*geometry.value.mesh;
                } else {
                    throw nb::value_error("Geometry has no component for field evaluation");
                }
                return PyTensor(context->evaluate_field(name, field_context));
            }

            nb::object prop(const std::string& name) const {
                const auto found = context->properties().find(name);
                return found == context->properties().end() ? nb::none()
                                                            : json_to_python(*found);
            }

            void output(const std::string& name, const nb::handle value) const {
                if (nb::isinstance<PyGeometry>(value))
                    context->set_output(name, nb::cast<PyGeometry>(value).value);
                else
                    context->set_output(name, python_to_value(value));
            }
        };

        std::string socket_type(std::string value) {
            if (value.starts_with("lfs."))
                return value;
            std::ranges::transform(value, value.begin(), [](const unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return "lfs." + value;
        }

        NodeTypeInfo node_type_from_python(const nb::object& cls) {
            NodeTypeInfo info;
            info.id = nb::cast<std::string>(cls.attr("id"));
            info.label = nb::hasattr(cls, "label") ? nb::cast<std::string>(cls.attr("label")) : info.id;
            info.category = nb::hasattr(cls, "category")
                                ? nb::cast<std::string>(cls.attr("category"))
                                : "Utilities";
            info.description = nb::hasattr(cls, "description")
                                   ? nb::cast<std::string>(cls.attr("description"))
                                   : std::string{};
            std::unordered_set<std::string> declared;
            if (!nb::hasattr(cls, "execute") || !PyCallable_Check(cls.attr("execute").ptr()))
                throw nb::type_error("Node classes must implement execute(self, ctx)");
            const auto append = [&](const nb::handle value, std::string_view collection) {
                if (nb::isinstance<PyInputDecl>(value)) {
                    const auto& declaration = nb::cast<const PyInputDecl&>(value);
                    const auto& identifier = declaration.identifier;
                    if (collection != "inputs" || identifier.empty() || !declared.insert("input:" + identifier).second)
                        throw nb::value_error("inputs must contain Input declarations with unique, non-empty identifiers");
                    info.inputs.push_back({identifier, identifier, declaration.type,
                                           declaration.default_value, declaration.min,
                                           declaration.max, std::nullopt, declaration.field});
                } else if (nb::isinstance<PyOutputDecl>(value)) {
                    const auto& declaration = nb::cast<const PyOutputDecl&>(value);
                    const auto& identifier = declaration.identifier;
                    if (collection != "outputs" || identifier.empty() || !declared.insert("output:" + identifier).second)
                        throw nb::value_error("outputs must contain Output declarations with unique, non-empty identifiers");
                    info.outputs.push_back({identifier, identifier, declaration.type});
                } else if (nb::isinstance<PyPropertyDecl>(value)) {
                    const auto& declaration = nb::cast<const PyPropertyDecl&>(value);
                    const auto& identifier = declaration.identifier;
                    if (collection != "properties" || identifier.empty() || !declared.insert("property:" + identifier).second)
                        throw nb::value_error("properties must contain Property declarations with unique, non-empty identifiers");
                    PropertyKind kind = PropertyKind::String;
                    if (declaration.type == "bool")
                        kind = PropertyKind::Bool;
                    else if (declaration.type == "int")
                        kind = PropertyKind::Int;
                    else if (declaration.type == "float")
                        kind = PropertyKind::Float;
                    else if (declaration.type == "enum")
                        kind = PropertyKind::Enum;
                    info.properties.push_back({identifier, identifier, kind,
                                               declaration.default_value, declaration.items});
                } else {
                    throw nb::type_error("Node declarations must be Input, Output or Property objects in their corresponding lists");
                }
            };
            for (const auto* collection : {"inputs", "outputs", "properties"}) {
                if (!nb::hasattr(cls, collection))
                    continue;
                if (!nb::isinstance<nb::list>(cls.attr(collection)))
                    throw nb::type_error("Node inputs, outputs and properties must be lists");
                for (const auto value : nb::cast<nb::list>(cls.attr(collection)))
                    append(value, collection);
            }
            for (const auto base : nb::cast<nb::tuple>(cls.attr("__mro__"))) {
                const auto attributes = nb::cast<nb::dict>(nb::module_::import_("builtins").attr("dict")(base.attr("__dict__")));
                for (const auto [key, value] : attributes) {
                    if (nb::isinstance<PyInputDecl>(value) || nb::isinstance<PyOutputDecl>(value) || nb::isinstance<PyPropertyDecl>(value))
                        throw nb::type_error("Attribute-style node declarations are unsupported; use inputs, outputs and properties lists");
                }
            }
            const std::string type_id = info.id;
            info.evaluate = [type_id](NodeContext& context) {
                SafeClass type;
                {
                    std::lock_guard lock(python_types_mutex());
                    const auto found = python_types().find(type_id);
                    if (found == python_types().end())
                        throw NodeError("Python node type is not registered");
                    type = found->second;
                }
                nb::gil_scoped_acquire gil;
                try {
                    auto instance = (**type)();
                    const auto callback = instance.attr("execute");
                    auto result = callback(PyNodeContext{&context});
                    if (!result.is_none()) {
                        if (nb::isinstance<PyGeometry>(result))
                            context.set_output("Geometry", nb::cast<PyGeometry>(result).value);
                        else if (nb::isinstance<nb::dict>(result))
                            for (const auto [key, value] : nb::cast<nb::dict>(result))
                                PyNodeContext{&context}.output(nb::cast<std::string>(key), value);
                    }
                } catch (const nb::python_error& error) {
                    LOG_WARN("Python node {} failed:\n{}", type_id, error.what());
                    const auto name = nb::cast<std::string>(error.type().attr("__name__"));
                    throw NodeError(name + ": " + nb::cast<std::string>(nb::str(error.value())));
                }
            };
            return info;
        }

        void install_python_type(const NodeTypeInfo& info) {
            standalone().registry.unregister_type(info.id);
            standalone().registry.register_type(info);
            invoke_on_viewer([info] {
                if (auto* manager = live_manager()) {
                    manager->registry().unregister_type(info.id);
                    manager->registry().register_type(info);
                    manager->markDirty();
                }
            });
        }

        bool unregister_python_type(const std::string& id) {
            standalone().registry.unregister_type(id);
            invoke_on_viewer([id] {
                if (auto* manager = live_manager()) {
                    manager->registry().unregister_type(id);
                    manager->markDirty();
                }
            });
            std::lock_guard lock(python_types_mutex());
            python_type_modules().erase(id);
            return python_types().erase(id) != 0;
        }

        size_t unregister_python_types_for_module(const std::string& prefix) {
            if (prefix.empty())
                return 0;
            std::vector<std::string> ids;
            {
                std::lock_guard lock(python_types_mutex());
                for (const auto& [id, module_name] : python_type_modules()) {
                    if (module_name == prefix || module_name.starts_with(prefix + "."))
                        ids.push_back(id);
                }
            }
            for (const auto& id : ids)
                unregister_python_type(id);
            return ids.size();
        }

        nb::dict evaluation_dict(const EvalResult& result) {
            nb::dict errors;
            for (const auto& [node, message] : result.errors)
                errors[nb::str(node.c_str())] = message;
            double total = 0.0;
            for (const auto& [_, time] : result.time_ms)
                total += time;
            nb::dict value;
            value["ok"] = result.ok;
            value["errors"] = std::move(errors);
            value["time_ms"] = total;
            return value;
        }

        std::optional<core::Uuid> scene_node_uuid(const std::string& name) {
            auto* scene_manager = get_scene_manager();
            if (!scene_manager)
                return std::nullopt;
            const auto* node = scene_manager->getScene().getNode(name);
            return node ? std::optional{node->uuid} : std::nullopt;
        }

        struct PyModifier {
            core::Uuid node_uuid;
            std::string uuid;
        };

        vis::Modifier& require_modifier(const PyModifier& handle) {
            auto* manager = live_manager();
            auto* stack = manager ? manager->stack(handle.node_uuid) : nullptr;
            if (stack) {
                const auto found = std::ranges::find(stack->modifiers, handle.uuid,
                                                     &vis::Modifier::uuid);
                if (found != stack->modifiers.end())
                    return *found;
            }
            throw nb::key_error("Modifier no longer exists");
        }
    } // namespace

    void register_nodes(nb::module_& module) {
        nb::class_<PySplats>(module, "Splats")
            .def("__init__", [](PySplats* self, PyTensor means, PyTensor sh0, PyTensor shN, PyTensor scaling, PyTensor rotation, PyTensor opacity, int sh_degree, float scene_scale) { new (self) PySplats{{means.tensor(), sh0.tensor(), shN.tensor(), scaling.tensor(), rotation.tensor(), opacity.tensor(), sh_degree, scene_scale, {}}}; }, nb::arg("means"), nb::arg("sh0"), nb::arg("shN"), nb::arg("scaling"), nb::arg("rotation"), nb::arg("opacity"), nb::arg("sh_degree") = 0, nb::arg("scene_scale") = 1.0f)
            .def_prop_ro("means", [](const PySplats& value) { return PyTensor(value.value.means); })
            .def_prop_ro("sh0", [](const PySplats& value) { return PyTensor(value.value.sh0); })
            .def_prop_ro("shN", [](const PySplats& value) { return PyTensor(value.value.shN); })
            .def_prop_ro("scaling", [](const PySplats& value) { return PyTensor(value.value.scaling); })
            .def_prop_ro("rotation", [](const PySplats& value) { return PyTensor(value.value.rotation); })
            .def_prop_ro("opacity", [](const PySplats& value) { return PyTensor(value.value.opacity); })
            .def("replace", &PySplats::replace);

        nb::class_<PyPoints>(module, "Points")
            .def("__init__", [](PyPoints* self, PyTensor positions, PyTensor colors) { new (self) PyPoints{{positions.tensor(), colors.tensor(), {}}}; }, nb::arg("positions"), nb::arg("colors"))
            .def_prop_ro("positions", [](const PyPoints& value) { return PyTensor(value.value.positions); })
            .def_prop_ro("colors", [](const PyPoints& value) { return PyTensor(value.value.colors); })
            .def("replace", &PyPoints::replace);

        nb::class_<PyMesh>(module, "Mesh")
            .def_prop_ro("vertices", [](const PyMesh& value) {
                return PyTensor(value.value.mesh ? value.value.mesh->vertices : core::Tensor{});
            })
            .def_prop_ro("indices", [](const PyMesh& value) {
                return PyTensor(value.value.mesh ? value.value.mesh->indices : core::Tensor{});
            })
            .def("replace", &PyMesh::replace);

        nb::class_<PyGeometry>(module, "Geometry")
            .def(nb::init<>())
            .def("__init__", [](PyGeometry* self, std::optional<PySplats> splats, std::optional<PyPoints> points, std::optional<PyMesh> mesh) { new (self) PyGeometry{{splats ? std::optional{splats->value} : std::nullopt,
                                                                                                                                                                       points ? std::optional{points->value} : std::nullopt,
                                                                                                                                                                       mesh ? std::optional{mesh->value} : std::nullopt}}; }, nb::arg("splats") = nb::none(), nb::arg("points") = nb::none(), nb::arg("mesh") = nb::none())
            .def_prop_ro("splats", [](const PyGeometry& value) -> std::optional<PySplats> { return value.value.splats ? std::optional{PySplats{*value.value.splats}} : std::nullopt; })
            .def_prop_ro("points", [](const PyGeometry& value) -> std::optional<PyPoints> { return value.value.points ? std::optional{PyPoints{*value.value.points}} : std::nullopt; })
            .def_prop_ro("mesh", [](const PyGeometry& value) -> std::optional<PyMesh> { return value.value.mesh ? std::optional{PyMesh{*value.value.mesh}} : std::nullopt; })
            .def_prop_ro("empty", [](const PyGeometry& value) { return value.value.empty(); })
            .def("replace", &PyGeometry::replace);

        nb::class_<PyInputDecl>(module, "Input")
            .def("__init__", [](PyInputDecl* self, std::string identifier, std::string type, nb::object default_value, std::optional<double> min, std::optional<double> max, bool field) { new (self) PyInputDecl{std::move(identifier), socket_type(std::move(type)), python_to_value(default_value), min, max, field}; }, nb::arg("identifier"), nb::arg("type"), nb::arg("default") = nb::none(), nb::arg("min") = nb::none(), nb::arg("max") = nb::none(), nb::arg("field") = false);
        nb::class_<PyOutputDecl>(module, "Output")
            .def("__init__", [](PyOutputDecl* self, std::string identifier, std::string type) { new (self) PyOutputDecl{std::move(identifier), socket_type(std::move(type))}; }, nb::arg("identifier"), nb::arg("type"));
        nb::class_<PyPropertyDecl>(module, "Property")
            .def("__init__", [](PyPropertyDecl* self, std::string identifier, std::string type, nb::object default_value, std::vector<std::string> items) { new (self) PyPropertyDecl{std::move(identifier), std::move(type), python_to_json(default_value), std::move(items)}; }, nb::arg("identifier"), nb::arg("type"), nb::arg("default") = nb::none(), nb::arg("items") = std::vector<std::string>{});
        nb::class_<PyNodeBase>(module, "Node", nb::dynamic_attr()).def(nb::init<>());
        nb::class_<PyNodeContext>(module, "NodeContext")
            .def("input", &PyNodeContext::input)
            .def("field", &PyNodeContext::field)
            .def("field", [](const PyNodeContext& context, const std::string& name,
                             const PySplats& component) {
                return context.field(name, PyGeometry{Geometry{.splats = component.value}});
            })
            .def("field", [](const PyNodeContext& context, const std::string& name, const PyPoints& component) {
                return context.field(name, PyGeometry{Geometry{.points = component.value}});
            })
            .def("field", [](const PyNodeContext& context, const std::string& name, const PyMesh& component) {
                return context.field(name, PyGeometry{Geometry{.mesh = component.value}});
            })
            .def("prop", &PyNodeContext::prop)
            .def("output", &PyNodeContext::output);

        nb::class_<PyNode>(module, "NodeHandle")
            .def_prop_ro("name", [](const PyNode& value) { return value.name; })
            .def_prop_ro("type_id", [](const PyNode& value) { return require_node(value).type_id; })
            .def_prop_rw("location", [](const PyNode& value) { return require_node(value).location; }, [](const PyNode& value, std::array<float, 2> location) { invoke_on_viewer([value, location] {
                                                                                                                                                                    auto& tree = *active_tree(value.tree_uuid);
                                                                                                                                                                    auto before = tree.to_json();
                                                                                                                                                                    require_node(value).location = location;
                                                                                                                                                                    record_tree_mutation(tree, std::move(before),
                                                                                                                                                                                         value.name + ":location");
                                                                                                                                                                }); })
            .def_prop_rw("muted", [](const PyNode& value) { return require_node(value).muted; }, [](const PyNode& value, bool muted) { invoke_on_viewer([value, muted] {
                                                                                                                                           auto& tree = *active_tree(value.tree_uuid);
                                                                                                                                           auto before = tree.to_json();
                                                                                                                                           require_node(value).muted = muted;
                                                                                                                                           record_tree_mutation(tree, std::move(before),
                                                                                                                                                                value.name + ":muted");
                                                                                                                                       }); })
            .def("set_input", [](const PyNode& node, const std::string& name, nb::object value) {
                auto converted = python_to_value(value);
                invoke_on_viewer([node, name, converted = std::move(converted)]() mutable {
                    if (auto* manager = live_manager()) {
                        const auto result = manager->setNodeInput(node.tree_uuid, node.name, name, std::move(converted));
                        if (!result)
                            throw std::invalid_argument(result.error().message);
                        return;
                    }
                    auto& tree = *active_tree(node.tree_uuid);
                    auto before = tree.to_json();
                    require_node(node).input_values[name] = std::move(converted);
                    record_tree_mutation(tree, std::move(before), node.name + ":input:" + name);
                }); })
            .def("get_input", [](const PyNode& node, const std::string& name) {
                const auto& values = require_node(node).input_values;
                const auto found = values.find(name);
                return found == values.end() ? nb::none() : value_to_python(found->second); })
            .def("set_property", [](const PyNode& node, const std::string& name, nb::object value) {
                auto converted = python_to_json(value);
                invoke_on_viewer([node, name, converted = std::move(converted)]() mutable {
                    auto& tree = *active_tree(node.tree_uuid);
                    auto before = tree.to_json();
                    require_node(node).properties[name] = std::move(converted);
                    record_tree_mutation(tree, std::move(before),
                                         node.name + ":property:" + name);
                }); })
            .def("get_property", [](const PyNode& node, const std::string& name) {
                const auto& properties = require_node(node).properties;
                const auto found = properties.find(name);
                return found == properties.end() ? nb::none() : json_to_python(*found); })
            .def_prop_ro("error", [](const PyNode& node) -> std::optional<std::string> {
                const auto found = standalone().evaluations.find(node.tree_uuid);
                if (found == standalone().evaluations.end())
                    return std::nullopt;
                const auto error = found->second.errors.find(node.name);
                return error == found->second.errors.end() ? std::nullopt
                                                           : std::optional{error->second}; })
            .def_prop_ro("eval_time_ms", [](const PyNode& node) {
                const auto found = standalone().evaluations.find(node.tree_uuid);
                if (found == standalone().evaluations.end())
                    return 0.0;
                const auto time = found->second.time_ms.find(node.name);
                return time == found->second.time_ms.end() ? 0.0 : time->second; });

        nb::class_<PyTree>(module, "NodeTree")
            .def_prop_ro("uuid", [](const PyTree& value) { return value.uuid; })
            .def_prop_rw("name", [](const PyTree& value) { return require_tree(value).name; }, [](const PyTree& value, std::string name) { invoke_on_viewer([value, name = std::move(name)]() mutable {
                                                                                                                                               auto& tree = require_tree(value);
                                                                                                                                               auto before = tree.to_json();
                                                                                                                                               tree.name = std::move(name);
                                                                                                                                               record_tree_mutation(tree, std::move(before), "tree:name");
                                                                                                                                           }); })
            .def_prop_ro("nodes", [](const PyTree& value) {
                std::vector<PyNode> result;
                for (const auto& node : require_tree(value).nodes)
                    result.push_back({value.uuid, node.name});
                return result; })
            .def_prop_ro("input_node", [](const PyTree& value) { return PyNode{value.uuid, require_tree(value).input_node().name}; })
            .def_prop_ro("output_node", [](const PyTree& value) { return PyNode{value.uuid, require_tree(value).output_node().name}; })
            .def("add_node", [](const PyTree& value, std::string type_id, std::string name, std::array<float, 2> location) {
                const auto node_name = invoke_on_viewer(
                    [value, type_id = std::move(type_id), name = std::move(name), location]() mutable {
                        auto& tree = require_tree(value);
                        auto before = tree.to_json();
                        auto& node = tree.add_node(std::move(type_id), std::move(name));
                        node.location = location;
                        const auto result = node.name;
                        record_tree_mutation(tree, std::move(before));
                        return result;
                    }, std::string{});
                return PyNode{value.uuid, node_name}; }, nb::arg("type_id"), nb::arg("name") = "", nb::arg("location") = std::array<float, 2>{0.0f, 0.0f})
            .def("remove_node", [](const PyTree& value, const std::string& name) { return invoke_on_viewer([value, name] {
                                                                                       auto& tree = require_tree(value);
                                                                                       auto before = tree.to_json();
                                                                                       const bool removed = tree.remove_node(name);
                                                                                       if (removed)
                                                                                           record_tree_mutation(tree, std::move(before));
                                                                                       return removed;
                                                                                   },
                                                                                                           false); })
            .def("link", [](const PyTree& value, const PyNode& from, const std::string& output, const PyNode& to, const std::string& input) {
                const auto error = invoke_on_viewer([value, from, output, to, input] {
                    auto& tree = require_tree(value);
                    auto before = tree.to_json();
                    std::string message;
                    if (tree.add_link({from.name, output, to.name, input}, &message))
                        record_tree_mutation(tree, std::move(before));
                    return message;
                }, std::string{"Viewer is unavailable"});
                if (!error.empty())
                    throw nb::value_error(error.c_str()); })
            .def("unlink", [](const PyTree& value, const PyNode& from, const std::string& output, const PyNode& to, const std::string& input) { return invoke_on_viewer([value, from, output, to, input] {
                                                                                                                                                    auto& tree = require_tree(value);
                                                                                                                                                    auto before = tree.to_json();
                                                                                                                                                    const bool removed = tree.remove_link({from.name, output, to.name, input});
                                                                                                                                                    if (removed)
                                                                                                                                                        record_tree_mutation(tree, std::move(before));
                                                                                                                                                    return removed;
                                                                                                                                                },
                                                                                                                                                                        false); })
            .def("add_input", [](const PyTree& value, std::string identifier, std::string type, nb::object default_value, std::optional<double> min, std::optional<double> max) {
                auto converted = python_to_value(default_value);
                invoke_on_viewer([value, identifier = std::move(identifier),
                                  type = socket_type(std::move(type)),
                                  converted = std::move(converted), min, max]() mutable {
                    auto& tree = require_tree(value);
                    auto before = tree.to_json();
                    tree.interface.inputs.push_back(
                        {identifier, identifier, std::move(type), std::move(converted),
                         min, max, std::nullopt});
                    record_tree_mutation(tree, std::move(before));
                }); }, nb::arg("identifier"), nb::arg("type"), nb::arg("default") = nb::none(), nb::arg("min") = nb::none(), nb::arg("max") = nb::none())
            .def("to_json", [](const PyTree& value) { return require_tree(value).to_json().dump(2); });

        nb::class_<PyModifier>(module, "Modifier")
            .def_prop_ro("name", [](const PyModifier& value) { return require_modifier(value).name; })
            .def_prop_rw("enabled", [](const PyModifier& value) { return require_modifier(value).enabled; }, [](const PyModifier& value, bool enabled) { invoke_on_viewer([value, enabled] {
                                                                                                                                                             auto before = modifier_stack_json(value.node_uuid);
                                                                                                                                                             require_modifier(value).enabled = enabled;
                                                                                                                                                             live_manager()->recordStackEdit(
                                                                                                                                                                 value.node_uuid, std::move(before),
                                                                                                                                                                 value.uuid + ":enabled");
                                                                                                                                                         }); })
            .def_prop_rw("show_viewport", [](const PyModifier& value) { return require_modifier(value).show_viewport; }, [](const PyModifier& value, bool visible) { invoke_on_viewer([value, visible] {
                                                                                                                                                                         auto before = modifier_stack_json(value.node_uuid);
                                                                                                                                                                         require_modifier(value).show_viewport = visible;
                                                                                                                                                                         live_manager()->recordStackEdit(
                                                                                                                                                                             value.node_uuid, std::move(before),
                                                                                                                                                                             value.uuid + ":show_viewport");
                                                                                                                                                                     }); })
            .def("set_input", [](const PyModifier& value, const std::string& name, nb::object input) {
                auto converted = python_to_value(input);
                invoke_on_viewer([value, name, converted = std::move(converted)]() mutable {
                    auto before = modifier_stack_json(value.node_uuid);
                    require_modifier(value).input_overrides[name] = std::move(converted);
                    live_manager()->recordStackEdit(
                        value.node_uuid, std::move(before),
                        value.uuid + ":input:" + name);
                }); })
            .def("capture_selection", [](const PyModifier& value, const std::string& node_name) {
                const auto error = invoke_on_viewer([value, node_name] {
                    auto before = modifier_stack_json(value.node_uuid);
                    const auto result = live_manager()->captureSelection(
                        value.node_uuid, require_modifier(value).name, node_name);
                    if (!result)
                        return result.error().message;
                    live_manager()->recordStackEdit(value.node_uuid, std::move(before));
                    return std::string{};
                }, std::string{"Viewer is unavailable"});
                if (!error.empty())
                    throw nb::value_error(error.c_str()); });

        module.def("node_types", [] {
            nb::list result;
            for (const auto& type : active_registry().list()) {
                nb::dict item;
                item["id"] = type->id;
                item["label"] = type->label;
                item["category"] = type->category;
                item["description"] = type->description;
                item["version"] = type->version;
                nb::list inputs;
                for (const auto& socket : type->inputs) {
                    nb::dict descriptor;
                    descriptor["identifier"] = socket.identifier;
                    descriptor["label"] = socket.label;
                    descriptor["type"] = socket.type;
                    descriptor["default"] = value_to_python(socket.default_value);
                    descriptor["min"] = socket.min ? nb::cast(*socket.min) : nb::none();
                    descriptor["max"] = socket.max ? nb::cast(*socket.max) : nb::none();
                    descriptor["soft_min"] = socket.soft_min ? nb::cast(*socket.soft_min) : nb::none();
                    descriptor["soft_max"] = socket.soft_max ? nb::cast(*socket.soft_max) : nb::none();
                    descriptor["step"] = socket.step ? nb::cast(*socket.step) : nb::none();
                    descriptor["field"] = socket.field;
                    descriptor["multi_input"] = socket.multi_input;
                    descriptor["hide_value"] = socket.hide_value;
                    inputs.append(std::move(descriptor));
                }
                item["inputs"] = std::move(inputs);
                nb::list outputs;
                for (const auto& socket : type->outputs) {
                    nb::dict descriptor;
                    descriptor["identifier"] = socket.identifier;
                    descriptor["label"] = socket.label;
                    descriptor["type"] = socket.type;
                    descriptor["multi_input"] = socket.multi_input;
                    outputs.append(std::move(descriptor));
                }
                item["outputs"] = std::move(outputs);
                nb::list properties;
                for (const auto& property : type->properties) {
                    nb::dict descriptor;
                    descriptor["identifier"] = property.identifier;
                    descriptor["label"] = property.label;
                    descriptor["kind"] = static_cast<int>(property.kind);
                    descriptor["default"] = json_to_python(property.default_value);
                    descriptor["items"] = property.items;
                    descriptor["min"] = property.min ? nb::cast(*property.min) : nb::none();
                    descriptor["max"] = property.max ? nb::cast(*property.max) : nb::none();
                    properties.append(std::move(descriptor));
                }
                item["properties"] = std::move(properties);
                result.append(std::move(item));
            }
            return result;
        });
        module.def("new_tree", [](std::string name) {
            const auto uuid = invoke_on_viewer([name = std::move(name)]() mutable {
                if (auto* manager = live_manager())
                    return manager->newTree(std::move(name)).uuid;
                auto tree = std::make_unique<NodeTree>(standalone().registry, standalone_tree_name(std::move(name)));
                const auto result = tree->uuid;
                standalone().trees[result] = std::move(tree);
                return result;
            }, std::string{});
            return PyTree{uuid}; }, nb::arg("name") = "Node Graph");
        module.def("get_tree", [](const std::string& name) -> std::optional<PyTree> {
            if (auto* manager = live_manager()) {
                const auto* tree = manager->tree(name);
                return tree ? std::optional{PyTree{tree->uuid}} : std::nullopt;
            }
            if (const auto found = standalone().trees.find(name); found != standalone().trees.end())
                return PyTree{found->first};
            for (const auto& [uuid, tree] : standalone().trees)
                if (tree->name == name)
                    return PyTree{uuid};
            return std::nullopt;
        });
        module.def("trees", [] {
            std::vector<PyTree> result;
            if (auto* manager = live_manager()) {
                for (const auto* tree : manager->trees())
                    result.push_back({tree->uuid});
            } else {
                for (const auto& [uuid, _] : standalone().trees)
                    result.push_back({uuid});
            }
            return result;
        });
        module.def("remove_tree", [](const std::string& name) {
            return invoke_on_viewer([name] {
                if (auto* manager = live_manager())
                    return manager->removeTree(name);
                const auto* tree = active_tree(name);
                return tree && standalone().trees.erase(tree->uuid) != 0;
            },
                                    false);
        });
        module.def("load_tree", [](const std::string& text) {
            const auto json = nlohmann::json::parse(text);
            const auto uuid = invoke_on_viewer([json] {
                if (auto* manager = live_manager())
                    return manager->loadTree(json).uuid;
                auto tree = std::make_unique<NodeTree>(
                    NodeTree::from_json(json, standalone().registry));
                tree->name = standalone_tree_name(tree->name, tree->uuid);
                const auto result = tree->uuid;
                standalone().trees[result] = std::move(tree);
                return result;
            },
                                               std::string{});
            return PyTree{uuid};
        });
        module.def("link", [](const PyTree& tree, const PyNode& from, const std::string& output,
                              const PyNode& to, const std::string& input) {
            const auto error = invoke_on_viewer([tree, from, output, to, input] {
                auto& value = require_tree(tree);
                auto before = value.to_json();
                std::string message;
                if (value.add_link({from.name, output, to.name, input}, &message))
                    record_tree_mutation(value, std::move(before));
                return message;
            },
                                                std::string{"Viewer is unavailable"});
            if (!error.empty())
                throw nb::value_error(error.c_str());
        });
        module.def("unlink", [](const PyTree& tree, const PyNode& from, const std::string& output,
                                const PyNode& to, const std::string& input) {
            return invoke_on_viewer([tree, from, output, to, input] {
                auto& value = require_tree(tree);
                auto before = value.to_json();
                const bool removed = value.remove_link({from.name, output, to.name, input});
                if (removed)
                    record_tree_mutation(value, std::move(before));
                return removed;
            },
                                    false);
        });
        module.def("evaluate_tree", [](const PyTree& tree, const PyGeometry& geometry) {
            auto result = evaluate(require_tree(tree), {.geometry = geometry.value});
            standalone().evaluations[tree.uuid] = result;
            if (!result.ok) {
                std::string messages;
                for (const auto& [node, message] : result.errors) {
                    if (!messages.empty())
                        messages += "; ";
                    messages += node + ": " + message;
                }
                throw nb::value_error(messages.c_str());
            }
            return PyGeometry{std::move(result.geometry)};
        });
        module.def("register_node", [](nb::object cls) {
            auto info = node_type_from_python(cls);
            const auto module_name = nb::cast<std::string>(cls.attr("__module__"));
            {
                std::lock_guard lock(python_types_mutex());
                python_types()[info.id] = safe_class(std::move(cls));
                python_type_modules()[info.id] = module_name;
            }
            install_python_type(info);
            return info.id;
        });
        module.def("unregister_node", &unregister_python_type);
        module.def("unregister_nodes_for_module", &unregister_python_types_for_module);

        module.def("add_modifier", [](const std::string& node_name, const PyTree& tree, std::string name) {
            const auto result = invoke_on_viewer(
                [node_name, tree, name = std::move(name)]() mutable
                    -> std::optional<PyModifier> {
                    auto uuid = scene_node_uuid(node_name);
                    if (!uuid || !live_manager())
                        return std::nullopt;
                    auto& modifier = live_manager()->addModifier(
                        *uuid, tree.uuid, std::move(name));
                    return PyModifier{*uuid, modifier.uuid};
                }, std::optional<PyModifier>{});
            if (!result)
                throw std::runtime_error("Scene node or scene manager is unavailable");
            return *result; }, nb::arg("node_name"), nb::arg("tree"), nb::arg("name") = "");
        module.def("modifiers", [](const std::string& node_name) {
            std::vector<PyModifier> result;
            const auto uuid = scene_node_uuid(node_name);
            const auto* stack = uuid && live_manager() ? live_manager()->stack(*uuid) : nullptr;
            if (stack)
                for (const auto& modifier : stack->modifiers)
                    result.push_back({*uuid, modifier.uuid});
            return result;
        });
        module.def("remove_modifier", [](const std::string& node_name, const std::string& name) {
            return invoke_on_viewer([node_name, name] {
                const auto uuid = scene_node_uuid(node_name);
                return uuid && live_manager() && live_manager()->removeModifier(*uuid, name);
            },
                                    false);
        });
        module.def("move_modifier", [](const std::string& node_name, const std::string& name,
                                       size_t index) {
            return invoke_on_viewer([node_name, name, index] {
                const auto uuid = scene_node_uuid(node_name);
                return uuid && live_manager() && live_manager()->moveModifier(*uuid, name, index);
            },
                                    false);
        });
        module.def("performance", [](const bool reset) {
            const auto result = invoke_on_viewer([reset] {
                return live_manager() ? live_manager()->performance(reset) : nlohmann::json::object();
            }, nlohmann::json::object());
            return json_to_python(result); }, nb::arg("reset") = false);
        module.def("evaluate", [](const std::string& node_name) {
            const auto result = invoke_on_viewer([node_name] {
                std::optional<nb::gil_scoped_release> release;
                if (PyGILState_Check())
                    release.emplace();
                const auto uuid = scene_node_uuid(node_name);
                return uuid && live_manager()
                           ? std::optional{live_manager()->evaluate(*uuid)}
                           : std::optional<vis::ModifierEvaluation>{};
            },
                                                 std::optional<vis::ModifierEvaluation>{});
            if (!result)
                throw std::runtime_error("Scene node or scene manager is unavailable");
            EvalResult core_result{.geometry = result->geometry, .ok = result->ok, .errors = result->errors, .time_ms = result->time_ms};
            return evaluation_dict(core_result);
        });
        module.def("evaluated", [](const std::string& node_name) -> std::optional<PyGeometry> {
            const auto geometry = invoke_on_viewer([node_name]() -> std::optional<Geometry> {
                const auto uuid = scene_node_uuid(node_name);
                return uuid && live_manager() ? live_manager()->evaluated(*uuid) : std::nullopt;
            },
                                                   std::optional<Geometry>{});
            return geometry ? std::optional{PyGeometry{*geometry}} : std::nullopt;
        });
        module.def("apply_modifier", [](const std::string& node_name, const std::string& name) {
            const auto error = invoke_on_viewer([node_name, name] {
                std::optional<nb::gil_scoped_release> release;
                if (PyGILState_Check())
                    release.emplace();
                const auto uuid = scene_node_uuid(node_name);
                if (!uuid || !live_manager())
                    return std::string{"Scene node or scene manager is unavailable"};
                const auto result = live_manager()->applyModifier(*uuid, name);
                return result ? std::string{} : result.error().message;
            },
                                                std::string{"Viewer is unavailable"});
            if (!error.empty())
                throw std::runtime_error(error);
        });
        // Release callback classes before Python tears down nanobind's types.
        // A module capsule runs too late and retains their declaration objects.
        nb::module_::import_("atexit").attr("register")(nb::cpp_function([] {
            std::lock_guard lock(python_types_mutex());
            python_types().clear();
            python_type_modules().clear();
        }));
    }
} // namespace lfs::python
