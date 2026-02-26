/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_rml.hpp"
#include "core/logger.hpp"
#include "python/python_runtime.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Context.h>
#include <cassert>
#include <cmath>
#include <nanobind/stl/optional.h>

namespace lfs::python {

    void register_builtin_transforms(Rml::DataModelConstructor& ctor);

    namespace {
        std::unordered_map<Rml::ElementDocument*, std::vector<Rml::ElementPtr>> s_held_elements;
        std::map<std::string, DataModelArrayStorage> s_model_storage;
        bool s_string_array_type_registered = false;
    } // namespace

    Rml::ElementPtr extractHeldElement(Rml::ElementDocument* doc, Rml::Element* raw) {
        auto it = s_held_elements.find(doc);
        if (it == s_held_elements.end())
            return nullptr;
        auto& vec = it->second;
        for (auto vi = vec.begin(); vi != vec.end(); ++vi) {
            if (vi->get() == raw) {
                auto ptr = std::move(*vi);
                vec.erase(vi);
                return ptr;
            }
        }
        return nullptr;
    }

    void storeHeldElement(Rml::ElementDocument* doc, Rml::ElementPtr elem) {
        s_held_elements[doc].push_back(std::move(elem));
    }

    void clearHeldElements(Rml::ElementDocument* doc) {
        s_held_elements.erase(doc);
    }

    nb::object variant_to_python(const Rml::Variant& v) {
        switch (v.GetType()) {
        case Rml::Variant::BOOL: return nb::cast(v.Get<bool>());
        case Rml::Variant::INT: return nb::cast(v.Get<int>());
        case Rml::Variant::INT64: return nb::cast(v.Get<int64_t>());
        case Rml::Variant::UINT: return nb::cast(v.Get<unsigned int>());
        case Rml::Variant::UINT64: return nb::cast(v.Get<uint64_t>());
        case Rml::Variant::FLOAT: return nb::cast(v.Get<float>());
        case Rml::Variant::DOUBLE: return nb::cast(v.Get<double>());
        case Rml::Variant::STRING: return nb::cast(v.Get<Rml::String>());
        default: return nb::none();
        }
    }

    Rml::Variant python_to_variant(const nb::handle& obj) {
        if (obj.is_none())
            return {};
        if (nb::isinstance<nb::bool_>(obj))
            return Rml::Variant(nb::cast<bool>(obj));
        if (nb::isinstance<nb::int_>(obj))
            return Rml::Variant(nb::cast<int>(obj));
        if (nb::isinstance<nb::float_>(obj))
            return Rml::Variant(nb::cast<double>(obj));
        if (nb::isinstance<nb::str>(obj))
            return Rml::Variant(nb::cast<std::string>(obj));
        return {};
    }

    // --- PyRmlContext ---

    nb::object PyRmlContext::create_data_model(const std::string& name) {
        auto ctor = ctx_->CreateDataModel(name);
        if (!ctor)
            return nb::none();
        register_builtin_transforms(ctor);
        return nb::cast(PyDataModelConstructor(std::move(ctor), name));
    }

    bool PyRmlContext::remove_data_model(const std::string& name) {
        s_model_storage.erase(name);
        return ctx_->RemoveDataModel(name);
    }

    // --- PyRmlEvent ---

    std::string PyRmlEvent::type() const { return event_->GetType(); }

    nb::object PyRmlEvent::target() {
        Rml::Element* t = event_->GetTargetElement();
        if (!t)
            return nb::none();
        return nb::cast(PyRmlElement(t));
    }

    nb::object PyRmlEvent::current_target() {
        Rml::Element* t = event_->GetCurrentElement();
        if (!t)
            return nb::none();
        return nb::cast(PyRmlElement(t));
    }

    void PyRmlEvent::stop_propagation() { event_->StopPropagation(); }

    std::string PyRmlEvent::get_parameter(const std::string& key, const std::string& default_val) {
        return event_->GetParameter<Rml::String>(key, default_val);
    }

    // --- PyRmlElement ---

    nb::object PyRmlElement::get_element_by_id(const std::string& id) {
        Rml::Element* e = elem_->GetElementById(id);
        if (!e)
            return nb::none();
        return nb::cast(PyRmlElement(e));
    }

    nb::list PyRmlElement::query_selector_all(const std::string& selector) {
        Rml::ElementList elements;
        elem_->QuerySelectorAll(elements, selector);
        nb::list result;
        for (auto* e : elements) {
            result.append(PyRmlElement(e));
        }
        return result;
    }

    nb::object PyRmlElement::query_selector(const std::string& selector) {
        Rml::Element* e = elem_->QuerySelector(selector);
        if (!e)
            return nb::none();
        return nb::cast(PyRmlElement(e));
    }

    nb::object PyRmlElement::parent() {
        Rml::Element* p = elem_->GetParentNode();
        if (!p)
            return nb::none();
        return nb::cast(PyRmlElement(p));
    }

    nb::list PyRmlElement::children() {
        nb::list result;
        for (int i = 0; i < elem_->GetNumChildren(); ++i) {
            result.append(PyRmlElement(elem_->GetChild(i)));
        }
        return result;
    }

    int PyRmlElement::num_children() { return elem_->GetNumChildren(); }

    nb::object PyRmlElement::append_child(const std::string& tag_name) {
        auto* doc = elem_->GetOwnerDocument();
        assert(doc);
        auto new_elem = doc->CreateElement(tag_name);
        if (!new_elem)
            return nb::none();
        Rml::Element* raw = new_elem.get();
        elem_->AppendChild(std::move(new_elem));
        return nb::cast(PyRmlElement(raw));
    }

    nb::object PyRmlElement::append_child_element(PyRmlElement& child) {
        auto* doc = elem_->GetOwnerDocument();
        assert(doc);
        auto held = extractHeldElement(doc, child.raw());
        if (!held) {
            LOG_ERROR("append_child: element not in holding area");
            return nb::none();
        }
        Rml::Element* raw = held.get();
        elem_->AppendChild(std::move(held));
        return nb::cast(PyRmlElement(raw));
    }

    nb::object PyRmlElement::insert_before(const std::string& tag_name, PyRmlElement& ref_child) {
        auto* doc = elem_->GetOwnerDocument();
        assert(doc);
        auto new_elem = doc->CreateElement(tag_name);
        if (!new_elem)
            return nb::none();
        Rml::Element* raw = new_elem.get();
        elem_->InsertBefore(std::move(new_elem), ref_child.raw());
        return nb::cast(PyRmlElement(raw));
    }

    nb::object PyRmlElement::insert_before_element(PyRmlElement& child, PyRmlElement& ref_child) {
        auto* doc = elem_->GetOwnerDocument();
        assert(doc);
        auto held = extractHeldElement(doc, child.raw());
        if (!held) {
            LOG_ERROR("insert_before: element not in holding area");
            return nb::none();
        }
        Rml::Element* raw = held.get();
        elem_->InsertBefore(std::move(held), ref_child.raw());
        return nb::cast(PyRmlElement(raw));
    }

    void PyRmlElement::remove_child(PyRmlElement& child) {
        elem_->RemoveChild(child.raw());
    }

    void PyRmlElement::set_inner_rml(const std::string& rml) { elem_->SetInnerRML(rml); }

    std::string PyRmlElement::get_inner_rml() { return elem_->GetInnerRML(); }

    void PyRmlElement::set_text(const std::string& text) { elem_->SetInnerRML(text); }

    void PyRmlElement::set_attribute(const std::string& name, const std::string& value) {
        elem_->SetAttribute(name, value);
    }

    std::string PyRmlElement::get_attribute(const std::string& name,
                                            const std::string& default_val) {
        return elem_->GetAttribute<Rml::String>(name, default_val);
    }

    bool PyRmlElement::has_attribute(const std::string& name) {
        return elem_->HasAttribute(name);
    }

    void PyRmlElement::remove_attribute(const std::string& name) {
        elem_->RemoveAttribute(name);
    }

    void PyRmlElement::set_class(const std::string& name, bool active) {
        elem_->SetClass(name, active);
    }

    bool PyRmlElement::is_class_set(const std::string& name) {
        return elem_->IsClassSet(name);
    }

    void PyRmlElement::set_class_names(const std::string& names) {
        elem_->SetClassNames(names);
    }

    std::string PyRmlElement::get_class_names() {
        return elem_->GetClassNames();
    }

    bool PyRmlElement::set_property(const std::string& name, const std::string& value) {
        return elem_->SetProperty(name, value);
    }

    void PyRmlElement::remove_property(const std::string& name) {
        elem_->RemoveProperty(name);
    }

    void PyRmlElement::add_event_listener(const std::string& event, nb::callable callback) {
        auto* listener = new PyEventListener(std::move(callback));
        elem_->AddEventListener(event, listener, false);
    }

    std::string PyRmlElement::id() { return elem_->GetId(); }
    void PyRmlElement::set_id(const std::string& id) { elem_->SetId(id); }
    std::string PyRmlElement::tag_name() { return elem_->GetTagName(); }

    float PyRmlElement::scroll_left() { return elem_->GetScrollLeft(); }
    float PyRmlElement::scroll_top() { return elem_->GetScrollTop(); }
    void PyRmlElement::set_scroll_left(float v) { elem_->SetScrollLeft(v); }
    void PyRmlElement::set_scroll_top(float v) { elem_->SetScrollTop(v); }
    float PyRmlElement::scroll_width() { return elem_->GetScrollWidth(); }
    float PyRmlElement::scroll_height() { return elem_->GetScrollHeight(); }
    void PyRmlElement::scroll_into_view(bool align_top) { elem_->ScrollIntoView(align_top); }

    bool PyRmlElement::focus() { return elem_->Focus(); }
    void PyRmlElement::blur() { elem_->Blur(); }

    // --- PyRmlDocument ---

    nb::object PyRmlDocument::create_element(const std::string& tag) {
        auto elem = doc_->CreateElement(tag);
        if (!elem)
            return nb::none();
        Rml::Element* raw = elem.get();
        storeHeldElement(doc_, std::move(elem));
        return nb::cast(PyRmlElement(raw));
    }

    nb::object PyRmlDocument::create_text_node(const std::string& text) {
        auto node = doc_->CreateTextNode(text);
        if (!node)
            return nb::none();
        Rml::Element* raw = node.get();
        storeHeldElement(doc_, std::move(node));
        return nb::cast(PyRmlElement(raw));
    }

    void PyRmlDocument::show() { doc_->Show(); }
    void PyRmlDocument::hide() { doc_->Hide(); }
    std::string PyRmlDocument::title() { return doc_->GetTitle(); }
    void PyRmlDocument::set_title(const std::string& t) { doc_->SetTitle(t); }

    nb::object PyRmlDocument::create_data_model(const std::string& name) {
        auto* ctx = doc_->GetContext();
        assert(ctx);
        auto ctor = ctx->CreateDataModel(name);
        if (!ctor)
            return nb::none();
        register_builtin_transforms(ctor);
        return nb::cast(PyDataModelConstructor(std::move(ctor), name));
    }

    bool PyRmlDocument::remove_data_model(const std::string& name) {
        auto* ctx = doc_->GetContext();
        assert(ctx);
        s_model_storage.erase(name);
        return ctx->RemoveDataModel(name);
    }

    // --- PyDataModelHandle ---

    void PyDataModelHandle::dirty(const std::string& name) {
        handle_.DirtyVariable(name);
    }

    void PyDataModelHandle::dirty_all() {
        handle_.DirtyAllVariables();
    }

    bool PyDataModelHandle::is_dirty(const std::string& name) {
        return handle_.IsVariableDirty(name);
    }

    void PyDataModelHandle::update_string_list(const std::string& name, nb::list items) {
        auto model_it = s_model_storage.find(model_name_);
        assert(model_it != s_model_storage.end());
        auto arr_it = model_it->second.string_arrays.find(name);
        assert(arr_it != model_it->second.string_arrays.end());
        auto& vec = arr_it->second;
        vec.clear();
        vec.reserve(nb::len(items));
        for (auto item : items)
            vec.push_back(nb::cast<std::string>(item));
        handle_.DirtyVariable(name);
    }

    // --- PyDataModelConstructor ---

    void PyDataModelConstructor::bind(const std::string& name, nb::callable getter,
                                      nb::object setter) {
        nb::callable get_cb = nb::borrow<nb::callable>(getter);
        prevent_gc_.push_back(nb::object(get_cb));

        Rml::DataGetFunc get_func = [get_cb](Rml::Variant& out) {
            nb::gil_scoped_acquire gil;
            try {
                nb::object result = get_cb();
                out = python_to_variant(result);
            } catch (const std::exception& e) {
                LOG_ERROR("Data model getter error: {}", e.what());
            }
        };

        Rml::DataSetFunc set_func;
        if (!setter.is_none()) {
            nb::callable set_cb = nb::borrow<nb::callable>(setter);
            prevent_gc_.push_back(nb::object(set_cb));

            set_func = [set_cb](const Rml::Variant& in) {
                nb::gil_scoped_acquire gil;
                try {
                    set_cb(variant_to_python(in));
                } catch (const std::exception& e) {
                    LOG_ERROR("Data model setter error: {}", e.what());
                }
            };
        }

        ctor_.BindFunc(name, std::move(get_func), std::move(set_func));
    }

    void PyDataModelConstructor::bind_func(const std::string& name, nb::callable getter) {
        bind(name, std::move(getter), nb::none());
    }

    void PyDataModelConstructor::bind_event(const std::string& name, nb::callable callback) {
        nb::callable cb = nb::borrow<nb::callable>(callback);
        prevent_gc_.push_back(nb::object(cb));
        const auto model_name = model_name_;

        ctor_.BindEventCallback(
            name, [cb, model_name](Rml::DataModelHandle handle, Rml::Event& event,
                                   const Rml::VariantList& args) {
                nb::gil_scoped_acquire gil;
                try {
                    nb::list py_args;
                    for (const auto& arg : args)
                        py_args.append(variant_to_python(arg));
                    cb(PyDataModelHandle(handle, model_name), PyRmlEvent(&event), py_args);
                } catch (const std::exception& e) {
                    LOG_ERROR("Data model event error: {}", e.what());
                }
            });
    }

    void PyDataModelConstructor::register_transform(const std::string& name, nb::callable func) {
        nb::callable cb = nb::borrow<nb::callable>(func);
        prevent_gc_.push_back(nb::object(cb));

        ctor_.RegisterTransformFunc(
            name, [cb](const Rml::VariantList& args) -> Rml::Variant {
                nb::gil_scoped_acquire gil;
                try {
                    nb::list py_args;
                    for (const auto& arg : args)
                        py_args.append(variant_to_python(arg));
                    nb::object result = cb(*py_args);
                    return python_to_variant(result);
                } catch (const std::exception& e) {
                    LOG_ERROR("Data model transform error: {}", e.what());
                    return {};
                }
            });
    }

    void PyDataModelConstructor::bind_string_list(const std::string& name) {
        if (!s_string_array_type_registered) {
            ctor_.RegisterArray<std::vector<Rml::String>>();
            s_string_array_type_registered = true;
        }
        auto& storage = s_model_storage[model_name_];
        storage.string_arrays[name]; // create empty vector
        ctor_.Bind(name, &storage.string_arrays[name]);
    }

    PyDataModelHandle PyDataModelConstructor::get_handle() {
        return PyDataModelHandle(ctor_.GetModelHandle(), model_name_);
    }

    void register_builtin_transforms(Rml::DataModelConstructor& ctor) {
        ctor.RegisterTransformFunc("format_float",
                                   [](const Rml::VariantList& args) -> Rml::Variant {
                                       if (args.empty())
                                           return {};
                                       double val = args[0].Get<double>();
                                       int precision = args.size() > 1 ? args[1].Get<int>() : 2;
                                       char buf[64];
                                       std::snprintf(buf, sizeof(buf), "%.*f", precision, val);
                                       return Rml::Variant(Rml::String(buf));
                                   });

        ctor.RegisterTransformFunc("format_int",
                                   [](const Rml::VariantList& args) -> Rml::Variant {
                                       if (args.empty())
                                           return {};
                                       return Rml::Variant(
                                           Rml::String(std::to_string(args[0].Get<int>())));
                                   });

        ctor.RegisterTransformFunc("format_percent",
                                   [](const Rml::VariantList& args) -> Rml::Variant {
                                       if (args.empty())
                                           return {};
                                       double val = args[0].Get<double>() * 100.0;
                                       char buf[64];
                                       std::snprintf(buf, sizeof(buf), "%.0f%%", val);
                                       return Rml::Variant(Rml::String(buf));
                                   });

        ctor.RegisterTransformFunc("to_degrees",
                                   [](const Rml::VariantList& args) -> Rml::Variant {
                                       if (args.empty())
                                           return {};
                                       double rad = args[0].Get<double>();
                                       double deg = rad * 180.0 / M_PI;
                                       char buf[64];
                                       std::snprintf(buf, sizeof(buf), "%.1f\xC2\xB0", deg);
                                       return Rml::Variant(Rml::String(buf));
                                   });
    }

    // --- PyEventListener ---

    void PyEventListener::ProcessEvent(Rml::Event& event) {
        nb::gil_scoped_acquire gil;
        try {
            callback_(PyRmlEvent(&event));
        } catch (const std::exception& e) {
            LOG_ERROR("RmlUI event listener error: {}", e.what());
        }
    }

    // --- RmlDocumentRegistry ---

    RmlDocumentRegistry& RmlDocumentRegistry::instance() {
        static RmlDocumentRegistry registry;
        return registry;
    }

    void RmlDocumentRegistry::register_document(const std::string& name,
                                                Rml::ElementDocument* doc) {
        auto it = documents_.find(name);
        if (it != documents_.end())
            clearHeldElements(it->second);
        documents_[name] = doc;
    }

    void RmlDocumentRegistry::unregister_document(const std::string& name) {
        auto it = documents_.find(name);
        if (it != documents_.end()) {
            clearHeldElements(it->second);
            documents_.erase(it);
        }
    }

    Rml::ElementDocument* RmlDocumentRegistry::get_document(const std::string& name) {
        auto it = documents_.find(name);
        return it != documents_.end() ? it->second : nullptr;
    }

    // --- Nanobind registration ---

    void register_rml_bindings(nb::module_& m) {
        auto rml = m.def_submodule("rml", "RmlUI DOM API");

        nb::class_<PyRmlContext>(rml, "RmlContext")
            .def("create_data_model", &PyRmlContext::create_data_model, nb::arg("name"))
            .def("remove_data_model", &PyRmlContext::remove_data_model, nb::arg("name"));

        nb::class_<PyRmlEvent>(rml, "RmlEvent")
            .def("type", &PyRmlEvent::type)
            .def("target", &PyRmlEvent::target)
            .def("current_target", &PyRmlEvent::current_target)
            .def("stop_propagation", &PyRmlEvent::stop_propagation)
            .def("get_parameter", &PyRmlEvent::get_parameter, nb::arg("key"),
                 nb::arg("default_val") = "");

        nb::class_<PyRmlElement>(rml, "RmlElement")
            .def("get_element_by_id", &PyRmlElement::get_element_by_id)
            .def("query_selector_all", &PyRmlElement::query_selector_all)
            .def("query_selector", &PyRmlElement::query_selector)
            .def("parent", &PyRmlElement::parent)
            .def("children", &PyRmlElement::children)
            .def("num_children", &PyRmlElement::num_children)
            .def("append_child", &PyRmlElement::append_child, nb::arg("tag_name"))
            .def("append_child", &PyRmlElement::append_child_element, nb::arg("child"))
            .def("insert_before", &PyRmlElement::insert_before, nb::arg("tag_name"),
                 nb::arg("ref_child"))
            .def("insert_before", &PyRmlElement::insert_before_element, nb::arg("child"),
                 nb::arg("ref_child"))
            .def("remove_child", &PyRmlElement::remove_child)
            .def("set_inner_rml", &PyRmlElement::set_inner_rml)
            .def("get_inner_rml", &PyRmlElement::get_inner_rml)
            .def("set_text", &PyRmlElement::set_text)
            .def("set_attribute", &PyRmlElement::set_attribute)
            .def("get_attribute", &PyRmlElement::get_attribute, nb::arg("name"),
                 nb::arg("default_val") = "")
            .def("has_attribute", &PyRmlElement::has_attribute)
            .def("remove_attribute", &PyRmlElement::remove_attribute)
            .def("set_class", &PyRmlElement::set_class)
            .def("is_class_set", &PyRmlElement::is_class_set)
            .def("set_class_names", &PyRmlElement::set_class_names)
            .def("get_class_names", &PyRmlElement::get_class_names)
            .def("set_property", &PyRmlElement::set_property)
            .def("remove_property", &PyRmlElement::remove_property)
            .def("add_event_listener", &PyRmlElement::add_event_listener)
            .def("set_id", &PyRmlElement::set_id)
            .def_prop_rw("id", &PyRmlElement::id, &PyRmlElement::set_id)
            .def_prop_ro("tag_name", &PyRmlElement::tag_name)
            .def_prop_rw("scroll_left", &PyRmlElement::scroll_left,
                         &PyRmlElement::set_scroll_left)
            .def_prop_rw("scroll_top", &PyRmlElement::scroll_top, &PyRmlElement::set_scroll_top)
            .def_prop_ro("scroll_width", &PyRmlElement::scroll_width)
            .def_prop_ro("scroll_height", &PyRmlElement::scroll_height)
            .def("scroll_into_view", &PyRmlElement::scroll_into_view,
                 nb::arg("align_top") = true)
            .def("focus", &PyRmlElement::focus)
            .def("blur", &PyRmlElement::blur);

        nb::class_<PyRmlDocument, PyRmlElement>(rml, "RmlDocument")
            .def("create_element", &PyRmlDocument::create_element)
            .def("create_text_node", &PyRmlDocument::create_text_node)
            .def("show", &PyRmlDocument::show)
            .def("hide", &PyRmlDocument::hide)
            .def("create_data_model", &PyRmlDocument::create_data_model, nb::arg("name"))
            .def("remove_data_model", &PyRmlDocument::remove_data_model, nb::arg("name"))
            .def_prop_rw("title", &PyRmlDocument::title, &PyRmlDocument::set_title);

        nb::class_<PyDataModelHandle>(rml, "DataModelHandle")
            .def("dirty", &PyDataModelHandle::dirty, nb::arg("name"))
            .def("dirty_all", &PyDataModelHandle::dirty_all)
            .def("is_dirty", &PyDataModelHandle::is_dirty, nb::arg("name"))
            .def("update_string_list", &PyDataModelHandle::update_string_list, nb::arg("name"),
                 nb::arg("items"));

        nb::class_<PyDataModelConstructor>(rml, "DataModelConstructor")
            .def("bind", &PyDataModelConstructor::bind, nb::arg("name"), nb::arg("getter"),
                 nb::arg("setter") = nb::none())
            .def("bind_func", &PyDataModelConstructor::bind_func, nb::arg("name"),
                 nb::arg("getter"))
            .def("bind_event", &PyDataModelConstructor::bind_event, nb::arg("name"),
                 nb::arg("callback"))
            .def("register_transform", &PyDataModelConstructor::register_transform,
                 nb::arg("name"), nb::arg("func"))
            .def("bind_string_list", &PyDataModelConstructor::bind_string_list, nb::arg("name"))
            .def("get_handle", &PyDataModelConstructor::get_handle);

        rml.def("get_document", [](const std::string& name) -> nb::object {
            auto* doc = RmlDocumentRegistry::instance().get_document(name);
            if (!doc)
                return nb::none();
            return nb::cast(PyRmlDocument(doc));
        });

        set_rml_doc_registry_callbacks(
            [](const char* name, void* doc) {
                RmlDocumentRegistry::instance().register_document(
                    name, static_cast<Rml::ElementDocument*>(doc));
            },
            [](const char* name) {
                RmlDocumentRegistry::instance().unregister_document(name);
            });
    }

} // namespace lfs::python
