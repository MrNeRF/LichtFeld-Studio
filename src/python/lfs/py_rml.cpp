/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_rml.hpp"
#include "core/logger.hpp"
#include "python/python_runtime.hpp"

#include <RmlUi/Core.h>
#include <cassert>
#include <nanobind/stl/optional.h>

namespace lfs::python {

    namespace {
        std::unordered_map<Rml::ElementDocument*, std::vector<Rml::ElementPtr>> s_held_elements;
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
            .def_prop_rw("title", &PyRmlDocument::title, &PyRmlDocument::set_title);

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
