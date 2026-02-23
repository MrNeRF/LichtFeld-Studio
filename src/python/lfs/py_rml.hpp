/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/EventListener.h>

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace lfs::python {

    class PyRmlEvent {
    public:
        explicit PyRmlEvent(Rml::Event* event) : event_(event) { assert(event_); }

        std::string type() const;
        nb::object target();
        nb::object current_target();
        void stop_propagation();
        std::string get_parameter(const std::string& key, const std::string& default_val = "");

    private:
        Rml::Event* event_;
    };

    class PyRmlElement {
    public:
        explicit PyRmlElement(Rml::Element* elem) : elem_(elem) { assert(elem_); }

        // DOM queries
        nb::object get_element_by_id(const std::string& id);
        nb::list query_selector_all(const std::string& selector);
        nb::object query_selector(const std::string& selector);
        nb::object parent();
        nb::list children();
        int num_children();

        // DOM mutation
        nb::object append_child(const std::string& tag_name);
        void remove_child(PyRmlElement& child);
        void set_inner_rml(const std::string& rml);
        std::string get_inner_rml();

        // Attributes
        void set_attribute(const std::string& name, const std::string& value);
        std::string get_attribute(const std::string& name, const std::string& default_val = "");
        bool has_attribute(const std::string& name);
        void remove_attribute(const std::string& name);

        // CSS classes
        void set_class(const std::string& name, bool active);
        bool is_class_set(const std::string& name);
        void set_class_names(const std::string& names);
        std::string get_class_names();

        // CSS properties
        bool set_property(const std::string& name, const std::string& value);
        void remove_property(const std::string& name);

        // Events
        void add_event_listener(const std::string& event, nb::callable callback);

        // Identity
        std::string id();
        void set_id(const std::string& id);
        std::string tag_name();

        // Scroll
        float scroll_left();
        float scroll_top();
        void set_scroll_left(float v);
        void set_scroll_top(float v);
        float scroll_width();
        float scroll_height();
        void scroll_into_view(bool align_top = true);

        // Focus
        bool focus();
        void blur();

        Rml::Element* raw() { return elem_; }

    private:
        Rml::Element* elem_;
    };

    class PyRmlDocument : public PyRmlElement {
    public:
        explicit PyRmlDocument(Rml::ElementDocument* doc)
            : PyRmlElement(doc),
              doc_(doc) {
            assert(doc_);
        }

        nb::object create_element(const std::string& tag);
        nb::object create_text_node(const std::string& text);
        void show();
        void hide();
        std::string title();
        void set_title(const std::string& t);

        Rml::ElementDocument* raw_doc() { return doc_; }

    private:
        Rml::ElementDocument* doc_;
    };

    class PyEventListener : public Rml::EventListener {
    public:
        explicit PyEventListener(nb::callable cb) : callback_(std::move(cb)) {}

        void ProcessEvent(Rml::Event& event) override;

    private:
        nb::callable callback_;
    };

    // Registry: document name -> PyRmlDocument, for Python access
    class RmlDocumentRegistry {
    public:
        static RmlDocumentRegistry& instance();

        void register_document(const std::string& name, Rml::ElementDocument* doc);
        void unregister_document(const std::string& name);
        Rml::ElementDocument* get_document(const std::string& name);

    private:
        std::unordered_map<std::string, Rml::ElementDocument*> documents_;
    };

    void register_rml_bindings(nb::module_& m);

} // namespace lfs::python
