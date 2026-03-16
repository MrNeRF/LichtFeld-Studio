/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Elements/ElementForm.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/EventListener.h>

#include <cassert>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace Rml {
    class Context;
} // namespace Rml

namespace lfs::python {

    class PyRmlContext {
    public:
        explicit PyRmlContext(Rml::Context* ctx) : ctx_(ctx) { assert(ctx_); }

        nb::object create_data_model(const std::string& name);
        bool remove_data_model(const std::string& name);

    private:
        Rml::Context* ctx_;
    };

    class PyRmlEvent {
    public:
        explicit PyRmlEvent(Rml::Event* event) : event_(event) { assert(event_); }

        std::string type() const;
        nb::object target();
        nb::object current_target();
        void stop_propagation();
        std::string get_parameter(const std::string& key, const std::string& default_val = "");
        bool get_bool_parameter(const std::string& key, bool default_val = false);

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
        nb::object append_child_element(PyRmlElement& child);
        nb::object insert_before(const std::string& tag_name, PyRmlElement& ref_child);
        nb::object insert_before_element(PyRmlElement& child, PyRmlElement& ref_child);
        void remove_child(PyRmlElement& child);
        void set_inner_rml(const std::string& rml);
        std::string get_inner_rml();
        void set_text(const std::string& text);

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

        // Animation
        bool animate(const std::string& property, const std::string& target_value, float duration,
                     const std::string& tween = "quadratic-out",
                     const std::optional<std::string>& start_value = std::nullopt,
                     bool remove_on_complete = false);

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
        float client_width();
        float client_height();
        float absolute_left();
        float absolute_top();
        float absolute_width();
        float absolute_height();
        float offset_top();
        float offset_height();
        void scroll_into_view(bool align_top = true);

        // Focus
        bool focus();
        void blur();
        void submit(const std::string& name = "", const std::string& value = "");

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

        nb::object create_data_model(const std::string& name);
        bool remove_data_model(const std::string& name);

        Rml::ElementDocument* raw_doc() { return doc_; }

    private:
        Rml::ElementDocument* doc_;
    };

    struct DynamicDataField {
        Rml::Variant value;

        bool operator==(const DynamicDataField& other) const { return value == other.value; }
    };

    struct DynamicDataRecord {
        std::map<std::string, DynamicDataField> fields;

        bool operator==(const DynamicDataRecord& other) const { return fields == other.fields; }
    };

    struct DataModelArrayStorage {
        std::map<std::string, std::vector<Rml::String>> string_arrays;
        std::map<std::string, std::vector<DynamicDataRecord>> record_arrays;
    };

    class PyDataModelHandle {
    public:
        PyDataModelHandle(Rml::DataModelHandle handle, std::string model_name,
                          Rml::Context* context = nullptr)
            : handle_(handle),
              model_name_(std::move(model_name)),
              context_(context) {}

        void dirty(const std::string& name);
        void dirty_all();
        bool is_dirty(const std::string& name);
        void update_string_list(const std::string& name, nb::list items);
        void update_record_list(const std::string& name, nb::list items);

    private:
        Rml::DataModelHandle handle_;
        std::string model_name_;
        Rml::Context* context_ = nullptr;
    };

    class PyDataModelConstructor {
    public:
        PyDataModelConstructor(Rml::DataModelConstructor ctor, std::string model_name,
                               Rml::Context* context = nullptr)
            : ctor_(std::move(ctor)),
              model_name_(std::move(model_name)),
              context_(context) {}

        void bind(const std::string& name, nb::callable getter, nb::object setter);
        void bind_func(const std::string& name, nb::callable getter);
        void bind_event(const std::string& name, nb::callable callback);
        void register_transform(const std::string& name, nb::callable func);
        void bind_string_list(const std::string& name);
        void bind_record_list(const std::string& name);
        PyDataModelHandle get_handle();

    private:
        Rml::DataModelConstructor ctor_;
        std::string model_name_;
        Rml::Context* context_ = nullptr;
        std::vector<nb::object> prevent_gc_;
    };

    class PyEventListener : public Rml::EventListener {
    public:
        explicit PyEventListener(nb::callable cb) : callback_(std::move(cb)) {}

        void ProcessEvent(Rml::Event& event) override;
        void OnDetach(Rml::Element*) override { delete this; }

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

    bool consume_document_dirty(Rml::ElementDocument* doc);
    bool is_document_dirty(Rml::ElementDocument* doc);
    void release_rml_context_state(Rml::Context* context);

    void register_rml_bindings(nb::module_& m);

} // namespace lfs::python
