#pragma once

#include <RmlUi/Core.h>
#include <RmlUi/Core/EventListener.h>

#include <functional>
#include <unordered_map>
#include <utility>

namespace lfs::vis::gui::rml_input {

    inline bool isTextEditableElement(Rml::Element* element) {
        if (!element)
            return false;

        const auto tag = element->GetTagName();
        if (tag == "textarea")
            return true;
        if (tag != "input")
            return false;

        const auto input_type = element->GetAttribute<Rml::String>("type", "text");
        return input_type.empty() || input_type == "text" || input_type == "password" ||
               input_type == "search" || input_type == "email" || input_type == "url";
    }

    template <typename Fn>
    void forEachTextEditableElement(Rml::Element* root, Fn&& fn) {
        auto visit = [&fn](auto&& self, Rml::Element* element) -> void {
            if (!element)
                return;

            if (isTextEditableElement(element))
                fn(*element);

            const int child_count = element->GetNumChildren();
            for (int i = 0; i < child_count; ++i)
                self(self, element->GetChild(i));
        };
        visit(visit, root);
    }

    inline bool hasFocusedKeyboardTarget(Rml::Element* element) {
        return element && element->GetTagName() != "body";
    }

    inline bool isSingleLineTextInput(Rml::Element* element) {
        return element && element->GetTagName() == "input" && isTextEditableElement(element);
    }

    inline bool isSelectRelatedElement(Rml::Element* element) {
        for (auto* current = element; current; current = current->GetParentNode()) {
            const auto tag = current->GetTagName();
            if (tag == "select" || tag == "selectbox")
                return true;
        }
        return false;
    }

    inline bool cancelFocusedElement(Rml::Context& context) {
        auto* const focused = context.GetFocusElement();
        if (!focused)
            return false;

        if (isTextEditableElement(focused)) {
            Rml::Dictionary params;
            focused->DispatchEvent("escapecancel", params);
            focused->Blur();
            return true;
        }

        if (!isSelectRelatedElement(focused))
            return false;

        focused->Blur();
        if (auto* const still_focused = context.GetFocusElement();
            still_focused && isSelectRelatedElement(still_focused)) {
            still_focused->Blur();
        }
        return true;
    }

    class TextInputEscapeRevertController final : public Rml::EventListener {
    public:
        using RestoreCallback = std::function<void(Rml::Element&)>;

        ~TextInputEscapeRevertController() override {
            clear();
        }

        void bind(Rml::Element* element, RestoreCallback restore_callback = {}) {
            if (!element || !isTextEditableElement(element) || bindings_.contains(element))
                return;

            Binding binding;
            binding.restore_callback = std::move(restore_callback);
            bindings_.emplace(element, std::move(binding));
            element->AddEventListener("focus", this);
            element->AddEventListener("blur", this);
            element->AddEventListener("escapecancel", this);
        }

        void clear() {
            for (auto& [element, _binding] : bindings_) {
                if (!element)
                    continue;
                element->RemoveEventListener("focus", this, false);
                element->RemoveEventListener("blur", this, false);
                element->RemoveEventListener("escapecancel", this, false);
            }
            bindings_.clear();
        }

        void ProcessEvent(Rml::Event& event) override {
            auto* const element = event.GetCurrentElement();
            if (!element)
                return;

            const auto it = bindings_.find(element);
            if (it == bindings_.end())
                return;

            const auto type = event.GetType();
            if (type == "focus") {
                it->second.snapshot = element->GetAttribute<Rml::String>("value", "");
                return;
            }

            if (type == "blur") {
                it->second.snapshot.clear();
                return;
            }

            if (type != "escapecancel")
                return;

            element->SetAttribute("value", it->second.snapshot);
            if (it->second.restore_callback)
                it->second.restore_callback(*element);
            event.StopPropagation();
        }

    private:
        struct Binding {
            Rml::String snapshot;
            RestoreCallback restore_callback;
        };

        std::unordered_map<Rml::Element*, Binding> bindings_;
    };

} // namespace lfs::vis::gui::rml_input
