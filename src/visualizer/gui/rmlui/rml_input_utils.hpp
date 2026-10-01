#pragma once

#include "core/logger.hpp"
#include "gui/rmlui/rml_text_input_handler.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "input/frame_input_buffer.hpp"
#include <cstdlib>

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

    inline bool isCustomTextInputElement(Rml::Element* element) {
        return element && element->GetAttribute<Rml::String>("data-text-input", "") == "true";
    }

    inline bool wantsTextInput(Rml::Element* element) {
        return isTextEditableElement(element) || isCustomTextInputElement(element);
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

    inline bool shouldCancelOnEscape(Rml::Element* element, const bool composing) {
        return !composing && element && (isTextEditableElement(element) || isSelectRelatedElement(element));
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

    inline bool isRepeatedDialogAction(const FrameInputEvent& event, Rml::Element* focused = nullptr) {
        if (event.kind != FrameInputEventKind::KeyDown || !event.repeat)
            return false;
        if (event.scancode == SDL_SCANCODE_ESCAPE)
            return true;
        if (event.scancode == SDL_SCANCODE_SPACE)
            return !wantsTextInput(focused);
        return (event.scancode == SDL_SCANCODE_RETURN || event.scancode == SDL_SCANCODE_KP_ENTER) &&
               (!focused || focused->GetTagName() != "textarea");
    }

    // Mouse transitions share the keyboard timeline; never flush future text on blur.
    template <typename Pointer, typename Keyboard>
    void replayInputEvents(const std::vector<FrameInputEvent>& events,
                           const std::vector<FrameMouseButtonEvent>& buttons,
                           Pointer&& pointer, Keyboard&& keyboard) {
        for (const auto& event : events) {
            if (event.kind == FrameInputEventKind::MouseButton) {
                if (event.mouse_button_index < buttons.size())
                    pointer(buttons[event.mouse_button_index]);
            } else {
                keyboard(event);
            }
        }
    }

    // Replay one SDL event. Callers retain their own submit/cancel policies, but
    // must invoke them at the event's position in the stream, before later text.
    inline bool processKeyboardEvent(Rml::Context& context, const FrameInputEvent& event,
                                     RmlTextInputHandler* handler = nullptr) {
        if (event.kind == FrameInputEventKind::MouseButton)
            return false;
        const auto consumed = [&](bool value) {
            if (event.dispatch)
                event.dispatch->consumed |= value;
            return value;
        };
        auto* focused = context.GetFocusElement();
        const bool editable = isTextEditableElement(focused);
        if (event.kind == FrameInputEventKind::TextEditing) {
            return consumed(editable && handler && handler->handleTextEditing(event.text, event.editing_start, event.editing_length));
        }
        if (event.kind == FrameInputEventKind::Text) {
            if (std::getenv("LFS_TRACE_INPUT"))
                LOG_INFO("INPUT consumer context={} element={} text={}", context.GetName(), focused ? focused->GetTagName() : "none", event.text);
            if (!wantsTextInput(focused))
                return false;
            if (!editable || !handler || !handler->handleTextInput(event.text))
                context.ProcessTextInput(event.text);
            consumed(true);
            return true;
        }
        const auto sc = event.scancode;
        if (handler && handler->isComposing() &&
            (sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER || sc == SDL_SCANCODE_ESCAPE))
            return false;
        if (wantsTextInput(focused) &&
            ((sc >= SDL_SCANCODE_KP_1 && sc <= SDL_SCANCODE_KP_0) || sc == SDL_SCANCODE_KP_PERIOD))
            return false;
        const auto key = sdlScancodeToRml(sc);
        if (key == Rml::Input::KI_UNKNOWN)
            return false;
        const int mods = sdlModsToRml(event.modifiers & SDL_KMOD_CTRL,
                                      event.modifiers & SDL_KMOD_SHIFT,
                                      event.modifiers & SDL_KMOD_ALT,
                                      event.modifiers & SDL_KMOD_GUI);
        if (event.kind == FrameInputEventKind::KeyDown) {
            if (!editable || !handler || !handler->handleKeyDown(key, mods))
                consumed(!context.ProcessKeyDown(key, mods));
            else
                consumed(true);
        } else {
            consumed(!context.ProcessKeyUp(key, mods));
        }
        return true;
    }

    class TextInputEscapeRevertController final : public Rml::EventListener {
    public:
        using RestoreCallback = std::function<void(Rml::Element&)>;

        ~TextInputEscapeRevertController() override {
            if (Rml::GetSystemInterface())
                clear();
            else
                bindings_.clear();
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
