/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "node_canvas_dom.hpp"

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementText.h>

namespace lfs::vis::gui::node_widgets {
    namespace {
        bool sameIdentity(const Rml::Element& a, const Rml::Element& b) {
            if (a.GetTagName() != b.GetTagName())
                return false;
            for (const char* attribute : {"id", "data-key", "data-node", "data-input", "data-property", "data-modifier"})
                if (a.GetAttribute<Rml::String>(attribute, "") != b.GetAttribute<Rml::String>(attribute, ""))
                    return false;
            return true;
        }

        void patchChildren(Rml::Element& target, Rml::Element& source);

        void patchElement(Rml::Element& target, Rml::Element& source) {
            if (auto* text = dynamic_cast<Rml::ElementText*>(&target)) {
                const auto* value = dynamic_cast<Rml::ElementText*>(&source);
                if (value && text->GetText() != value->GetText())
                    text->SetText(value->GetText());
                return;
            }
            const auto attributes = target.GetAttributes();
            for (const auto& [key, _] : attributes)
                if (!source.HasAttribute(key))
                    target.RemoveAttribute(key);
            for (const auto& [key, value] : source.GetAttributes()) {
                const auto* previous = target.GetAttribute(key);
                if (!previous || previous->Get<Rml::String>() != value.Get<Rml::String>())
                    target.SetAttribute(key, value);
            }
            if (!source.HasAttribute("data-preserve-content"))
                patchChildren(target, source);
        }

        void patchChildren(Rml::Element& target, Rml::Element& source) {
            int index = 0;
            while (source.GetNumChildren() > 0) {
                auto* desired = source.GetChild(0);
                // GetChild also indexes RmlUi's private children (scrollbars,
                // select controls). Only DOM children belong to this patch.
                auto* current = index < target.GetNumChildren() ? target.GetChild(index) : nullptr;
                if (current && sameIdentity(*current, *desired)) {
                    patchElement(*current, *desired);
                    source.RemoveChild(desired);
                } else {
                    auto child = source.RemoveChild(desired);
                    if (current)
                        target.ReplaceChild(std::move(child), current);
                    else
                        target.AppendChild(std::move(child));
                }
                ++index;
            }
            while (target.GetNumChildren() > index)
                target.RemoveChild(target.GetChild(index));
        }
    } // namespace

    void patchMarkup(Rml::Element& parent, const std::string& markup) {
        auto fragment = parent.GetOwnerDocument()->CreateElement("div");
        fragment->SetInnerRML(markup);
        patchChildren(parent, *fragment);
    }
} // namespace lfs::vis::gui::node_widgets
