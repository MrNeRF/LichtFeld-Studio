/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/rml_template_browser.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_input.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_pointer_dispatch.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "internal/resource_paths.hpp"
#include "python/python_runtime.hpp"
#include "python/ui_hooks.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <map>
#include <ranges>

namespace lfs::vis::gui {
    namespace {
        std::string escaped(const std::string& text) { return Rml::StringUtilities::EncodeRml(text); }
        std::string lower(std::string text) {
            std::ranges::transform(text, text.begin(), [](unsigned char c) { return std::tolower(c); });
            return text;
        }
        void text(Rml::Element* element, const std::string& value) {
            if (element)
                element->SetInnerRML(element->GetTagName() == "button" ? "<span>" + escaped(value) + "</span>" : escaped(value));
        }
        std::string categoryLabel(const std::string& category) {
            static const std::map<std::string, std::string> keys{
                {"Clean-up", "cleanup"},
                {"Colour", "colour"},
                {"Selection", "selection"},
                {"Geometry", "geometry"},
                {"Animation", "animation"}};
            const auto found = keys.find(category);
            return found == keys.end() ? category : LOC("node_template.category." + found->second);
        }
    } // namespace

    RmlTemplateBrowser::RmlTemplateBrowser(RmlUIManager& rml, std::function<void(std::string)> on_added)
        : rml_(rml), on_added_(std::move(on_added)) {}

    RmlTemplateBrowser::~RmlTemplateBrowser() {
        lfs::python::unregister_rml_document("node_templates");
        if (rml_.isInitialized()) {
            rml_.releaseCachedVulkanContext(cache_);
            if (context_)
                rml_.destroyContext("node_templates");
        }
    }

    void RmlTemplateBrowser::init() {
        if (context_)
            return;
        context_ = rml_.createContext("node_templates", std::max(1, width_), std::max(1, height_));
        if (!context_)
            return;
        document_ = rml_documents::loadDocument(context_, getAssetPath("rmlui/node_templates.rml"));
        if (!document_)
            return;
        document_->AddEventListener("click", this);
        document_->AddEventListener("change", this);
        document_->Show();
        lfs::python::register_rml_document("node_templates", document_);
        theme_signature_ = 0;
    }

    void RmlTemplateBrowser::open(ModifierManager& manager, const core::Uuid target, std::string save_tree) {
        manager_ = &manager;
        target_ = target;
        save_tree_ = std::move(save_tree);
        rename_id_.clear();
        category_ = "all";
        query_.clear();
        selected_id_.clear();
        menu_id_.clear();
        last_click_id_.clear();
        open_ = true;
        init();
        refreshTemplates();
        if (document_) {
            document_->Show();
            if (auto* input = document_->GetElementById(save_tree_.empty() ? "template-search" : "template-name"))
                input->Focus();
        }
        lfs::python::request_redraw();
    }

    void RmlTemplateBrowser::close() {
        open_ = false;
        if (document_)
            document_->Hide();
        if (context_)
            rml_.deactivateInput(context_);
        tooltip_.setHover({}, nullptr);
        std::fill(std::begin(pointer_down_), std::end(pointer_down_), false);
        lfs::python::request_redraw();
    }

    void RmlTemplateBrowser::reloadResources() {
        if (!context_)
            return;
        lfs::python::unregister_rml_document("node_templates");
        rml_.releaseCachedVulkanContext(cache_);
        rml_.destroyContext("node_templates");
        context_ = nullptr;
        document_ = nullptr;
        if (open_) {
            init();
            rebuild();
        }
    }

    void RmlTemplateBrowser::refreshTemplates() {
        templates_ = manager_->templates();
        language_generation_ = event::LocalizationManager::getInstance().getCurrentLanguageGeneration();
        rebuild();
    }

    const NodeGraphTemplate* RmlTemplateBrowser::selected() const {
        const auto it = std::ranges::find(templates_, selected_id_, &NodeGraphTemplate::id);
        return it == templates_.end() ? nullptr : &*it;
    }

    void RmlTemplateBrowser::rebuild() {
        if (!document_)
            return;
        dirty_ = true;
        columns_ = 0;
        document_->GetElementById("window-frame")->SetClass("is-editing", !save_tree_.empty() || !rename_id_.empty());
        text(document_->GetElementById("title-text"), LOC(!save_tree_.empty() ? "node_editor.save_as_template" : !rename_id_.empty() ? "node_editor.template_rename"
                                                                                                                                     : "node_editor.templates_title"));
        if (!save_tree_.empty() || !rename_id_.empty()) {
            const auto* value = selected();
            std::string html = "<label>" + escaped(LOC("node_editor.template_name")) +
                               "<input id=\"template-name\" type=\"text\" value=\"" +
                               escaped(value && !rename_id_.empty() ? value->name : std::string{}) + "\"/></label>";
            if (rename_id_.empty()) {
                html += "<label>" + escaped(LOC("node_editor.template_description")) +
                        "<input id=\"template-description\" type=\"text\"/></label><label>" +
                        escaped(LOC("node_editor.template_category")) +
                        "<input id=\"template-category\" type=\"text\"/></label>";
            }
            document_->GetElementById("template-edit-content")->SetInnerRML(html);
            text(document_->GetElementById("template-confirm"), LOC("node_editor.save"));
            document_->GetElementById("template-confirm")->RemoveAttribute("disabled");
            document_->GetElementById("template-confirm")->SetClass("disabled", false);
            document_->GetElementById("template-name")->Focus();
            return;
        }
        std::map<std::string, int> categories;
        int user_count = 0;
        for (const auto& value : templates_) {
            if (value.builtin)
                ++categories[value.category];
            else
                ++user_count;
        }
        const auto nav = [&](const std::string& id, const std::string& label, int count, bool mine = false) {
            return "<button class=\"btn preferences-nav-item" + std::string(category_ == id ? " active" : "") +
                   (mine ? " my-templates" : "") + "\" data-action=\"category\" data-category=\"" + escaped(id) +
                   "\"><span>" + escaped(label) + "</span><span class=\"template-count\">" +
                   std::to_string(count) + "</span></button>";
        };
        std::string navigation = nav("all", LOC("node_editor.template_all"), static_cast<int>(templates_.size()));
        for (const auto& [category, count] : categories)
            navigation += nav(category, categoryLabel(category), count);
        navigation += nav("mine", LOC("node_editor.my_templates"), user_count, true);
        document_->GetElementById("template-nav")->SetInnerRML(navigation);
        std::string cards;
        const auto more_icon = rml_theme::pathToRmlImageSource(getAssetPath("icon/more-vertical.png"));
        for (const auto& value : templates_) {
            cards += "<div class=\"asset-card" + std::string(value.id == selected_id_ ? " is-selected" : "") +
                     "\" tabindex=\"0\" data-action=\"select\" data-template=\"" + escaped(value.id) +
                     "\"><div class=\"template-card-heading\"><span class=\"asset-card-title\">" + escaped(value.name) + "</span>";
            if (!value.builtin)
                cards += "<button class=\"asset-button asset-button--small-icon asset-card-menu\" data-action=\"menu\" title=\"" +
                         escaped(LOC("common.more")) + "\"><img src=\"" + more_icon + "\"/></button>";
            cards += "</div><span class=\"template-card-description\" title=\"" + escaped(value.description) + "\">" +
                     escaped(value.description) + "</span><span class=\"template-card-adjust\" title=\"" +
                     escaped(value.adjust) + "\">" + escaped(LOC("node_editor.template_adjust_short")) + " " +
                     escaped(value.adjust) + "</span><div class=\"template-card-bottom\"><div class=\"template-chips\">";
            for (const auto& kind : value.scene_kinds)
                cards += "<span class=\"template-chip\">" + escaped(LOC("node_template.kind." + kind)) + "</span>";
            cards += "</div><button class=\"btn btn--secondary\" data-action=\"add\"><span>" +
                     escaped(LOC("node_editor.template_add")) + "</span></button></div>";
            if (value.id == menu_id_)
                cards += "<div class=\"template-card-menu-items\"><button class=\"btn btn--secondary\" data-action=\"rename\"><span>" +
                         escaped(LOC("node_editor.template_rename")) +
                         "</span></button><button class=\"btn btn--secondary\" data-action=\"delete\"><span>" +
                         escaped(LOC("node_editor.delete")) + "</span></button></div>";
            cards += "</div>";
        }
        document_->GetElementById("template-grid")->SetInnerRML(cards);
        auto* search = static_cast<Rml::ElementFormControlInput*>(document_->GetElementById("template-search"));
        search->SetValue(query_);
        filter();
    }

    void RmlTemplateBrowser::filter() {
        const auto needle = lower(query_);
        auto* grid = document_->GetElementById("template-grid");
        int visible = 0;
        bool selection_visible = false;
        for (int i = 0; i < grid->GetNumChildren(); ++i) {
            auto* card = grid->GetChild(i);
            const auto& value = templates_.at(i);
            const bool category_match = category_ == "all" || (category_ == "mine" ? !value.builtin : value.builtin && value.category == category_);
            const bool match = category_match && lower(value.name + " " + value.description + " " + value.adjust).find(needle) != std::string::npos;
            card->SetClass("filtered", !match);
            visible += match;
            selection_visible |= match && value.id == selected_id_;
        }
        if (!selection_visible)
            selected_id_.clear();
        document_->GetElementById("template-search-placeholder")->SetClass("hidden", !query_.empty());
        text(document_->GetElementById("template-results-label"),
             std::vformat(LOC(visible == 1 ? "node_editor.template_count_one" : "node_editor.template_count"), std::make_format_args(visible)));
        auto* empty = document_->GetElementById("template-empty");
        empty->SetClass("hidden", visible != 0);
        text(empty, LOC(category_ == "mine" && query_.empty() ? "node_editor.template_mine_empty" : "node_editor.template_no_results"));
        document_->GetElementById("template-scroll")->SetScrollTop(0);
        updateFooter();
        dirty_ = true;
    }

    void RmlTemplateBrowser::select(std::string id) {
        selected_id_ = std::move(id);
        auto* grid = document_->GetElementById("template-grid");
        for (int i = 0; i < grid->GetNumChildren(); ++i) {
            auto* card = grid->GetChild(i);
            card->SetClass("is-selected", card->GetAttribute<std::string>("data-template", "") == selected_id_);
        }
        updateFooter();
        dirty_ = true;
    }

    void RmlTemplateBrowser::updateFooter() {
        const auto* value = selected();
        text(document_->GetElementById("template-selection"), value ? value->name : std::string{});
        text(document_->GetElementById("template-confirm"), LOC("node_editor.template_add_modifier"));
        for (const auto* id : {"template-confirm", "template-export"}) {
            auto* button = document_->GetElementById(id);
            if (value)
                button->RemoveAttribute("disabled");
            else
                button->SetAttribute("disabled", true);
            button->SetClass("disabled", !value);
        }
    }

    void RmlTemplateBrowser::setError(std::string message) {
        auto* element = document_->GetElementById("template-error");
        text(element, message);
        element->SetProperty("display", message.empty() ? "none" : "block");
        dirty_ = true;
    }

    void RmlTemplateBrowser::apply(std::string id) {
        const auto result = manager_->applyTemplate(target_, id);
        if (!result) {
            setError(result.error().message);
            return;
        }
        const auto modifier = (*result)->uuid;
        close();
        on_added_(modifier);
    }

    void RmlTemplateBrowser::edit(std::string id) {
        selected_id_ = id;
        rename_id_ = std::move(id);
        rebuild();
    }

    void RmlTemplateBrowser::ProcessEvent(Rml::Event& event) {
        auto* target = event.GetTargetElement();
        if (!target || !open_)
            return;
        if (event.GetType() == "change" && target->GetId() == "template-search") {
            query_ = static_cast<Rml::ElementFormControlInput*>(target)->GetValue();
            filter();
            return;
        }
        if (event.GetType() != "click")
            return;
        std::string action, id;
        Rml::Element* action_element = nullptr;
        for (auto* element = target; element; element = element->GetParentNode()) {
            if (action.empty() && element->HasAttribute("data-action")) {
                action = element->GetAttribute<std::string>("data-action", "");
                action_element = element;
            }
            if (id.empty())
                id = element->GetAttribute<std::string>("data-template", "");
        }
        if (action.empty())
            return;
        event.StopPropagation();
        dirty_ = true;
        setError({});
        if (action == "close") {
            close();
        } else if (action == "category") {
            category_ = action_element->GetAttribute<std::string>("data-category", "all");
            menu_id_.clear();
            rebuild();
        } else if (action == "select") {
            select(id);
            const auto now = std::chrono::steady_clock::now();
            if (last_click_id_ == id && now - last_click_ < std::chrono::milliseconds(400))
                apply(id);
            last_click_id_ = id;
            last_click_ = now;
        } else if (action == "add") {
            apply(id);
        } else if (action == "menu") {
            selected_id_ = id;
            menu_id_ = menu_id_ == id ? "" : id;
            rebuild();
            context_->Update();
            if (auto* menu = document_->QuerySelector(".template-card-menu-items"))
                menu->ScrollIntoView(false);
        } else if (action == "rename") {
            edit(id);
        } else if (action == "delete") {
            const auto result = manager_->deleteTemplate(id);
            if (!result)
                setError(result.error().message);
            else
                refreshTemplates();
        } else if (action == "import") {
            const auto path = OpenJsonFileDialog();
            if (!path.empty()) {
                const auto result = manager_->importTemplate(path);
                if (!result)
                    setError(result.error().message);
                else {
                    category_ = "mine";
                    selected_id_ = result->id;
                    refreshTemplates();
                }
            }
        } else if (action == "export" && selected()) {
            const auto path = SaveJsonFileDialog(selected()->name + ".json");
            if (!path.empty()) {
                const auto result = manager_->exportTemplate(selected_id_, path);
                if (!result)
                    setError(result.error().message);
            }
        } else if (action == "confirm") {
            const auto value = [&](const char* name) {
                const auto* input = dynamic_cast<Rml::ElementFormControlInput*>(document_->GetElementById(name));
                return input ? input->GetValue() : std::string{};
            };
            if (!rename_id_.empty()) {
                const auto result = manager_->renameTemplate(rename_id_, value("template-name"));
                if (!result)
                    setError(result.error().message);
                else {
                    rename_id_.clear();
                    menu_id_.clear();
                    refreshTemplates();
                }
            } else if (!save_tree_.empty()) {
                const auto result = manager_->saveTemplate(save_tree_, value("template-name"), value("template-description"), value("template-category"));
                if (!result)
                    setError(result.error().message);
                else
                    close();
            } else if (selected()) {
                apply(selected_id_);
            }
        }
    }

    void RmlTemplateBrowser::updateLayout() {
        const auto dp = std::max(0.01f, context_->GetDensityIndependentPixelRatio());
        auto* window = document_->GetElementById("window-frame");
        const auto preferred = std::clamp(width_ / dp * 0.7f, 960.0f, 1180.0f);
        if (save_tree_.empty() && rename_id_.empty())
            window->SetProperty("width", std::format("{}dp", preferred));
        else
            window->SetProperty("width", "480dp");
        window->SetClass("is-narrow", width_ / dp < 820.0f);
        context_->Update();
        const float available = document_->GetElementById("template-grid")->GetClientWidth() / dp - 6.0f;
        const int columns = available >= 900 ? 3 : available >= 520 ? 2
                                                                    : 1;
        const float card_width = std::max(80.0f, (available - 12.0f * (columns - 1)) / columns);
        if (columns != columns_ || std::abs(card_width - card_width_) > 0.5f) {
            columns_ = columns;
            card_width_ = card_width;
            auto* grid = document_->GetElementById("template-grid");
            for (int i = 0; i < grid->GetNumChildren(); ++i)
                grid->GetChild(i)->SetProperty("width", std::format("{}dp", card_width));
            context_->Update();
            dirty_ = true;
        }
    }

    void RmlTemplateBrowser::processInput(const PanelInputState& input, const bool blocked) {
        if (!open_ || !context_ || blocked) {
            if (context_)
                rml_.deactivateInput(context_);
            return;
        }
        if (rml_.routeInput(context_, input, [this](const PanelInputState& event) { processInput(event, false); }, true))
            return;
        rml_.trackContextFrame(context_, 0, 0);
        auto& focus = guiFocusState();
        focus.want_capture_mouse = true;
        focus.want_capture_keyboard = true;
        focus.want_text_input |= rml_input::isTextEditableElement(context_->GetFocusElement());
        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift, input.key_alt, input.key_super);
        std::vector<FrameMouseButtonEvent> fallback;
        if (input.mouse_button_events.empty()) {
            if (input.mouse_clicked[0])
                fallback.push_back({.button = 0, .down = true, .x = input.mouse_x, .y = input.mouse_y});
            if (input.mouse_released[0])
                fallback.push_back({.button = 0, .down = false, .x = input.mouse_x, .y = input.mouse_y});
        }
        const auto& events = input.mouse_button_events.empty() ? fallback : input.mouse_button_events;
        const auto dimensions = context_->GetDimensions();
        dirty_ |= rml_input::replayButtonEvents(*context_, pointer_down_, events,
                                                {input.screen_x, input.screen_y}, {dimensions.x, dimensions.y}, mods, false,
                                                [](const Rml::Element* element) { return element != nullptr; });
        const int x = static_cast<int>(input.mouse_x - input.screen_x);
        const int y = static_cast<int>(input.mouse_y - input.screen_y);
        if (x != mouse_x_ || y != mouse_y_) {
            mouse_x_ = x;
            mouse_y_ = y;
            context_->ProcessMouseMove(x, y, mods);
            dirty_ = true;
        }
        if (input.mouse_wheel != 0 || input.mouse_wheel_x != 0) {
            context_->ProcessMouseWheel(Rml::Vector2f(-input.mouse_wheel_x, -input.mouse_wheel), mods);
            dirty_ = true;
        }
        for (const auto& event : input.input_events) {
            if (event.kind == FrameInputEventKind::KeyDown && event.scancode == SDL_SCANCODE_ESCAPE) {
                close();
                return;
            }
            dirty_ |= rml_input::processKeyboardEvent(*context_, event, rml_.getTextInputHandler());
        }
        auto* hover = context_->GetHoverElement();
        tooltip_.setHover(resolveRmlTooltip(hover), hover);
        if (const auto deadline = tooltip_.revealDeadline())
            lfs::python::request_redraw_after(std::max(0.0, std::chrono::duration<double>(*deadline - std::chrono::steady_clock::now()).count()));
    }

    bool RmlTemplateBrowser::needsAnimationFrame() const {
        return open_ && (dirty_ || tooltip_.revealDue());
    }

    void RmlTemplateBrowser::render(const int width, const int height) {
        if (!open_ || !document_ || width <= 0 || height <= 0)
            return;
        dirty_ |= lfs::python::consume_pending_rml_document_updates(document_);
        if (width != width_ || height != height_) {
            width_ = width;
            height_ = height;
            context_->SetDimensions({width, height});
            dirty_ = true;
        }
        const auto signature = rml_theme::currentThemeSignature();
        if (theme_signature_ != signature) {
            theme_signature_ = signature;
            rml_theme::applyTheme(document_,
                                  rml_theme::loadBaseRCSS("rmlui/preferences.rcss") +
                                      rml_theme::loadBaseRCSS("rmlui/asset_manager.rcss") +
                                      rml_theme::loadBaseRCSS("rmlui/node_templates.rcss"),
                                  rml_theme::loadBaseRCSS("rmlui/preferences.theme.rcss") +
                                      rml_theme::loadBaseRCSS("rmlui/asset_manager.theme.rcss") +
                                      rml_theme::loadBaseRCSS("rmlui/node_templates.theme.rcss"));
            dirty_ = true;
        }
        if (language_generation_ != event::LocalizationManager::getInstance().getCurrentLanguageGeneration()) {
            rml_documents::refreshLocalizedContent(document_);
            refreshTemplates();
        }
        if (dirty_)
            updateLayout();
        dirty_ |= tooltip_.apply(document_, mouse_x_, mouse_y_, width_, height_);
        if (dirty_)
            context_->Update();
        rml_.trackContextFrame(context_, 0, 0);
        rml_.queueCachedVulkanContext({.context = context_, .cache = &cache_, .cache_width = width, .cache_height = height, .offset_x = 0, .offset_y = 0, .draw_width = static_cast<float>(width), .draw_height = static_cast<float>(height), .refresh = dirty_ || cache_.texture == 0, .foreground = true, .clip = {}});
        dirty_ = false;
    }
} // namespace lfs::vis::gui
