/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/startup_overlay.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/string_keys.hpp"
#include "internal/resource_paths.hpp"
#include "preferences.hpp"
#include "theme/theme.hpp"
#include "visualizer/app_store.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/Input.h>
#include <algorithm>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <utility>

#ifdef _WIN32
#include <shellapi.h>
#include <windows.h>
#endif

namespace lfs::vis::gui {

    class LinkClickListener final : public Rml::EventListener {
    public:
        void ProcessEvent(Rml::Event& event) override {
            auto* el = event.GetCurrentElement();
            if (!el)
                return;
            auto url = el->GetAttribute("data-url", Rml::String(""));
            if (!url.empty())
                StartupOverlay::openURL(url.c_str());
        }
    };

    class LangChangeListener final : public Rml::EventListener {
    public:
        explicit LangChangeListener(RmlUIManager* mgr) : mgr_(mgr) {}

        void ProcessEvent(Rml::Event& event) override {
            auto* el = event.GetCurrentElement();
            if (!el)
                return;
            auto* select = dynamic_cast<Rml::ElementFormControlSelect*>(el);
            if (!select)
                return;
            int idx = select->GetSelection();
            if (idx < 0)
                return;

            auto& loc = lfs::event::LocalizationManager::getInstance();
            const auto available = loc.getAvailableLanguages();
            if (idx >= static_cast<int>(available.size()))
                return;

            const auto& lang = available[idx];
            if (loc.setLanguage(lang)) {
                lfs::vis::saveLanguagePreference(lang);
                lfs::vis::publish_language_generation();
            }
            if (mgr_ && (lang == "ja" || lang == "ko" || lang == "zh"))
                mgr_->ensureCjkFontsLoaded();
        }

    private:
        RmlUIManager* mgr_ = nullptr;
    };

    void StartupOverlay::openURL(const char* url) {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::string cmd = "xdg-open \"" + std::string(url) + "\" &";
        if (const int status = std::system(cmd.c_str()); status != 0) {
            LOG_WARN("StartupOverlay: xdg-open failed for '{}': status {}", url, status);
        }
#endif
    }

    void StartupOverlay::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("startup_overlay", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("StartupOverlay: failed to create RML context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/startup.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("StartupOverlay: failed to load startup.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("StartupOverlay: resource not found: {}", e.what());
            return;
        }

        populateLanguages();
        updateLocalizedText();

        link_listener_ = new LinkClickListener();
        for (const char* id : {"link-discord", "link-x", "link-youtube", "link-donate",
                               "link-core11", "link-volinga"}) {
            auto* el = document_->GetElementById(id);
            if (el)
                el->AddEventListener(Rml::EventId::Click, link_listener_);
        }

        lang_listener_ = new LangChangeListener(rml_manager_);
        auto* lang_select = document_->GetElementById("lang-select");
        if (lang_select)
            lang_select->AddEventListener(Rml::EventId::Change, lang_listener_);

        updateTheme();
    }

    void StartupOverlay::shutdown() {
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("startup_overlay");
        rml_context_ = nullptr;
        document_ = nullptr;
        delete link_listener_;
        link_listener_ = nullptr;
        delete lang_listener_;
        lang_listener_ = nullptr;
    }

    void StartupOverlay::reloadResources() {
        if (!rml_context_)
            return;

        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        has_theme_signature_ = false;
        has_language_generation_ = false;
        shown_frames_ = 0;
        width_ = 0;
        height_ = 0;
        fitted_dp_ratio_ = 0.0f;
        content_dirty_ = true;
        last_mouse_valid_ = false;

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/startup.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("StartupOverlay: failed to reload startup.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("StartupOverlay: resource not found during reload: {}", e.what());
            return;
        }

        populateLanguages();
        updateLocalizedText();

        if (!link_listener_)
            link_listener_ = new LinkClickListener();
        for (const char* id : {"link-discord", "link-x", "link-youtube", "link-donate",
                               "link-core11", "link-volinga"}) {
            auto* el = document_->GetElementById(id);
            if (el)
                el->AddEventListener(Rml::EventId::Click, link_listener_);
        }

        if (!lang_listener_)
            lang_listener_ = new LangChangeListener(rml_manager_);
        auto* lang_select = document_->GetElementById("lang-select");
        if (lang_select)
            lang_select->AddEventListener(Rml::EventId::Change, lang_listener_);

        updateTheme();
    }

    void StartupOverlay::dismiss() {
        if (!visible_)
            return;
        if (lfs::vis::loadLanguagePreference().empty()) {
            const auto& language = lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
            if (!language.empty())
                lfs::vis::saveLanguagePreference(language);
        }
        visible_ = false;
        input_ = nullptr;
        last_mouse_valid_ = false;
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
    }

    void StartupOverlay::setPluginLoadState(const bool started,
                                            const bool active,
                                            float progress,
                                            std::string stage) {
        progress = std::clamp(progress, 0.0f, 1.0f);
        std::lock_guard lock(plugin_load_mutex_);
        assert(started || !active);
        assert(!plugin_load_state_started_ || started);
        plugin_load_state_.active = active;
        plugin_load_state_.progress = progress;
        plugin_load_state_.stage = std::move(stage);
        plugin_load_state_started_ = started;
        plugin_load_complete_ = !started || !active;
        content_dirty_ = true;
    }

    bool StartupOverlay::needsAnimationFrame() const {
        if (!visible_)
            return false;
        if (shown_frames_ < 3 || content_dirty_)
            return true;
        if (!has_theme_signature_ ||
            rml_theme::currentThemeSignature() != last_theme_signature_)
            return true;
        if (!has_language_generation_ ||
            app_store().language_generation.get() != last_language_generation_)
            return true;
        {
            std::lock_guard lock(plugin_load_mutex_);
            if (plugin_load_state_.active)
                return true;
        }
        return false;
    }

    bool StartupOverlay::isPluginLoadComplete() const {
        std::lock_guard lock(plugin_load_mutex_);
        return plugin_load_complete_;
    }

    bool StartupOverlay::blocksUnderlayInput() const {
        if (!visible_)
            return false;
        std::lock_guard lock(plugin_load_mutex_);
        return !plugin_load_state_started_;
    }

    static std::string escapeRmlText(const std::string& input) {
        std::string out;
        out.reserve(input.size());
        for (const char ch : input) {
            switch (ch) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out += ch; break;
            }
        }
        return out;
    }

    template <typename... Args>
    static std::string formatRuntime(const char* fmt, Args&&... args) {
        return std::vformat(std::string_view{fmt}, std::make_format_args(args...));
    }

    static bool extractProgressCounts(const std::string& input, std::size_t& current, std::size_t& total) {
        std::size_t found = 0;
        const char* const begin = input.data();
        const char* const end = begin + input.size();
        const char* it = begin;
        while (it < end) {
            if (!std::isdigit(static_cast<unsigned char>(*it))) {
                ++it;
                continue;
            }

            const char* num_begin = it;
            while (it < end && std::isdigit(static_cast<unsigned char>(*it)))
                ++it;

            std::size_t value = 0;
            const auto [ptr, ec] = std::from_chars(num_begin, it, value);
            if (ec != std::errc{})
                return false;

            if (found == 0) {
                current = value;
            } else if (found == 1) {
                total = value;
                return true;
            }
            ++found;
        }
        return false;
    }

    void StartupOverlay::populateLanguages() {
        auto* select_el = document_->GetElementById("lang-select");
        if (!select_el)
            return;
        auto* select = dynamic_cast<Rml::ElementFormControlSelect*>(select_el);
        if (!select)
            return;

        auto& loc = lfs::event::LocalizationManager::getInstance();
        if (auto* row = document_->GetElementById("lang-row")) {
            row->SetProperty("display", lfs::vis::loadLanguagePreference().empty() ? "flex" : "none");
        }
        const auto langs = loc.getAvailableLanguages();
        const auto names = loc.getAvailableLanguageNames();
        const auto& current = loc.getCurrentLanguage();

        if (rml_manager_) {
            if (current == "ja" || current == "ko" || current == "zh") {
                ensureLanguageDropdownFontsLoaded();
            }
        }

        for (size_t i = 0; i < langs.size(); ++i) {
            select->Add(names[i], langs[i]);
            if (langs[i] == current)
                select->SetSelection(static_cast<int>(i));
        }
    }

    void StartupOverlay::updateLocalizedText() {
        if (!document_)
            return;

        auto set_text = [&](const char* id, const char* key) {
            auto* el = document_->GetElementById(id);
            if (el)
                el->SetInnerRML(LOC(key));
        };

        set_text("supported-text", lichtfeld::Strings::Startup::SUPPORTED_BY);
        set_text("lang-label", lichtfeld::Strings::Preferences::LANGUAGE);
        set_text("discord-link", lichtfeld::Strings::Startup::DISCORD);
        set_text("donate-link", lichtfeld::Strings::Startup::DONATE);
        has_applied_plugin_load_state_ = false;
        updateClickHintUI();
    }

    void StartupOverlay::updateClickHintUI() {
        if (!document_)
            return;

        auto* hint = document_->GetElementById("click-hint");
        if (!hint)
            return;

        if (visible_ && shown_frames_ > 2) {
            hint->SetInnerRML(LOC(lichtfeld::Strings::Startup::CLICK_TO_CONTINUE));
            hint->SetProperty("visibility", "visible");
        } else {
            hint->SetInnerRML(LOC(lichtfeld::Strings::Startup::CLICK_TO_CONTINUE));
            hint->SetProperty("visibility", "hidden");
        }
    }

    bool StartupOverlay::updatePluginLoadUI() {
        if (!document_)
            return false;

        PluginLoadState current;
        {
            std::lock_guard lock(plugin_load_mutex_);
            current = plugin_load_state_;
        }

        if (has_applied_plugin_load_state_ &&
            current.active == applied_plugin_load_state_.active &&
            current.progress == applied_plugin_load_state_.progress &&
            current.stage == applied_plugin_load_state_.stage) {
            return false;
        }

        applied_plugin_load_state_ = current;
        has_applied_plugin_load_state_ = true;

        auto* row = document_->GetElementById("plugin-load-row");
        auto* track = document_->GetElementById("plugin-load-track");
        auto* stage = document_->GetElementById("plugin-load-stage");
        auto* percent = document_->GetElementById("plugin-load-percent");
        auto* fill = document_->GetElementById("plugin-load-fill");
        if (!row || !track || !stage || !percent || !fill)
            return true;

        if (!plugin_load_state_started_) {
            row->SetProperty("display", "none");
            updateClickHintUI();
            return true;
        }

        row->SetProperty("display", "flex");
        if (plugin_load_complete_) {
            track->SetProperty("display", "block");
            track->SetProperty("width", "360dp");
            track->SetProperty("height", "18dp");
            track->SetProperty("background-color", "transparent");
            track->SetProperty("border-color", "transparent");
            fill->SetProperty("display", "none");
            percent->SetProperty("display", "none");
            row->SetProperty("height", "18dp");
            row->SetProperty("margin-top", "14dp");
            std::size_t current_count = 0;
            std::size_t total_count = 0;
            if (extractProgressCounts(current.stage, current_count, total_count)) {
                stage->SetInnerRML(
                    formatRuntime(LOC(lichtfeld::Strings::Startup::LOADED_PLUGINS), current_count, total_count));
            } else {
                stage->SetInnerRML(escapeRmlText(current.stage));
            }
            stage->SetProperty("width", "360dp");
            stage->SetProperty("height", "18dp");
            stage->SetProperty("margin-left", "0");
            stage->SetProperty("margin-top", "0");
            stage->SetProperty("line-height", "18dp");
            stage->SetProperty("text-align", "center");
        } else {
            track->SetProperty("display", "block");
            track->SetProperty("width", "300dp");
            track->SetProperty("height", "14dp");
            track->SetProperty("background-color", "rgba(255, 255, 255, 0.06)");
            track->SetProperty("border-color", "rgba(255, 255, 255, 0.15)");
            fill->SetProperty("display", "block");
            percent->SetProperty("display", "block");
            row->SetProperty("height", "16dp");
            row->SetProperty("margin-top", "14dp");
            stage->SetProperty("width", "286dp");
            stage->SetProperty("height", "14dp");
            stage->SetProperty("margin-left", "7dp");
            stage->SetProperty("margin-top", "-14dp");
            stage->SetProperty("line-height", "14dp");
            stage->SetProperty("text-align", "center");
            stage->SetInnerRML(escapeRmlText(current.stage));
        }
        percent->SetInnerRML(std::format("{:.0f}%", current.progress * 100.0f));
        fill->SetProperty("width", std::format("{:.1f}%", current.progress * 100.0f));
        updateClickHintUI();
        return true;
    }

    void StartupOverlay::updateTheme() {
        if (!document_)
            return;

        const auto& t = theme();
        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        const bool is_light = t.isLightTheme();
        const auto logo_path = lfs::vis::getAssetPath(
            is_light ? "lichtfeld-splash-logo-dark.png" : "lichtfeld-splash-logo.png");
        auto* logo = document_->GetElementById("logo");
        if (logo) {
            logo->SetAttribute("src", rml_theme::pathToRmlImageSource(logo_path));
            auto [w, h, c] = lfs::core::get_image_info(logo_path);
            if (w > 0 && h > 0) {
                logo->SetProperty("width", std::format("{:.0f}dp", w * 1.3f));
                logo->SetProperty("height", std::format("{:.0f}dp", h * 1.3f));
            }
        }

        const auto core11_path = lfs::vis::getAssetPath(
            is_light ? "core11-logo-dark.png" : "core11-logo.png");
        auto* core11 = document_->GetElementById("core11-logo");
        if (core11) {
            core11->SetAttribute("src", rml_theme::pathToRmlImageSource(core11_path));
            auto [w, h, c] = lfs::core::get_image_info(core11_path);
            if (w > 0 && h > 0) {
                core11->SetProperty("width", std::format("{:.0f}dp", w * 0.5f));
                core11->SetProperty("height", std::format("{:.0f}dp", h * 0.5f));
            }
        }

        const auto volinga_path = lfs::vis::getAssetPath(
            is_light ? "volinga-logo-dark.png" : "volinga-logo.png");
        auto* volinga = document_->GetElementById("volinga-logo");
        if (volinga) {
            volinga->SetAttribute("src", rml_theme::pathToRmlImageSource(volinga_path));
            auto [w, h, c] = lfs::core::get_image_info(volinga_path);
            if (w > 0 && h > 0) {
                constexpr float TARGET_HEIGHT_DP = 24.0f;
                const float scale = TARGET_HEIGHT_DP / static_cast<float>(h);
                volinga->SetProperty("width", std::format("{:.0f}dp", w * scale));
                volinga->SetProperty("height", std::format("{:.0f}dp", h * scale));
            }
        }

        auto base_rcss = rml_theme::loadBaseRCSS("rmlui/startup.rcss");
        rml_theme::applyTheme(document_, base_rcss, rml_theme::loadBaseRCSS("rmlui/startup.theme.rcss"));
    }

    bool StartupOverlay::hasInputActivity(const PanelInputState& input) const {
        if (!last_mouse_valid_ ||
            std::abs(input.mouse_x - last_mouse_x_) > 0.5f ||
            std::abs(input.mouse_y - last_mouse_y_) > 0.5f) {
            return true;
        }
        for (int i = 0; i < 3; ++i) {
            if (input.mouse_clicked[i] || input.mouse_released[i])
                return true;
        }
        return input.mouse_wheel != 0.0f ||
               !input.keys_pressed.empty() ||
               !input.keys_repeated.empty() ||
               !input.keys_released.empty() ||
               !input.text_codepoints.empty() ||
               !input.text_inputs.empty() ||
               input.has_text_editing;
    }

    bool StartupOverlay::isLanguageSelectOpen() const {
        auto* lang_el = document_ ? document_->GetElementById("lang-select") : nullptr;
        auto* sel = dynamic_cast<Rml::ElementFormControlSelect*>(lang_el);
        return sel && sel->IsSelectBoxVisible();
    }

    bool StartupOverlay::isLanguageSelectHit(const float local_x, const float local_y) const {
        auto* lang_el = document_ ? document_->GetElementById("lang-select") : nullptr;
        const auto bounds = elementBorderRect(lang_el);
        return bounds && bounds->contains(local_x, local_y);
    }

    std::optional<StartupOverlayRect> StartupOverlay::elementBorderRect(Rml::Element* const element) const {
        if (!element)
            return std::nullopt;

        const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
        if (size.x <= 0.0f || size.y <= 0.0f)
            return std::nullopt;

        const auto offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
        return StartupOverlayRect{
            .left = offset.x,
            .top = offset.y,
            .right = offset.x + size.x,
            .bottom = offset.y + size.y,
        };
    }

    std::optional<StartupOverlayRect> StartupOverlay::languageDropdownRect() const {
        auto* lang_el = document_ ? document_->GetElementById("lang-select") : nullptr;
        auto* select = dynamic_cast<Rml::ElementFormControlSelect*>(lang_el);
        if (!select || !select->IsSelectBoxVisible())
            return std::nullopt;

        for (int i = 0; i < select->GetNumChildren(true); ++i) {
            auto* const child = select->GetChild(i);
            if (child && child->GetTagName() == "selectbox")
                return elementBorderRect(child);
        }
        return std::nullopt;
    }

    bool StartupOverlay::isLanguageDropdownHit(const float local_x, const float local_y) const {
        const auto bounds = languageDropdownRect();
        return bounds && bounds->contains(local_x, local_y);
    }

    bool StartupOverlay::blocksPointerInput(const float window_x, const float window_y) const {
        if (!visible_ || !document_)
            return false;

        const auto card = elementBorderRect(document_->GetElementById("overlay-box"));
        if (!card)
            return false;
        return startupOverlayBlocksPointer(*card, languageDropdownRect(), window_x, window_y);
    }

    bool StartupOverlay::isLinkHit(const float local_x, const float local_y) const {
        if (!rml_context_ || !document_)
            return false;

        for (auto* el = rml_context_->GetElementAtPoint(Rml::Vector2f(local_x, local_y)); el;
             el = el->GetParentNode()) {
            if (!el->GetAttribute("data-url", Rml::String("")).empty())
                return true;
        }
        return false;
    }

    void StartupOverlay::ensureLanguageDropdownFontsLoaded() {
        if (language_dropdown_fonts_requested_ || !rml_manager_)
            return;

        rml_manager_->ensureCjkFontsLoaded();
        language_dropdown_fonts_requested_ = true;
    }

    StartupOverlay::InputForwardResult StartupOverlay::forwardInput(
        const PanelInputState& input, float overlay_x,
        float overlay_y, float overlay_w, float overlay_h) {
        assert(rml_context_);
        InputForwardResult result;
        if (rml_manager_) {
            rml_manager_->trackContextFrame(rml_context_,
                                            static_cast<int>(overlay_x - input.screen_x),
                                            static_cast<int>(overlay_y - input.screen_y));
        }

        const float local_x = input.mouse_x - overlay_x;
        const float local_y = input.mouse_y - overlay_y;

        const bool hovered = local_x >= 0 && local_y >= 0 &&
                             local_x < overlay_w && local_y < overlay_h;
        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);
        bool replayed_button_events = false;
        for (const auto& event : input.mouse_button_events) {
            if (event.button >= 3)
                continue;
            const float event_x = event.x - overlay_x;
            const float event_y = event.y - overlay_y;
            if (event_x < 0.0f || event_y < 0.0f ||
                event_x >= overlay_w || event_y >= overlay_h)
                continue;
            if (event.down && isLanguageSelectHit(event_x, event_y))
                ensureLanguageDropdownFontsLoaded();
            rml_context_->ProcessMouseMove(static_cast<int>(event_x),
                                           static_cast<int>(event_y), mods);
            if (event.down)
                rml_context_->ProcessMouseButtonDown(event.button, mods);
            else
                rml_context_->ProcessMouseButtonUp(event.button, mods);
            replayed_button_events = true;
            result.event_forwarded = true;
        }

        if (hovered) {
            const bool opening_language_select =
                input.mouse_clicked[0] && isLanguageSelectHit(local_x, local_y);
            if (isLanguageSelectOpen() || opening_language_select)
                ensureLanguageDropdownFontsLoaded();
            rml_context_->ProcessMouseMove(static_cast<int>(local_x),
                                           static_cast<int>(local_y), mods);
            result.event_forwarded = true;

            if (!replayed_button_events && input.mouse_clicked[0]) {
                rml_context_->ProcessMouseButtonDown(0, mods);
                result.event_forwarded = true;
            }
            if (!replayed_button_events && input.mouse_released[0]) {
                rml_context_->ProcessMouseButtonUp(0, mods);
                result.event_forwarded = true;
            }

            if (input.mouse_wheel != 0.0f) {
                rml_context_->ProcessMouseWheel(
                    Rml::Vector2f(-input.mouse_wheel_x, -input.mouse_wheel), mods);
                result.event_forwarded = true;
            }
        }

        if (!input.viewport_keyboard_focus &&
            rml_input::hasFocusedKeyboardTarget(rml_context_->GetFocusElement())) {
            const int focused_mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                                  input.key_alt, input.key_super);
            for (int sc : input.keys_pressed) {
                if (sc == SDL_SCANCODE_ESCAPE && rml_input::cancelFocusedElement(*rml_context_)) {
                    result.escape_consumed = true;
                    result.event_forwarded = true;
                    continue;
                }

                const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    rml_context_->ProcessKeyDown(rml_key, focused_mods);
                    result.event_forwarded = true;
                }
            }

            for (int sc : input.keys_released) {
                if (result.escape_consumed && sc == SDL_SCANCODE_ESCAPE)
                    continue;

                const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    rml_context_->ProcessKeyUp(rml_key, focused_mods);
                    result.event_forwarded = true;
                }
            }
        }

        last_mouse_valid_ = true;
        last_mouse_x_ = input.mouse_x;
        last_mouse_y_ = input.mouse_y;
        return result;
    }

    bool StartupOverlay::applyFitRatio(const int context_width,
                                       const int context_height,
                                       const float maximum_ratio) {
        assert(rml_context_);
        assert(document_);

        const auto card = elementBorderRect(document_->GetElementById("overlay-box"));
        if (!card)
            return false;

        const float layout_ratio = rml_context_->GetDensityIndependentPixelRatio();
        const float shadow_margin = rml_theme::layeredShadowPadding(theme(), 4) * layout_ratio;
        constexpr float FIT_GUARD_PIXELS = 1.0f;
        StartupOverlayRect visual_bounds{
            .left = card->left - shadow_margin - FIT_GUARD_PIXELS,
            .top = card->top - shadow_margin - FIT_GUARD_PIXELS,
            .right = card->right + shadow_margin + FIT_GUARD_PIXELS,
            .bottom = card->bottom + shadow_margin + FIT_GUARD_PIXELS,
        };
        if (const auto dropdown = languageDropdownRect()) {
            visual_bounds.left = std::min(visual_bounds.left, dropdown->left);
            visual_bounds.top = std::min(visual_bounds.top, dropdown->top);
            visual_bounds.right = std::max(visual_bounds.right, dropdown->right);
            visual_bounds.bottom = std::max(visual_bounds.bottom, dropdown->bottom);
        }

        const float fitted_ratio = startupOverlayFitDpRatio(
            maximum_ratio, layout_ratio,
            static_cast<float>(context_width), static_cast<float>(context_height),
            visual_bounds);
        if (std::abs(fitted_ratio - layout_ratio) <= 0.001f) {
            fitted_dp_ratio_ = fitted_ratio;
            return false;
        }

        rml_context_->SetDensityIndependentPixelRatio(fitted_ratio);
        fitted_dp_ratio_ = fitted_ratio;
        return true;
    }

    void StartupOverlay::render(const float window_x, const float window_y,
                                const float window_width, const float window_height,
                                const bool drag_hovering) {
        if (!visible_)
            return;

        if (!rml_context_ || !document_)
            return;

        if (blocksUnderlayInput()) {
            auto& focus = guiFocusState();
            focus.want_capture_mouse = true;
            focus.want_capture_keyboard = true;
        }

        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;

        const int ctx_w = static_cast<int>(window_width);
        const int ctx_h = static_cast<int>(window_height);
        const bool size_changed = ctx_w != width_ || ctx_h != height_;
        const float maximum_dp_ratio = rml_manager_->getDpRatio();
        const bool density_changed =
            std::abs(rml_context_->GetDensityIndependentPixelRatio() - fitted_dp_ratio_) > 0.001f;
        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        const bool theme_changed = !has_theme_signature_ || theme_signature != last_theme_signature_;
        const auto language_generation = app_store().language_generation.get();
        const bool language_changed =
            !has_language_generation_ || language_generation != last_language_generation_;
        bool refresh_cache = content_dirty_ || size_changed || density_changed ||
                             theme_changed || language_changed || shown_frames_ < 3;

        if (theme_changed) {
            updateTheme();
            refresh_cache = true;
        }
        if (language_changed) {
            updateLocalizedText();
            last_language_generation_ = language_generation;
            has_language_generation_ = true;
            refresh_cache = true;
        }

        if (size_changed) {
            width_ = ctx_w;
            height_ = ctx_h;
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
            rml_context_->SetDimensions(Rml::Vector2i(ctx_w, ctx_h));
            document_->SetProperty("width", std::format("{}px", ctx_w));
            document_->SetProperty("height", std::format("{}px", ctx_h));
            last_mouse_valid_ = false;
            refresh_cache = true;
        }

        if (shown_frames_ == 3) {
            updateClickHintUI();
            refresh_cache = true;
        }

        if (updatePluginLoadUI())
            refresh_cache = true;

        bool updated_this_frame = false;
        if (refresh_cache) {
            rml_context_->SetDensityIndependentPixelRatio(maximum_dp_ratio);
            rml_context_->Update();
            if (applyFitRatio(ctx_w, ctx_h, maximum_dp_ratio))
                rml_context_->Update();
            updated_this_frame = true;
        }

        bool escape_consumed = false;
        bool rml_select_open = isLanguageSelectOpen();
        const bool rml_select_was_open = rml_select_open;
        bool input_event_forwarded = false;
        const bool plugin_load_complete = isPluginLoadComplete();
        if (input_ && hasInputActivity(*input_) &&
            (plugin_load_complete || rml_select_open ||
             isLanguageSelectHit(input_->mouse_x - window_x,
                                 input_->mouse_y - window_y) ||
             isLinkHit(input_->mouse_x - window_x,
                       input_->mouse_y - window_y))) {
            const auto input_result = forwardInput(*input_, window_x, window_y,
                                                   window_width, window_height);
            escape_consumed = input_result.escape_consumed;
            refresh_cache = refresh_cache || input_result.event_forwarded;
            input_event_forwarded = input_result.event_forwarded;
            rml_select_open = rml_select_open || isLanguageSelectOpen();
        }
        const auto language_generation_after_input = app_store().language_generation.get();
        if (language_generation_after_input != last_language_generation_) {
            updateLocalizedText();
            last_language_generation_ = language_generation_after_input;
            has_language_generation_ = true;
            refresh_cache = true;
        }
        if (input_event_forwarded || (refresh_cache && !updated_this_frame)) {
            rml_context_->Update();
            if (applyFitRatio(ctx_w, ctx_h, maximum_dp_ratio))
                rml_context_->Update();
        }

        const bool select_interaction_active = rml_select_open || isLanguageSelectOpen();
        ++shown_frames_;

        bool clicked_language_select = false;
        bool clicked_language_dropdown = false;
        bool clicked_link = false;
        if (input_) {
            const float local_x = input_->mouse_x - window_x;
            const float local_y = input_->mouse_y - window_y;
            clicked_language_select = input_->mouse_clicked[0] && isLanguageSelectHit(local_x, local_y);
            clicked_language_dropdown = input_->mouse_clicked[0] && isLanguageDropdownHit(local_x, local_y);
            clicked_link = input_->mouse_clicked[0] && isLinkHit(local_x, local_y);
        }

        if (shown_frames_ > 2 && !drag_hovering && input_) {
            const bool mouse_clicked =
                input_->mouse_clicked[0] || input_->mouse_clicked[1] || input_->mouse_clicked[2];
            const bool key_action = (!escape_consumed && !select_interaction_active &&
                                     hasKey(input_->keys_pressed, SDL_SCANCODE_ESCAPE)) ||
                                    (!select_interaction_active &&
                                     (hasKey(input_->keys_pressed, SDL_SCANCODE_SPACE) ||
                                      hasKey(input_->keys_pressed, SDL_SCANCODE_RETURN) ||
                                      hasKey(input_->keys_pressed, SDL_SCANCODE_KP_ENTER)));

            if (key_action) {
                LOG_DEBUG("StartupOverlay: dismissed by key action");
                dismiss();
            } else if (mouse_clicked && !rml_select_was_open && !clicked_language_select &&
                       !clicked_language_dropdown && !clicked_link) {
                LOG_DEBUG("StartupOverlay: dismissed by mouse click");
                dismiss();
            }
        }

        if (!visible_)
            return;

        const float offset_x = window_x;
        const float offset_y = window_y;
        rml_manager_->trackContextFrame(rml_context_,
                                        static_cast<int>(offset_x),
                                        static_cast<int>(offset_y));
        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = ctx_w,
            .cache_height = ctx_h,
            .offset_x = offset_x,
            .offset_y = offset_y,
            .draw_width = window_width,
            .draw_height = window_height,
            .refresh = refresh_cache,
            .foreground = true,
            .clip_enabled = true,
            .clip = {
                .x1 = offset_x,
                .y1 = offset_y,
                .x2 = offset_x + window_width,
                .y2 = offset_y + window_height,
            },
        });
        content_dirty_ = false;
    }

} // namespace lfs::vis::gui
