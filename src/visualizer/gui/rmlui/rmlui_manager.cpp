/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_manager.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/elements/chromaticity_element.hpp"
#include "gui/rmlui/elements/color_picker_element.hpp"
#include "gui/rmlui/elements/crf_curve_element.hpp"
#include "gui/rmlui/elements/loss_graph_element.hpp"
#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "gui/rmlui/rml_fbo.hpp"
#include "gui/rmlui/rml_text_input_handler.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/rmlui/rmlui_system_interface.hpp"
#include "internal/resource_paths.hpp"
#include "python/python_runtime.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Debugger.h>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <string_view>

namespace lfs::vis::gui {

    namespace {
        bool envFlagEnabled(const char* name) {
            const char* value = std::getenv(name);
            if (!value || !*value)
                return false;
            return std::string_view(value) != "0";
        }
    } // namespace

    RmlUIManager::RmlUIManager() = default;

    RmlUIManager::~RmlUIManager() {
        if (initialized_)
            shutdown();
    }

    bool RmlUIManager::init(SDL_Window* window, float dp_ratio) {
        assert(!initialized_);
        assert(window);
        assert(dp_ratio >= 1.0f);

        dp_ratio_ = dp_ratio;
        window_ = window;
        debugger_enabled_ = envFlagEnabled("LFS_RML_DEBUGGER");

        system_interface_ = std::make_unique<RmlSystemInterface>(window);
        render_interface_ = std::make_unique<RmlRenderInterface>();
        text_input_handler_ = std::make_unique<RmlTextInputHandler>();

        Rml::SetSystemInterface(system_interface_.get());
        Rml::SetRenderInterface(render_interface_.get());
        Rml::SetTextInputHandler(text_input_handler_.get());

        if (!Rml::Initialise()) {
            LOG_ERROR("Failed to initialize RmlUI");
            return false;
        }

        static Rml::ElementInstancerGeneric<ChromaticityElement> chromaticity_instancer;
        static Rml::ElementInstancerGeneric<ColorPickerElement> color_picker_instancer;
        static Rml::ElementInstancerGeneric<CRFCurveElement> crf_curve_instancer;
        static Rml::ElementInstancerGeneric<LossGraphElement> loss_graph_instancer;
        static Rml::ElementInstancerGeneric<SceneGraphElement> scene_graph_instancer;
        Rml::Factory::RegisterElementInstancer("chromaticity-diagram", &chromaticity_instancer);
        Rml::Factory::RegisterElementInstancer("color-picker", &color_picker_instancer);
        Rml::Factory::RegisterElementInstancer("crf-curve", &crf_curve_instancer);
        Rml::Factory::RegisterElementInstancer("loss-graph", &loss_graph_instancer);
        Rml::Factory::RegisterElementInstancer("scene-graph", &scene_graph_instancer);

        try {
            const auto regular_path = lfs::vis::getAssetPath("fonts/Inter-Regular.ttf");
            if (Rml::LoadFontFace(regular_path.string(), true)) {
                LOG_INFO("RmlUI: loaded font {}", regular_path.string());
            } else {
                LOG_WARN("RmlUI: failed to load Inter-Regular.ttf");
            }
            const auto bold_path = lfs::vis::getAssetPath("fonts/Inter-SemiBold.ttf");
            if (Rml::LoadFontFace(bold_path.string(), false)) {
                LOG_INFO("RmlUI: loaded font {}", bold_path.string());
            } else {
                LOG_WARN("RmlUI: failed to load Inter-SemiBold.ttf");
            }

            const auto jp_path = lfs::vis::getAssetPath("fonts/NotoSansJP-Regular.ttf");
            if (Rml::LoadFontFace(jp_path.string(), true)) {
                LOG_INFO("RmlUI: loaded font {}", jp_path.string());
            } else {
                LOG_WARN("RmlUI: failed to load NotoSansJP-Regular.ttf");
            }

            const auto kr_path = lfs::vis::getAssetPath("fonts/NotoSansKR-Regular.ttf");
            if (Rml::LoadFontFace(kr_path.string(), true)) {
                LOG_INFO("RmlUI: loaded font {}", kr_path.string());
            } else {
                LOG_WARN("RmlUI: failed to load NotoSansKR-Regular.ttf");
            }

            const auto mono_path = lfs::vis::getAssetPath("fonts/JetBrainsMono-Regular.ttf");
            if (Rml::LoadFontFace(mono_path.string(), false)) {
                LOG_INFO("RmlUI: loaded font {}", mono_path.string());
            } else {
                LOG_WARN("RmlUI: failed to load JetBrainsMono-Regular.ttf");
            }
        } catch (const std::exception& e) {
            LOG_WARN("RmlUI: font not found: {}", e.what());
        }

        initialized_ = true;
        LOG_INFO("RmlUI initialized");
        return true;
    }

    void RmlUIManager::shutdown() {
        if (!initialized_)
            return;

        if (debugger_initialized_) {
            Rml::Debugger::Shutdown();
            debugger_initialized_ = false;
        }

        for (auto& [name, ctx] : contexts_) {
            Rml::RemoveContext(name);
        }
        contexts_.clear();

        if (Rml::GetTextInputHandler() == text_input_handler_.get())
            Rml::SetTextInputHandler(nullptr);
        Rml::Shutdown();
        render_interface_.reset();
        text_input_handler_.reset();
        system_interface_.reset();
        resize_deferring_ = false;
        initialized_ = false;

        LOG_INFO("RmlUI shut down");
    }

    void RmlUIManager::setDpRatio(float ratio) {
        assert(ratio >= 1.0f);
        if (!initialized_)
            return;
        dp_ratio_ = ratio;
        for (auto& [name, ctx] : contexts_) {
            ctx->SetDensityIndependentPixelRatio(ratio);
        }
    }

    Rml::Context* RmlUIManager::createContext(const std::string& name, int width, int height) {
        assert(initialized_);

        auto it = contexts_.find(name);
        if (it != contexts_.end()) {
            return it->second;
        }

        Rml::Context* ctx = Rml::CreateContext(name, Rml::Vector2i(width, height));
        if (!ctx) {
            LOG_ERROR("RmlUI: failed to create context '{}'", name);
            return nullptr;
        }

        ctx->SetDensityIndependentPixelRatio(dp_ratio_);
        ctx->SetDefaultScrollBehavior(Rml::ScrollBehavior::Instant, 1.0f);
        if (!active_theme_id_.empty())
            ctx->ActivateTheme(active_theme_id_, true);

        if (debugger_enabled_ && !debugger_initialized_) {
            debugger_initialized_ = Rml::Debugger::Initialise(ctx);
            if (debugger_initialized_) {
                Rml::Debugger::SetVisible(true);
                LOG_INFO("RmlUI debugger enabled on context '{}'", name);
            } else {
                LOG_WARN("RmlUI debugger requested but failed to initialize on context '{}'", name);
            }
        }

        contexts_[name] = ctx;
        return ctx;
    }

    Rml::Context* RmlUIManager::getContext(const std::string& name) {
        auto it = contexts_.find(name);
        return it != contexts_.end() ? it->second : nullptr;
    }

    void RmlUIManager::destroyContext(const std::string& name) {
        auto it = contexts_.find(name);
        if (it != contexts_.end()) {
            if (system_interface_)
                system_interface_->releaseContext(it->second);
            if (auto fn = lfs::python::get_rml_context_destroy_handler())
                fn(it->second);
            Rml::RemoveContext(name);
            contexts_.erase(it);
        }
    }

    void RmlUIManager::activateTheme(const std::string& theme_id) {
        if (theme_id == active_theme_id_)
            return;
        for (auto& [name, ctx] : contexts_) {
            if (!active_theme_id_.empty())
                ctx->ActivateTheme(active_theme_id_, false);
            ctx->ActivateTheme(theme_id, true);
        }
        active_theme_id_ = theme_id;
    }

    void RmlUIManager::beginFrameCursorTracking() {
        if (system_interface_)
            system_interface_->beginFrame();
    }

    void RmlUIManager::trackContextFrame(const Rml::Context* const context,
                                         const int window_x,
                                         const int window_y) {
        if (system_interface_)
            system_interface_->trackContext(context, window_x, window_y);
    }

    RmlCursorRequest RmlUIManager::consumeCursorRequest() {
        return system_interface_ ? system_interface_->consumeCursorRequest()
                                 : RmlCursorRequest::None;
    }

    bool RmlUIManager::shouldDeferFboUpdate(const RmlFBO& fbo) const {
        return resize_deferring_ && fbo.valid();
    }

} // namespace lfs::vis::gui
