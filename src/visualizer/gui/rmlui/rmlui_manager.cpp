/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_manager.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/elements/chromaticity_element.hpp"
#include "gui/rmlui/elements/color_picker_element.hpp"
#include "gui/rmlui/elements/crf_curve_element.hpp"
#include "gui/rmlui/elements/loss_graph_element.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/rmlui/rmlui_system_interface.hpp"
#include "gui/rmlui/stb_font_engine.hpp"
#include "internal/resource_paths.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Factory.h>
#include <cassert>
#include <filesystem>

namespace lfs::vis::gui {

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

        system_interface_ = std::make_unique<RmlSystemInterface>(window);
        render_interface_ = std::make_unique<RmlRenderInterface>();
        font_engine_ = std::make_unique<StbFontEngine>();

        Rml::SetSystemInterface(system_interface_.get());
        Rml::SetRenderInterface(render_interface_.get());
        Rml::SetFontEngineInterface(font_engine_.get());

        if (!Rml::Initialise()) {
            LOG_ERROR("Failed to initialize RmlUI");
            return false;
        }

        static Rml::ElementInstancerGeneric<ChromaticityElement> chromaticity_instancer;
        static Rml::ElementInstancerGeneric<ColorPickerElement> color_picker_instancer;
        static Rml::ElementInstancerGeneric<CRFCurveElement> crf_curve_instancer;
        static Rml::ElementInstancerGeneric<LossGraphElement> loss_graph_instancer;
        Rml::Factory::RegisterElementInstancer("chromaticity-diagram", &chromaticity_instancer);
        Rml::Factory::RegisterElementInstancer("color-picker", &color_picker_instancer);
        Rml::Factory::RegisterElementInstancer("crf-curve", &crf_curve_instancer);
        Rml::Factory::RegisterElementInstancer("loss-graph", &loss_graph_instancer);

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

        for (auto& [name, ctx] : contexts_) {
            Rml::RemoveContext(name);
        }
        contexts_.clear();

        Rml::Shutdown();
        font_engine_.reset();
        render_interface_.reset();
        system_interface_.reset();
        initialized_ = false;

        LOG_INFO("RmlUI shut down");
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
            Rml::RemoveContext(name);
            contexts_.erase(it);
        }
    }

} // namespace lfs::vis::gui
