/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

struct SDL_Window;

namespace Rml {
    class Context;
}

namespace lfs::vis::gui {

    class RmlFBO;
    class RmlSystemInterface;
    class RmlRenderInterface;

    class RmlUIManager {
    public:
        RmlUIManager();
        ~RmlUIManager();

        bool init(SDL_Window* window, float dp_ratio = 1.0f);
        void shutdown();

        float getDpRatio() const { return dp_ratio_; }
        void setDpRatio(float ratio);

        Rml::Context* createContext(const std::string& name, int width, int height);
        Rml::Context* getContext(const std::string& name);
        void destroyContext(const std::string& name);

        void setResizeDeferring(bool defer) { resize_deferring_ = defer; }
        [[nodiscard]] bool shouldDeferFboUpdate(const RmlFBO& fbo) const;

        RmlRenderInterface* getRenderInterface() const { return render_interface_.get(); }
        SDL_Window* getWindow() const { return window_; }

    private:
        std::unique_ptr<RmlSystemInterface> system_interface_;
        std::unique_ptr<RmlRenderInterface> render_interface_;
        std::unordered_map<std::string, Rml::Context*> contexts_;
        SDL_Window* window_ = nullptr;
        float dp_ratio_ = 1.0f;
        bool resize_deferring_ = false;
        bool initialized_ = false;
    };

} // namespace lfs::vis::gui
