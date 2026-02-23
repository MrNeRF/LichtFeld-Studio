/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

struct GLFWwindow;

namespace Rml {
    class Context;
}

namespace lfs::vis::gui {

    class RmlSystemInterface;
    class RmlRenderInterface;
    class StbFontEngine;

    class RmlUIManager {
    public:
        RmlUIManager();
        ~RmlUIManager();

        bool init(GLFWwindow* window, float dp_ratio = 1.0f);
        void shutdown();

        float getDpRatio() const { return dp_ratio_; }

        Rml::Context* createContext(const std::string& name, int width, int height);
        Rml::Context* getContext(const std::string& name);
        void destroyContext(const std::string& name);

        RmlRenderInterface* getRenderInterface() const { return render_interface_.get(); }

    private:
        std::unique_ptr<RmlSystemInterface> system_interface_;
        std::unique_ptr<RmlRenderInterface> render_interface_;
        std::unique_ptr<StbFontEngine> font_engine_;
        std::unordered_map<std::string, Rml::Context*> contexts_;
        float dp_ratio_ = 1.0f;
        bool initialized_ = false;
    };

} // namespace lfs::vis::gui
