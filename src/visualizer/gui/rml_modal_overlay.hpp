/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/modal_request.hpp"
#include "gui/rmlui/rml_fbo.hpp"

#include <RmlUi/Core/EventListener.h>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    class RmlUIManager;

    class LFS_VIS_API RmlModalOverlay {
    public:
        explicit RmlModalOverlay(RmlUIManager* rml_manager);
        ~RmlModalOverlay();

        RmlModalOverlay(const RmlModalOverlay&) = delete;
        RmlModalOverlay& operator=(const RmlModalOverlay&) = delete;

        void enqueue(lfs::core::ModalRequest request);
        void processInput();
        void render(int screen_w, int screen_h, float vp_x, float vp_y, float vp_w, float vp_h);
        void destroyGLResources();

        [[nodiscard]] bool isOpen() const;

    private:
        void initContext();
        void syncTheme();
        std::string generateThemeRCSS(const lfs::vis::Theme& t) const;
        void cacheElements();

        void showNext();
        void dismiss(const std::string& button_label);
        void cancel();
        lfs::core::ModalResult collectFormValues() const;

        struct OverlayEventListener : Rml::EventListener {
            RmlModalOverlay* overlay = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        RmlUIManager* rml_manager_;
        OverlayEventListener listener_;

        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        RmlFBO fbo_;

        Rml::Element* el_backdrop_ = nullptr;
        Rml::Element* el_dialog_ = nullptr;
        Rml::Element* el_title_ = nullptr;
        Rml::Element* el_content_ = nullptr;
        Rml::Element* el_input_row_ = nullptr;
        Rml::Element* el_input_ = nullptr;
        Rml::Element* el_button_row_ = nullptr;

        bool elements_cached_ = false;

        mutable std::mutex queue_mutex_;
        std::deque<lfs::core::ModalRequest> queue_;
        std::optional<lfs::core::ModalRequest> active_;

        std::string base_rcss_;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        int width_ = 0;
        int height_ = 0;
    };

} // namespace lfs::vis::gui
