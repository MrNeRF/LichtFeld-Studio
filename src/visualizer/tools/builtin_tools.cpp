/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "builtin_tools.hpp"
#include "core/editor_context.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "unified_tool_registry.hpp"

#include <initializer_list>

namespace lfs::vis {

    namespace {

        constexpr int ORDER_SELECT = 10;
        constexpr int ORDER_TRANSLATE = 20;
        constexpr int ORDER_ROTATE = 30;
        constexpr int ORDER_SCALE = 40;
        constexpr int ORDER_MIRROR = 50;
        constexpr int ORDER_BRUSH = 60;
        constexpr int ORDER_ALIGN = 70;

        bool pollTool(const ToolType tool) {
            const auto* editor = services().editorOrNull();
            return editor && editor->isToolAvailable(tool);
        }

        void invokeTool(const ToolType tool) {
            lfs::core::events::tools::SetToolbarTool{.tool_mode = static_cast<int>(tool)}.emit();
        }

        void addTool(const char* id, const char* label, const char* icon, const char* shortcut, const char* group,
                     const int order, const ToolType tool_type,
                     std::initializer_list<SubModeDescriptor> submodes = {}) {
            ToolDescriptor desc;
            desc.id = id;
            desc.label = label;
            desc.icon = icon;
            desc.shortcut = shortcut;
            desc.group = group;
            desc.order = order;
            desc.source = ToolSource::CPP;
            desc.submodes.assign(submodes.begin(), submodes.end());
            desc.poll_fn = [tool_type] { return pollTool(tool_type); };
            desc.invoke_fn = [tool_type] { invokeTool(tool_type); };
            UnifiedToolRegistry::instance().registerTool(std::move(desc));
        }

    } // namespace

    void registerBuiltinTools() {
        addTool("builtin.select", "Select", "selection", "1", "select", ORDER_SELECT, ToolType::Selection,
                {
                    {"centers", "Centers", "circle-dot"},
                    {"rectangle", "Rectangle", "rectangle"},
                    {"polygon", "Polygon", "polygon"},
                    {"lasso", "Lasso", "lasso"},
                    {"rings", "Rings", "ring"},
                });
        addTool("builtin.translate", "Translate", "translation", "2", "transform", ORDER_TRANSLATE, ToolType::Translate);
        addTool("builtin.rotate", "Rotate", "rotation", "3", "transform", ORDER_ROTATE, ToolType::Rotate);
        addTool("builtin.scale", "Scale", "scaling", "4", "transform", ORDER_SCALE, ToolType::Scale);
        addTool("builtin.mirror", "Mirror", "mirror", "5", "transform", ORDER_MIRROR, ToolType::Mirror);
        addTool("builtin.brush", "Paint", "painting", "6", "paint", ORDER_BRUSH, ToolType::Brush);
        addTool("builtin.align", "Align", "align", "7", "align", ORDER_ALIGN, ToolType::Align);
    }

} // namespace lfs::vis
