/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "operator_id.hpp"
#include <array>
#include <cassert>

namespace lfs::vis::op {

    namespace {

        struct OpInfo {
            const char* id_string;
            const char* label;
        };

        constexpr std::array<OpInfo, static_cast<size_t>(BuiltinOp::_Count)> OP_INFO = {{
            {"selection.stroke", "Selection Stroke"},
            {"brush.stroke", "Brush Stroke"},
            {"transform.set", "Set Transform"},
            {"transform.translate", "Translate"},
            {"transform.rotate", "Rotate"},
            {"transform.scale", "Scale"},
            {"transform.apply_batch", "Apply Batch Transform"},
            {"align.pick_point", "Align to Ground"},
            {"ed.undo", "Undo"},
            {"ed.redo", "Redo"},
            {"ed.delete", "Delete"},
            {"selection.clear", "Clear Selection"},
            {"scene.select_node", "Select Node"},
            {"crop_box.add", "Add Crop Box"},
            {"crop_box.set", "Set Crop Box"},
            {"crop_box.fit", "Fit Crop Box"},
            {"crop_box.reset", "Reset Crop Box"},
            {"ellipsoid.add", "Add Ellipsoid"},
            {"ellipsoid.set", "Set Ellipsoid"},
            {"ellipsoid.fit", "Fit Ellipsoid"},
            {"ellipsoid.reset", "Reset Ellipsoid"},
        }};

        struct ToolInfo {
            const char* id_string;
            const char* label;
        };

        constexpr std::array<ToolInfo, static_cast<size_t>(BuiltinTool::_Count)> TOOL_INFO = {{
            {"builtin.select", "Select"},
            {"builtin.translate", "Move"},
            {"builtin.rotate", "Rotate"},
            {"builtin.scale", "Scale"},
            {"builtin.mirror", "Mirror"},
            {"builtin.brush", "Brush"},
            {"builtin.align", "Align"},
        }};

    } // namespace

    const char* to_string(BuiltinOp op) {
        const auto idx = static_cast<size_t>(op);
        assert(idx < OP_INFO.size());
        return OP_INFO[idx].id_string;
    }

    std::optional<BuiltinOp> builtin_op_from_string(std::string_view s) {
        for (size_t i = 0; i < OP_INFO.size(); ++i) {
            if (s == OP_INFO[i].id_string) {
                return static_cast<BuiltinOp>(i);
            }
        }
        return std::nullopt;
    }

    const char* builtin_op_label(BuiltinOp op) {
        const auto idx = static_cast<size_t>(op);
        assert(idx < OP_INFO.size());
        return OP_INFO[idx].label;
    }

    const char* to_string(BuiltinTool tool) {
        const auto idx = static_cast<size_t>(tool);
        assert(idx < TOOL_INFO.size());
        return TOOL_INFO[idx].id_string;
    }

    std::optional<BuiltinTool> builtin_tool_from_string(std::string_view s) {
        for (size_t i = 0; i < TOOL_INFO.size(); ++i) {
            if (s == TOOL_INFO[i].id_string) {
                return static_cast<BuiltinTool>(i);
            }
        }
        return std::nullopt;
    }

    const char* builtin_tool_label(BuiltinTool tool) {
        const auto idx = static_cast<size_t>(tool);
        assert(idx < TOOL_INFO.size());
        return TOOL_INFO[idx].label;
    }

} // namespace lfs::vis::op
