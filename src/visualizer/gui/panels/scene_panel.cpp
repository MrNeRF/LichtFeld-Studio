/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <glad/glad.h>

#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/panels/scene_panel.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/windows_utils.hpp"
#include "gui/windows/image_preview.hpp"
#include "internal/resource_paths.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"

#include <algorithm>
#include <format>
#include <ranges>
#include <imgui.h>

namespace lfs::vis::gui {

    using namespace lichtfeld::Strings;

    using namespace lfs::core::events;
    using lfs::core::ExportFormat;

    namespace {
        unsigned int loadSceneIcon(const std::string& name) {
            try {
                const auto path = lfs::vis::getAssetPath("icon/scene/" + name);
                const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

                unsigned int texture_id;
                glGenTextures(1, &texture_id);
                glBindTexture(GL_TEXTURE_2D, texture_id);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                const GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
                glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

                lfs::core::free_image(data);
                glBindTexture(GL_TEXTURE_2D, 0);
                return texture_id;
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load scene icon {}: {}", name, e.what());
                return 0;
            }
        }

        void deleteTexture(unsigned int& tex) {
            if (tex) {
                glDeleteTextures(1, &tex);
                tex = 0;
            }
        }

        void showDisabledDeleteTooltip(const bool is_protected) {
            if (is_protected && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("%s", LOC(lichtfeld::Strings::Scene::CANNOT_DELETE_TRAINING));
            }
        }
    } // namespace

    ScenePanel::ScenePanel(std::shared_ptr<const TrainerManager> trainer_manager)
        : m_trainerManager(std::move(trainer_manager)) {
        m_imagePreview = std::make_unique<ImagePreview>();
        setupEventHandlers();
    }

    ScenePanel::~ScenePanel() {
        shutdownIcons();
    }

    void ScenePanel::initIcons() {
        if (m_icons.initialized)
            return;

        m_icons.visible = loadSceneIcon("visible.png");
        m_icons.hidden = loadSceneIcon("hidden.png");
        m_icons.group = loadSceneIcon("group.png");
        m_icons.dataset = loadSceneIcon("dataset.png");
        m_icons.camera = loadSceneIcon("camera.png");
        m_icons.splat = loadSceneIcon("splat.png");
        m_icons.cropbox = loadSceneIcon("cropbox.png");
        m_icons.pointcloud = loadSceneIcon("pointcloud.png");
        m_icons.mask = loadSceneIcon("mask.png");
        m_icons.trash = loadSceneIcon("trash.png");
        m_icons.grip = loadSceneIcon("grip.png");
        m_icons.initialized = true;
    }

    void ScenePanel::shutdownIcons() {
        if (!m_icons.initialized)
            return;

        deleteTexture(m_icons.visible);
        deleteTexture(m_icons.hidden);
        deleteTexture(m_icons.group);
        deleteTexture(m_icons.dataset);
        deleteTexture(m_icons.camera);
        deleteTexture(m_icons.splat);
        deleteTexture(m_icons.cropbox);
        deleteTexture(m_icons.pointcloud);
        deleteTexture(m_icons.mask);
        deleteTexture(m_icons.trash);
        deleteTexture(m_icons.grip);
        m_icons.initialized = false;
    }

    void ScenePanel::setupEventHandlers() {
        cmd::GoToCamView::when([this](const auto& e) { handleGoToCamView(e); });

        state::SceneCleared::when([this](const auto&) {
            m_imagePaths.clear();
            m_pathToCamId.clear();
            m_currentDatasetPath.clear();
            m_selectedImageIndex = -1;
            m_highlightedCamUid = -1;
            m_needsScrollToCam = false;
        });

        state::DatasetLoadCompleted::when([this](const auto& e) {
            if (e.success) {
                loadImageCams(e.path);
            }
        });
    }

    void ScenePanel::handleGoToCamView(const cmd::GoToCamView& event) {
        for (const auto& [path, cam_id] : m_pathToCamId) {
            if (cam_id == event.cam_id) {
                if (const auto it = std::find(m_imagePaths.begin(), m_imagePaths.end(), path); it != m_imagePaths.end()) {
                    m_selectedImageIndex = static_cast<int>(std::distance(m_imagePaths.begin(), it));
                    m_needsScrollToSelection = true;
                }
                break;
            }
        }
        m_highlightedCamUid = event.cam_id;
        m_needsScrollToCam = true;
    }

    bool ScenePanel::hasImages() const {
        return !m_imagePaths.empty();
    }

    bool ScenePanel::hasPLYs(const UIContext* ctx) const {
        if (!ctx || !ctx->viewer)
            return false;
        const auto* sm = ctx->viewer->getSceneManager();
        if (!sm)
            return false;
        return sm->getScene().hasNodes();
    }

    void ScenePanel::render(bool* p_open, const UIContext* ctx) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(theme().palette.surface_bright, 0.8f));

        if (!ImGui::Begin(LOC(Window::SCENE), p_open)) {
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        renderContent(ctx);

        ImGui::End();
        ImGui::PopStyleColor();
    }

    void ScenePanel::renderContent(const UIContext* ctx) {
        if (m_showImagePreview && m_imagePreview) {
            m_imagePreview->render(&m_showImagePreview);
        }
        if (hasPLYs(ctx)) {
            renderPLYSceneGraph(ctx);
        } else {
            ImGui::TextDisabled("%s", LOC(lichtfeld::Strings::Scene::NO_DATA_LOADED));
            ImGui::TextDisabled("%s", LOC(lichtfeld::Strings::Scene::USE_FILE_MENU));
        }
    }

    void ScenePanel::renderPLYSceneGraph(const UIContext* ctx) {
        if (!ctx || !ctx->viewer)
            return;

        // Lazy-load icons
        if (!m_icons.initialized)
            initIcons();

        auto* scene_manager = ctx->viewer->getSceneManager();
        if (!scene_manager)
            return;

        // Update flash intensity
        const auto* rm = ctx->viewer->getRenderingManager();
        m_selectionFlashIntensity = rm ? rm->getSelectionFlashIntensity() : 0.0f;

        const auto& scene = scene_manager->getScene();
        const auto selected_names_vec = scene_manager->getSelectedNodeNames();
        std::unordered_set<std::string> selected_names(selected_names_vec.begin(), selected_names_vec.end());
        const auto& t = theme();
        const float scale = getDpiScale();

        // Search filter
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f * scale, 2.0f * scale));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, withAlpha(t.palette.surface, 0.5f));
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##filter", LOC(lichtfeld::Strings::Scene::FILTER), m_filterText, sizeof(m_filterText));
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();

        ImGui::Spacing();

        // Compact outliner style
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f * scale, 1.0f * scale));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f * scale, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 14.0f * scale);
        ImGui::PushStyleColor(ImGuiCol_Header, withAlpha(t.palette.primary, 0.3f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, withAlpha(t.palette.primary, 0.4f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, withAlpha(t.palette.primary, 0.5f));

        ImGui::BeginChild("SceneGraph", {0, 0}, ImGuiChildFlags_None);

        m_rowIndex = 0;

        // Keyboard shortcuts
        if (ImGui::IsWindowFocused() && !m_renameState.is_renaming) {
            if (ImGui::IsKeyPressed(ImGuiKey_F2) && !selected_names.empty()) {
                startRenaming(*selected_names.begin());
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Escape) && !selected_names.empty()) {
                scene_manager->clearSelection();
                ui::NodeDeselected{}.emit();
            }
        }

        renderModelsFolder(scene, selected_names);

        ImGui::EndChild();

        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(3);
    }

    void ScenePanel::renderModelsFolder(const Scene& scene, const std::unordered_set<std::string>& selected_names) {
        static constexpr ImGuiTreeNodeFlags FOLDER_FLAGS =
            ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow;

        // Count only splat nodes
        const auto nodes = scene.getNodes();
        const size_t splat_count = std::ranges::count_if(nodes,
                                                         [](const SceneNode* n) { return n->type == NodeType::SPLAT; });

        const std::string label = std::vformat(LOC(lichtfeld::Strings::Scene::MODELS), std::make_format_args(splat_count));
        if (!ImGui::TreeNodeEx(label.c_str(), FOLDER_FLAGS))
            return;

        // Drop target for moving nodes to root
        handleDragDrop("", true);

        // Context menu for folder
        theme().pushContextMenuStyle();
        if (ImGui::BeginPopupContextItem("##ModelsMenu")) {
            if (!m_trainerManager->hasTrainer()) {
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::ADD_PLY))) {
                    cmd::ShowWindow{.window_name = "file_browser", .show = true}.emit();
#ifdef _WIN32
                    OpenPlyFileDialogNative();
                    cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif
                }
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::ADD_GROUP))) {
                    cmd::AddGroup{.name = "New Group", .parent_name = ""}.emit();
                }
                ImGui::Separator();
            }
            if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::EXPORT), nullptr, false, splat_count > 0)) {
                cmd::ShowWindow{.window_name = "export_dialog", .show = true}.emit();
            }
            ImGui::EndPopup();
        }
        Theme::popContextMenuStyle();

        // Render root-level nodes (parent_id == NULL_NODE)
        for (const auto* node : nodes) {
            if (node->parent_id == NULL_NODE) {
                renderModelNode(*node, scene, selected_names);
            }
        }

        if (!scene.hasNodes()) {
            ImGui::TextDisabled("%s", LOC(lichtfeld::Strings::Scene::NO_MODELS_LOADED));
            ImGui::TextDisabled("%s", LOC(lichtfeld::Strings::Scene::RIGHT_CLICK_TO_ADD));
        }

        ImGui::TreePop();
    }

    void ScenePanel::renderModelNode(const SceneNode& node, const Scene& scene,
                                     const std::unordered_set<std::string>& selected_names,
                                     const int depth) {
        // Filter check
        if (m_filterText[0] != '\0') {
            std::string lower_name = node.name;
            std::string lower_filter = m_filterText;
            std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
            std::transform(lower_filter.begin(), lower_filter.end(), lower_filter.begin(), ::tolower);
            if (lower_name.find(lower_filter) == std::string::npos) {
                for (const auto child_id : node.children) {
                    if (const auto* child = scene.getNodeById(child_id))
                        renderModelNode(*child, scene, selected_names, depth + 1);
                }
                return;
            }
        }

        // Draw indentation guides
        renderIndentGuides(depth);

        ImGui::PushID(node.id);

        const bool is_visible = node.visible.get();
        const bool is_selected = selected_names.contains(node.name);
        const bool is_group = (node.type == NodeType::GROUP);
        const bool is_cropbox = (node.type == NodeType::CROPBOX);
        const bool is_dataset = (node.type == NodeType::DATASET);
        const bool is_camera_group = (node.type == NodeType::CAMERA_GROUP);
        const bool is_camera = (node.type == NodeType::CAMERA);
        const bool is_pointcloud = (node.type == NodeType::POINTCLOUD);
        const bool has_children = !node.children.empty();
        const bool has_mask = is_camera && !node.mask_path.empty();
        const bool is_highlighted_cam = is_camera && node.camera_uid == m_highlightedCamUid;

        const auto* parent_node = scene.getNodeById(node.parent_id);
        const bool parent_is_dataset = parent_node && parent_node->type == NodeType::DATASET;

        const auto& t = theme();
        const float scale = getDpiScale();
        ImDrawList* const draw_list = ImGui::GetWindowDrawList();

        const float ROW_PADDING = 2.0f * scale;
        constexpr ImU32 HIGHLIGHT_COLOR = IM_COL32(80, 120, 180, 180);
        constexpr ImU32 SELECTION_COLOR_BASE = IM_COL32(60, 100, 160, 200);
        constexpr ImU32 SELECTION_COLOR_FLASH = IM_COL32(140, 180, 240, 230);

        const ImVec2 row_min = ImGui::GetCursorScreenPos();
        const float window_left = ImGui::GetWindowPos().x;
        const float window_right = window_left + ImGui::GetWindowWidth();
        const float row_height = ImGui::GetTextLineHeight() + ROW_PADDING;

        // Compute row color
        ImU32 row_color;
        if (is_selected) {
            if (m_selectionFlashIntensity > 0.0f) {
                const ImVec4 base = ImGui::ColorConvertU32ToFloat4(SELECTION_COLOR_BASE);
                const ImVec4 flash = ImGui::ColorConvertU32ToFloat4(SELECTION_COLOR_FLASH);
                const float t = 1.0f - m_selectionFlashIntensity;
                row_color = ImGui::ColorConvertFloat4ToU32(ImVec4(
                    flash.x + (base.x - flash.x) * t,
                    flash.y + (base.y - flash.y) * t,
                    flash.z + (base.z - flash.z) * t,
                    flash.w + (base.w - flash.w) * t));
            } else {
                row_color = SELECTION_COLOR_BASE;
            }
        } else if (is_highlighted_cam) {
            row_color = HIGHLIGHT_COLOR;
        } else {
            row_color = (m_rowIndex++ % 2 == 0) ? t.row_even_u32() : t.row_odd_u32();
        }

        draw_list->AddRectFilled(
            ImVec2(window_left, row_min.y),
            ImVec2(window_right, row_min.y + row_height),
            row_color);

        if (is_highlighted_cam && m_needsScrollToCam) {
            ImGui::SetScrollHereY(0.5f);
            m_needsScrollToCam = false;
        }

        // Scene graph icon layout constants
        const float ICON_SIZE = 16.0f * scale;
        const float ICON_SPACING = 2.0f * scale;
        const ImVec2 icon_sz{ICON_SIZE, ICON_SIZE};

        const bool can_drag = canReparent(node, nullptr, scene);
        const bool is_training_protected = isNodeProtectedDuringTraining(node, scene);
        const bool is_deletable = !is_camera && !is_camera_group && !is_cropbox && !parent_is_dataset && !is_training_protected;

        // Button style for all icon buttons
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, withAlpha(t.palette.surface_bright, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, withAlpha(t.palette.surface_bright, 0.7f));

        // [Grip] - drag indicator
        if (m_icons.grip) {
            const ImVec4 grip_tint = can_drag
                                         ? withAlpha(t.palette.text_dim, 0.5f)
                                         : ImVec4(0, 0, 0, 0);
            ImGui::Image(static_cast<ImTextureID>(m_icons.grip), icon_sz, {0, 0}, {1, 1}, grip_tint, {0, 0, 0, 0});
            ImGui::SameLine(0.0f, ICON_SPACING);
        }

        // [Visibility] - toggle visible/hidden
        if (const unsigned int vis_tex = is_visible ? m_icons.visible : m_icons.hidden) {
            const ImVec4 vis_tint = is_visible
                                        ? ImVec4(0.4f, 0.9f, 0.4f, 1.0f)
                                        : ImVec4(0.6f, 0.4f, 0.4f, 0.7f);
            if (ImGui::ImageButton("##vis", static_cast<ImTextureID>(vis_tex), icon_sz, {0, 0}, {1, 1}, {0, 0, 0, 0}, vis_tint))
                cmd::SetPLYVisibility{.name = node.name, .visible = !is_visible}.emit();
            ImGui::SameLine(0.0f, ICON_SPACING);
        }

        // [Trash] - delete node
        if (is_deletable && m_icons.trash) {
            const ImVec4 trash_tint = is_selected
                                          ? ImVec4(1.0f, 0.6f, 0.6f, 0.9f)
                                          : withAlpha(t.palette.text_dim, 0.5f);
            if (ImGui::ImageButton("##del", static_cast<ImTextureID>(m_icons.trash), icon_sz, {0, 0}, {1, 1}, {0, 0, 0, 0}, trash_tint))
                cmd::RemovePLY{.name = node.name, .keep_children = false}.emit();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", LOC(lichtfeld::Strings::Scene::DELETE_NODE));
            ImGui::SameLine(0.0f, ICON_SPACING);
        }

        ImGui::PopStyleColor(3);

        const bool is_renaming = m_renameState.is_renaming && m_renameState.renaming_node_name == node.name;

        if (is_renaming) {
            if (m_renameState.focus_input) {
                ImGui::SetKeyboardFocusHere();
                m_renameState.focus_input = false;
            }

            static constexpr ImGuiInputTextFlags RENAME_INPUT_FLAGS =
                ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue;
            const bool entered = ImGui::InputText("##rename", m_renameState.buffer,
                                                  sizeof(m_renameState.buffer), RENAME_INPUT_FLAGS);
            const bool is_focused = ImGui::IsItemFocused();
            if (ImGui::IsItemActive())
                m_renameState.input_was_active = true;

            if (entered) {
                // Need non-const scene_manager for rename - get from context indirectly via event
                finishRenaming(nullptr);
            } else if (ImGui::IsKeyPressed(ImGuiKey_Escape) ||
                       (m_renameState.input_was_active && !is_focused)) {
                cancelRenaming();
            }
        } else {
            // Type indicator icon
            unsigned int type_tex = m_icons.splat;
            ImVec4 type_tint(0.6f, 0.8f, 1.0f, 0.9f); // Blue for splats

            if (is_group) {
                type_tex = m_icons.group;
                type_tint = ImVec4(0.7f, 0.7f, 0.7f, 0.8f);
            } else if (is_dataset) {
                type_tex = m_icons.dataset;
                type_tint = ImVec4(0.5f, 0.7f, 1.0f, 0.9f);
            } else if (is_camera_group || is_camera) {
                type_tex = m_icons.camera;
                type_tint = is_camera_group
                                ? ImVec4(0.6f, 0.7f, 0.9f, 0.8f)
                                : ImVec4(0.5f, 0.6f, 0.8f, 0.6f);
            } else if (is_cropbox) {
                type_tex = m_icons.cropbox;
                type_tint = ImVec4(1.0f, 0.7f, 0.3f, 0.9f);
            } else if (is_pointcloud) {
                type_tex = m_icons.pointcloud;
                type_tint = ImVec4(0.8f, 0.5f, 1.0f, 0.8f);
            }

            // [Type] - node type indicator
            if (type_tex) {
                ImGui::Image(static_cast<ImTextureID>(type_tex), icon_sz, {0, 0}, {1, 1}, type_tint, {0, 0, 0, 0});
            } else {
                ImGui::Dummy(icon_sz);
            }

            // [Mask] - indicator for cameras with masks
            if (has_mask && m_icons.mask) {
                ImGui::SameLine(0.0f, ICON_SPACING);
                ImGui::Image(static_cast<ImTextureID>(m_icons.mask), icon_sz, {0, 0}, {1, 1},
                             ImVec4(0.9f, 0.5f, 0.6f, 0.8f), {0, 0, 0, 0});
            }

            ImGui::SameLine(0.0f, ICON_SPACING + 2.0f);

            static constexpr ImGuiTreeNodeFlags BASE_FLAGS = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
            ImGuiTreeNodeFlags flags = BASE_FLAGS;
            if (!has_children)
                flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            if (is_group || is_dataset)
                flags |= ImGuiTreeNodeFlags_DefaultOpen;

            // Build label with count suffix
            std::string label = node.name;
            if (is_pointcloud) {
                const size_t count = node.point_cloud ? node.point_cloud->size() : 0;
                label += std::format("  ({:L})", count);
            } else if (!is_group && !is_dataset && !is_camera_group && !is_camera && !is_cropbox) {
                label += std::format("  ({:L})", node.gaussian_count);
            }

            const bool is_open = ImGui::TreeNodeEx(label.c_str(), flags);

            const bool hovered = ImGui::IsItemHovered();
            const bool clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
            const bool toggled = ImGui::IsItemToggledOpen();

            if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                ImGui::OpenPopup(("##ctx_" + node.name).c_str());
            }

            // Drag source (only for reparentable nodes)
            if (can_drag && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                ImGui::SetDragDropPayload("SCENE_NODE", node.name.c_str(), node.name.size() + 1);
                ImGui::Text(LOC(lichtfeld::Strings::Scene::MOVE_NODE), node.name.c_str());
                ImGui::EndDragDropSource();
            }

            // Drop target (only groups accept children)
            if (is_group)
                handleDragDrop(node.name, true);

            // Selection - emit event, let SceneManager handle state
            // Camera nodes don't participate in selection - they have their own interactions
            if (clicked && !toggled && !is_camera) {
                if (is_selected) {
                    ui::NodeDeselected{}.emit();
                } else {
                    // Determine node type string for event
                    std::string type_str = "PLY";
                    if (is_group)
                        type_str = "Group";
                    else if (is_dataset)
                        type_str = "Dataset";
                    else if (is_camera_group)
                        type_str = "CameraGroup";
                    else if (is_pointcloud)
                        type_str = "PointCloud";

                    ui::NodeSelected{
                        .path = node.name,
                        .type = type_str,
                        .metadata = {{"name", node.name},
                                     {"gaussians", std::to_string(node.gaussian_count)},
                                     {"visible", is_visible ? "true" : "false"}}}
                        .emit();
                }
            }

            // Double-click opens image preview for cameras
            if (is_camera && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                const ImVec2 item_min = ImGui::GetItemRectMin();
                const ImVec2 item_max = ImGui::GetItemRectMax();
                const ImVec2 mouse = ImGui::GetMousePos();
                const bool in_item = mouse.x >= item_min.x && mouse.x <= item_max.x &&
                                     mouse.y >= item_min.y && mouse.y <= item_max.y;
                if (in_item && !node.image_path.empty() && m_imagePreview) {
                    // Collect camera paths with names for sorting
                    struct CameraEntry {
                        std::string name;
                        std::filesystem::path image_path;
                        std::filesystem::path mask_path;
                    };
                    std::vector<CameraEntry> entries;

                    for (const auto* n : scene.getNodes()) {
                        if (n->type == NodeType::CAMERA && !n->image_path.empty()) {
                            entries.push_back({n->name, n->image_path, n->mask_path});
                        }
                    }

                    // Sort by name (which is typically the image filename)
                    std::ranges::sort(entries, {}, &CameraEntry::name);

                    // Build sorted path vectors and find current index
                    std::vector<std::filesystem::path> camera_paths;
                    std::vector<std::filesystem::path> mask_paths;
                    size_t current_idx = 0;

                    for (size_t i = 0; i < entries.size(); ++i) {
                        if (entries[i].name == node.name)
                            current_idx = i;
                        camera_paths.push_back(entries[i].image_path);
                        mask_paths.push_back(entries[i].mask_path);
                    }

                    if (!camera_paths.empty()) {
                        m_imagePreview->openWithOverlay(camera_paths, mask_paths, current_idx);
                        m_showImagePreview = true;
                    }
                }
            }

            // Context menu
            theme().pushContextMenuStyle();
            if (ImGui::BeginPopup(("##ctx_" + node.name).c_str())) {
                // Helper to finish node after early menu exit
                const auto finishNode = [&]() {
                    ImGui::EndPopup();
                    Theme::popContextMenuStyle();
                    if (is_open && has_children) {
                        renderNodeChildren(node.id, scene, selected_names, depth + 1);
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                };

                if (is_camera) {
                    if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::GO_TO_CAMERA_VIEW))) {
                        cmd::GoToCamView{.cam_id = node.camera_uid}.emit();
                    }
                    finishNode();
                    return;
                }

                if (is_camera_group || parent_is_dataset) {
                    ImGui::TextDisabled("%s", LOC(lichtfeld::Strings::Scene::NO_ACTIONS));
                    finishNode();
                    return;
                }

                if (is_dataset) {
                    if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::DELETE_ITEM), nullptr, false, !is_training_protected)) {
                        cmd::RemovePLY{.name = node.name, .keep_children = false}.emit();
                    }
                    showDisabledDeleteTooltip(is_training_protected);
                    finishNode();
                    return;
                }

                if (is_cropbox) {
                    if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::FIT_TO_SCENE))) {
                        cmd::FitCropBoxToScene{.use_percentile = false}.emit();
                    }
                    if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::FIT_TO_SCENE_TRIMMED))) {
                        cmd::FitCropBoxToScene{.use_percentile = true}.emit();
                    }
                    finishNode();
                    return;
                }

                if (is_group) {
                    if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::ADD_GROUP_ELLIPSIS))) {
                        cmd::AddGroup{.name = "New Group", .parent_name = node.name}.emit();
                    }
                    if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::MERGE_TO_SINGLE_PLY))) {
                        cmd::MergeGroup{.name = node.name}.emit();
                    }
                    ImGui::Separator();
                }
                if (!is_group && ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::EXPORT))) {
                    cmd::ShowWindow{.window_name = "export_dialog", .show = true}.emit();
                }
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::RENAME)))
                    startRenaming(node.name);
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::DUPLICATE_ITEM)))
                    cmd::DuplicateNode{.name = node.name}.emit();

                if (ImGui::BeginMenu(LOC(lichtfeld::Strings::Scene::MOVE_TO))) {
                    const auto* parent_node = scene.getNodeById(node.parent_id);
                    if (parent_node) {
                        if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::MOVE_TO_ROOT))) {
                            cmd::ReparentNode{.node_name = node.name, .new_parent_name = ""}.emit();
                        }
                        ImGui::Separator();
                    }
                    bool found_group = false;
                    for (const auto* other : scene.getNodes()) {
                        if (other->type == NodeType::GROUP && other->name != node.name &&
                            (parent_node == nullptr || other->name != parent_node->name)) {
                            found_group = true;
                            if (ImGui::MenuItem(other->name.c_str())) {
                                cmd::ReparentNode{.node_name = node.name, .new_parent_name = other->name}.emit();
                            }
                        }
                    }
                    if (!found_group && !parent_node) {
                        ImGui::TextDisabled("%s", LOC(lichtfeld::Strings::Scene::NO_GROUPS_AVAILABLE));
                    }
                    ImGui::EndMenu();
                }

                ImGui::Separator();
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::DELETE_ITEM), nullptr, false, !is_training_protected)) {
                    cmd::RemovePLY{.name = node.name, .keep_children = false}.emit();
                }
                showDisabledDeleteTooltip(is_training_protected);
                ImGui::EndPopup();
            }
            Theme::popContextMenuStyle();

            if (is_open && has_children) {
                renderNodeChildren(node.id, scene, selected_names, depth + 1);
                ImGui::TreePop();
            }
        }

        ImGui::PopID();
    }

    void ScenePanel::renderNodeChildren(NodeId parent_id, const Scene& scene,
                                        const std::unordered_set<std::string>& selected_names,
                                        const int depth) {
        const auto* parent = scene.getNodeById(parent_id);
        if (!parent)
            return;

        for (const NodeId child_id : parent->children) {
            if (const auto* child = scene.getNodeById(child_id))
                renderModelNode(*child, scene, selected_names, depth);
        }
    }

    void ScenePanel::renderIndentGuides(const int depth) const {
        if (depth <= 0)
            return;

        const float scale = getDpiScale();
        const float INDENT_WIDTH = 14.0f * scale;
        const float LINE_OFFSET_X = 7.0f * scale;

        const auto& t = theme();
        const ImU32 guide_color = toU32(withAlpha(t.palette.text_dim, 0.25f));

        ImDrawList* const dl = ImGui::GetWindowDrawList();
        const ImVec2 cursor = ImGui::GetCursorScreenPos();
        const float row_height = ImGui::GetTextLineHeight() + 2.0f * scale;

        for (int i = 0; i < depth; ++i) {
            const float x = cursor.x + i * INDENT_WIDTH + LINE_OFFSET_X;
            dl->AddLine({x, cursor.y}, {x, cursor.y + row_height}, guide_color, 1.0f);
        }
    }

    bool ScenePanel::canReparent(const SceneNode& node, const SceneNode* target, const Scene& scene) {
        // Only SPLAT, GROUP, and POINTCLOUD nodes at root level can be reparented
        if (node.type != NodeType::SPLAT && node.type != NodeType::GROUP && node.type != NodeType::POINTCLOUD)
            return false;

        // Check if node is inside a DATASET - these cannot be moved out
        const auto* parent = scene.getNodeById(node.parent_id);
        while (parent) {
            if (parent->type == NodeType::DATASET)
                return false;
            parent = scene.getNodeById(parent->parent_id);
        }

        // Target must be GROUP or root (nullptr)
        if (target && target->type != NodeType::GROUP)
            return false;

        // Cannot reparent to self or descendant
        if (target) {
            const auto* check = target;
            while (check) {
                if (check->id == node.id)
                    return false;
                check = scene.getNodeById(check->parent_id);
            }
        }

        return true;
    }

    bool ScenePanel::handleDragDrop(const std::string& target_name, const bool is_group_target) {
        if (!ImGui::BeginDragDropTarget())
            return false;

        bool handled = false;
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_NODE")) {
            const char* dragged_name = static_cast<const char*>(payload->Data);
            if (dragged_name != target_name) {
                cmd::ReparentNode{
                    .node_name = std::string(dragged_name),
                    .new_parent_name = is_group_target ? target_name : ""}
                    .emit();
                handled = true;
            }
        }

        ImGui::EndDragDropTarget();
        return handled;
    }

    void ScenePanel::startRenaming(const std::string& node_name) {
        m_renameState.is_renaming = true;
        m_renameState.renaming_node_name = node_name;
        m_renameState.focus_input = true;
        strncpy(m_renameState.buffer, node_name.c_str(), sizeof(m_renameState.buffer) - 1);
        m_renameState.buffer[sizeof(m_renameState.buffer) - 1] = '\0';
    }

    void ScenePanel::finishRenaming(SceneManager* /*scene_manager*/) {
        if (!m_renameState.is_renaming)
            return;

        std::string new_name(m_renameState.buffer);
        // Trim whitespace
        if (const auto pos = new_name.find_last_not_of(" \t\n\r"); pos != std::string::npos) {
            new_name = new_name.substr(0, pos + 1);
        }

        if (!new_name.empty() && new_name != m_renameState.renaming_node_name) {
            cmd::RenamePLY{
                .old_name = m_renameState.renaming_node_name,
                .new_name = new_name}
                .emit();
        }

        cancelRenaming();
    }

    void ScenePanel::cancelRenaming() {
        m_renameState.is_renaming = false;
        m_renameState.renaming_node_name.clear();
        m_renameState.focus_input = false;
        m_renameState.input_was_active = false;
        m_renameState.escape_pressed = false;
        memset(m_renameState.buffer, 0, sizeof(m_renameState.buffer));
    }

    bool ScenePanel::isNodeProtectedDuringTraining(const SceneNode& /*node*/, const Scene& /*scene*/) const {
        return m_trainerManager && !m_trainerManager->canPerform(TrainingAction::DeleteTrainingNode);
    }

    void ScenePanel::renderImageList() {
        ImGui::BeginChild("ImageList", ImVec2(0, 0), true);

        if (!m_imagePaths.empty()) {
            ImGui::Text(LOC(lichtfeld::Strings::Scene::IMAGES), m_imagePaths.size());
            ImGui::Separator();

            for (size_t i = 0; i < m_imagePaths.size(); ++i) {
                const auto& imagePath = m_imagePaths[i];
                const std::string filename = imagePath.filename().string();
                const std::string unique_id = std::format("{}##{}", filename, i);
                const bool is_selected = (m_selectedImageIndex == static_cast<int>(i));

                if (is_selected) {
                    const auto& t = theme();
                    ImGui::PushStyleColor(ImGuiCol_Header, withAlpha(t.palette.info, 0.8f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, withAlpha(t.palette.info, 0.9f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, withAlpha(t.palette.info, 0.8f));
                }

                if (ImGui::Selectable(unique_id.c_str(), is_selected)) {
                    m_selectedImageIndex = static_cast<int>(i);
                    onImageSelected(imagePath);
                }

                if (is_selected && m_needsScrollToSelection) {
                    ImGui::SetScrollHereY(0.5f);
                    m_needsScrollToSelection = false;
                }

                if (is_selected) {
                    ImGui::PopStyleColor(3);
                }

                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    onImageDoubleClicked(i);
                }

                const std::string context_menu_id = std::format("context_menu_{}", i);
                theme().pushContextMenuStyle();
                if (ImGui::BeginPopupContextItem(context_menu_id.c_str())) {
                    if (ImGui::MenuItem(LOC(lichtfeld::Strings::Scene::GO_TO_CAM_VIEW))) {
                        if (const auto cam_it = m_pathToCamId.find(imagePath); cam_it != m_pathToCamId.end()) {
                            cmd::GoToCamView{.cam_id = cam_it->second}.emit();
                        }
                    }
                    ImGui::EndPopup();
                }
                Theme::popContextMenuStyle();
            }
        } else {
            ImGui::TextUnformatted(LOC(lichtfeld::Strings::Scene::NO_IMAGES));
            ImGui::TextUnformatted(LOC(lichtfeld::Strings::Scene::USE_FILE_BROWSER));
        }

        ImGui::EndChild();
    }

    void ScenePanel::loadImageCams(const std::filesystem::path& path) {
        m_currentDatasetPath = path;
        m_imagePaths.clear();
        m_pathToCamId.clear();
        m_selectedImageIndex = -1;

        if (!m_trainerManager) {
            LOG_ERROR("TrainerManager not set");
            return;
        }

        const auto cams = m_trainerManager->getCamList();
        for (const auto& cam : cams) {
            m_imagePaths.emplace_back(cam->image_path());
            m_pathToCamId[cam->image_path()] = cam->uid();
        }

        std::ranges::sort(m_imagePaths, [](const auto& a, const auto& b) {
            return a.filename() < b.filename();
        });

        LOG_INFO("Loaded {} cameras from dataset", m_imagePaths.size());
    }

    void ScenePanel::setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback) {
        m_onDatasetLoad = std::move(callback);
    }

    void ScenePanel::onImageSelected(const std::filesystem::path& imagePath) {
        ui::NodeSelected{
            .path = imagePath.string(),
            .type = "Images",
            .metadata = {{"filename", imagePath.filename().string()}, {"path", imagePath.string()}}}
            .emit();
    }

    void ScenePanel::onImageDoubleClicked(const size_t imageIndex) {
        if (imageIndex >= m_imagePaths.size())
            return;

        if (m_imagePreview) {
            m_imagePreview->open(m_imagePaths, imageIndex);
            m_showImagePreview = true;
        }
    }

} // namespace lfs::vis::gui
