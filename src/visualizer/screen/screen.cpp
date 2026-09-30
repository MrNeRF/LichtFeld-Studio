/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen/screen.hpp"
#include "screen/view3d_space.hpp"

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>

namespace lfs::vis::screen {

    namespace {
        constexpr int kFormatVersion = 1;
    } // namespace

    SpaceData* Area::space(const std::string_view editor_id) const {
        const auto it = spaces.find(editor_id);
        return it != spaces.end() ? it->second.get() : nullptr;
    }

    Screen::Screen(const EditorTypeRegistry& registry)
        : registry_(&registry) {}

    Screen::Screen(const Screen& other)
        : registry_(other.registry_),
          layout_(other.layout_),
          active_view_(other.active_view_),
          maximized_(other.maximized_),
          next_area_(other.next_area_),
          generation_(other.generation_) {
        for (const auto& [id, area] : other.areas_) {
            Area copy;
            copy.id = area.id;
            copy.editor = area.editor;
            for (const auto& [editor, space] : area.spaces) {
                if (space)
                    copy.spaces.emplace(editor, space->clone());
            }
            areas_.emplace(id, std::move(copy));
        }
    }

    Screen& Screen::operator=(const Screen& other) {
        if (this != &other) {
            Screen copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    Screen Screen::makeDefault(const EditorTypeRegistry& registry) {
        Screen screen(registry);
        const AreaId view = screen.addArea(std::string(editors::kView3D));
        screen.layout_ = ScreenLayout(view);
        screen.active_view_ = view;
        const AreaId scene = screen.addArea(std::string(editors::kScene));
        screen.layout_.split(view, scene, SplitAxis::Columns, 0.22f);
        const AreaId properties = screen.addArea(std::string(editors::kProperties));
        screen.layout_.split(scene, properties, SplitAxis::Rows, 0.6f);
        return screen;
    }

    const Area* Screen::area(const AreaId id) const {
        const auto it = areas_.find(id);
        return it != areas_.end() ? &it->second : nullptr;
    }

    Area* Screen::area(const AreaId id) {
        const auto it = areas_.find(id);
        return it != areas_.end() ? &it->second : nullptr;
    }

    AreaId Screen::findEditor(const std::string_view editor) const {
        for (const AreaId id : layout_.areas()) {
            if (const auto* a = area(id); a && a->editor == editor)
                return id;
        }
        return {};
    }

    std::vector<AreaId> Screen::views() const {
        std::vector<AreaId> out;
        for (const AreaId id : layout_.areas()) {
            if (isView(id))
                out.push_back(id);
        }
        return out;
    }

    View3DSpace* Screen::view(const AreaId id) const {
        const auto* a = area(id);
        if (!a || a->editor != editors::kView3D)
            return nullptr;
        return dynamic_cast<View3DSpace*>(a->activeSpace());
    }

    bool Screen::setActiveView(const AreaId id) {
        if (!isView(id))
            return false;
        if (active_view_ != id) {
            active_view_ = id;
            touch();
        }
        return true;
    }

    bool Screen::toggleMaximized(const AreaId id) {
        const AreaId next = (id.valid() && area(id) && maximized_ != id) ? id : AreaId{};
        if (next == maximized_)
            return false;
        maximized_ = next;
        touch();
        return true;
    }

    AreaId Screen::split(const AreaId id, const SplitAxis axis, const float fraction, const bool new_first) {
        const Area* source = area(id);
        if (!source)
            return {};
        const bool duplicate = isMultiInstance(source->editor);
        const std::string editor = duplicate ? source->editor : std::string(editors::kView3D);

        const AreaId added{next_area_};
        if (!layout_.split(id, added, axis, fraction, new_first))
            return {};
        ++next_area_;
        Area copy;
        copy.id = added;
        copy.editor = editor;
        if (duplicate) {
            for (const auto& [type, space] : source->spaces) {
                if (space)
                    copy.spaces.emplace(type, space->clone());
            }
        } else if (auto space = newSpace(editor)) {
            copy.spaces.emplace(editor, std::move(space));
        }
        if (auto* view = dynamic_cast<View3DSpace*>(copy.space(editors::kView3D)))
            view->settings.split_view_mode = SplitViewMode::Disabled;
        areas_.emplace(added, std::move(copy));
        if (maximized_.valid())
            maximized_ = {};
        touch();
        return added;
    }

    bool Screen::canJoin(const AreaId kept, const AreaId absorbed) const {
        if (!layout_.canJoin(kept, absorbed))
            return false;
        return !isLastView(absorbed);
    }

    bool Screen::join(const AreaId kept, const AreaId absorbed) {
        if (!canJoin(kept, absorbed) || !layout_.join(kept, absorbed))
            return false;
        eraseArea(absorbed);
        repair();
        touch();
        return true;
    }

    bool Screen::canClose(const AreaId id) const {
        return area(id) && layout_.areaCount() > 1 && !isLastView(id);
    }

    bool Screen::close(const AreaId id) {
        if (!canClose(id) || !layout_.remove(id))
            return false;
        eraseArea(id);
        repair();
        touch();
        return true;
    }

    bool Screen::swap(const AreaId a, const AreaId b) {
        if (!layout_.swap(a, b))
            return false;
        touch();
        return true;
    }

    bool Screen::setEditor(const AreaId id, const std::string_view editor) {
        Area* target = area(id);
        if (!target || !registry_->contains(editor))
            return false;
        if (target->editor == editor)
            return true;
        if (!isMultiInstance(editor)) {
            if (const AreaId other = findEditor(editor); other.valid())
                return swap(id, other);
        }
        if (isLastView(id))
            return false;
        if (!target->space(editor)) {
            // A new 3D viewport starts where the user is looking.
            std::unique_ptr<SpaceData> space;
            if (editor == editors::kView3D) {
                if (const auto* active = view(active_view_))
                    space = active->clone();
                if (auto* view = dynamic_cast<View3DSpace*>(space.get()))
                    view->settings.split_view_mode = SplitViewMode::Disabled;
            }
            if (!space)
                space = newSpace(editor);
            if (space)
                target->spaces.emplace(std::string(editor), std::move(space));
        }
        target->editor = std::string(editor);
        repair();
        touch();
        return true;
    }

    AreaId Screen::openEditor(const std::string_view editor) {
        const auto type = registry_->find(editor);
        if (!type)
            return {};
        if (!type->multi_instance) {
            if (const AreaId existing = findEditor(editor); existing.valid()) {
                if (maximized_.valid() && maximized_ != existing) {
                    maximized_ = {};
                    touch();
                }
                return existing;
            }
        }

        const auto& placement = type->placement;
        const bool new_first = placement.side == Side::Left || placement.side == Side::Top;
        AreaId anchor;
        if (placement.anchor == EditorPlacement::Anchor::ActiveView)
            anchor = active_view_;
        else if (placement.anchor == EditorPlacement::Anchor::Editor)
            anchor = findEditor(placement.editor);

        const AreaId added{next_area_};
        const bool placed =
            anchor.valid()
                ? layout_.split(anchor, added, axisAcross(placement.side), placement.fraction, new_first)
                : layout_.insertAtEdge(added, placement.edge_side, placement.edge_fraction);
        if (!placed)
            return {};
        ++next_area_;
        Area created;
        created.id = added;
        created.editor = std::string(editor);
        std::unique_ptr<SpaceData> space;
        if (editor == editors::kView3D) {
            if (const auto* active = view(active_view_))
                space = active->clone();
            if (auto* view = dynamic_cast<View3DSpace*>(space.get()))
                view->settings.split_view_mode = SplitViewMode::Disabled;
        }
        if (!space)
            space = newSpace(editor);
        if (space)
            created.spaces.emplace(std::string(editor), std::move(space));
        areas_.emplace(added, std::move(created));
        maximized_ = {};
        touch();
        return added;
    }

    bool Screen::closeEditor(const std::string_view editor) {
        bool closed = false;
        for (const AreaId id : layout_.areas()) {
            const Area* a = area(id);
            if (a && a->editor == editor && canClose(id))
                closed = close(id) || closed;
        }
        return closed;
    }

    bool Screen::moveDivider(const DividerGeometry& divider, const float position) {
        if (!layout_.moveDivider(divider, position))
            return false;
        touch();
        return true;
    }

    bool Screen::toggleQuadView(const AreaId id, const float viewport_height) {
        if (!isView(id))
            return false;
        if (const auto quad = layout_.quadAround(id)) {
            const bool all_views =
                std::all_of(quad->begin(), quad->end(), [this](const AreaId area) { return isView(area); });
            if (all_views) {
                maximized_ = {};
                for (const AreaId area : *quad) {
                    if (area != id && layout_.remove(area))
                        eraseArea(area);
                }
                active_view_ = id;
                repair();
                touch();
                return true;
            }
        }
        // Top | original over Front | Right, as Blender lays its quad view out.
        const AreaId left = split(id, SplitAxis::Columns, 0.5f, true);
        if (!left.valid())
            return false;
        const AreaId bottom_right = split(id, SplitAxis::Rows, 0.5f);
        const AreaId bottom_left = split(left, SplitAxis::Rows, 0.5f);
        const auto orient = [&](const AreaId area, const ViewAxis axis) {
            if (auto* space = view(area)) {
                space->settings.orthographic = false;
                setAxisView(*space, axis, viewport_height);
            }
        };
        orient(left, ViewAxis::Top);
        orient(bottom_left, ViewAxis::Front);
        orient(bottom_right, ViewAxis::Right);
        active_view_ = id;
        touch();
        return true;
    }

    bool Screen::toggleSideView(const AreaId id) {
        if (!isView(id))
            return false;
        for (const Side side : {Side::Right, Side::Left}) {
            const AreaId neighbour = layout_.joinableNeighbour(id, side);
            if (neighbour.valid() && isView(neighbour))
                return join(id, neighbour);
        }
        return split(id, SplitAxis::Columns, 0.5f).valid();
    }

    bool Screen::isView(const AreaId id) const {
        const Area* a = area(id);
        return a && a->editor == editors::kView3D;
    }

    bool Screen::isLastView(const AreaId id) const {
        if (!isView(id))
            return false;
        return views().size() <= 1;
    }

    bool Screen::isMultiInstance(const std::string_view editor) const {
        const auto type = registry_->find(editor);
        return type && type->multi_instance;
    }

    std::unique_ptr<SpaceData> Screen::newSpace(const std::string_view editor) const {
        return registry_->createSpace(editor);
    }

    AreaId Screen::addArea(std::string editor) {
        const AreaId id{next_area_++};
        Area created;
        created.id = id;
        if (auto space = newSpace(editor))
            created.spaces.emplace(editor, std::move(space));
        created.editor = std::move(editor);
        areas_.emplace(id, std::move(created));
        return id;
    }

    void Screen::eraseArea(const AreaId id) {
        areas_.erase(id);
        if (maximized_ == id)
            maximized_ = {};
    }

    void Screen::repair() {
        if (!isView(active_view_)) {
            const auto all = views();
            active_view_ = all.empty() ? AreaId{} : all.front();
        }
        if (maximized_.valid() && !area(maximized_))
            maximized_ = {};
    }

    nlohmann::json Screen::save() const {
        nlohmann::json areas = nlohmann::json::array();
        for (const AreaId id : layout_.areas()) {
            const Area* a = area(id);
            if (!a)
                continue;
            nlohmann::json spaces = nlohmann::json::object();
            for (const auto& [editor, space] : a->spaces) {
                if (space)
                    spaces[editor] = space->save();
            }
            areas.push_back({{"id", id.value}, {"editor", a->editor}, {"spaces", std::move(spaces)}});
        }
        return {{"version", kFormatVersion},
                {"layout", layout_.toJson()},
                {"areas", std::move(areas)},
                {"active_view", active_view_.value}};
    }

    std::optional<Screen> Screen::load(const nlohmann::json& json, const EditorTypeRegistry& registry) {
        if (!json.is_object() || json.value("version", 0) != kFormatVersion)
            return std::nullopt;
        const auto layout_it = json.find("layout");
        const auto areas_it = json.find("areas");
        if (layout_it == json.end() || areas_it == json.end() || !areas_it->is_array())
            return std::nullopt;
        auto layout = ScreenLayout::fromJson(*layout_it);
        if (!layout)
            return std::nullopt;

        Screen screen(registry);
        screen.layout_ = std::move(*layout);
        std::uint32_t largest = 0;
        for (const auto& item : *areas_it) {
            if (!item.is_object())
                return std::nullopt;
            const auto id_it = item.find("id");
            const auto editor_it = item.find("editor");
            if (id_it == item.end() || !id_it->is_number_unsigned() || editor_it == item.end() ||
                !editor_it->is_string() || editor_it->get<std::string>().empty())
                return std::nullopt;
            const auto raw_id = id_it->get<std::uint64_t>();
            if (raw_id == 0 || raw_id > std::numeric_limits<std::uint32_t>::max())
                return std::nullopt;
            const AreaId id{static_cast<std::uint32_t>(raw_id)};
            if (!screen.layout_.contains(id) || screen.areas_.contains(id))
                return std::nullopt;
            Area restored;
            restored.id = id;
            restored.editor = editor_it->get<std::string>();
            if (const auto spaces_it = item.find("spaces"); spaces_it != item.end()) {
                if (!spaces_it->is_object())
                    return std::nullopt;
                for (const auto& [editor, space_json] : spaces_it->items()) {
                    auto space = registry.createSpace(editor);
                    // Spaces of editors this build does not know are dropped;
                    // a known editor's malformed state fails the load.
                    if (!space)
                        continue;
                    if (!space->load(space_json))
                        return std::nullopt;
                    restored.spaces.emplace(editor, std::move(space));
                }
            }
            if (!restored.space(restored.editor)) {
                if (auto space = registry.createSpace(restored.editor))
                    restored.spaces.emplace(restored.editor, std::move(space));
            }
            screen.areas_.emplace(id, std::move(restored));
            largest = std::max(largest, id.value);
        }
        if (screen.areas_.size() != screen.layout_.areaCount() || screen.views().empty())
            return std::nullopt;
        screen.next_area_ = largest + 1;
        if (const auto active_it = json.find("active_view");
            active_it != json.end() && active_it->is_number_unsigned() &&
            active_it->get<std::uint64_t>() <= std::numeric_limits<std::uint32_t>::max()) {
            screen.active_view_ = AreaId{active_it->get<std::uint32_t>()};
        }
        screen.repair();
        return screen;
    }

    void registerBuiltinEditorTypes(EditorTypeRegistry& registry) {
        registry.add(EditorType{
            .id = std::string(editors::kView3D),
            .label = "3D Viewport",
            .label_key = "editor.view3d",
            .icon = "editor-view3d",
            .multi_instance = true,
            .placement = {.anchor = EditorPlacement::Anchor::ActiveView, .side = Side::Right, .fraction = 0.5f},
            .create_space = [] { return std::make_unique<View3DSpace>(); },
        });
        registry.add(EditorType{
            .id = std::string(editors::kScene),
            .label = "Scene",
            .label_key = "editor.scene",
            .icon = "editor-scene",
            .placement = {.anchor = EditorPlacement::Anchor::Editor,
                          .side = Side::Top,
                          .fraction = 0.4f,
                          .editor = std::string(editors::kProperties)},
        });
        registry.add(EditorType{
            .id = std::string(editors::kProperties),
            .label = "Properties",
            .label_key = "editor.properties",
            .icon = "editor-properties",
            .placement = {.anchor = EditorPlacement::Anchor::Editor,
                          .side = Side::Bottom,
                          .fraction = 0.6f,
                          .editor = std::string(editors::kScene)},
        });
        registry.add(EditorType{
            .id = std::string(editors::kConsole),
            .label = "Python Console",
            .label_key = "editor.console",
            .icon = "editor-console",
            .placement = {.anchor = EditorPlacement::Anchor::ActiveView, .side = Side::Right, .fraction = 0.4f},
        });
    }

} // namespace lfs::vis::screen
