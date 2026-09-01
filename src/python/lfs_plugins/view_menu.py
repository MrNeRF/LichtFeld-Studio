# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""View menu implementation."""

import lichtfeld as lf
from .layouts.menus import menu_action, menu_separator, register_menu, menu_submenu, menu_toggle

__lfs_menu_classes__ = ["ViewMenu"]


def _tr_fallback(key: str, fallback: str) -> str:
    result = lf.ui.tr(key)
    if result and result != key:
        return result
    return fallback


@register_menu
class ViewMenu:
    """View menu for the menu bar."""

    label = "menu.view"
    location = "MENU_BAR"
    order = 30

    _SCALE_OPTIONS = (
        (0.0, "menu.view.ui_scale.auto"),
        (1.0, "100%"),
        (1.25, "125%"),
        (1.5, "150%"),
        (1.75, "175%"),
        (2.0, "200%"),
    )

    def menu_items(self):
        tr = lf.ui.tr
        theme_catalog = sorted(
            lf.ui.themes(),
            key=lambda theme: (theme.get("order", 0), theme.get("name", theme.get("id", ""))),
        )
        families = {}
        for theme in theme_catalog:
            family_id = theme.get("family_id") or theme["id"]
            family = families.setdefault(
                family_id,
                {
                    "id": family_id,
                    "name": theme.get("family_name") or theme.get("name") or family_id,
                    "order": theme.get("order", 0),
                    "variants": [],
                },
            )
            family["order"] = min(family["order"], theme.get("order", 0))
            family["variants"].append(theme)

        current_family = lf.ui.get_theme_family()
        current_mode = lf.ui.get_theme_mode()
        theme_items = []
        for family in sorted(
            families.values(), key=lambda item: (item["order"], item["name"])
        ):
            variants = sorted(
                family["variants"],
                key=lambda variant: (variant.get("order", 0), variant.get("mode", "")),
            )
            available_modes = {variant["mode"] for variant in variants}
            if len(variants) == 1:
                mode = variants[0]["mode"]
                theme_items.append(
                    menu_toggle(
                        family["name"],
                        lambda family_id=family["id"], variant_mode=mode: lf.ui.set_theme_family(
                            family_id, variant_mode
                        ),
                        current_family == family["id"],
                    )
                )
                continue

            variant_items = []
            for variant in variants:
                mode = variant["mode"]
                label = variant.get("variant_name") or variant.get("name") or mode.title()
                variant_items.append(
                    menu_toggle(
                        label,
                        lambda family_id=family["id"], variant_mode=mode: lf.ui.set_theme_family(
                            family_id, variant_mode
                        ),
                        current_family == family["id"] and current_mode == mode,
                    )
                )

            if (
                {"dark", "light"}.issubset(available_modes)
                and lf.ui.supports_system_theme()
            ):
                variant_items.append(
                    menu_toggle(
                        tr("menu.view.theme.auto"),
                        lambda family_id=family["id"]: lf.ui.set_theme_family(
                            family_id, "auto"
                        ),
                        current_family == family["id"] and current_mode == "auto",
                    )
                )

            theme_items.append(menu_submenu(family["name"], variant_items))

        pref = lf.ui.get_ui_scale_preference()
        scale_items = []
        for scale_val, label_key in self._SCALE_OPTIONS:
            label = tr(label_key) if scale_val == 0.0 else label_key
            scale_items.append(
                menu_toggle(
                    label,
                    lambda scale=scale_val: lf.ui.set_ui_scale(scale),
                    abs(pref - scale_val) < 0.01,
                )
            )

        return [
            menu_submenu(tr("menu.view.theme"), theme_items),
            menu_submenu(tr("menu.view.ui_scale"), scale_items),
            menu_separator(),
            menu_toggle(
                tr("menu.view.performance_hud"),
                lf.ui.toggle_vram_hud,
                bool(getattr(lf.ui, "is_perf_hud_visible", lambda: False)()),
                shortcut="F10",
            ),
            menu_action(_tr_fallback("image_preview.reset_view", "Reset View"), lf.reset_camera),
            menu_action(_tr_fallback("main_panel.console", "Console"), lf.ui.toggle_system_console),
        ]



def register():
    pass


def unregister():
    pass
