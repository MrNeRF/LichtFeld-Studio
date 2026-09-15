---
sidebar_position: 5
---
# Themes

LichtFeld Studio groups related appearances into theme families. Choose a family in **Preferences → Appearance → Theme**, then select its available variant. Variant names may be custom names such as **Night** and **Day**, while still representing dark and light modes.

Families containing only one variant are selected directly. When a family provides both dark and light variants, the mode buttons share the full selector width. If automatic system detection is available, a third **Auto** button appears.

The same choices are available from **View → Theme**. The menu label includes the current family and selection, for example **Theme · Signal · Day**, so the active appearance remains visible without opening every submenu. Families with multiple variants open a second submenu; single-variant families remain direct menu entries. The selected family is repeated with its active variant inside the submenu.

## Built-in appearance families

Every family can provide a dark variant, a light variant, or both. **LichtFeld** keeps the familiar application appearance, while **Signal** is the more expressive built-in family: it uses layered blue/violet surfaces, gradient section chrome, stronger panel separation, and translucent viewport controls. Signal exposes the concise variant names **Night** and **Day** while still participating in semantic dark/light selection and automatic system matching.

Gradient definitions are optional. Families that do not provide them remain valid and receive palette-derived flat fallbacks.

**Preferences → Appearance → Viewport controls** is a global presentation choice rather than part of a theme file. **Solid**, **Translucent**, and **Frosted glass** keep the active family's colors and contrast rules; changing family therefore still gives viewport chrome an identifiable tint. Frosted glass adds a low-resolution blur and slight optical bend behind the controls when the Vulkan swapchain supports the required blit path. Unsupported configurations retain the themed translucent surface instead of losing contrast.

## Automatic mode

Automatic mode follows the operating system's light or dark preference:

- Windows reads the current application theme preference.
- Linux uses SDL3 system-theme detection. SDL can obtain the preference through the desktop integration available in the current session, including the standardized [XDG Desktop Portal appearance setting](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.Settings.html) where supported.

**Auto** is shown only when the running system returns a definite light or dark preference. If the preference is unavailable or unknown, LichtFeld hides **Auto** and uses the dark fallback for an older saved automatic choice.

Theme family, variant, and automatic-mode choices are saved with the other user preferences. If a selected variant later disappears, LichtFeld uses the remaining variant in that family; if the entire family is missing, it returns to the built-in dark theme.

## Custom themes

The current application catalog is loaded from packaged theme assets. There is not yet an end-user import/delete command or a user-theme directory, so copying a JSON file into the user preferences tree does not install it.

Theme authors and contributors can use the [theme format v2 guide](../development/theme-format.md) and its downloadable template. During development, add the family file to `src/visualizer/gui/assets/themes/` and register it in the adjacent `manifest.json`; theme resources then participate in hot reload. This developer workflow is distinct from the planned user-facing custom-theme importer.
