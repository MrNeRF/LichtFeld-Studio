---
sidebar_position: 5
---
# Themes

LichtFeld Studio groups related appearances into theme families. Choose a family in **Preferences → Appearance → Theme**, then select its available variant. Variant names may be custom names such as **Night** and **Day**, while still representing dark and light modes.

Families containing only one variant are selected directly. When a family provides both dark and light variants, the mode buttons share the full selector width. If automatic system detection is available, a third **Auto** button appears.

The same choices are available from **View → Theme**. Families with multiple variants open a second submenu; single-variant families remain direct menu entries.

## Automatic mode

Automatic mode follows the operating system's light or dark preference:

- Windows reads the current application theme preference.
- Linux uses SDL3 system-theme detection. SDL can obtain the preference through the desktop integration available in the current session, including the standardized [XDG Desktop Portal appearance setting](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.Settings.html) where supported.

**Auto** is shown only when the running system returns a definite light or dark preference. If the preference is unavailable or unknown, LichtFeld hides **Auto** and uses the dark fallback for an older saved automatic choice.

Theme family, variant, and automatic-mode choices are saved with the other user preferences. If a selected variant later disappears, LichtFeld uses the remaining variant in that family; if the entire family is missing, it returns to the built-in dark theme.
