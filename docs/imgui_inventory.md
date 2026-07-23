# ImGui / ImPlot Removal — Closure Note

**Status: complete (2026-07), PR #1395.**

Dear ImGui and ImPlot are fully removed from LichtFeld Studio. RmlUi is the
only GUI stack. There are no `imgui` / `implot` vcpkg dependencies, no CMake
`find_package` or link targets, no production includes, and no remaining
ImGui/ImPlot symbols in app or library binaries.

## What replaced what

| Former ImGui / ImPlot surface | Replacement |
| --- | --- |
| Frame bootstrap, input capture, panel shell, native chrome | RmlUi documents, shell, and SDL-native input |
| Python plugin immediate-mode panels | `RmlImModeLayout` / `RmlImModePanelAdapter` — im-mode bridge over retained RmlUi with slot-based reconciliation and equality-gated updates |
| Viewport immediate drawing (`draw_window_*`, `draw_*`) | `ScreenOverlayRenderer` (viewport-scoped); Python path via the viewport overlay bridge |
| Pie-menu icons | `IconCache` + `ScreenOverlayRenderer::addImage` |
| Floating-panel and color-picker drop shadows | RmlUi `box-shadow` theme tokens in RCSS |
| Chromaticity / CRF plot widgets | Retained Rml custom elements `<chromaticity-diagram>` / `<crf-curve>`; the legacy layout methods warn once and direct authors to these elements |
| Zep text editor display backend | `ZepDisplay_Rml` |

Hooks that cannot host interactive widgets log a one-shot warning instead of
silently no-oping.

Viewport-overlay Python drawing is available only while the
`ScreenOverlayRenderer` frame is active. Both `draw_*` and `draw_window_*`
take absolute screen coordinates; the legacy `background` arguments are
compatibility-only. The renderer has one command stream, packed into a shape
batch followed by a text/image batch, with enqueue order retained within each
batch.

## Mechanical acceptance (post-merge)

```sh
# Source / build graph: no hits
rg -n 'imgui|implot|ImGui|ImPlot' src/ CMakeLists.txt vcpkg.json \
  src/**/CMakeLists.txt cmake/ tests/CMakeLists.txt

# Project binaries: no leftover symbol or linked DSO
nm -C build/LichtFeld-Studio build/liblfs_*.so \
  build/src/python/lichtfeld*.so 2>/dev/null | rg -i 'imgui|implot'  # empty
ldd build/LichtFeld-Studio 2>/dev/null | rg -i 'imgui|implot'        # empty
```

Phase 1 exit criteria for the broader Vulkan track live in
[`docs/development/vulkan-elite-roadmap.md`](development/vulkan-elite-roadmap.md)
(Phase 1 — legacy immediate-mode UI exorcism).

## Known follow-up (outside this PR)

Pre-existing on master, not a PR #1395 regression: the tools-hook chain is
dead. `python_runtime.cpp::draw_tools_section()` has no in-tree caller, and
`panels.py` does not import or register the first-party
`cropbox_controls.py` / `ellipsoid_controls.py` modules. Those modules would
register on `"tools"/"transform"` if loaded. Track this as a separate
resurrect-or-retire cleanup.
