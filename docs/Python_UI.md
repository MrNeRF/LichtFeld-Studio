# Python UI Guide

`docs/Python_UI.md` is retained only as a compatibility redirect for older links.

The canonical plugin UI documentation lives in:

- `docs/plugins/getting-started.md`
- `docs/plugins/api-reference.md`
- `docs/plugins/examples/README.md`

The first public panel API uses:

- `id`, not `idname`
- `poll_dependencies`, not `poll_deps`
- `lf.ui.PanelSpace` enums only
- enum values for panel metadata, not string literals

Minimal example:

```python
import lichtfeld as lf


class MyPanel(lf.ui.Panel):
    id = "my_plugin.panel"
    label = "My Custom Panel"
    space = lf.ui.PanelSpace.FLOATING
    poll_dependencies = {
        lf.ui.PollDependency.SCENE,
        lf.ui.PollDependency.SELECTION,
    }

    def draw(self, ui):
        ui.heading("Settings")
        ui.text_disabled("See docs/plugins/ for the full UI guide.")


lf.register_class(MyPanel)
```
