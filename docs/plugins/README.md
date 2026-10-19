# Python API and Plugin Docs

Start here for the Python API shipped with this checkout. Run native API code
inside LichtFeld Studio; SDK stubs are for IDE completion, not a standalone
Python runtime.

| Page | Use it for |
|---|---|
| [Development guide](getting-started.md) | Scaffolding, panels, operators, scene edits, training hooks, dependencies, debugging, and publishing |
| [Application API](api-reference.md) | Scene/data types, selection, transforms, training, rendering, file/project IO, tensors, and automation |
| [Plugin and UI API](plugin-reference.md) | Panel/menu contracts, widgets, retained UI, runtime state, properties, tools, capabilities, and plugin lifecycle |
| [Examples](examples/README.md) | Small examples and complete plugin packages |

Each guide has an on-page contents list. Limitations are documented next to the
API they affect, including layout-specific widget support and off-cursor
picking semantics.

The [SDK stubs](../../src/python/stubs/lichtfeld/) contain overloads and full
method lists for large surfaces such as tensors and meshes. Verify behavior in
[native bindings](../../src/python/lfs/) and [Python helpers](../../src/python/lfs_plugins/);
`help()` in the app describes the version currently running.

Related guides: [MCP automation](../docs/development/mcp/),
[RmlUI styling](../docs/development/rmlui-styling.md), and
[UI design language](../docs/development/ui-design-language.md).
