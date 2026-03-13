---
sidebar_position: 3
---

# Operators And Modal Workflows

Use this flow when the task depends on registered GUI operators rather than a dedicated high-level tool.

## Sequence

1. Read `lichtfeld://operators/registry` or call `operator.list`.
2. Call `operator.describe` for the chosen id and inspect its schema and poll state.
3. Call `operator.invoke`.
4. If the result is modal, switch to `operator.modal_state`, `operator.modal_event`, `operator.cancel_modal`, and the normalized runtime job `operator.modal`.

## Discover Operators

```json
{
  "tool": "operator.list",
  "arguments": {
    "include_schema": true,
    "include_poll": true
  }
}
```

## Describe One Operator

```json
{
  "tool": "operator.describe",
  "arguments": {
    "operator_id": "transform.translate",
    "include_schema": true,
    "include_poll": true
  }
}
```

## Invoke The Operator

Use the schema returned by `operator.describe` instead of guessing the payload:

```json
{
  "tool": "operator.invoke",
  "arguments": {
    "operator_id": "transform.translate",
    "arguments": {
      "...": "use fields from the operator schema"
    }
  }
}
```

## Handle Modal State

Inspect current modal state:

```json
{
  "tool": "operator.modal_state",
  "arguments": {}
}
```

Dispatch a modal event:

```json
{
  "tool": "operator.modal_event",
  "arguments": {
    "type": "mouse_move",
    "x": 320,
    "y": 240,
    "dx": 8,
    "dy": -4,
    "mods": 0
  }
}
```

Cancel the active modal operator:

```json
{
  "tool": "operator.cancel_modal",
  "arguments": {}
}
```

## Notes

- Use `operator.describe` before `operator.invoke`
- `operator.invoke` may finish immediately or return `running_modal`
- modal operator state is also normalized as runtime job `operator.modal`
