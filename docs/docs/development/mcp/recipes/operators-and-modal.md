---
sidebar_position: 3
---

# Operators And Modal Workflows

Use this flow when the task depends on registered GUI operators rather than a dedicated high-level tool.

## Sequence

1. Read `lichtfeld://operators/registry` or call `operator_list`.
2. Call `operator_describe` for the chosen id and inspect its schema and poll state.
3. Call `operator_invoke`.
4. If the result is modal, switch to `operator_modal_state`, `operator_modal_event`, `operator_cancel_modal`, and the normalized runtime job `operator.modal`.

## Discover Operators

```json
{
  "tool": "operator_list",
  "arguments": {
    "include_schema": true,
    "include_poll": true
  }
}
```

## Describe One Operator

```json
{
  "tool": "operator_describe",
  "arguments": {
    "operator_id": "transform.translate",
    "include_schema": true,
    "include_poll": true
  }
}
```

## Invoke The Operator

Use the schema returned by `operator_describe` instead of guessing the payload:

```json
{
  "tool": "operator_invoke",
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
  "tool": "operator_modal_state",
  "arguments": {}
}
```

Dispatch a modal event:

```json
{
  "tool": "operator_modal_event",
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
  "tool": "operator_cancel_modal",
  "arguments": {}
}
```

## Notes

- Use `operator_describe` before `operator_invoke`
- `operator_invoke` may finish immediately or return `running_modal`
- modal operator state is also normalized as runtime job `operator.modal`
