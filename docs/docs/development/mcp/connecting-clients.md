---
sidebar_position: 2
---

# Connecting MCP Clients

LichtFeld Studio embeds its own MCP server. By default it starts automatically
with the GUI and listens on:

```
http://127.0.0.1:45677/mcp
```

The transport is plain HTTP JSON-RPC. The default loopback binding is reachable
only from the same computer. Preferences can explicitly expose the endpoint on
the local network; that mode has no authentication or TLS and must only be used
on a trusted network. The port, listener state, usable endpoints, and optional
request logging are also managed from Preferences or the MCP status-bar chip.

## Choose a Connection Path

| Client | Path | App lifecycle |
| --- | --- | --- |
| Claude Code, other clients with native HTTP transport | Direct HTTP | Launch LichtFeld Studio yourself |
| Claude Desktop | `mcp-remote` stdio proxy | Launch LichtFeld Studio yourself |
| Coding agents working inside this repository (Codex) | `.mcp.json` stdio bridge | Bridge launches the app on demand |

## Direct HTTP

Use this whenever the client supports the streamable HTTP transport. Start LichtFeld Studio, then register the endpoint. For Claude Code:

```bash
claude mcp add --transport http lichtfeld http://127.0.0.1:45677/mcp
```

No proxy process, no extra runtime. The only requirement is that the app is running before the client connects.

## Claude Desktop

Claude Desktop cannot use the endpoint directly:

- The custom connector UI only accepts `https://` URLs, so the plain-HTTP local endpoint is rejected.
- Its stdio transport expects newline-delimited JSON, which the repo bridge does not speak (see below).

The working path is the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) proxy (requires Node.js). Add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lichtfeldstudio": {
      "command": "cmd",
      "args": [
        "/c",
        "npx",
        "mcp-remote",
        "http://127.0.0.1:45677/mcp",
        "--allow-http"
      ]
    }
  }
}
```

On Linux, replace `"command": "cmd"` and the `"/c"` argument with `"command": "npx"` and drop `"/c"`.

Notes:

- `--allow-http` is required; `mcp-remote` refuses plain-HTTP URLs without it.
  The default loopback endpoint does not leave the computer. If network
  exposure is enabled, plain HTTP traffic is visible to the local network.
- Launch LichtFeld Studio before starting the conversation. `mcp-remote` connects to a running instance; it does not launch the app.

## In-Repo stdio Bridge

The repository's [`.mcp.json`](https://github.com/MrNeRF/LichtFeld-Studio/blob/master/.mcp.json) launches `scripts/lichtfeld_mcp_bridge.py`, a stdio-to-HTTP proxy that also starts LichtFeld Studio on demand: it pings the endpoint, launches the executable if nothing answers, then forwards JSON-RPC messages to HTTP.

This path targets coding agents that operate inside a checkout of this repository. Current limitations to know about:

- The bridge frames stdio messages with `Content-Length` headers. Clients that use spec-standard newline-delimited framing (including Claude Desktop) will hang; use `mcp-remote` or direct HTTP for those instead.
- Auto-launch passes `--no-splash`, which portable builds (the released binaries) do not recognize, so the launched app exits immediately. Use a regular development build, or set `LFS_EXECUTABLE` to one.

See [issue #1399](https://github.com/MrNeRF/LichtFeld-Studio/issues/1399) for the status of these limitations.

The bridge overrides are listed with the rest of the canonical environment
surface in [Developer flags and diagnostics](../flags#mcp-bridge-variables).

## Verify the Connection

With LichtFeld Studio running, list the tools straight from the endpoint:

```bash
curl -s -X POST http://127.0.0.1:45677/mcp \
  -H "content-type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

A JSON response with a `tools` array means the server is up and any of the paths
above should work. If the connection is refused, inspect the MCP status chip:
the server may be disabled, configured on another port, or unable to bind
because another process already owns the port. The displayed endpoint list
contains client-ready URLs; `0.0.0.0` is only the listener bind address and is
not presented as a client endpoint.

## Runtime Controls and Diagnostics

The MCP section in Preferences controls:

- whether the listener is enabled;
- loopback-only or local-network binding;
- the HTTP port;
- opt-in request activity logging.

Port edits are drafts until confirmed with Enter, the adjacent confirmation
button, or the Preferences footer OK button. Closing Preferences from the title
bar discards an unconfirmed port draft, preventing it from being applied later. The
other controls apply immediately. The status bar offers remappable
`Ctrl+Shift+M` and `Ctrl+Shift+N` actions for server power and bind scope.
If a bind fails, Preferences keeps the structured failure visible, identifies
the attempted address and port, and does not advertise an active endpoint. A
retry of the same configuration performs a real listener restart. Choose another
port when the configured one is already in use or reserved by the operating system.
The MCP listener deliberately claims single-process ownership of its address and
port, so a second listener cannot silently share the configured endpoint.

When request logging is enabled, complete JSONL records are appended to the
current per-session file without rewriting earlier requests. Its path is shown
in Preferences and the log directory can be opened from there. Error records state
whether the failure came from the JSON-RPC envelope or from an MCP tool result,
plus the failure stage and stable diagnostic codes. Each request record also
includes the source and destination socket IP addresses and ports. Parameters,
tool names, payloads, free-form messages, and error details are not logged.
Network addresses can identify devices and interfaces, so logging remains an
explicit opt-in. Safe mode forces both the listener and request logging off
without modifying the persisted configuration; the MCP section reports why and
disables its configuration controls for the lifetime of that safe-mode process.

## After Connecting

Follow the discovery-first workflow in [Bootstrap](bootstrap.md): call `initialize`, list tools and resources once, and read the `lichtfeld://` state resources before invoking tools.
