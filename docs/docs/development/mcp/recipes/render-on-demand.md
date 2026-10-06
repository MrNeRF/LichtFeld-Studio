# Frame demand and idle verification

`runtime.frame_ledger` reports presented GUI frames, rendered views, per-reason counters, active holders, and stale-view detections. Pass `{"reset": true}` to reset counters before a measurement window. `lichtfeld://runtime/state` includes the same snapshot under `frames`.

For an idle check, first load the desired scene and wait for its initial redraw to finish. Reset the ledger, leave the pointer still, and sample for at least 60 seconds. The expected idle result is zero presented frames, zero rendered views, zero stale detections, and no live holders. A visible training-status animation or VRAM HUD may produce GUI-only frames while visible; those frames must not render a 3D view.

The GPU integration test is `tests/python/test_render_on_demand_idle.py`. It needs an isolated home directory, display `:92`, the MCP port `45692`, and a manifest of public fixture setup calls for each render mode. Run it on the configured GPU runner with:

```sh
pytest tests/python/test_render_on_demand_idle.py -m 'gpu and integration' -q
```

The manifest is runner configuration and must not be committed with local dataset paths. For an idle floor comparison, sample GPU utilization before the app starts; unrelated GPU work can make utilization unsuitable as a per-process rendering measurement.

`ui_fps` counts successful UI presentations in the trailing second, excluding the single idle-clear presentation. `viewport_fps` counts fresh view outputs (including empty-scene clears); cached and deferred results do not count. Both reach zero at idle. The ledger still counts the idle-clear presentation in `frames_presented`.
