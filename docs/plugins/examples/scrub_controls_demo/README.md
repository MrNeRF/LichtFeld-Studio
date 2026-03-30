# Scrub Controls Demo Plugin

This is a runnable plugin package you can drop into your plugin folder to verify
that `lfs_plugins.ScrubFieldController` is working end-to-end.

## Install this demo

```bash
cp -r docs/plugins/examples/scrub_controls_demo ~/.lichtfeld/plugins/scrub_controls_demo
```

Then restart LichtFeld Studio or reload plugins from the plugin registry view. The
plugin ID is `docs.scrub_controls_demo.panel`.

The panel exposes three scrub controls:

- `Strength`
- `Quality`
- `Threshold`

Use the knob-like scrub controls to verify pointer interaction, direct typing, and
model-bound updates.
