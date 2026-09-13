# Studio ⇄ portal gallery E2E

`scripts/validate_gallery_sync.py` drives the real Asset Manager through Studio's MCP editor. It launches its own Studio process, Xvfb display and fresh `LFS_HOME`; local mode also migrates and seeds a temporary SQLite portal and runs its processor. It never configures, builds, or commits. An existing native build with gallery plugins and an NVIDIA-capable Linux runtime is required.

## Local release check

From the Studio checkout, with the portal dependencies already installed:

```sh
python3 -B scripts/validate_gallery_sync.py \
  --build-dir /home/gauss/lfs-worktrees/galleryam/build \
  --portal-source /home/gauss/projects/lichtfeld-portal \
  --portal-python /home/gauss/lfs-runs/galleryam/portalenv/bin/python \
  --display :93 \
  --format sog \
  --report /home/gauss/lfs-runs/galleryam/gallery-sync-e2e.md \
  --junit /home/gauss/lfs-runs/galleryam/gallery-sync-e2e.xml
```

The script starts at `--display` (default `:93`) and skips occupied socket or lock paths until it finds a free display. It records the selected display in the report and never attaches to an existing X server. To require a particular display, pass both `--display :93` and `--strict-display`; occupancy then fails the run. `--strict-display` alone still permits fallback from the default. Install Xvfb and provide the GPU/Vulkan runtime in the runner. An inherited `VK_ICD_FILENAMES` is respected; otherwise the launcher selects an installed NVIDIA ICD from `/usr/share/vulkan/icd.d`. Studio uses `SDL_VIDEODRIVER=x11` and `--no-splash --log-level debug`. This is a GUI check on a virtual display, not Studio's CLI renderer.

Local mode is the default. Without either portal argument, it uses `~/projects/lichtfeld-portal`. `--portal-python` selects an already provisioned interpreter; otherwise the portal's `.venv/bin/python`, then the script interpreter, is used. No dependency installation is attempted. Local startup overrides the database, external origin, storage path, R2 bucket and relevant Django gallery flags; it does not source an existing portal environment or reuse its database. The seeded disposable account is `native-sync@example.com` with the local-only desktop bearer `local-test-access` and `desktop.basic gallery.sync` scopes.

Fixtures must exist at `tests/data/portable-{ply,sog,ssog,multi}.licht`. They are copied to the fresh profile's `projects` folder. In CI, provision these fixtures, a compatible existing Studio build, the portal Python environment, Xvfb and GPU access before running the command. Use separate display numbers and report paths for concurrent runs. Run each of `sog`, `ssog`, `spz`, and `studio` for format coverage; the default is `sog`.

## Real origin

```sh
python3 -B scripts/validate_gallery_sync.py \
  --build-dir build \
  --portal-origin https://portal.lichtfeld.io \
  --display :94 --format sog \
  --report /tmp/gallery-sync-live.md --junit /tmp/gallery-sync-live.xml
```

Real origins require HTTPS with no URL path or credentials. The script starts Studio's device flow, prints the verification URL and code, and waits for browser approval. Use a gallery-enabled account. No real tokens are supplied in arguments or embedded in the validator. This mode requires a human sign-in; use local mode for unattended CI. HTTP is reserved for the script-owned `127.0.0.1` portal.

Each run uses a unique title prefix, `E2E validate <random run id> `. Cleanup stops the producer, restarts Studio with the same private credentials, cancels this run's outstanding uploads, and deletes this run's live scenes through the authenticated API. Other scenes and earlier E2E runs are excluded. Deleted tombstones are expected in the sync listing and do not count as cleanup failures. If cleanup fails, the script prints the affected IDs/error and run prefix, retains the private profile, and exits nonzero. Inspect that account's gallery for the printed prefix and remove any remaining entries; do not delete other E2E runs by the common prefix alone.

## Assertions and artifacts

Every reached step is timed and captured using `render_capture_window`; a capture failure fails the step. A launch failure may have no screenshot because Studio never became available. The Markdown report is rewritten after every step, names the exact failure, and includes the last 30 relevant log lines. PNGs and full logs live beside the report in a unique artifact directory and survive temporary-profile cleanup. `--junit` writes CI-compatible results. `--keep` retains the database, storage and profile as well; it still stops every process owned by the script.

Every editor result is enclosed by `<<LFSE2E>>` and `<</LFSE2E>>`. Parsing ignores surrounding diagnostics, including the account service's non-default portal host notice. Those diagnostics are saved in `editor-diagnostics.log`, linked from the report, and included in its log excerpt. Missing, duplicate or incomplete sentinel pairs fail explicitly. The parser handles terminal row wrapping; the emitter escapes spaces and angle brackets so row trimming and marker-like values cannot corrupt JSON.

MCP `editor_run` defaults to 20,000 output bytes (`src/app/mcp_gui_tools.cpp`); the validator explicitly requests that limit and caps each compact ASCII JSON payload at 8,000 bytes before printing. Job, remote-scene and cleanup queries return the fields needed for assertions rather than entire snapshots. Oversized payloads, truncated output and unfinished editor calls fail with a diagnostic instead of being accepted as partial results.

Exit status is `0` only when no failure was recorded. Any failed step, cleanup failure, late log error or exception outside a step returns nonzero. A passing log scan cannot clear an earlier failure. Exceptions during workflow, cleanup or shutdown are also retained in the report when report storage remains writable.

The check asserts:

1. Signed-out gallery sidebar rows and the rendered “Sign in to see your gallery” hint.
2. Successful sign-in and gallery refresh.
3. All four fixtures appear with “Not published”.
4. Publishing closed `portable-multi` completes a new upload job, creates a ready portal scene, and shows “Up to date”. Both native closed-project export and the Open → Publish fallback are accepted. Local mode inspects every `bundle_index.manifest.nodes[*].file`: `.sog`, `.ssog`, `.spz`, or `.ply` for Studio format. Both modes verify the scene through authenticated `GET /api/gallery/v1/splats/{id}`.
5. Opening the published fixture, changing exposure by 0.1 and saving produces “Saved since last publish”; Update completes and restores “Up to date”. Current controllers can PATCH an exposure-only change without creating another upload job. Both that path and a completed replacement upload must change the remote revision and return the saved exposure through GET; the report records which path ran.
6. Unlink requires the native “Continue” confirmation and produces a remote-only card.
7. Pull review and start complete a download, register exactly one new linked local project, keep Asset Manager enabled, and preserve the open document path, node identities/names and exposure.
8. Killing and relaunching Studio restores the link and all transfer IDs from the journal, followed by authenticated refresh and restored panel links.
9. Remove requires confirmation, shows “Removed on portal”, and produces exactly one Needs attention row and rendered count.
10. The transfers tray opens and its model and rendered rows contain every job with human-readable byte sizes.
11. Studio and portal logs contain no `Traceback`, `[error]`, `Syntax error parsing property`, or `Missing localization key` lines. The only default exception is the specific unrelated `xdg-open: no method available for opening` failure, not all lines mentioning xdg-open.

Jobs are identified relative to earlier job IDs and must reach `completed` with complete byte counters and server processing finished. A metadata-only Update is verified separately and does not fabricate a transfer job; the tray must contain every actual job. Local processing calls `processing.run_one()` in the temporary portal environment; real origins use their deployed workers. `--timeout` defaults to 300 seconds per asynchronous operation; `--auth-timeout` defaults to 600 seconds for browser approval. Progress is printed while waiting. SIGINT and SIGTERM enter cleanup; an uncatchable SIGKILL cannot run cleanup.

The restart assertion reads the new `GallerySync` instance's constructor-loaded journal **before that instance's first refresh**. Public `snapshot()['links']` deliberately stays empty until authenticated refresh establishes ownership. This checks persisted recovery without claiming that private links are exposed before account verification. `_gallery_counts()` counts selected projects; the Needs attention count is `len(p._gallery_rows(attention=True))`, matching the UI binding.

The script uses the same launch and RPC sequence as `validate_gallery_resilience.py`. That script's helpers are closures inside `main()` and cannot be imported independently; with the new-files-only constraint, this validator reuses the exported `launch_environment()` from `lichtfeld_mcp_bridge.py` instead. It initializes MCP, discovers tool metadata/resources and reads the five baseline state resources before editing the panel.

## Pure helper tests

No running Studio, native Python extension, portal, or network is needed:

```sh
python3 -B -m unittest discover \
  -s tests/python -p test_validate_gallery_sync_script.py -v
```

These cover argument/origin validation, automatic and strict display selection, format and manifest checks, log filtering, framed JSON with noisy and wrapped terminal output, payload limits and truncation, JSON-RPC/editor/PNG handling, Markdown/JUnit rendering, cleanup scope and failure exit status through `main()`. Passing helper tests does not establish that the native E2E passed: run the release command and retain its report and screenshots as separate evidence.
