---
title: Gallery sync stress harness
---

Run `scripts/stress_gallery_sync.py` against an existing LichtFeld Studio build and a local
portal checkout. It never configures, builds, installs dependencies, or commits.
It imports Lane H's `validate_gallery_sync.py` for portal seeding, MCP discovery,
private Xvfb launch, sign-in injection and panel access. Additional reusable pure
helpers and the fault proxy live in `gallery_sync_e2e_common.py`.

```sh
cd /home/gauss/lfs-worktrees/galleryam
python3 -B scripts/stress_gallery_sync.py \
  --build-dir /home/gauss/lfs-worktrees/galleryam/build \
  --portal-source /home/gauss/lfs-worktrees/portal-galleryapi \
  --portal-python /home/gauss/lfs-runs/galleryam/portalenv/bin/python \
  --display :94 --cycles 3 --seed 731 \
  --watchdog-seconds 130 \
  --artifacts-root /home/gauss/lfs-runs/galleryam/gallery-sync-stress-artifacts \
  --report /home/gauss/lfs-runs/galleryam/gallery-sync-stress.md
```

The portal Python must already have the checkout's Django and image-processing
dependencies. Xvfb, working Vulkan/CUDA runtime libraries and an unused display
are required. The scenario with two LichtFeld Studio instances also reserves the following display (:95
in this example). Select a different display if either is occupied. MCP and web
ports are allocated on loopback. The supplied binary must include the integrated
native APIs; reports record its path/mtime and the actual plugin source path.

Every scenario receives its own migrated SQLite database, local storage directory,
two paid test users/DesktopSessions and a fresh `LFS_HOME`. LichtFeld Studio connects through
an owned HTTP reverse proxy. The portal starts with
`GALLERY_STUDIO_BUNDLES_ENABLED=1`, local storage enabled and production bucket
settings cleared, following the E2E launcher. The harness starts the looping
`manage.py process_gallery_uploads` worker, and stops it explicitly for watchdog
and retained-part observations. Each scenario stops its producers and removes its
database scenes on exit, including after failure. `--keep` retains the disposable
database/profile/storage for diagnosis; scene rows are still deleted after their
evidence is captured. The user's normal profile and portal are never used.

Use `--scenarios round_trip_cycles,listing_scale` for a subset. All fourteen names
below are accepted, in the specified order. `--cycles` controls round trips and the
rate-limit variant; `--seed` derives independent scenario seeds. A scenario's seed
is unchanged when other scenarios are omitted. Seeded fixture values and the HTTP
fault schedule are reproducible; operating-system timings, UUIDs, ports and the
order of concurrent requests are not deterministic. The proxy rejects one request
in each ten-request block with alternating 429/503 statuses. It does not implement
retries on LichtFeld Studio's behalf. The 429 assertion measures the next same-method,
same-path request against the response's `Retry-After` interval.

| Scenario | Required observations |
| --- | --- |
| `round_trip_cycles` | SOG publish, repeated saved exposure updates and portal view/camera pulls, matching domain tokens, stable scene identity, bounded jobs, Up to date. |
| `web_edits_during_idle` | Press Refresh after title, description, visibility and camera edits to show Portal changes. Poster/presentation changes preserve shared-domain freshness and cause no 409 on the next Update. |
| `conflict_both_sides` | Changed here and on portal; actual resolution dialogs exercise Mine, Portal and camera Both; chosen title, camera frames and journal domain tokens checked. |
| `replaced_elsewhere` | Independent HTTP replacement upload, stale domain write returns 409/currentRevisions, LichtFeld Studio cannot overwrite replacement content, portal-first resolution then successful Update. |
| `kill_during_upload` | Expected SIGKILL at 30–70%, relaunch on the same `LFS_HOME` with sign-in re-injected, paused/Interrupted journal job restored, explicitly press Resume on the same job, retained UploadPart ETags/sizes preserved, fewer bytes sent than the full file, one live scene. Unexpected exits of the relaunched process still fail. |
| `portal_down_mid_transfer` | Portal stopped during both upload and download, Paused (connection lost) in both halves, editor remains responsive, no partial project registration; restart the portal, explicitly press Resume, and verify completion. |
| `slow_processing_watchdog` | Worker withheld for `--watchdog-seconds`; responsive editor, visibly checking or recoverable job, cancel stops waiting and discards upload. |
| `remote_delete_and_recreate` | Soft deletion, Removed on portal, Publish again creates a different scene ID. |
| `account_switch` | A's links/jobs/posters exist first; switching to B clears cards, posters and tray; switching back restores A's links. |
| `two_studios_one_account` | Distinct displays/MCP ports share a profile and portal backend; B owns only its LichtFeld Studio/display processes. B sees and updates A's link; A's stale write leaves B's disk digest unchanged, and refresh loads that digest and link while clearing the stale state. |
| `listing_scale` | Exactly 300 unique remote cards across pagination, measured refresh and editor response under two seconds, second list request returns 304. An explicit one-second proxy hold makes the concurrent responsiveness probe observable. |
| `thumbnail_cache` | Fifty real PNG posters; traverse the Published grid with the panel's scroll-window API until every card is visited, using viewport dimensions rather than a fixed pixel range. Both the proxy and portal log must show exactly 50 initial 200 responses and 50 conditional 304 responses with no image bytes. Cache bytes must fit the configured `posterCacheMiB` bound. |
| `bad_downloads` | Inflated declared download length and truncated stored .licht fail with specific messages, no registration, temporary download/import staging removed. |
| `rate_limit_429` | Round-trip scenario through the seeded fault schedule; successful exchanges and measured Retry-After compliance. |

Revision domain tokens are content hashes. They cannot be ordered numerically or
lexicographically. The harness requires unique metadata tokens for deliberately
distinct round-trip edits, nondecreasing exchange timestamps and exact equality
between the link and current portal domain tokens after each exchange.

For interruption tests, add `--large-fixture /absolute/path/large.licht` to reuse a
valid saved project with more than 24 MiB. Otherwise the harness writes five million
deterministic Gaussian PLY records, loads them using LichtFeld Studio's `lf.io`, and saves a
native `.licht`. The generated file must be at least 200 MB. It does not append
invalid padding to a container. Generation needs substantial memory, GPU memory,
disk space and time. Polling waits default to 120 seconds and are capped at 120
even if `--timeout` is higher; smaller values shorten the waits. Transfer pacing is
4 MiB/s so the kill/down window can be observed without racing localhost speed.

Set `--watchdog-seconds` above the product's configured no-progress bound. The watchdog flags a stuck transfer; further checking requires Resume.

The Markdown report has PASS/FAIL and duration per scenario; each scenario has a
JSON file with timed observations, portal rows, sanitized transfer summaries and
the last 30 lines of each process log. Every FAIL row also includes `message`,
`reason`, cached `jobs` (including their message/reason), the full structured
`last_wait`, and a `log_excerpt` of the last 20 relevant LichtFeld Studio/editor lines with
frame-performance chatter removed. These fields survive a crash or failure while
collecting evidence; startup failures explicitly explain unavailable logs/jobs.
`modal_presses` captures each attempted press's title, body, complete button list,
chosen label and outcome before calling MCP, including Resolve and Cancel. A
second LichtFeld Studio's evidence and diagnostics are included under `second_studio`.
Failure screenshots are captured through
`render_capture_window`; if startup/capture fails, JSON explains why no screenshot
exists. Log and screenshot artifacts survive profile cleanup. Each invocation
creates a unique UTC timestamp/random-suffix directory under `--artifacts-root`
(default: `<report stem>-artifacts` beside the report). Existing roots are accepted;
prior runs' files remain untouched. Reusing the report filename updates its links
to the new run's directory, including JSON links for failures. Any invariant, setup
or cleanup failure exits 1;
later scenarios still run unless interrupted. SIGINT/SIGTERM enter cleanup.

Run the helper tests without LichtFeld Studio or Django:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/lfs-tensor-test-venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/python/test_stress_gallery_sync_script.py \
  tests/python/test_validate_gallery_sync_script.py
```

The pytest interpreter path above is the existing local environment; any Python
with pytest works. Tests cover CLI validation, hash-token semantics, resume byte
accounting, retry timing, seeded fault density, proxy body transformation/log
redaction, report escaping, large MCP snapshot pagination and continued
execution/cleanup after failures. Lane S verification: **38 stress tests and
47 Lane H helper tests passed (85 total)**. `--help` also succeeds. No native
scenario, configure, build, staging or commit was run in this lane.

Lane S implementation handoff: all fourteen scenarios have implementations; native
scenario outcomes remain unverified until the integrated binary is exercised.
Source-review candidates to check with that run:

- `src/python/lfs_plugins/gallery_controller.py:338`: the metadata-only resolution
  branch builds a selected portal view and calls `service.edit`; it does not visibly
  restore that view/camera track to the open local project. The idle web-edit
  camera assertion should expose this if no other completion path applies it.
- `src/python/lfs_plugins/gallery_controller.py:1857`: `combine_camera_tracks`
  rewrites appended `time` fields as `t`; the portal's native camera-path schema
  uses `time`. The Both scenario checks whether the merged path is accepted and
  preserves both tracks.
- `src/python/lfs_plugins/portal_gallery.py:56`: `_request` delegates once to the
  account transport. Retry-After is parsed by the account service, but the gallery
  transport examined here has no general 429/503 retry loop. The proxy scenario
  must fail if recovery never happens or retries are too early.

These are source-review suspicions, not reproduced product bugs. Other lanes may
change the source before execution. No plugin files are edited by this harness.
The repository's broad `*.py` ignore rule ignores both new scripts; the integrating
maintainer must explicitly include them when staging. Lane S does not stage or
change ignore rules.
