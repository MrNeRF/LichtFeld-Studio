# Projects panel, Inspector and Gallery publishing: concept v3 (for owner review)

Date: 2026-09-14. Branch: feat/projects-panel (from master 536c23a60). Author: orchestrator (Claude). Evidence under `.codex_tmp/am_concept/`: four read-only audit lanes (Codex gpt-6-astra: `ui`, `portal`; Grok 4.6: `model`, `inspect`), two research lanes (`research_desktop`, `research_sync`), live screenshots of the owner's catalog (`shots/`), and two critiques of v2 (`critique_grok` folded in; `critique_codex` pending, folded in before P0 starts). Visual mockup at three widths: https://claude.ai/artifact/RYzTKjpma1TuDL5bZvsjNy (source `docs/static/design/projects_panel_mockup.html`).

Owner notes folded in: decide whether a project can be uploaded; inspect a .licht and prune checkpoints while keeping resume-from-checkpoint; import a dataset into the file; the full operation set; update the thumbnail; aligned and styled buttons; every part resizable; no style breaks; gpt-6-astra reviews; a human-voice review of what makes sense and what does not; plain commit messages on a feature branch, no AI attribution.

Nothing here is implemented.

## 0. Straight talk

What makes sense today and stays:

- The single-file `.licht` with two checksummed head slots and append-only saves. It is the best thing in this area and the reason an in-app Inspector is worth building.
- Identity by container id and freshness by commit id and revision tokens, never clocks. That logic is right; it is the presentation that is wrong.
- Thumbnails, drag a card to the viewport, keyboard navigation, the shared confirm and context-menu machinery, the pure geometry module `asset_layout.py` as a test seam.
- The guarded portal updates and the durable, resumable transfer journal. Nobody needs to rewrite the transport.

What is nonsense:

- A panel called Asset Manager whose only content is projects, with a primary button "Open Project..." that does not open anything.
- Seventeen badge words for one cloud icon, plus a separate eight-word local status, plus a "Needs attention" bucket that the panel opens on by default, empty.
- The sidebar stacked above the results and sized to 40 percent of the height, so a 980 by 620 floating window shows one row of 90 px thumbnail stubs.
- An Info panel that spends its 220 px on a stretched thumbnail and then lists Folder, Size, Path, Created, Modified. The file knows its iteration, gaussian count, dataset, save history and how much of it is dead bytes; the panel tells you none of it.
- "Remove" next to "Delete from portal..." in the same menu; "Up to date" on a viewing copy that is neither a backup nor the same file.
- A CLI inspection tool that ignores head checksums and crashes on a truncated file while the app's own reader classifies it correctly.

What feels superficial and cumbersome:

- Every list row carries a button; every card carries 60 px of empty body. Buttons that are always visible stop being actions and become noise.
- A cycling "Sort by: Name" button on its own toolbar row; a "Gallery checked." toast that never leaves; "Update all (0)".
- The Transfers window: a second floating window to say "No transfers yet." with three buttons.
- Recovering from anything: Locate lives in a region that is 0 dp tall at common heights; a missing file offers Publish; a corrupt file offers Refresh; an overwritten file needed three manual steps and a session of explanation to heal.
- The same project has four names (card, chooser, recent menu, gallery title) and three states depending on where you look.

And in my own v2, killed by the critique: replacing Open with Resume training when a checkpoint exists (a first-time user would restart training when they wanted to look), an overwrite dialog with three equal buttons where one destroys the only remaining published copy, two card columns claimed at 340 dp with a default card width that yields one, "Update gallery" next to "Check gallery", and an "Unsupported" bucket that hides three different problems with three different fixes.

## 1. The short version

1. **Projects**, not Asset Manager. A .licht file is a project; watched disk folders are Folders; the portal is the Gallery; a project's published copy is a published scene. "Asset", "portal", "pull", "splat" and "scene" (for local files) leave the panel.
2. Layout rebuilt around the browser: navigator beside the results, a thumbnail size slider, cards that end where their text ends, an Inspector on the right when wide and a band or strip when narrow, a Transfers tray in the footer, a floating window that works. Every region resizes with a visible handle and remembers its size.
3. State is three independent facts: file health (with a fix each), gallery relationship, activity. "Needs attention" is a count.
4. The Inspector is the .licht operations hub: facts in three cost tiers and every operation on a file in one place, including resume from an older checkpoint, reduce size (prune history, drop an embedded dataset, compact), embed dataset, export as a format, update the thumbnail. The CLI tool becomes a client of the same native reader.
5. The Gallery is a publishing destination for viewing copies, not a backup and not a sync peer. The UI says so at every step that matters.
6. Identity: entry keyed by the container id with a unique path; overwrite at the same path is a visible state with a safe default; rename and move relocate; missing files heal by Locate or by a scan; the gallery badge is never persisted in the catalog again.
7. Visual quality is a contract with a gate reviewed by gpt-6-astra on every phase.

## 2. What is wrong today (evidence)

| Fact | Value | Source |
| --- | --- | --- |
| Strings owned by this surface | 250 | `en.json`; ui report C |
| Competing label choices for seven concept clusters | 27 | ui report C |
| Concepts a user must connect for one local-to-gallery round trip | 24 | ui report C |
| Visible sync states on a card | 17 | `gallery_controller.py:1595-1674` |
| Local file statuses, of which distinct labels | 8, three share "Unverified" | `asset_index.py:409-421`; ui report B.4 |
| Stores that must agree | `library.json` (with a persisted gallery badge), `scan_cache.json`, `gallery/sync.json`, posters | model report A |
| Sidebar above the results at every width, 40 percent of height | no breakpoint makes it a rail | `asset_manager.rcss:266`; `asset_layout.py:18` |
| Info panel allocation at 600 dp panel height, at 400 | 23 dp, 0 dp | ui report E |
| Grid column arithmetic | gaps not subtracted, native px compared with dp | ui report E, finding 3 |
| Gallery listing refresh | first-page ETag; edits past scene 100 stay hidden | portal report E2 |
| Reclaimable bytes in the owner's files | garden 49.9 percent (355 MB), bicycle 28.9, stump 20.6; nobody is told | inspect report B |

Screenshots: `.codex_tmp/am_concept/shots/`, notes in `shots/observations.md`. The 24 ranked usability findings are in `ui/report.md` section F; the ones that shaped this concept are cited inline below as "ui N".

## 3. Principles

1. One noun per thing.
2. Three facts, not one word: file health, gallery relationship, activity, each in its own place.
3. Every problem carries its fix, reachable from the context menu as well as the Inspector (ui 1).
4. Narrow collapses structure, it does not shrink it. Open stays one click away at every width.
5. Cheap facts first, never on the UI thread. Cards render from the cached Library record.
6. Manual gallery contact with a visible clock.
7. One action table computed from state, used by card, row, menu, Inspector and tray (ui 10).
8. One component, shared with the startup chooser and the Gallery scope; the startup chooser is rebound in the same phase, not "later".
9. Safe defaults on every destructive choice; the destructive button is red and names what cannot be undone.
10. The design contract applies (`docs/docs/development/ui-design-language.md`) plus section 11 here.

## 4. Vocabulary

Panel label **Projects** (panel id `lfs.asset_manager` unchanged). The navigator has no "Library" heading; the rows are All projects and Recent. "Library" survives only inside "Remove from Library" and "Add existing..." wording (critique G-vocab: two nouns for one place is one too many).

| Concept | UI word | Never | Notes |
| --- | --- | --- | --- |
| A .licht file | project | asset, file, scene, splat | "file" only in Show in folder, Move file to trash, the File section |
| A watched folder | folder | asset folder, location | Add folder..., Remove folder from Library |
| The portal | Gallery | portal, cloud, remote | Status chip becomes "Janusch, Gallery" in the same phase (G19) |
| A project's published copy | published scene | scene, splat, upload | In dialogs and the Inspector; cards use the short state labels |
| Portal collection | (not shown) | project | Portal renames its tab to Collections (decision 8) |
| Register an existing file | Add existing... | import, open, add project | Picker title "Choose an existing .licht" (G9) |
| Load into the viewport | Open | load, open in application | Always the primary; never replaced (G1) |
| First upload | Publish... | upload, push, sync | |
| Replace the published copy | Publish changes | update gallery, sync, push | Same family as Publish; no collision with Check gallery (G8) |
| Get a copy back as a new local project | Download as new project | pull | Dialog first line: "A viewing copy without training history. A new file on this machine." (G7) |
| Bring gallery edits into the local project | Apply gallery changes... | pull, apply portal changes | Title, visibility, view settings; scene content defaults to Mine when the local file has a checkpoint (G14) |
| Refresh gallery state | Check gallery | refresh, verify, sync | Button plus "checked 5 min ago" |
| Discover new or changed files | Rescan folders | refresh, scan | |
| Forget a project, keep the file | Remove from Library | remove, delete | |
| Delete the file | Move file to trash... | delete | New (decision 4), red |
| Delete the published scene | Delete published scene... | remove from gallery, delete from portal, unpublish | Portal says Delete scene; same word (G-vocab) |
| Break the link, keep both | Unlink | detach | Overflow only |
| Details region | Inspector | Info | |
| Container save points | save (Save 3 of 3), Save history | version, commit, generation | "version" is overloaded by portal revisions and app versions (G15); "generation" only in the File section |
| A saved training state | checkpoint | resume file | |
| Image on the card | thumbnail | preview, poster, cover | Portal keeps "cover" |
| Attention items | Needs attention (count) | | Filter chip and badge |
| Account | signed in as name | connected, portal | |

Locale keys `asset_manager.*` and `asset_manager.gallery.*` become `projects.*` in ten languages with real translations; `account.error.*` strings that say "portal" are rewritten in the same pass; the portal's `native_upload_help.html` changes in the same phase as Publish. Format names stay English.

Search matches display name, file stem, parent folder name, full path and the published title (critique D).

## 5. Information architecture

```
+------------------------------------------------------------------------------+
| [Search projects...........] [Filter v] [Sort v] [grid|list] [o----]         |
| [Check gallery  checked 5 min ago]                    [Add existing...] [...]|
+---------------+----------------------------------------------+---------------+
|  All projects | browser: virtualized grid or list            | INSPECTOR     |
|  Recent       |                                              |  bicycle      |
| FOLDERS    +  |  [thumb] [thumb] [thumb] [thumb]             |  mrnf/bicycle |
|  mrnf      3  |  name    name    name    name                |  [Open] [Resume training] [Publish changes] [...]
|  projects  3  |  502 MB  ...                                 |  Problem      |
| GALLERY       |  cloud^  ok      !missing                    |  Project ...  |
|  Published 7  |                                              |  Gallery ...  |
|  Janusch      |                                              |  File ...     |
|               |                                              |  Operations   |
+---------------+----------------------------------------------+---------------+
| Uploading garden 43 percent, 12 MB/s, 1 queued        [Pause] [Show all]    |
+------------------------------------------------------------------------------+
```

1. **Toolbar**, one row from 640 dp, two rows below: search (fills), Filter (Needs attention, Not published, Published, Gallery only, Missing files, Has checkpoint, Has dataset), Sort (Name, Saved, Opened, Size, Iteration once cached, Published), grid/list, size slider (grid only), Check gallery with time, Add existing..., overflow (Add folder..., Rescan folders, Projects settings...). Below 640 dp Filter, Sort and the slider fold into one View menu; Check gallery becomes an icon with the time in its tooltip; Add existing... stays visible as an icon button with a tooltip, never only in the overflow (critique person 1).
2. **Navigator**, a left column from 640 dp (120 to 240, resizable), a scope dropdown in the toolbar below. Rows are real focusable controls (ui 16). All projects, Recent (last 10 by real open time, ui 19); Folders with count and kebab (Rescan, Show in folder, Remove folder from Library, Clean up missing entries n); Gallery: Published, the account line. Attention counts as amber badges on All projects and Published. Add folder... asks "This folder only" or "Include subfolders" (critique person 2).
3. **Browser**: grid or list; multi-select; keyboard; drag a card to the viewport opens it; drop a .licht onto the panel adds it (decision 7) with hover copy "Add to Library"; Space opens Quick look. Batch actions act on the visible selection only (ui 6).
4. **Inspector**: right column from 900 dp; bottom band 640 to 899; strip below 640 that hosts a real Open (or the state verb) button, so Open is never only inside an overlay (G5). Sections in 7.3.
5. **Transfers tray**: 32 dp footer only while a transfer is queued, running, paused, interrupted, applying or failed; expands into the list; rows link to their project and offer the same verbs as the card (ui 9). The separate Transfers window is removed.

Explicit states: no folders yet (Add folder..., Add existing..., File > Open also opens a project directly, drop a .licht here); folder with no projects; no search matches with Clear; signed out (one line plus Sign in); offline (last check time, cached); scanning (count and Stop); verifying (its own count and Stop, ui 15); first paint of an unverified entry shows "Reading..." rather than an openable-looking card (critique state 6).

## 6. Layout, sizing and resizing (dp)

Breakpoints on the panel content width W: compact below 420, narrow 420 to 639, medium 640 to 899, wide 900 and above. Breakpoints, thumbnail heights and column counts are computed in `asset_layout.py` and applied as a root class on the content element, because RCSS in this tree has no media queries, grid, aspect-ratio or sticky (critique layout section).

| Region | compact | narrow | medium | wide |
| --- | --- | --- | --- | --- |
| Toolbar | 2 rows | 2 rows | 1 row | 1 row |
| Navigator | dropdown | dropdown | column 160 (120 to 240) | column 200 (160 to 240) |
| Inspector | strip 32 with Open button, overlay 200 as a sibling of the scroll region | strip, band 180 | band 200 (120 to H/2) | column 280 (240 to 420) |
| Tray | 32 when active | 32 | 32 | 32 |
| Default card width | 112 | 136 | 168 | 168 |
| Grid columns at the default | 2 at 260 and up | 2 to 3 | 2 to 4 | 3 and up |

Resizable regions, each with an 8 dp visible handle and a 12 dp hit target (theme minimum grab, `components.rcss:99`; G12), hover highlight, min and max, a remembered value, double-click to reset: navigator width, Inspector width or height, tray expanded height (120 to H/2), list column widths, thumbnail size (slider 112 to 320, persisted per breakpoint class).

Cards: columns = floor((browser width minus 24) divided by (card + 12)), gap subtracted per column (ui 3); cards stretch up to 15 percent to fill the row; thumbnail 16:10 with height computed at runtime; card height = thumbnail + 40 (name line 16, meta line 16, padding 8; on the 8 dp grid, G18); name 12 dp, 14 dp only on the selected card (G12). All arithmetic in dp; native scroll offsets are divided by the UI scale before comparison (ui E).

List: row 48 with a 32 by 20 thumbnail and two text lines (name; gallery label), or row 40 with one line when the gallery column is visible; columns Thumb 32, Name flex (min 80 at compact, 120 otherwise), Gallery 128 (24 dp glyph only below 480), Size 72 (hidden below 360, G4), Saved 96 (hidden below 560), Folder 100 (hidden below 700). No buttons in rows. The header is split out of the scroll region in RML so it stays fixed (no sticky in RmlUi).

Inspector band: thumbnail 160 by 100 left, facts in two columns, action row on top. Column: 16:10 thumbnail, then sections. Never a full-width thumbnail that hides the facts.

Floating window: default 1100 by 700, minimum 480 by 360 (compact rules apply there, with Open on the strip). Docked minimum 260.

Control heights: 24 dp in toolbars, cards and tray; 28 dp in dialogs and the Inspector header. The 24 dp variant is added to `components.rcss` as a shared class, not to the panel (critique contradiction on `.btn`).

## 7. Card, row, Inspector and states

### 7.1 Card

```
+-----------------------+
| thumbnail 16:10       |  "!" top-left only with a file problem
|                   [o] |  activity ring top-right only while a transfer runs
+-----------------------+
| bicycle               |  name 12 dp, ellipsis; full path and any copy path in the tooltip
| 502 MB  cloud^ Changes here |  size, then the gallery glyph and short label
+-----------------------+
```

Name rule: the user's display name if set; else the file stem; except that `project.licht` resolves to its parent folder name. Decision 10 adds an optional PROJ `title` so a name travels with the file (PROJ has no name today).

Kebab visible on the selected card and on hover elsewhere (G20); the card is the drag handle; click selects; double click opens. On compact the selected card additionally shows a 24 dp Open button in its meta row (G5).

### 7.2 The three facts

File health (one fix each):

| State | Meaning | Fix |
| --- | --- | --- |
| Available | master, openable, id matches | none |
| Reading | not inspected yet in this session | none; card is not offered as openable until read |
| Missing | path gone | Locate..., or Use found location when a scan found the same id elsewhere; folder kebab Clean up missing entries |
| Replaced a published project | a new project id at the path of an entry that had a published scene; stable until the user answers | the question in 8.2 |
| Unreadable | stat ok, open failed | Verify (shows the diagnostic), Show in folder |
| Needs repair | no fully valid head (RepairOnly) | Repair... (decision 11) or, until it exists, "Cannot be repaired by this version"; Versions... may still list older saves |
| Saved by a newer version | min reader version above this app | "Update LichtFeld Studio" |

Autosave sidecars are never Library entries; a master whose sidecar is newer shows "Autosave newer than the file" in the File section with Recover.

Gallery relationship (eight labels; glyphs reuse `gallery-cloud*.png`):

| Label | Glyph | Meaning | Primary verb |
| --- | --- | --- | --- |
| Not published | none | no link | Publish... |
| Published | cloud-check | linked, content and metadata match | Open in Gallery |
| Changes here | cloud-up | local save newer than the published copy | Publish changes |
| Changes in gallery | cloud-down | title, visibility or view settings changed on the portal | Apply gallery changes... |
| Cover or story changed | cloud-check with a dot | presentation revision changed; nothing to apply locally | Open in Gallery (G6) |
| Conflict | cloud-updown, warning | both changed, or a transfer conflict | Resolve... |
| Gallery only | cloud, info | no local project for this published scene | Download as new project |
| Removed from gallery | cloud-strike, dim | link kept, scene gone | Publish again... |

Before the first check after sign-in the last known label is shown greyed with the check time; a project with a journal link is never shown as Not published (critique state 6.2 hole).

Activity (replaces the glyph while active; the tray text carries the verb, "Publishing changes" or "Publishing", so the card never has to): Preparing, Uploading 43 percent, Processing (with "Keep waiting" after the timeout, never auto-Failed), Downloading 43 percent, Applying (no dismiss), Paused (Resume), Interrupted (Resume, "may restart from the beginning", G11), Failed (reason, Retry).

Precedence: a file problem hides gallery verbs (ui 4); activity hides the relationship glyph; Conflict wins over ahead or behind.

Needs attention count = Conflict + Failed + Interrupted + Missing + Replaced a published project + Removed from gallery + Unreadable + Needs repair. "Saved by a newer version" is not counted (nothing to fix in the panel). A damaged journal ("Gallery data on this machine needs recovery", model report storage_issue) is a panel-level notice with Open recovery folder, not a per-card state.

### 7.3 Inspector

Header: name (editable), folder and file name, buttons: Open (always, primary), Resume training (`btn--success`, when a resumable checkpoint exists and the file is not a downloaded viewing copy), the gallery verb for the state, overflow.

Sections, collapsible, remembered; empty rows are hidden (G17):

- **Problem** (first, when a flag is set): message and the fix button.
- **Project**: Saved (date, Save 3 of 3), Opened, Training (iteration, strategy, resumable or converted to edit mode), Model (gaussians, SH degree, from the checkpoint header), Dataset (external path with reachable or missing, or "embedded, 391 images, complete"; the dataset node name), Metrics (only when METR has samples), License (only when set).
- **Gallery**: relationship label and reason; "Published as SOG, 34 MB, 2026-09-01, a viewing copy"; Title and Visibility (editable, decision 5); blocking reasons only when publishing is blocked ("No visible splats", "Dataset not embedded", "Not signed in", "Format not supported by this Gallery"; the remaining-space check waits for usedBytes on /me, portal report A3); Open in Gallery, Copy link; Delete published scene..., Unlink (overflow). One line: "Cover, highlights and share links are edited in the Gallery."
- **File**: path (selectable, Show in folder; copies at other paths listed), size, saves and reclaimable bytes ("2 saves, 49.9 percent reclaimable"), autosave newer than the file (Recover), Verify with its last result; writer version and format in the overflow.
- **Operations**: collapsed by default unless a Problem is set or reclaimable bytes exceed 10 percent: Versions..., Reduce size..., Embed dataset, Locate dataset..., Export as..., Update thumbnail, Set license.

Multi-selection: count, total size, batch actions that apply to every selected project (Publish n, Publish changes n, Remove from Library n, Move to trash n); remote-only selections count correctly (ui D).

### 7.4 Context menu

Open; Resume training (when applicable); Quick look; Publish... / Publish changes / Apply gallery changes... / Download as new project / Resolve...; Open in Gallery; Copy link; Versions...; Export as...; Rename; Show in folder; Locate... (when missing); separator; Remove from Library; Move file to trash...; Delete published scene...; Unlink.

### 7.5 Conflict dialog

One dialog listing the differing parts (title and description, visibility, view settings, camera track, scene content) with both values visible (ui 8) and Mine / Gallery / Both where Both exists (camera tracks). Scene content defaults to Mine when the local file holds a checkpoint and the incoming copy does not, with the warning "The Gallery copy has no training history" (G14). Apply, then Publish changes. The local project is backed up first.

## 8. Model: identity, stores, healing

### 8.1 Entry and stores

- Library entry key: the container's `project_uuid`; `path` is a unique locator. Persisted: id, path, display name (user override), folder id, last-good inspection facts for instant paint (including iteration, so Sort by Iteration never decodes a checkpoint on the UI path). Derived on read: exists, health, open state, role, size and mtime. The gallery badge is not persisted (v5 `projects[].gallery` is ignored on load and dropped on save); the greyed last-known label is derived from the journal link until the first check.
- Stores stay separate: `library.json` device-local and usable signed out; `gallery/sync.json` account-scoped and lock-fenced, links keyed by project id. The panel joins them in memory.
- No clocks for freshness. The scan cache keeps its 2 second guard. Symlinked `.licht` files are listed; path keys fold case on case-insensitive filesystems.
- Every store write goes through the panel worker; the UI reads snapshots.
- Removing a folder from the Library removes its entries and says so ("N projects leave the Library. Files on disk and published scenes stay."); journal links are never dropped by a folder removal (G13).
- File > Open and Save As register the file when it lies inside a Library folder (as the current write poll does); Save As with a new id creates a new entry.

### 8.2 Reconciliation rules

- Same id at a new path while the catalog path is missing or holds another id: relocate silently; the display name and link follow.
- New id at the catalog path, no published scene on the old entry: replace the entry; the display name follows.
- New id at the catalog path, old entry had a published scene: replace the entry, keep the old link attached to it as "previous published scene", set the stable state "Replaced a published project", count it in Needs attention, and ask once in the Inspector Problem section (never as a modal on scan):
  - **Keep both** (default): this project starts Not published; the previous published scene stays in the Gallery and appears as Gallery only with the note "previously published from this path".
  - **Replace the published scene...** (red): "Viewers of the current link will see this run. The previous published copy cannot be restored." Uses the portal replacement upload; keeps title, link and visibility.
  - Unlink the previous scene (overflow).
  Locate is not offered, because the path exists (G2). The question stays available in the Inspector until answered; the state clears when the user answers, publishes, or deletes the previous scene.
- Same id at two paths: one entry; the other path listed under File and in the card tooltip.
- Missing: keep the last facts, flag Missing; heal by Locate, by a later scan, or by Clean up missing entries.
- Migration: v3 and v4 load as today; v5 drops the gallery projection on the next save; `sync.json` v2 unchanged. One-shot reconciliation on first load runs in the worker with the scan status line visible, never silently while the panel is closed.

## 9. Operations

Every operation on a project, its placement, and what exists today (inspect report C).

| Operation | Placement | Today | Work |
| --- | --- | --- | --- |
| Open | double click, Enter, drag to viewport, header, menu, compact strip button | exists | none |
| Add existing... | toolbar, drop onto the panel | exists, mislabeled | relabel, drop target with hover copy |
| Add, remove, rescan folders | navigator | exists | keyboard rows, subfolder choice, safe removal wording |
| Resume training | header button when resumable | through open plus the Training panel; CLI `--resume` | one-click path; disabled on downloaded viewing copies |
| Resume from an older checkpoint | Versions...: pick a save that holds a checkpoint, "Open as new project" or "Resume from here" | restore only in `tools/inspect_licht.py --generation` | native restore binding plus resume |
| Versions (Save history) | dialog: save number, kind, date, iteration, bytes added; restore to a new file | CLI only | binding, dialog |
| Reduce size (prune) | dialog: what occupies the file (checkpoint history, embedded dataset, reclaimable bytes) with checkboxes: keep only the latest checkpoint, drop the embedded dataset when the source folder is reachable (decision 12), compact; shows the resulting size | compaction exists (File > Compact, idle auto at 50 percent dead); pruning live history does not | writer support to tombstone selected historical checkpoint and dataset chapters, then compact; never drops the resumable checkpoint or the only dataset copy |
| Compact | inside Reduce size and as its own button above 10 percent reclaimable | exists | expose reclaimable bytes |
| Embed dataset | Operations, when the dataset is external | exists for the open project | closed projects too, progress in the tray |
| Locate dataset | Problem or Operations when REFS is unreachable | API only | dialog, save |
| Export as... | Operations and menu: PLY, SOG, SSOG, SPZ, from a closed project | Export panel for the open scene; publication preparation already extracts visible splats from a closed file | reuse that path for a local export |
| Update thumbnail | Operations: from the current viewport (when open), from the first dataset image, from an image file | capture and writer support exist | one operation writing THMB and the head locator |
| Verify | File section | reader has `verify_all`; not exposed | binding, progress, result |
| Recover autosave | File section when a newer sidecar exists | startup and open prompt | same disposition; the lock-taking recovery inspection is never called from a card (inspect report E) |
| Repair | Problem section | does not exist | decision 11 |
| Set license | Operations | API only | inline editor |
| Rename | header, menu | catalog alias only | PROJ title when decision 10 is taken |
| Move file to trash | menu, overflow | none | new, red, confirmation names the path |
| Publishing block reasons | Gallery section | scattered errors after clicking | derived from the inspection |
| Publish, Publish changes, Apply gallery changes, Download as new project, Open in Gallery, Copy link, Delete published scene, Unlink, Publish again | per state | exist under other names | rename, action table |
| Check gallery | toolbar | exists | one button, visible time, no conditional shortcut until the portal ETag is fixed (G16 test: a title change on scene 101 is visible after a check) |
| Quick look | Space | none | large thumbnail plus cached facts |
| Batch | multi-select | Publish n, Update n | Remove n, Trash n; visible selection only |
| Keyboard | Enter open, Ctrl+Enter gallery verb, Delete remove (confirm), F5 rescan, Ctrl+F search, Space quick look | most exist | complete and document |

Native surface (inspect report E, model report F.7; keeps `inspect_project` for compatibility):

- `inspect_project_card(path)`: ids, save number, times, size, role, open state including RepairOnly without throwing, has_preview, min reader version, diagnostic. A new open path that does not validate every live chunk header (391 embedded-image headers cost 2.9 ms warm and random IO cold); target under 1 ms.
- `inspect_project_details(path)`: storage stats, save history (kind, date, bytes added, checkpoint iteration), chapter table, PROJ and license, scene-graph summary, parameters summary, references with reachability, checkpoint header by logical prefix with a byte budget, metrics, sidecar presence by stat. Tens of ms.
- `inspect_project_preview(path)`: thumbnail bytes.
- Operations: `restore_save`, `prune_project`, `compact_project` (exists), `verify_project`, `set_preview`, `relink_dataset`, `export_project_as`.
- Rules: GIL released; worker thread; results cached on the entry keyed by size, mtime and commit id; a card never triggers a read; a file open for writing by another instance is shown as "In use by another LichtFeld Studio".

`tools/inspect_licht.py` keeps its output but reads through the binding (or a `--inspect` flag, decision 6); today it ignores head checksums, walks the whole file and crashes on a truncated tail while the app's reader returns RepairOnly (inspect report D).

## 10. Gallery publishing

The portal stores prepared viewing copies: one save, visible splats embedded, no checkpoint, no history, new container ids; it records no origin project; it is not a backup and not a multi-device sync of the master (portal report B1, E4, E5). The UI says so: empty state "Publish a viewing copy of a project to your Gallery"; Publish dialog "A prepared copy is uploaded. Your project file and its history stay on this machine"; Download dialog "A viewing copy without training history. A new file on this machine". "Keep .licht as saved" becomes "Keep the saved splat encoding".

- Check gallery: on demand, on panel open when signed in (decision 3), after the user's own transfer; never in the background; a full listing walk until the portal ETag is fixed.
- Publish...: title (default project name), description, visibility, format with a size estimate; disabled with the blocking reason; Sign in inline when signed out (ui 13).
- Publish changes: for Changes here. A downloaded viewing copy offers "Publish as a new scene" first and "Publish changes" second (critique person 4).
- Apply gallery changes...: for Changes in gallery; lists what changes; backup kept; Undo lives in the tray history until the next action (ui 12).
- Download as new project: for Gallery only; "Download and open" second.
- Conflict: 7.5.
- Errors inline with cause and one action; every failed user action shows a notice (ui 14).
- Portal follow-ups (portal report F): listing ETag over the whole owner state or a change feed; origin project and commit ids and timestamps on scenes; usedBytes on /me; cover API; share-link lifecycle; collections API; Collections rename; help text.

## 11. Visual quality contract (owner note)

1. Shared components only; the 24 dp toolbar variant is added to `components.rcss`.
2. One control height per row: 24 dp in toolbars, cards and tray; 28 dp in dialogs and the Inspector header; icon buttons square; 8 dp side padding; labels never wrap (ellipsis with tooltip).
3. 8 dp spacing grid, 4 dp inside a control; section padding 12; gutters 12; navigator rows 24; list rows 40 or 48.
4. Baselines align across a row; numbers in JetBrains Mono, tabular, right aligned; labels ellipsize, never clip.
5. dp everywhere in RCSS; runtime geometry converts native pixels with the UI scale.
6. Every region has a handle that looks like one; nothing else looks like a handle (ui 18).
7. Icon-only controls carry a tooltip and an accessible label (ui 17); every state has a word next to its color.
8. All ten locales fit at every breakpoint (longest translation checked).
9. No inline static styles in RML; palette in the theme file; correct in every registered theme.

Gate on every phase, run by Claude, reviewed by gpt-6-astra (max):

- Screenshots at 260, 320, 500, 760 and 1100 dp by 400, 600 and 900 dp, docked and floating, dark and light theme, English and German, UI scale 1.0 and 1.5.
- A geometry dump from the live document (bounds of every toolbar control, navigator row, card, list column, Inspector row), checked by a script: alignment within 1 dp, no overflow, no overlap, one control height per row.
- Zero error or warning lines in the log while driving every scope, view mode and breakpoint.
- The astra review report names every misalignment with a screenshot crop and a file:line; the phase does not close with an open item.

## 12. Phases and lane protocol

Branch `feat/projects-panel` from master. Commits in plain words, one change per commit, no AI attribution, no trailers. Each phase: frozen spec (this document plus the lane evidence), one implementation lane (Codex gpt-5.6-luna high, or Grok 4.6 xhigh, own worktree and build dir), one review lane (gpt-6-astra max against the spec and the gate), Claude verification: build, `pytest tests/python/test_asset_*.py tests/python/test_gallery_*.py`, `tools/check_locale_completeness.py`, the gate, an app run on a copy of the owner's catalog. Estimates from the Grok critique.

| Phase | Content | Lane-days | Riskiest seam |
| --- | --- | --- | --- |
| P0 bugs (two parallel lanes) | Published row click; first-open scope; sticky "Gallery checked."; "Update all (0)" visibility and scope; full-panel Info thumbnail; grid arithmetic and px/dp; missing file offering Publish; log-only failures; 8 second Undo; list gallery column truncation; focusable navigator rows | 4 | `asset_gallery_ui.py`, `asset_layout.py` |
| P1 layout and words | navigator column and dropdown, breakpoints, resizable regions, slider, card and row anatomy, Inspector placement with Open on the strip, strings in ten locales plus the status chip and account errors, preferences page, startup chooser and Open Recent rebound to the shared name rule (path-based `project.licht` rule; PROJ title waits for decision 10) | 10 | `asset_manager.rml`, `asset_manager.rcss`, `asset_layout.py`, locales |
| P2 model | entry shape, reconciliation rules with the Replaced state and question, gallery projection removed, cached facts including iteration, folder removal contract, migration in the worker, tests on the owner's catalog copy and the four real files, symlinks and case folding | 8 | `asset_index.py`, `asset_watch.py` |
| P3 Inspector and operations | card and details inspection with the cheap open path, storage stats, Save history, restore, prune, verify, thumbnail, dataset embed and relink, export from a closed project, license, Inspector sections, Quick look, CLI on the binding | 14 | `project_container.cpp` cheap open; the prune writer (ships last; Inspector can ship read-only plus Compact) |
| P4 publishing | action table, Check gallery without the shortcut, tray with Interrupted and Applying, conflict dialog with values and the checkpoint default, inline errors, Apply gallery changes, Download as new project, blocking reasons, Transfers window removal, the scene-101 test | 9 | `gallery_controller.py` |
| P5 portal | listing ETag or change feed, origin ids and times, usedBytes, cover API, share links, Collections rename, help text; then desktop exposure of visibility, cover and collections | 12 plus 4 | portal `api.py`, `sync.py` |

Order: P0, then P1 and P2 in parallel worktrees (P1 does not touch the model; P2 does not touch RML), then P3 and P4, then P5. The Codex critique of v2 (`critique_codex/report.md`) is folded into the P0 to P2 specs before those lanes start.

## 13. Decisions for the owner

1. Panel name: "Projects" (recommended) or keep "Asset Manager".
2. Inspector default placement: right column when wide, band otherwise (recommended); tear-off Inspector window later.
3. Check gallery on panel open while signed in (recommended yes; no background polling).
4. "Move file to trash..." in the panel (recommended yes).
5. Edit title, description and visibility of the published scene from the Inspector (recommended yes).
6. CLI tool on the native binding (recommended) or a `--inspect` flag.
7. Dropping a .licht from outside a known folder: add to the Library (recommended) or only open.
8. Portal renames its "Projects" tab to "Collections".
9. Thumbnail slider 112 to 320, defaults 112 / 136 / 168 by breakpoint.
10. Optional `title` in the PROJ chapter (format 1.2, additive) so a name travels with the file (recommended).
11. Repair for RepairOnly files (rebuild the head from the newest valid save): in P3 (recommended) or later.
12. Prune may drop the embedded dataset only when the source folder is reachable (recommended), or always with a warning.

## 14. Non-goals

Background sync, multi-account, a master-file backup service on the portal, managing non-.licht files in the Library, a second catalog format, a native C++ rewrite of the panel.

## Appendix A: mapping from today's keys

| Today | Concept |
| --- | --- |
| unlinked "Not published" | Not published |
| equal "Up to date" | Published |
| local "Saved since last publish" | Changes here |
| remote "Portal changes" | Changes in gallery |
| (presentation revision, ignored today) | Cover or story changed |
| diverged "Changed here and on portal" | Conflict |
| remote_only "In gallery only" | Gallery only |
| local_missing "Local file missing" | Missing (file health) on the linked entry; the entry stays until cleaned up |
| remote_deleted "Removed on portal" | Removed from gallery |
| unknown "Not checked" | greyed last-known label with the check time |
| queued, paused, interrupted | Waiting, Paused, Interrupted |
| uploading, processing, downloading, preparing, applying | activity with progress; Applying shown |
| error "Needs attention" | Failed with reason; counted |
| UNVERIFIED | Reading |
| UNREADABLE | Unreadable |
| REPAIR_ONLY | Needs repair |
| UNSUPPORTED_NEWER | Saved by a newer version |
| UNSUPPORTED (non-master) | never listed |
| IDENTITY_MISMATCH | Replaced a published project, or a plain replacement when no scene was published |
| MISSING | Missing |
