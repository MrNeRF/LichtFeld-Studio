# Depth-window state and locking

`RenderingManager` owns the two panel windows and the legacy global projection.
Each window contains near/far, scale X/Y and offset X/Y. In independent view,
the projection follows the focused panel. Unsynced edits affect one panel;
sync edits copy to both. In independent view, a same-mode unsynced settings
write without SELECTION intent preserves the current projection, including a
live drag preview.

## The three identities

| Identity | Purpose | Lifetime |
|---|---|---|
| Mode epoch | Refuse drag writes and undo entries from an expired mode lifetime. | Changes at depth-relevant mode boundaries and project reset. |
| Drag token | Prevent a replaced or superseded drag from writing, restoring or releasing a slot it no longer owns. | Minted at invoke; never reused. |
| Reference lineage | Tell polling consumers that slot-derived Size references need recovery, including boundaries they did not observe. | Stamped by destructive slot replacement or retained-pair invalidation, with source panel and kind. |

These identities answer different questions. An epoch cannot replace lineage:
entering a mode can expire a drag without destroying a stored Size reference.
The GT parked pair is a restoration snapshot, not an editable third viewport.

## GT retention and global editing

Independent -> GT -> Disabled -> Independent restores the original pair. Repeated
GT entry during the retained Disabled interval keeps that same pair. On GT exit
to Disabled, the live slots follow the authoritative global projection; the saved
pair remains separate. The next Independent entry restores and consumes it.

Project/scene reset, another comparison mode, changed Disabled global geometry,
a changed Disabled drag commit, or an actual Disabled sync change discard the
saved pair. A same-value sync request preserves it. A Disabled sync change checks
drag ownership first, then discards retention before the dormant-pair refusal
guard. A request refused by drag ownership cannot discard the pair.

Compare geometry across all six fields exactly after the writing path's existing
normalization. A drag commit compares with its committed pre-drag backup, not the
preview. Releasing an existing handle within 0.001 screen pixel of its press
restores the exact owned pre-drag state without committing or adding undo.
Equal normalized geometry, enable/viz-only changes and cancelled or
subthreshold drags preserve retention. GT-time global writes preserve it too.

An explicit-panel window setter during a parked GT session returns refusal before
mutation. The Python binding raises for enabled=True and returns without changes
for enabled=False, before its trailing global-enable write. This differs from
the sync setter's silent refusal and actual-flag return. The legacy panel=None
global route and GT entered without a parked pair remain writable.

The Python Size consumer keeps the two earlier reference dimensions alongside a
lineage witness. Valid observed or coalesced GT excursions preserve that witness;
invalidating the native pair stamps `RetainedPairDiscard` under the settings lock,
except project restore, which retains its `ProjectRestore` stamp.
The consumer must invalidate old references on that stamp rather than interpreting
it as a focused-panel copy. It cannot reconstruct user reference history from
before the consumer was mounted.

A synced park keeps the shared Size baseline in the pair-reference entries and
restores shared from them on return. Unreported reference history remains lost:
equal-slot sync ON followed by GT between polls carries no sync-source stamp,
so the consumer cannot distinguish different source panels for that sequence.

## Drag ownership and rollback

Ownership lasts from invoke to destruction, including a subthreshold press.
Preview lasts only while the latch is active. Replacement invokes the incoming
operator before destroying the outgoing one, so ownership counts can overlap.

An owned slot retains its original pre-drag backup across replacement. Claiming
an unowned slot refreshes that backup from live state. Successful commits and
non-drag slot writes supersede the affected backups; teardown restores only
slots still owned by that drag. Rollback applies each owned slot without sync
fan-out, under one settings lock. Closing ownership alone does not discard a
backup needed to remove an abandoned preview at a mode transition.

## Lock order and publication

For operations requiring both, acquire `depth_window_transition_mutex_` before
`settings_mutex_`. Never acquire the transition mutex while holding settings
or history locks. The settings lock protects window state and each atomic
epoch/token check plus mutation. Drag release enters with neither settings nor
history locked, releases settings before pushing history, and keeps transition
locked through latch release and draw publication. Mode changes use the same
transition lock. The sync setter also releases settings before its history
push, but does not acquire transition. This ordering does not make every
viewport operation thread-safe.

`updateSettings` acquires the transition lock only for an actual mode change.
It releases the settings lock first and rechecks after acquisition. Equal-mode
writes must remain reentrant: latch release reapplies current tool settings
while the release sequence already holds the nonrecursive transition lock.

Do not remove this ordering based on the ordinary GUI caller. In
`src/core/event_bridge/event_bridge.cpp`, `EventBridge::emit` copies handlers
under its own mutex, then directly invokes them on the emitter's thread. The
Python GT toggle emits through this bridge; the rendering-settings callback in
`visualizer_impl.cpp` directly calls the manager. A future viewer-thread command
boundary must include these entrances, project restore and release, and define
synchronous returns, reentrancy and shutdown before replacing the lock.

## Consumer and undo snapshots

Native mode snapshots read windows, sync and epoch under one settings lock.
A drag baseline substitutes recorded backups for the slots that token owns.
Undo restores compare epoch and apply the absolute state under the same lock.
Drag undo preserves sync and projects whichever panel is focused at execution.
Sync undo also restores sync, then stamps reference lineage from its caller
under a second settings-lock acquisition. Slot restore and lineage stamp are
not one atomic transaction.

The lineage record reads source, generation and kind together. Consumers must
distinguish copying one surviving window from restoring two absolute windows;
the latter requires recovering each reference from its own slot. Independent
Python reads of mode, focus and windows still need their existing generation
revalidation. That check alone does not close the sync restore-to-stamp interval;
combining that write and exposing a complete native consumer snapshot would be
a separate API change.
