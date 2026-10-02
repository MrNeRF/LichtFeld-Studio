# Selection depth window

Each 3D view has one selection depth window. Its near/far range, scale, offset,
and transform are stored in `ViewSettings`. Comparison panels share that window.
GT comparison suspends filtering without changing the stored window.

A drag owns a monotonic token and a pre-drag snapshot. Replacement drags retain
the original snapshot; explicit settings writes revoke ownership. Preview,
cancellation, and commit check both the owner token and the mode epoch before
writing. This prevents stale modal teardown from overwriting a newer edit.

A successful commit records one absolute snapshot for undo and redo. Project
replacement expires old snapshots. GT boundaries cancel the active drag before
changing the mode and advance the epoch to reject any surviving preview write.
