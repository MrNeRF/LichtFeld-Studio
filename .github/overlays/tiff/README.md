# TIFF allocation failure handling

This overlay keeps the baseline libtiff 4.7.1 and its existing feature defaults.
Only the downstream port revision changes. The port and existing patches are
copied from vcpkg baseline c3867e714dd3a51c272826eea77267876517ed99.

The additional patch preserves the old field table if realloc fails, reports
failed default-field initialization, and stops ordinary/custom directory reads
before using a missing field table. Normal pixel, metadata, codec and file
format policies are unchanged. It does not add codecs or dependencies.

TiffMemory.ProviderFailedFieldTableGrowthPreservesExistingFields exercises the
realloc path using the public custom-tag API and a cumulative allocation budget.
It loses the IMAGEWIDTH field with the original provider and preserves it with
this patch. TiffMemory.ProviderAllocationBudgetsFailCleanlyAndRecover checks
bounded provider rejection and subsequent healthy decoding. Linux transient
malloc-failure sweeps additionally exercise encoder/owning/direct/file readers;
the private harness is not linked into production or CI targets.

## Upstream status and removal criteria

Checked against libtiff master
[d23249978a7a99ea2e0752a476611914501577de](https://gitlab.com/libtiff/libtiff/-/commit/d23249978a7a99ea2e0752a476611914501577de).
[!775](https://gitlab.com/libtiff/libtiff/-/merge_requests/775) addressed the
original NULL field-table dereference. [!865](https://gitlab.com/libtiff/libtiff/-/merge_requests/865)
subsequently preserved the old realloc pointer, but its failure path still
clears tif_nfields and makes existing fields inaccessible.

[Upstream !989](https://gitlab.com/libtiff/libtiff/-/merge_requests/989) preserves
that count and adds a public allocation-budget regression. The regression fails
on the unmodified master and passes with the fix. The fork's basic-build-fork
CI passed. This MR is open; it is not an integrated upstream fix.

!989 covers field-table growth only. The default-field initialization and
ordinary/custom directory-read guards in this 4.7.1 backport are not included
in that MR. Their absence has not been demonstrated to cause the original
4.7.1 crash on current master. They remain downstream changes requiring
separate upstream evaluation. Merging !989 alone is not enough to remove the
whole overlay.

At the next libtiff bump, verify preserved pointer AND count on failed field-table
growth, failed default-field setup propagation, and ordinary/custom-reader
handling before dropping the corresponding backport hunks. Remove the overlay
only once all retained hunks are covered or shown unnecessary for the new
provider. Keep the allocation-budget and file/memory regressions.
