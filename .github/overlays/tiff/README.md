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
Master already uses a temporary realloc pointer. Its failure branch still
clears tif_nfields, so keeping the pointer alone does not retain usable fields
or normal cleanup. The prepared upstream change retains that count and adds
the default-field initialization/ordinary/custom-read checks. Its regression
fails against that master and passes with the change; test_open_options and
test_directory pass afterward.

Upstream submission/link is pending; do not treat a prepared patch as a merged
or submitted fix. At the next libtiff bump, remove this overlay only after
verifying all of: preserved pointer AND count on failed field-table growth,
failed default-field setup propagation, and checks in ordinary/custom readers.
Keep the allocation-budget and file/memory regressions when removing it.
