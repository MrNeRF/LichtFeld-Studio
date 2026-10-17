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
