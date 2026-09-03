# Offline video reconstruction contract

Video export stores a backend-neutral reconstruction selection independently
from the interactive viewport:

- `backend_id`: stable backend identifier;
- `preset_id`: stable preset identifier;
- `fallback`: either `abort` or `native`.

The persisted object has its own schema version. Projects that do not contain
the object are migrated to `native`/`native` with the `abort` policy, which is
the behavior of video export before reconstruction selection was introduced.
Numeric vendor quality values and input scales are not persisted. Those values
belong to a backend's read-only catalog descriptor and are resolved when an
export starts.

## Resolution boundary

`VideoReconstructionCatalog` is metadata-only. Looking up a descriptor must not
load plugin code, create GPU objects, or probe a device. The resolver snapshots
that metadata into an immutable `VideoReconstructionPlan` before rendering the
first frame. The plan records:

- requested and effective backend and preset identifiers;
- requested and effective provider version and digest;
- output and internal render extents;
- projection and required resource flags;
- fallback policy, resolution issue, and unavailability reason.

`native` is supplied by the host and is always available. It preserves the
exact output extent and does not perform reconstruction. A missing,
unavailable, incompatible, or malformed non-native backend either fails plan
resolution or resolves to native according to the saved fallback policy. No
silent fallback is allowed.

The host localizes the user-facing failure summary. Structured issue IDs and
provider-specific unavailability reason IDs remain stable for logs and for
future resolution through owner-scoped plugin catalogs.

The initial catalog intentionally exposes only `native`. DLSS, FSR, and other
implementations are not named or loaded by this contract. Later viewer-only
plugin work can populate the same read-only interface after static discovery
and device availability checks.

## Offline invariants

- Export selection never inherits viewport selection or viewport history.
- Project parsing and plan serialization never load a plugin.
- Fallback is resolved before scene capture and frame rendering when possible.
- The plan carries the requested/effective pair as the source for later export
  provenance integration.
- The current native export continues to use its existing renderer and encoder;
  this contract does not execute reconstruction yet.
- The contract contains no CUDA types or CUDA-specific behavior and adds no
  CUDA linkage beyond the existing video target.
- Equirectangular support must be declared explicitly by a backend descriptor.
- Splat precision remains a separate concern and is not part of this selection.

## Persisted shape

```json
{
  "version": 1,
  "backend_id": "native",
  "preset_id": "native",
  "fallback": "abort"
}
```

The object is stored under `SEQR.preferences.reconstruction`. Existing SEQR
chapter version 1 remains valid when the field is absent.
