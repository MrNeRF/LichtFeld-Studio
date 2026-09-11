# Local patches vs upstream TinyUSDZ

This tree is vendored from TinyUSDZ commit `46c2e36b1d4`. Record
LichtFeld-only deltas here so upstream syncs do not silently drop them.

## ascii-parser.cc — register half-precision role types

- **File:** `src/ascii-parser.cc`
- **Delta:** Registered `value::kNormal3h`, `value::kPoint3h`, and
  `value::kVector3h` with `RegisterPrimAttrTypes`, and dispatched those types
  to their `ParseBasicPrimAttr` implementations. Added the missing optional
  `ReadBasicType` wrappers used by those implementations.
- **Why:** The ASCII parser has the half-role value implementations, but both
  the declaration type names, attribute parser dispatch, and optional reader
  overloads must recognize them before USDA attributes can be parsed.

## crate-reader.cc / crate-reader.hh — flat accessors

- **Files:** `src/crate-reader.cc`, `src/crate-reader.hh`
- **Delta:** Added `GetUncompressedArrayData` for direct access to eligible
  uncompressed crate arrays and `UnpackValueRepForFlat` as the flat reader's
  access point for value unpacking.
- **Why:** The flat USDC reader needs bounded, allocation-free access to the
  array payloads it can represent directly.

## io-util.hh — clamp tile coordinates

- **File:** `src/io-util.hh`
- **Delta:** `UDIMIndex` and `UVTILEIndex` clamp both tile coordinates with
  `std::min(9u, value)`.
- **Why:** The upstream implementation incorrectly used `std::max` for `v`,
  producing invalid tile indices for values below 9.
