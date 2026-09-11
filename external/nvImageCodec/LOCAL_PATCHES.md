# Local patches vs upstream nvImageCodec

This tree is vendored under `external/nvImageCodec`. Record LichtFeld-only deltas here so upstream syncs do not silently drop them.

## static_switch.h — remove Boost.Preprocessor dependency

- **File:** `src/imgproc/static_switch.h`
- **Why:** The vendored image-processing conversion kernel uses only a bounded variadic for-each and parenthesis removal from Boost.Preprocessor.
- **Delta:** Replaced those Boost.Preprocessor macros with standard C++ variadic macros supporting up to 16 list entries. The `TYPE_SWITCH`, `VALUE_SWITCH`, and `BOOL_SWITCH` interfaces and documented argument shapes are unchanged.
- **Upstream:** Re-apply if the upstream header changes its switch macro interfaces.
