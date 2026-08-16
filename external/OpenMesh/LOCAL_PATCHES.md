# Local patches vs upstream OpenMesh

This tree is vendored under `external/OpenMesh`. Record LichtFeld-only deltas here so upstream syncs do not silently drop them.

## Timer.cc — buffer size + snprintf for float formats

- **File:** `src/OpenMesh/Tools/Utils/Timer.cc`
- **Function:** `Timer::as_string(double, Timer::Format)`
- **Why:** CodeQL `cpp/overrunning-write-with-float` — `sprintf` of float conversions into a 32-byte stack buffer may require ~316 bytes.
- **Delta:**
  - `char string[32]` → `char string[512]`
  - Cases `Seconds` / `HSeconds` / `MSeconds` / `MicroSeconds` / `NanoSeconds`: `sprintf` → `snprintf` with remaining capacity; **format strings unchanged**
- **Upstream:** not applied upstream as of vendored pin; re-apply on OpenMesh upgrades.
