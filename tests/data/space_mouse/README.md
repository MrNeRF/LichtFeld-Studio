# SpaceMouse Compact capture

`compact_capture.txt` is an unmodified capture of the owner's own 256f:c635
SpaceMouse Compact on Linux, recorded on 2026-10-26 with SDL3 HID. No scene data
is involved. The first line identifies the device; subsequent lines contain
milliseconds since capture start, byte count, and hexadecimal report bytes.
There are 851 translation reports, 852 rotation reports and four button reports.
`compact_descriptor.hex` contains the same device's 239-byte HID report descriptor,
read from sysfs. It advertises a logical axis range of −350 to +350.

The operator performed right, forward (away), up, down, nose down, clockwise
twist (seen from above), right roll, left button, right button, with pauses.
The observed raw frame maps to camera coordinates as
`t = (tx, -tz, ty)` and `r = (rx, -rz, ry)`.

| Gesture | Full interval (ms) | Camera axis/sign | Dominance-check window (ms) |
| --- | --- | --- | --- |
| Right | 7900-9100 | translation X+ | 8800-9000 |
| Forward | 11000-12500 | translation Z− | 11000-12500 |
| Up | 12500-14500 | translation Y+ | 12500-14500 |
| Down | 14500-16500 | translation Y− | 14500-16500 |
| Nose down | 17500-20900 | rotation X− | 17500-20900 |
| Clockwise twist | 21000-24000 | rotation Y− | 23400-23600 |
| Right roll | 24500-27000 | rotation Z− | 24500-27000 |

The small cap is strongly cross-coupled. Over the entire right gesture, Z−
translation exceeds X+; over the twist, pitch and roll have larger RMS values
than yaw. The short right/twist windows above are representative samples where
the intended axis dominates within its translation/rotation group. Tests also
check the intended sign across each full interval. They do not claim that these
were isolated single-axis gestures. A cleaner motion capture requires the owner.

At 28357 ms the left button reports `03 02 00`; at 31805 ms the right button
reports `03 01 00`. Both are followed by release reports.
