// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Shared scalar mathematics for the CPU implementation and all Slang backends.
#ifdef __cplusplus
#define COLOR_INLINE  inline
#define COLOR_PRECISE volatile
#else
#define COLOR_INLINE
// Slang's precise modifier is emitted verbatim for CUDA. The program is
// compiled with -fp-mode precise and CUDA --fmad=false instead.
#define COLOR_PRECISE
#endif
COLOR_INLINE int colorRgbByte(float value) {
    if (isnan(value))
        return 0;
    value = min(max(value, 0.0f), 1.0f);
    COLOR_PRECISE float scaled = value * 255.0f;
    COLOR_PRECISE float rounded = scaled + 0.5f;
    return int(floor(rounded));
}
COLOR_INLINE int colorLuma(int r, int g, int b) { return (66 * r + 129 * g + 25 * b + 128) / 256 + 16; }
COLOR_INLINE int colorChromaU(int r, int g, int b) { return min(max(int(floor(float(-38 * r - 74 * g + 112 * b + 128) / 256.0f)) + 128, 0), 255); }
COLOR_INLINE int colorChromaV(int r, int g, int b) { return min(max(int(floor(float(112 * r - 94 * g - 18 * b + 128) / 256.0f)) + 128, 0), 255); }
#undef COLOR_INLINE
#undef COLOR_PRECISE
