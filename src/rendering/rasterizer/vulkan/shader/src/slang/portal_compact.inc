/* Compact container math adapted from PlayCanvas.
MIT License

Copyright (c) 2011-2026 PlayCanvas Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Display-only emulation of PlayCanvas 2.22.0 containerCompactWrite/Read.
// Source Gaussian data and Studio/training precision remain unchanged.
// The portal packs logarithmic scales and opacity into 8 bits, RGB into
// 11/11/10 bits over [0,4], and a canonical quaternion into 11/11/10 bits.
LFS_PORTAL_INLINE float lfsPortalCompactScale(float scale) {
    if (scale < 1e-10f) return 0.0f;
    float code = floor(clamp((log(scale) + 12.0f) * (255.0f / 21.0f) + 0.5f, 1.0f, 255.0f));
    return exp(code * (21.0f / 255.0f) - 12.0f);
}
LFS_PORTAL_INLINE LFS_PORTAL_VEC3 lfsPortalCompactScales(LFS_PORTAL_VEC3 scales) {
    return LFS_PORTAL_VEC3(lfsPortalCompactScale(scales.x), lfsPortalCompactScale(scales.y), lfsPortalCompactScale(scales.z));
}
LFS_PORTAL_INLINE float lfsPortalCompactOpacity(float opacity) {
    return floor(clamp(opacity, 0.0f, 1.0f) * 255.0f + 0.5f) / 255.0f;
}
LFS_PORTAL_INLINE LFS_PORTAL_VEC3 lfsPortalCompactColor(LFS_PORTAL_VEC3 color) {
    LFS_PORTAL_VEC3 levels = LFS_PORTAL_VEC3(2047.0f, 2047.0f, 1023.0f);
    return floor(clamp(color, LFS_PORTAL_VEC3(0.0f), LFS_PORTAL_VEC3(4.0f)) * (levels / 4.0f) + 0.5f) * (4.0f / levels);
}
LFS_PORTAL_INLINE LFS_PORTAL_VEC4 lfsPortalCompactRotation(LFS_PORTAL_VEC4 rotation) {
    float norm2 = dot(rotation, rotation);
    LFS_PORTAL_VEC4 q = norm2 > 1e-8f ? rotation / sqrt(norm2) : LFS_PORTAL_VEC4(1.0f, 0.0f, 0.0f, 0.0f);
    // Native quaternions are wxyz. Portal loads PLY with a 180-degree
    // rotation about Z, packs in that frame, then this helper maps back.
    q = LFS_PORTAL_VEC4(-q.w, -q.z, q.y, q.x);
    if (q.x < 0.0f) q = -q;
    LFS_PORTAL_VEC3 p = LFS_PORTAL_VEC3(q.y, q.z, q.w) / sqrt(1.0f + q.x);
    LFS_PORTAL_VEC3 levels = LFS_PORTAL_VEC3(2047.0f, 2047.0f, 1023.0f);
    p = floor(clamp(p * 0.5f + 0.5f, LFS_PORTAL_VEC3(0.0f), LFS_PORTAL_VEC3(1.0f)) * levels + 0.5f) / levels * 2.0f - 1.0f;
    float norm_p2 = dot(p, p);
    LFS_PORTAL_VEC3 v = sqrt(max(0.0f, 2.0f - norm_p2)) * p;
    return LFS_PORTAL_VEC4(v.z, v.y, -v.x, norm_p2 - 1.0f);
}
