/* Tone curves adapted from PlayCanvas.
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

// Tone operators match the pinned PlayCanvas 2.22.0 curves. Inputs are
// display RGB for Studio composition; standard splats decode/encode below.
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsFilmicCurve(LFS_COLOR_VEC3 x) {
    return ((x*(0.15f*x+0.05f)+0.004f)/(x*(0.15f*x+0.50f)+0.06f))-0.02f/0.30f;
}
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsDisplayTone(LFS_COLOR_VEC3 color, LFS_COLOR_UINT mode, float exposure) {
    if (mode == 0u) return color;
    color *= exposure;
    if (mode == 1u) return color;
    if (mode == 2u) return lfsFilmicCurve(color) / lfsFilmicCurve(LFS_COLOR_VEC3(11.2f));
    if (mode == 3u) {
        LFS_COLOR_VEC3 h = max(LFS_COLOR_VEC3(0.0f), color - LFS_COLOR_VEC3(0.004f));
        return (h*(0.275f*h+LFS_COLOR_VEC3(0.0375f))+LFS_COLOR_VEC3(0.0025f)) /
               (h*(0.22f*h+LFS_COLOR_VEC3(0.3f))+LFS_COLOR_VEC3(0.06f)) - LFS_COLOR_VEC3(1.25f*0.01f/0.3f);
    }
    if (mode == 4u) return (color*(2.51f*color+0.03f))/(color*(2.43f*color+0.59f)+0.14f);
    if (mode == 5u) {
        color /= 0.6f;
        color = LFS_COLOR_VEC3(dot(color,LFS_COLOR_VEC3(0.59719f,0.35458f,0.04823f)),
                              dot(color,LFS_COLOR_VEC3(0.07600f,0.90834f,0.01566f)),
                              dot(color,LFS_COLOR_VEC3(0.02840f,0.13383f,0.83777f)));
        color = (color*(color+0.0245786f)-0.000090537f)/(color*(0.983729f*color+0.4329510f)+0.238081f);
        color = LFS_COLOR_VEC3(dot(color,LFS_COLOR_VEC3(1.60475f,-0.53108f,-0.07367f)),
                              dot(color,LFS_COLOR_VEC3(-0.10208f,1.10813f,-0.00605f)),
                              dot(color,LFS_COLOR_VEC3(-0.00327f,-0.07276f,1.07602f)));
        return clamp(color,LFS_COLOR_VEC3(0.0f),LFS_COLOR_VEC3(1.0f));
    }
    if (mode == 6u) {
        float x = min(color.x,min(color.y,color.z));
        float offset = x < 0.08f ? x-6.25f*x*x : 0.04f;
        color -= offset;
        float peak = max(color.x,max(color.y,color.z));
        if (peak < 0.76f) return color;
        float newPeak = 1.0f-0.24f*0.24f/(peak+0.24f-0.76f);
        color *= newPeak/peak;
        float g = 1.0f-1.0f/(0.15f*(peak-newPeak)+1.0f);
        return LFS_COLOR_MIX(color,LFS_COLOR_VEC3(newPeak),g);
    }
    return color;
}
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsDisplaySplat(LFS_COLOR_VEC3 color, LFS_COLOR_UINT mode, float exposure) {
    if (mode == 0u) return color;
    LFS_COLOR_VEC3 linear = pow(max(color,LFS_COLOR_VEC3(0.0f)),LFS_COLOR_VEC3(2.2f));
    return pow(lfsDisplayTone(linear,mode,exposure)+LFS_COLOR_VEC3(0.0000001f),LFS_COLOR_VEC3(1.0f/2.2f));
}
