/*
MIT License

Copyright (c) 2025 Niantic Labs
Copyright (c) 2025 Adobe Inc.

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

#ifndef SPZ_SPLAT_C_TYPES_H_
#define SPZ_SPLAT_C_TYPES_H_

#define _USE_MATH_DEFINES
#include <math.h>
#include <stddef.h>
#include <stdint.h>

// These types are used to bridge between the C++ API and C (to interop with Swift and C#).

typedef struct {
    size_t count;
    float* data;
} SpzFloatBuffer;

// Forward declaration - full definition is in splat-extensions.h
#ifdef SPZ_BUILD_EXTENSIONS
#include "splat-extensions.h"
#else
typedef struct SpzExtensionNode SpzExtensionNode;
#endif

typedef struct {
    int32_t numPoints;
    int32_t shDegree;
    bool antialiased;
    SpzFloatBuffer positions;
    SpzFloatBuffer scales;
    SpzFloatBuffer rotations;
    SpzFloatBuffer alphas;
    SpzFloatBuffer colors;
    SpzFloatBuffer sh;
    SpzExtensionNode* extensions;
} GaussianCloudData;

#endif // SPZ_SPLAT_C_TYPES_H_
