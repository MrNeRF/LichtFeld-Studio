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
} GaussianCloudData;

#endif // SPZ_SPLAT_C_TYPES_H_
