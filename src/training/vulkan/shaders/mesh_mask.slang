// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Mesh coverage for evaluation masks, ported from training/kernels/mesh_mask.cu. Operation 0
// prepares every face and rasterizes small ones; operation 1 drains the queue of large faces
// with one workgroup per face.
#include "../../../core/tensor/backend/vulkan/shaders/distortion_model.slangh"

[SpecializationConstant] const uint operation = 0;

static const int MAX_POLYGON_VERTICES = 8;
static const int SMALL_BOX_PIXELS = 64;
static const float DISTORTED_BOUNDS_MARGIN = 2.0f;
static const int MAX_EDGE_SAMPLES = 65536;
static const uint THREADS = 256;

struct Parameters {
    uint64_t vertices;
    uint64_t indices;
    uint64_t samples;
    uint64_t largeFaces;
    uint64_t counters;
    uint64_t mask;
    uint64_t camera;
    int vertexCount;
    int faceCount;
    float zNear;
    int padding;
};

// Read word by word from MeshMaskCameraBlock (ops/mesh_mask_camera.hpp).
struct Camera {
    float rows[12];
    float fx, fy, cx, cy;
    int width, height;
    float guardMinX, guardMaxX, guardMinY, guardMaxY;
    bool distorted;
    int model;
    int numDistortion;
    float distortion[12];
    float srcFx, srcFy, srcCx, srcCy, dstFx, dstFy;
};

struct Polygon {
    float2 vertices[MAX_POLYGON_VERTICES];
    int count;
    int minX, maxX, minY, maxY;
};

Camera loadCamera(uint64_t address) {
    Ptr<uint, Access.Read> words = (Ptr<uint, Access.Read>)address;
    Camera c;
    for (int i = 0; i < 12; ++i)
        c.rows[i] = asfloat(words[i]);
    c.fx = asfloat(words[12]);
    c.fy = asfloat(words[13]);
    c.cx = asfloat(words[14]);
    c.cy = asfloat(words[15]);
    c.width = asint(words[16]);
    c.height = asint(words[17]);
    c.guardMinX = asfloat(words[18]);
    c.guardMaxX = asfloat(words[19]);
    c.guardMinY = asfloat(words[20]);
    c.guardMaxY = asfloat(words[21]);
    c.distorted = asint(words[22]) != 0;
    c.model = asint(words[23]);
    c.numDistortion = asint(words[24]);
    for (int i = 0; i < 12; ++i)
        c.distortion[i] = asfloat(words[25 + i]);
    c.srcFx = asfloat(words[37]);
    c.srcFy = asfloat(words[38]);
    c.srcCx = asfloat(words[39]);
    c.srcCy = asfloat(words[40]);
    c.dstFx = asfloat(words[41]);
    c.dstFy = asfloat(words[42]);
    return c;
}

float3 transformPoint(Camera c, float3 p) {
    return float3(c.rows[0] * p.x + c.rows[1] * p.y + c.rows[2] * p.z + c.rows[3],
                  c.rows[4] * p.x + c.rows[5] * p.y + c.rows[6] * p.z + c.rows[7],
                  c.rows[8] * p.x + c.rows[9] * p.y + c.rows[10] * p.z + c.rows[11]);
}

float planeDistance(float3 p, int plane, Camera c, float zNear) {
    switch (plane) {
    case 0:
        return p.z - zNear;
    case 1:
        return p.x - c.guardMinX * p.z;
    case 2:
        return c.guardMaxX * p.z - p.x;
    case 3:
        return p.y - c.guardMinY * p.z;
    default:
        return c.guardMaxY * p.z - p.y;
    }
}

int clipAgainstPlane(float3 input[MAX_POLYGON_VERTICES], int inputCount,
                     inout float3 output[MAX_POLYGON_VERTICES], int plane, Camera c, float zNear) {
    int outputCount = 0;
    float3 previous = input[inputCount - 1];
    float previousDistance = planeDistance(previous, plane, c, zNear);
    bool previousInside = previousDistance >= 0.0f;
    for (int i = 0; i < inputCount; ++i) {
        const float3 current = input[i];
        const float currentDistance = planeDistance(current, plane, c, zNear);
        const bool currentInside = currentDistance >= 0.0f;
        if (currentInside != previousInside) {
            const float t = previousDistance / (previousDistance - currentDistance);
            output[outputCount++] = float3(previous.x + t * (current.x - previous.x),
                                           previous.y + t * (current.y - previous.y),
                                           previous.z + t * (current.z - previous.z));
        }
        if (currentInside)
            output[outputCount++] = current;
        previous = current;
        previousDistance = currentDistance;
        previousInside = currentInside;
    }
    return outputCount;
}

bool computePixelBounds(inout Polygon polygon, Camera c) {
    const float inf = asfloat(0x7f800000u);
    float boundsMinX = inf;
    float boundsMaxX = -inf;
    float boundsMinY = inf;
    float boundsMaxY = -inf;
    if (!c.distorted) {
        for (int i = 0; i < polygon.count; ++i) {
            const float px = polygon.vertices[i].x * c.fx + c.cx;
            const float py = polygon.vertices[i].y * c.fy + c.cy;
            boundsMinX = min(boundsMinX, px);
            boundsMaxX = max(boundsMaxX, px);
            boundsMinY = min(boundsMinY, py);
            boundsMaxY = max(boundsMaxY, py);
        }
    } else {
        const float sampleFx = max(abs(c.srcFx), abs(c.dstFx));
        const float sampleFy = max(abs(c.srcFy), abs(c.dstFy));
        for (int edge = 0; edge < polygon.count; ++edge) {
            const float2 a = polygon.vertices[edge];
            const float2 b = polygon.vertices[(edge + 1) % polygon.count];
            const float edgePixels = max(abs(b.x - a.x) * sampleFx, abs(b.y - a.y) * sampleFy);
            const int steps = max(1, min(MAX_EDGE_SAMPLES, int(ceil(edgePixels))));
            for (int sample = 0; sample <= steps; ++sample) {
                const float t = float(sample) / float(steps);
                float dx, dy;
                apply_distortion(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), c.model, c.distortion,
                                 c.numDistortion, dx, dy);
                const float px = dx * c.srcFx + c.srcCx;
                const float py = dy * c.srcFy + c.srcCy;
                if (!isfinite(px) || !isfinite(py))
                    continue;
                boundsMinX = min(boundsMinX, px);
                boundsMaxX = max(boundsMaxX, px);
                boundsMinY = min(boundsMinY, py);
                boundsMaxY = max(boundsMaxY, py);
            }
        }
    }

    if (!isfinite(boundsMinX) || !isfinite(boundsMaxX) || !isfinite(boundsMinY) || !isfinite(boundsMaxY))
        return false;
    const float margin = c.distorted ? DISTORTED_BOUNDS_MARGIN : 0.0f;
    const float firstX = boundsMinX - 0.5f - margin;
    const float lastX = boundsMaxX - 0.5f + margin;
    const float firstY = boundsMinY - 0.5f - margin;
    const float lastY = boundsMaxY - 0.5f + margin;
    if (lastX < 0.0f || lastY < 0.0f || firstX > float(c.width - 1) || firstY > float(c.height - 1))
        return false;
    polygon.minX = max(0, int(floor(max(firstX, -1.0f))));
    polygon.maxX = min(c.width - 1, int(ceil(min(lastX, float(c.width)))));
    polygon.minY = max(0, int(floor(max(firstY, -1.0f))));
    polygon.maxY = min(c.height - 1, int(ceil(min(lastY, float(c.height)))));
    return polygon.minX <= polygon.maxX && polygon.minY <= polygon.maxY;
}

bool prepareTriangle(Parameters p, Camera c, int face, out Polygon polygon) {
    polygon = {};
    Ptr<float, Access.Read> vertices = (Ptr<float, Access.Read>)p.vertices;
    Ptr<int, Access.Read> indices = (Ptr<int, Access.Read>)p.indices;
    float3 clippedA[MAX_POLYGON_VERTICES];
    float3 clippedB[MAX_POLYGON_VERTICES];
    for (int corner = 0; corner < 3; ++corner) {
        const int vertex = indices[3 * face + corner];
        if (vertex < 0 || vertex >= p.vertexCount)
            return false;
        clippedA[corner] = transformPoint(c, float3(vertices[3 * vertex], vertices[3 * vertex + 1],
                                                    vertices[3 * vertex + 2]));
    }

    int count = 3;
    for (int plane = 0; plane < 5; ++plane) {
        count = clipAgainstPlane(clippedA, count, clippedB, plane, c, p.zNear);
        if (count < 3)
            return false;
        clippedA = clippedB;
    }

    polygon.count = count;
    for (int i = 0; i < count; ++i) {
        const float inverseZ = 1.0f / clippedA[i].z;
        polygon.vertices[i] = float2(clippedA[i].x * inverseZ, clippedA[i].y * inverseZ);
    }
    float twiceArea = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float2 a = polygon.vertices[i];
        const float2 b = polygon.vertices[(i + 1) % count];
        twiceArea += a.x * b.y - a.y * b.x;
    }
    if (!isfinite(twiceArea) || twiceArea == 0.0f)
        return false;
    if (twiceArea < 0.0f) {
        for (int i = 0; i < count / 2; ++i) {
            const float2 swapped = polygon.vertices[i];
            polygon.vertices[i] = polygon.vertices[count - 1 - i];
            polygon.vertices[count - 1 - i] = swapped;
        }
    }
    return computePixelBounds(polygon, c);
}

float edgeFunction(float2 a, float2 b, float2 point) {
    return fma(b.x - a.x, point.y - a.y, -(b.y - a.y) * (point.x - a.x));
}

bool insideEdge(float2 a, float2 b, float2 point) {
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float px = point.x - a.x;
    const float py = point.y - a.y;
    const float tolerance = 4.0f * 1.1920928955078125e-7f * (abs(dx * py) + abs(dy * px));
    return fma(dx, py, -dy * px) >= -tolerance;
}

bool pointInPolygonFan(Polygon polygon, float2 point) {
    const float2 first = polygon.vertices[0];
    for (int triangle = 1; triangle + 1 < polygon.count; ++triangle) {
        const float2 second = polygon.vertices[triangle];
        const float2 third = polygon.vertices[triangle + 1];
        if (edgeFunction(first, second, third) == 0.0f)
            continue;
        if (insideEdge(first, second, point) && insideEdge(second, third, point) && insideEdge(third, first, point))
            return true;
    }
    return false;
}

float2 samplePoint(Parameters p, Camera c, int x, int y) {
    if (c.distorted) {
        Ptr<float, Access.Read> samples = (Ptr<float, Access.Read>)p.samples;
        const int index = 2 * (y * c.width + x);
        return float2(samples[index], samples[index + 1]);
    }
    return float2((float(x) + 0.5f - c.cx) / c.fx, (float(y) + 0.5f - c.cy) / c.fy);
}

// Every write stores 1, so racing writers to one word only need the bit set.
void markCovered(Parameters p, int pixel) {
    Ptr<uint, Access.ReadWrite> words = (Ptr<uint, Access.ReadWrite>)p.mask;
    InterlockedOr(words[uint(pixel) >> 2], 1u << ((uint(pixel) & 3u) * 8u));
}

void coverPixel(Parameters p, Camera c, Polygon polygon, int x, int y) {
    const float2 sample = samplePoint(p, c, x, y);
    if (isfinite(sample.x) && isfinite(sample.y) && pointInPolygonFan(polygon, sample))
        markCovered(p, y * c.width + x);
}

groupshared Polygon sharedPolygon;
groupshared int sharedFace;
groupshared bool sharedPrepared;

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p) {
    uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const Camera c = loadCamera(p.camera);
    if (operation == 2) {
        Ptr<float, Access.Read> means = (Ptr<float, Access.Read>)p.vertices;
        const int radius = p.padding;
        for (uint i = groupId.x * THREADS + lane; i < uint(p.vertexCount); i += groups.x * THREADS) {
            const float3 view = transformPoint(c, float3(means[3*i], means[3*i+1], means[3*i+2]));
            if (!(view.z > 1.0e-6f)) continue;
            float nx = view.x / view.z, ny = view.y / view.z;
            if (c.distorted) {
                float dx, dy;
                apply_distortion(nx, ny, c.model, c.distortion, c.numDistortion, dx, dy);
                nx = dx; ny = dy;
            }
            const float u = c.fx * nx + c.cx, v = c.fy * ny + c.cy;
            if (!isfinite(u) || !isfinite(v) || u < -radius || v < -radius || u >= c.width + radius || v >= c.height + radius) continue;
            const int cu = int(floor(u)), cv = int(floor(v));
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int x = cu + dx, y = cv + dy;
                    if (x >= 0 && x < c.width && y >= 0 && y < c.height && dx*dx + dy*dy <= radius*radius)
                        markCovered(p, y*c.width+x);
                }
        }
        return;
    }
    Ptr<int, Access.ReadWrite> counters = (Ptr<int, Access.ReadWrite>)p.counters;
    Ptr<int, Access.ReadWrite> largeFaces = (Ptr<int, Access.ReadWrite>)p.largeFaces;

    if (operation == 0) {
        for (int face = int(groupId.x * THREADS + lane); face < p.faceCount; face += int(groups.x * THREADS)) {
            Polygon polygon;
            if (!prepareTriangle(p, c, face, polygon))
                continue;
            const int pixels = (polygon.maxX - polygon.minX + 1) * (polygon.maxY - polygon.minY + 1);
            if (pixels <= SMALL_BOX_PIXELS) {
                for (int y = polygon.minY; y <= polygon.maxY; ++y)
                    for (int x = polygon.minX; x <= polygon.maxX; ++x)
                        coverPixel(p, c, polygon, x, y);
            } else {
                int slot;
                InterlockedAdd(counters[0], 1, slot);
                largeFaces[slot] = face;
            }
        }
        return;
    }

    // The large queue holds counters[0] faces; counters[1] is the next one to claim.
    while (true) {
        if (lane == 0) {
            int claimed;
            InterlockedAdd(counters[1], 1, claimed);
            sharedFace = claimed < counters[0] ? largeFaces[claimed] : -1;
            Polygon polygon;
            sharedPrepared = sharedFace >= 0 && prepareTriangle(p, c, sharedFace, polygon);
            sharedPolygon = polygon;
        }
        GroupMemoryBarrierWithGroupSync();
        if (sharedFace < 0)
            return;
        if (sharedPrepared) {
            const Polygon polygon = sharedPolygon;
            const int boxWidth = polygon.maxX - polygon.minX + 1;
            for (int y = polygon.minY; y <= polygon.maxY; ++y)
                for (int boxX = int(lane); boxX < boxWidth; boxX += int(THREADS))
                    coverPixel(p, c, polygon, polygon.minX + boxX, y);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
