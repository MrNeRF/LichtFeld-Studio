// Ray-crossing parity over a triangle tree; the same test as internal/triangle_tree.hpp, walked depth
// first with only the current level and node as state.
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 120;
static const uint kFanoutBits = 5;
static const uint kFanout = 32;
static const uint kMaxLevels = 8;

struct Parameters {
    uint64_t points;
    uint64_t visit;
    uint64_t triangles;
    uint64_t boxes;
    uint64_t output;
    uint count;
    uint references;
    uint levels;
    uint pad0;
    uint levelOffset[kMaxLevels];
    uint levelCount[kMaxLevels];
};

float3 load3(uint64_t address, uint64_t index) {
    Ptr<float, Access.Read> data = (Ptr<float, Access.Read>)address;
    const uint64_t base = index * 3;
    return float3(data[base], data[base + 1], data[base + 2]);
}

bool rayCrossesTriangle(float3 p, uint64_t triangle) {
    Ptr<float, Access.Read> t = (Ptr<float, Access.Read>)triangle;
    const float e1x = t[3], e1y = t[4], e1z = t[5];
    const float e2x = t[6], e2y = t[7], e2z = t[8];
    const float px = 0.37139067f * e2z - 0.52911311f * e2y;
    const float py = 0.52911311f * e2x - e2z;
    const float pz = e2y - 0.37139067f * e2x;
    const float determinant = e1x * px + e1y * py + e1z * pz;
    if (!(abs(determinant) > 1e-8f))
        return false;
    const float inverse = 1.0f / determinant;
    const float tx = p.x - t[0], ty = p.y - t[1], tz = p.z - t[2];
    const float u = (tx * px + ty * py + tz * pz) * inverse;
    if (!(u >= 0.0f && u <= 1.0f))
        return false;
    const float qx = ty * e1z - tz * e1y;
    const float qy = tz * e1x - tx * e1z;
    const float qz = tx * e1y - ty * e1x;
    const float v = (qx + 0.37139067f * qy + 0.52911311f * qz) * inverse;
    return v >= 0.0f && u + v <= 1.0f && (e2x * qx + e2y * qy + e2z * qz) * inverse > 1e-7f;
}

int treeParity(Parameters p, float3 point) {
    if (p.references == 0u)
        return 0;
    const float3 q = float3(dot(float3(0.348155307f, -0.937436869f, 0.0f), point),
                            dot(float3(0.416551804f, 0.154703454f, -0.895852351f), point),
                            dot(float3(0.839805023f, 0.311895750f, 0.444351848f), point));
    const float slack = 1e-6f * (abs(point.x) + abs(point.y) + abs(point.z));
    Ptr<float, Access.Read> boxes = (Ptr<float, Access.Read>)p.boxes;
    const uint top = p.levels - 1u;
    uint level = top;
    uint node = 0u;
    int parity = 0;
    // Set when node is a parent whose children are done: the next step only moves past it.
    bool climbing = false;
    while (true) {
        if (!climbing) {
            const uint64_t box = (uint64_t(p.levelOffset[level]) + node) * 6;
            const bool open = q.x + slack >= boxes[box] && q.x - slack <= boxes[box + 3] &&
                              q.y + slack >= boxes[box + 1] && q.y - slack <= boxes[box + 4] &&
                              q.z - slack <= boxes[box + 5];
            if (open && level != 0u) {
                --level;
                node <<= kFanoutBits;
                continue;
            }
            if (open) {
                const uint64_t first = uint64_t(node) << kFanoutBits;
                const uint64_t last = min(first + kFanout, uint64_t(p.references));
                for (uint64_t j = first; j < last; ++j)
                    parity ^= rayCrossesTriangle(point, p.triangles + j * 36) ? 1 : 0;
            }
        }
        climbing = false;
        // Next sibling, or the parent once its group of children is done.
        ++node;
        if (level == top) {
            if (node >= p.levelCount[top])
                return parity;
        } else if ((node & (kFanout - 1u)) == 0u || node >= p.levelCount[level]) {
            node = (node - 1u) >> kFanoutBits;
            ++level;
            climbing = true;
        }
    }
    return parity;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)] void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters p) {
    const uint3 numWorkgroups = spirv_asm {
result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    Ptr<int, Access.Read> visit = (Ptr<int, Access.Read>)p.visit;
    Ptr<int, Access.ReadWrite> output = (Ptr<int, Access.ReadWrite>)p.output;
    for (uint t = dispatchThreadId.x; t < p.count; t += stride) {
        const uint i = uint(visit[t]);
        const float3 point = load3(p.points, i);
        output[i] = all(isfinite(point)) ? treeParity(p, point) : 0;
    }
}
