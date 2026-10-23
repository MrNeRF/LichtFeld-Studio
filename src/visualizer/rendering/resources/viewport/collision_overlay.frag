#version 450

layout(set = 0, binding = 0, std430) readonly buffer Occupancy {
    uint cells[];
};
layout(push_constant) uniform CollisionPush {
    mat4 inverseMvp;
    vec4 originSize;
    uvec4 dimensionsFlags;
    vec4 viewport;
    vec4 camera;
} push;
layout(location = 0) out vec4 outColor;

int sliceHeight() {
    return int(ceil(float(push.dimensionsFlags.y) * float((push.dimensionsFlags.w >> 1u) & 65535u) / 65535.0));
}

bool contains(ivec3 cell) {
    return all(greaterThanEqual(cell, ivec3(0))) &&
           all(lessThan(cell, ivec3(push.dimensionsFlags.xyz))) && cell.y < sliceHeight();
}

ivec3 cacheMin = ivec3(-1);
int cacheSize = 0;
uvec2 cacheMask = uvec2(0);
bool cacheMixed = false;
bool cacheSolid = false;
uint traversalPath[22];
int traversalLevel;

bool sampleCell(ivec3 cell, out ivec3 leafMin, out ivec3 leafMax) {
    if (cacheSize > 0 && all(greaterThanEqual(cell, cacheMin)) && all(lessThan(cell, cacheMin + cacheSize))) {
        if (cacheMixed) {
            ivec3 local = cell - cacheMin;
            uint bit = uint(local.x + 4 * local.y + 16 * local.z);
            leafMin = cell; leafMax = cell + 1;
            return (cacheMask[bit >> 5u] & (1u << (bit & 31u))) != 0u;
        }
        leafMin = cacheMin;
        leafMax = min(cacheMin + cacheSize, ivec3(push.dimensionsFlags.xyz));
        return cacheSolid;
    }
    int size = 1 << traversalLevel;
    uint index = traversalPath[traversalLevel];
    uint word = cells[index];
    while (word > 1u && (word & 0x80000000u) == 0u) {
        --traversalLevel;
        size /= 2;
        ivec3 bit = (cell >> traversalLevel) & 1;
        index = word - 2u + uint(bit.x + 2 * bit.y + 4 * bit.z);
        traversalPath[traversalLevel] = index;
        word = cells[index];
    }
    leafMin = cell & ~(size - 1);
    cacheMin = leafMin;
    cacheSize = size;
    cacheMixed = (cells[index] & 0x80000000u) != 0u;
    cacheSolid = cells[index] != 0u;
    if (cacheMixed) {
        ivec3 local = cell - leafMin;
        uint bit = uint(local.x + 4 * local.y + 16 * local.z);
        uint base = cells[index] & 0x7fffffffu;
        cacheMask = uvec2(cells[base], cells[base + 1u]);
        leafMin = cell;
        leafMax = cell + 1;
        return (cacheMask[bit >> 5u] & (1u << (bit & 31u))) != 0u;
    }
    leafMax = min(leafMin + size, ivec3(push.dimensionsFlags.xyz));
    return cells[index] != 0u;
}

bool occupied(ivec3 cell) {
    if (!contains(cell))
        return false;
    ivec3 lo, hi;
    return sampleCell(cell, lo, hi);
}

vec4 shadeSurface(vec3 hit, float pixelWidth, bool insideSolid) {
    // Tree traversal can cross many cells in one uniform leaf. Shade the actual
    // hit voxel, never the stale cell from before that skip (notably on exits).
    vec3 local = fract((hit - push.originSize.xyz) / push.originSize.w);
    vec3 boundaryDistance = min(local, 1.0 - local);
    float smallest = min(boundaryDistance.x, min(boundaryDistance.y, boundaryDistance.z));
    float largest = max(boundaryDistance.x, max(boundaryDistance.y, boundaryDistance.z));
    float edgeDistance = boundaryDistance.x + boundaryDistance.y + boundaryDistance.z - smallest - largest;
    float screenScale = (push.dimensionsFlags.w & (1u << 23u)) != 0u ? 0.5 : 1.0;
    float edge = 1.0 - smoothstep(0.0, clamp(pixelWidth * screenScale * 1.5, 0.01, 0.5), edgeDistance);
    float brightness = boundaryDistance.y <= boundaryDistance.x && boundaryDistance.y <= boundaryDistance.z
                           ? 0.65 : boundaryDistance.z <= boundaryDistance.x ? 0.45 : 0.85;
    vec3 tint = insideSolid ? vec3(1.0, 0.35, 0.08) :
                (push.dimensionsFlags.w & 1u) != 0u ? vec3(0.65, 0.85, 1.0) : vec3(1.0);
    float alpha = mix(0.4, 0.8, edge) * push.camera.w;
    return vec4(tint * mix(brightness, 0.2, edge) * alpha, alpha);
}

void main() {
    outColor = vec4(0.0);
    vec2 ndc = (gl_FragCoord.xy - push.viewport.xy) / push.viewport.zw * 2.0 - 1.0;
    vec4 nearPoint = push.inverseMvp * vec4(ndc, 0.0, 1.0);
    vec4 farPoint = push.inverseMvp * vec4(ndc, 1.0, 1.0);
    vec3 ro = nearPoint.xyz / nearPoint.w;
    vec3 rd = normalize(farPoint.xyz / farPoint.w - ro);
    vec3 roDx = dFdx(ro), roDy = dFdy(ro);
    vec3 rdDx = dFdx(rd), rdDy = dFdy(rd);
    float size = push.originSize.w;
    vec3 gridMin = push.originSize.xyz;
    vec3 gridMax = gridMin + vec3(push.dimensionsFlags.x, sliceHeight(), push.dimensionsFlags.z) * size;
    if (sliceHeight() <= 0 || push.camera.w <= 0.0) return;

    // Parallel rays need finite slab distances, including when on a grid plane.
    vec3 safeDirection = mix(vec3(1e-20), rd, greaterThan(abs(rd), vec3(1e-20)));
    vec3 inverseDirection = 1.0 / safeDirection;
    vec3 first = (gridMin - ro) * inverseDirection;
    vec3 last = (gridMax - ro) * inverseDirection;
    vec3 entry = min(first, last);
    vec3 exit = max(first, last);
    float t = max(0.0, max(entry.x, max(entry.y, entry.z)));
    float end = min(exit.x, min(exit.y, exit.z));
    if (t >= end)
        return;

    bool insideSolid = (push.dimensionsFlags.w & (1u << 22u)) != 0u;
    traversalLevel = int((push.dimensionsFlags.w >> 17u) & 31u);
    traversalPath[traversalLevel] = 0u;
    vec3 point = ro + rd * (t + size * 1e-4);
    ivec3 cell = clamp(ivec3(floor((point - gridMin) / size)),
                      ivec3(0), ivec3(push.dimensionsFlags.xyz) - 1);
    uint steps = push.dimensionsFlags.x + push.dimensionsFlags.y + push.dimensionsFlags.z + 1u;
    for (uint i = 0u; i < steps; ++i) {
        ivec3 leafMin, leafMax;
        if (sampleCell(cell, leafMin, leafMax) != insideSolid) {
            float pixelWidth = max(length(roDx + t * rdDx), length(roDy + t * rdDy)) / size;
            outColor = shadeSurface(ro + rd * t, pixelWidth, insideSolid);
            return;
        }
        leafMax.y = min(leafMax.y, sliceHeight());
        ivec3 boundary = mix(leafMin, leafMax, greaterThan(safeDirection, vec3(0.0)));
        vec3 nextPlane = gridMin + vec3(boundary) * size;
        vec3 nextT = (nextPlane - ro) * inverseDirection;
        float next = min(nextT.x, min(nextT.y, nextT.z));
        if (next >= end) {
            // Outside the grid is empty, including a completely occupied volume's exit.
            if (insideSolid) {
                float pixelWidth = max(length(roDx + end * rdDx), length(roDy + end * rdDy)) / size;
                outColor = shadeSurface(ro + rd * end, pixelWidth, true);
            }
            return;
        }
        // Advance all tied axes to avoid visiting zero-width cells at corners.
        bvec3 crossed = lessThanEqual(nextT, vec3(next));
        ivec3 previousCell = cell;
        cell = ivec3(floor((ro + rd * next - gridMin) / size));
        ivec3 across = mix(leafMin - 1, leafMax, greaterThan(safeDirection, vec3(0.0)));
        cell = mix(cell, across, crossed);
        // Rounding at an edge can put a non-crossed axis back into the
        // previous cell. Clamp every axis to the ray's direction so tied
        // crossings cannot bounce between two cells until the safety limit.
        cell = mix(min(cell, previousCell), max(cell, previousCell),
                   greaterThan(safeDirection, vec3(0.0)));
        t = next;
        if (!contains(cell)) {
            if (insideSolid) {
                float pixelWidth = max(length(roDx + t * rdDx), length(roDy + t * rdDy)) / size;
                outColor = shadeSurface(ro + rd * t, pixelWidth, true);
            }
            return;
        }
        // Retain the ray's ancestor path. Ascend only as far as the most
        // significant changed coordinate bit; never restart at the root.
        uvec3 changed = uvec3(previousCell ^ cell);
        int ancestor = findMSB(changed.x | changed.y | changed.z) + 1;
        traversalLevel = max(traversalLevel, ancestor);
    }
}
