#include "../../../../include/core/rounded_math.slangh"

// Neighbour counts over a point tree; the same counts as internal/point_tree.hpp, walked depth first
// with only the current level and node as state.
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 160;
static const uint kFanoutBits = 5;
static const uint kFanout = 32;
static const uint kMaxLevels = 8;

// 0 neighbour counts, 1 join each point with its mutual neighbours (labels in output).
[SpecializationConstant] const uint kMode = 0;

struct Parameters {
    uint64_t points;
    uint64_t sorted;
    uint64_t boxes;
    uint64_t visit;
    uint64_t radii;
    uint64_t queries;
    uint64_t output;
    uint64_t boxRadii;
    uint64_t sortedRadii;
    uint count;
    uint references;
    uint levels;
    int maxCount;
    float radius;
    uint pad0;
    uint levelOffset[kMaxLevels];
    uint levelCount[kMaxLevels];
};

#include "byte_read.slangh"

float3 load3(uint64_t address, uint64_t index) {
    Ptr<float, Access.Read> data = (Ptr<float, Access.Read>)address;
    const uint64_t base = index * 3;
    return float3(data[base], data[base + 1], data[base + 2]);
}

bool within(float3 a, float3 b, float radius) {
    float3 d = float3(roundedSub(a.x, b.x), roundedSub(a.y, b.y), roundedSub(a.z, b.z));
    float r2 = roundedMul(radius, radius);
    if (r2 < 1.17549435e-38f || !isfinite(r2)) {
        d = float3(roundedDiv(d.x, radius), roundedDiv(d.y, radius), roundedDiv(d.z, radius));
        r2 = 1.0f;
    }
    return dotRounded(d, d) <= r2;
}

// near > limit: no point of the box is within the radius; far <= limit: every point is.
float3 boxReach(float3 p, uint64_t box, float radius) {
    Ptr<float, Access.Read> bounds = (Ptr<float, Access.Read>)box;
    const float3 low = float3(bounds[0], bounds[1], bounds[2]);
    const float3 high = float3(bounds[3], bounds[4], bounds[5]);
    float3 near = float3(max(0.0f, max(roundedSub(low.x, p.x), roundedSub(p.x, high.x))),
                         max(0.0f, max(roundedSub(low.y, p.y), roundedSub(p.y, high.y))),
                         max(0.0f, max(roundedSub(low.z, p.z), roundedSub(p.z, high.z))));
    float3 far = float3(max(abs(roundedSub(p.x, low.x)), abs(roundedSub(p.x, high.x))),
                        max(abs(roundedSub(p.y, low.y)), abs(roundedSub(p.y, high.y))),
                        max(abs(roundedSub(p.z, low.z)), abs(roundedSub(p.z, high.z))));
    const float r2 = roundedMul(radius, radius);
    float limit = r2;
    if (r2 < 1.17549435e-38f || !isfinite(r2)) {
        near = float3(roundedDiv(near.x, radius), roundedDiv(near.y, radius), roundedDiv(near.z, radius));
        far = float3(roundedDiv(far.x, radius), roundedDiv(far.y, radius), roundedDiv(far.z, radius));
        limit = 1.0f;
    }
    return float3(dotRounded(near, near), dotRounded(far, far), limit);
}

int treeCount(Parameters p, float3 q, int rank, float radius) {
    if (!(radius > 0.0f) || !isfinite(radius) || p.references == 0u)
        return 0;
    const uint top = p.levels - 1u;
    uint level = top;
    uint node = 0u;
    int found = 0;
    // Set when node is a parent whose children are done: the next step only moves past it.
    bool climbing = false;
    while (true) {
        const float3 reach = climbing ? float3(1.0f, 1.0f, 0.0f)
                                      : boxReach(q, p.boxes + (uint64_t(p.levelOffset[level]) + node) * 24, radius);
        climbing = false;
        if (reach.x <= reach.z) {
            const uint shift = kFanoutBits * (level + 1u);
            const uint64_t first = uint64_t(node) << shift;
            const uint64_t span = uint64_t(1) << shift;
            const uint64_t last = first + span < uint64_t(p.references) ? first + span : uint64_t(p.references);
            if (reach.y <= reach.z) {
                const bool self = rank >= 0 && uint64_t(rank) >= first && uint64_t(rank) < last;
                // A subtree holds at most references points, so its size fits int.
                found += int(last - first) - (self ? 1 : 0);
                if (found >= p.maxCount)
                    return p.maxCount;
            } else if (level != 0u) {
                --level;
                node <<= kFanoutBits;
                continue;
            } else {
                for (uint64_t j = first; j < last; ++j) {
                    if (int64_t(j) != int64_t(rank) && within(q, load3(p.sorted, j), radius)) {
                        ++found;
                        if (found == p.maxCount)
                            return found;
                    }
                }
            }
        }
        // Next sibling, or the parent once its group of children is done.
        ++node;
        if (level == top) {
            if (node >= p.levelCount[top])
                return found;
        } else if ((node & (kFanout - 1u)) == 0u || node >= p.levelCount[level]) {
            node = (node - 1u) >> kFanoutBits;
            ++level;
            climbing = true;
        }
    }
    return found;
}

int componentRoot(Ptr<int, Access.ReadWrite> parent, int node) {
    int current = parent[node];
    if (current != node) {
        int previous = node;
        int ancestor = parent[current];
        while (current > ancestor) {
            parent[previous] = ancestor;
            previous = current;
            current = ancestor;
            ancestor = parent[current];
        }
    }
    return current;
}

// Joins the point at sorted position rank with every later reference within both radii; mirrors
// pointTreeMutualNeighbors in internal/point_tree.hpp.
void joinMutual(Parameters p, uint rank) {
    Ptr<int, Access.Read> visit = (Ptr<int, Access.Read>)p.visit;
    Ptr<float, Access.Read> radii = (Ptr<float, Access.Read>)p.radii;
    Ptr<float, Access.Read> boxRadii = (Ptr<float, Access.Read>)p.boxRadii;
    Ptr<float, Access.Read> sortedRadii = (Ptr<float, Access.Read>)p.sortedRadii;
    Ptr<int, Access.ReadWrite> parent = (Ptr<int, Access.ReadWrite>)p.output;
    const int self = visit[rank];
    const float radius = radii[self];
    if (!(radius > 0.0f) || !isfinite(radius))
        return;
    const float3 q = load3(p.points, uint64_t(self));
    int mine = componentRoot(parent, self);
    const uint top = p.levels - 1u;
    uint level = top;
    uint node = 0u;
    bool climbing = false;
    while (true) {
        if (!climbing) {
            const uint shift = kFanoutBits * (level + 1u);
            const uint64_t first = uint64_t(node) << shift;
            const uint64_t span = uint64_t(1) << shift;
            const uint64_t last = first + span < uint64_t(p.references) ? first + span : uint64_t(p.references);
            bool open = last > uint64_t(rank) + 1;
            if (open) {
                const uint64_t index = uint64_t(p.levelOffset[level]) + node;
                const float reachRadius = min(radius, boxRadii[index]);
                if (reachRadius > 0.0f) {
                    const float3 reach = boxReach(q, p.boxes + index * 24, reachRadius);
                    open = reach.x <= reach.z;
                } else {
                    open = false;
                }
            }
            if (open && level != 0u) {
                --level;
                node <<= kFanoutBits;
                continue;
            }
            if (open) {
                for (uint64_t j = max(first, uint64_t(rank) + 1); j < last; ++j) {
                    const float3 other = load3(p.sorted, j);
                    if (!within(q, other, radius) || !within(q, other, sortedRadii[j]))
                        continue;
                    int root = componentRoot(parent, visit[j]);
                    while (mine != root) {
                        int seen;
                        if (mine < root) {
                            InterlockedCompareExchange(parent[root], root, mine, seen);
                            if (seen == root)
                                break;
                            root = seen;
                        } else {
                            InterlockedCompareExchange(parent[mine], mine, root, seen);
                            if (seen == mine)
                                break;
                            mine = seen;
                        }
                    }
                }
            }
        }
        climbing = false;
        ++node;
        if (level == top) {
            if (node >= p.levelCount[top])
                return;
        } else if ((node & (kFanout - 1u)) == 0u || node >= p.levelCount[level]) {
            node = (node - 1u) >> kFanoutBits;
            ++level;
            climbing = true;
        }
    }
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)] void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters p) {
    const uint3 numWorkgroups = spirv_asm {
result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    if (kMode == 1) {
        for (uint t = dispatchThreadId.x; t < p.references; t += stride)
            joinMutual(p, t);
        return;
    }
    Ptr<int, Access.Read> visit = (Ptr<int, Access.Read>)p.visit;
    Ptr<float, Access.Read> radii = (Ptr<float, Access.Read>)p.radii;
    Ptr<int, Access.ReadWrite> output = (Ptr<int, Access.ReadWrite>)p.output;
    for (uint t = dispatchThreadId.x; t < p.count; t += stride) {
        const uint i = uint(visit[t]);
        int result = 0;
        if (p.queries == 0 || loadByte(p.queries, i) != 0u) {
            const float3 q = load3(p.points, i);
            if (all(isfinite(q)))
                result = treeCount(p, q, t < p.references ? int(t) : -1, radii[i]);
        }
        output[i] = result;
    }
}
