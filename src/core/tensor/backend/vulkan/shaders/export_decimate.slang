#include "df64.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 48;

[SpecializationConstant]
const uint kPhase = 0;

struct Parameters {
    uint64_t pos;
    uint64_t order;
    uint64_t boxes;
    uint64_t neighbors;
    uint n;
    uint leaves;
    uint begin;
    uint padding;
};

float box_distance(float3 p, Ptr<float, Access.ReadWrite> boxes, uint node) {
    float d = 0;
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        const float low = boxes[node * 6 + a];
        const float high = boxes[node * 6 + 3 + a];
        const float delta = max(max(low - p[a], p[a] - high), 0);
        d += delta * delta;
    }
    return d;
}

// A squared distance is m * 2^e with m in [1, 2). Each pair is scaled by its own largest
// delta, so no scene-wide scale can underflow nearby distances or overflow distant ones.
struct Wd {
    int e;
    Df m;
};

static const int WD_ZERO = -1000000;
static const int WD_INF = 1000000;

Wd wd_inf() {
    Wd r;
    r.e = WD_INF;
    r.m = df_from(1);
    return r;
}

int float_exponent(float x) { return int((asuint(x) >> 23) & 0xffu) - 127; }

bool wd_gt(Wd a, Wd b) { return a.e != b.e ? a.e > b.e : df_gt(a.m, b.m); }

Wd squared_distance(Ptr<float, Access.Read> pos, uint i, uint j) {
    Df delta[3];
    int top = WD_ZERO;
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        delta[a] = df_sub(df_from(pos[i * 3 + a]), df_from(pos[j * 3 + a]));
        if (delta[a].hi != 0) top = max(top, float_exponent(delta[a].hi));
    }
    Wd r;
    if (top == WD_ZERO) {
        r.e = WD_ZERO;
        r.m = df_from(0);
        return r;
    }
    Df d2 = df_from(0);
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        const Df scaled = df_scale2(delta[a], -top);
        d2 = df_add(d2, df_mul(scaled, scaled));
    }
    if (!isfinite(d2.hi)) return wd_inf();
    const int t = float_exponent(d2.hi);
    r.e = 2 * top + t;
    r.m = df_scale2(d2, -t);
    return r;
}

struct NeighbourHeap {
    Wd distance[16];
    uint ids[16];
    Wd worst;
    float search;
    uint worstId;
};

float unscaled_limit(Wd worst) {
    if (worst.e >= 128) return asfloat(0x7f800000u);
    // Below the normal float range the float pre-filter cannot resolve distances; keep all.
    if (worst.e < -120) return asfloat(8u << 23);
    // Slightly wider than the double round-up so a short float-float residual
    // cannot prune a neighbour the double heap would keep.
    return df_round_up(df_scale2(df_mul(worst.m, df_from(1.000004)), worst.e));
}

void consider(inout NeighbourHeap heap, Ptr<float, Access.Read> pos, uint i, uint j) {
    if (j == i) return;
    float approx = 0;
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        const float delta = pos[i * 3 + a] - pos[j * 3 + a];
        approx += delta * delta;
    }
    if (approx > heap.search) return;
    const Wd d2 = squared_distance(pos, i, j);
    const bool fartherThanWorst = wd_gt(d2, heap.worst);
    const bool equalWorst = !fartherThanWorst && !wd_gt(heap.worst, d2);
    if (fartherThanWorst || (equalWorst && j >= heap.worstId)) return;
    int slot = 0;
    while (slot * 2 + 1 < 16) {
        int child = slot * 2 + 1;
        if (child + 1 < 16 && wd_gt(heap.distance[child + 1], heap.distance[child])) ++child;
        else if (child + 1 < 16 && !wd_gt(heap.distance[child], heap.distance[child + 1]) && heap.ids[child + 1] > heap.ids[child]) ++child;
        const bool childFarther = wd_gt(heap.distance[child], d2) || (!wd_gt(d2, heap.distance[child]) && heap.ids[child] > j);
        if (!childFarther) break;
        heap.distance[slot] = heap.distance[child];
        heap.ids[slot] = heap.ids[child];
        slot = child;
    }
    heap.distance[slot] = d2;
    heap.ids[slot] = j;
    heap.worst = heap.ids[0] == 0xffffffffu ? wd_inf() : heap.distance[0];
    heap.worstId = heap.ids[0];
    heap.search = unscaled_limit(heap.worst);
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Ptr<float, Access.Read> pos = (Ptr<float, Access.Read>)p.pos;
    Ptr<uint, Access.Read> order = (Ptr<uint, Access.Read>)p.order;
    Ptr<float, Access.ReadWrite> boxes = (Ptr<float, Access.ReadWrite>)p.boxes;
    if (kPhase == 0) {
        for (uint leaf = gid.x * 256u + tid.x; leaf < p.leaves; leaf += groups.x * 256u) {
            const float inf = asfloat(0x7f800000u);
            float3 lo = float3(inf, inf, inf);
            float3 hi = float3(-inf, -inf, -inf);
            for (uint t = leaf * 8u; t < (leaf + 1u) * 8u && t < p.n; ++t) {
                const uint i = order[t];
                const float3 v = float3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
                lo = min(lo, v);
                hi = max(hi, v);
            }
            const uint node = p.leaves + leaf;
            [ForceUnroll] for (uint a = 0; a < 3; ++a) {
                boxes[node * 6 + a] = lo[a];
                boxes[node * 6 + 3 + a] = hi[a];
            }
        }
    } else if (kPhase == 1) {
        for (uint t = gid.x * 256u + tid.x; t < p.begin; t += groups.x * 256u) {
            const uint i = p.begin + t;
            [ForceUnroll] for (uint a = 0; a < 3; ++a) {
                boxes[i * 6 + a] = min(boxes[(i * 2) * 6 + a], boxes[(i * 2 + 1) * 6 + a]);
                boxes[i * 6 + 3 + a] = max(boxes[(i * 2) * 6 + 3 + a], boxes[(i * 2 + 1) * 6 + 3 + a]);
            }
        }
    } else {
        Ptr<uint, Access.ReadWrite> result = (Ptr<uint, Access.ReadWrite>)p.neighbors;
        for (uint row = gid.x * 256u + tid.x; row < p.n; row += groups.x * 256u) {
            const uint i = order[row];
            const float3 point = float3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
            NeighbourHeap heap;
            [ForceUnroll] for (uint a = 0; a < 16; ++a) {
                heap.distance[a] = wd_inf();
                heap.ids[a] = 0xffffffffu;
            }
            heap.worst = wd_inf();
            heap.search = asfloat(0x7f800000u);
            heap.worstId = 0xffffffffu;
            const uint seedBegin = row > 32u ? row - 32u : 0u;
            const uint seedEnd = min(p.n, seedBegin + 65u);
            for (uint cursor = seedBegin; cursor < seedEnd; ++cursor)
                consider(heap, pos, i, order[cursor]);
            uint node = 1u;
            while (node != 0) {
                if (!(box_distance(point, boxes, node) > heap.search)) {
                    if (node < p.leaves) {
                        node *= 2u;
                        continue;
                    }
                    const uint begin = (node - p.leaves) * 8u;
                    for (uint cursor = begin; cursor < begin + 8u && cursor < p.n; ++cursor) {
                        if (cursor < seedBegin || cursor >= seedEnd)
                            consider(heap, pos, i, order[cursor]);
                    }
                }
                // In the implicit tree, an even node has an unvisited right sibling.
                while ((node & 1u) != 0) node >>= 1u;
                if (node != 0) ++node;
            }
            [ForceUnroll] for (uint a = 0; a < 16; ++a) result[i * 16u + a] = heap.ids[a];
        }
    }
}
