// Float-float edge costs. The final value is rounded to float32, which is the
// precision the decimator sorts; the double host oracle is the reference.
#include "df64.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 96;

struct Parameters {
    uint64_t pos;
    uint64_t rot;
    uint64_t scale;
    uint64_t opacity;
    uint64_t dc;
    uint64_t sh;
    uint64_t neighbors;
    uint64_t cache;
    uint64_t idx;
    uint64_t cost;
    uint n;
    uint rest;
    uint pad0;
    uint pad1;
};

struct GCache {
    float r[9];
    float v[3];
    float invv[3];
    float sigma[9];
    float logdet;
    float mass;
    Df sample0;
    Df sample1;
    Df sample2;
    Df selfLog;
};

float df_as(Df a) { return a.hi + a.lo; }
Df df_floor(Df a, float bound) { return df_as(a) > bound ? a : df_from(bound); }
Df df_sample(GCache c, uint axis) {
    if (axis == 0) return c.sample0;
    if (axis == 1) return c.sample1;
    return c.sample2;
}

static const Df K_4PI = {12.566370964050293, -3.4969110629390343e-07};
static const Df K_LOG2PI = {1.8378770351409912, 3.126835323996602e-08};
static const Df K_P = {1.6074999570846558, 4.2915345943583816e-08};
static const Df K_INVP = {0.6220839619636536, 1.9373826987134635e-08};
static const Df K_Z0 = {1.6264322996139526, 8.576314769470628e-09};
static const Df K_Z1 = {0.003369733924046159, 1.0921582388467854e-10};
static const Df K_Z2 = {1.0509958267211914, 1.7497320925485838e-08};
static const Df K_JITTER = {9.9999999392252903e-09, 6.077470660972466e-17};

Df df_area(Df x, Df y, Df z) {
    const Df xy = df_pow(df_mul(x, y), K_P);
    const Df xz = df_pow(df_mul(x, z), K_P);
    const Df yz = df_pow(df_mul(y, z), K_P);
    const Df mean = df_div(df_add(df_add(xy, xz), yz), df_from(3));
    return df_mul(K_4PI, df_pow(mean, K_INVP));
}

void cache_one(Ptr<float, Access.Read> pos, Ptr<float, Access.Read> rot, Ptr<float, Access.Read> scale,
               Ptr<float, Access.Read> opacity, uint i, out GCache c) {
    Df variance[3];
    Df axes[3];
    Df ld = df_from(0);
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        const Df s = df_floor(df_exp_precise(df_from(scale[i * 3 + a])), 1e-12);
        axes[a] = s;
        variance[a] = df_add(df_mul(s, s), K_JITTER);
        c.v[a] = df_as(variance[a]);
        c.invv[a] = df_as(df_div(df_from(1), df_floor(variance[a], 1e-30)));
        ld = df_add(ld, df_log(df_floor(variance[a], 1e-30)));
    }
    c.logdet = df_as(ld);
    const Df sigmoid = df_sigmoid(opacity[i]);
    c.mass = df_as(df_add(df_mul(sigmoid, df_area(axes[0], axes[1], axes[2])), df_from(1e-12)));

    const Df q0 = df_from(rot[i * 4 + 0]);
    const Df q1 = df_from(rot[i * 4 + 1]);
    const Df q2 = df_from(rot[i * 4 + 2]);
    const Df q3 = df_from(rot[i * 4 + 3]);
    const Df inv = df_div(df_from(1), df_floor(df_sqrt(df_add(df_add(df_mul(q0, q0), df_mul(q1, q1)), df_add(df_mul(q2, q2), df_mul(q3, q3)))), 1e-12));
    const Df w = df_mul(q0, inv);
    const Df x = df_mul(q1, inv);
    const Df y = df_mul(q2, inv);
    const Df z = df_mul(q3, inv);
    const Df two = df_from(2);
    c.r[0] = df_as(df_sub(df_from(1), df_mul(two, df_add(df_mul(y, y), df_mul(z, z)))));
    c.r[1] = df_as(df_mul(two, df_sub(df_mul(x, y), df_mul(w, z))));
    c.r[2] = df_as(df_mul(two, df_add(df_mul(x, z), df_mul(w, y))));
    c.r[3] = df_as(df_mul(two, df_add(df_mul(x, y), df_mul(w, z))));
    c.r[4] = df_as(df_sub(df_from(1), df_mul(two, df_add(df_mul(x, x), df_mul(z, z)))));
    c.r[5] = df_as(df_mul(two, df_sub(df_mul(y, z), df_mul(w, x))));
    c.r[6] = df_as(df_mul(two, df_sub(df_mul(x, z), df_mul(w, y))));
    c.r[7] = df_as(df_mul(two, df_add(df_mul(y, z), df_mul(w, x))));
    c.r[8] = df_as(df_sub(df_from(1), df_mul(two, df_add(df_mul(x, x), df_mul(y, y)))));
    for (uint a = 0; a < 3; ++a) {
        for (uint b = a; b < 3; ++b) {
            Df acc = df_from(0);
            [ForceUnroll] for (uint k = 0; k < 3; ++k)
                acc = df_add(acc, df_mul(df_mul(df_from(c.r[a * 3 + k]), df_from(c.r[b * 3 + k])), variance[k]));
            const float stored = df_as(acc);
            c.sigma[a * 3 + b] = stored;
            c.sigma[b * 3 + a] = stored;
        }
    }
    const Df zed[3] = {K_Z0, K_Z1, K_Z2};
    Df scaled[3];
    [ForceUnroll] for (uint a = 0; a < 3; ++a)
        scaled[a] = df_mul(zed[a], df_sqrt(df_floor(df_from(c.v[a]), 0)));
    Df sample[3];
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        sample[a] = df_add(df_from(pos[i * 3 + a]),
                           df_add(df_add(df_mul(scaled[0], df_from(c.r[a * 3])), df_mul(scaled[1], df_from(c.r[a * 3 + 1]))),
                                  df_mul(scaled[2], df_from(c.r[a * 3 + 2]))));
    }
    c.sample0 = sample[0];
    c.sample1 = sample[1];
    c.sample2 = sample[2];
    Df d[3];
    [ForceUnroll] for (uint a = 0; a < 3; ++a)
        d[a] = df_sub(sample[a], df_from(pos[i * 3 + a]));
    Df q = df_from(0);
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        const Df yv = df_add(df_add(df_mul(d[0], df_from(c.r[a])), df_mul(d[1], df_from(c.r[3 + a]))), df_mul(d[2], df_from(c.r[6 + a])));
        q = df_add(q, df_mul(df_mul(yv, yv), df_from(c.invv[a])));
    }
    c.selfLog = df_mul(df_from(-0.5), df_add(df_add(df_mul(df_from(3), K_LOG2PI), df_from(c.logdet)), q));
}

Df df_logpdf(Df x0, Df x1, Df x2, float m0, float m1, float m2, GCache c) {
    const Df d0 = df_sub(x0, df_from(m0));
    const Df d1 = df_sub(x1, df_from(m1));
    const Df d2 = df_sub(x2, df_from(m2));
    const Df d[3] = {d0, d1, d2};
    Df q = df_from(0);
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        const Df yv = df_add(df_add(df_mul(d[0], df_from(c.r[a])), df_mul(d[1], df_from(c.r[3 + a]))), df_mul(d[2], df_from(c.r[6 + a])));
        q = df_add(q, df_mul(df_mul(yv, yv), df_from(c.invv[a])));
    }
    return df_mul(df_from(-0.5), df_add(df_add(df_mul(df_from(3), K_LOG2PI), df_from(c.logdet)), q));
}

Df df_logadd(Df a, Df b) {
    const Df m = df_as(a) > df_as(b) ? a : b;
    return df_add(m, df_log(df_add(df_exp(df_sub(a, m)), df_exp(df_sub(b, m)))));
}

Df df_det(Df s[9]) {
    const Df t0 = df_sub(df_mul(s[4], s[8]), df_mul(s[5], s[7]));
    const Df t1 = df_sub(df_mul(s[3], s[8]), df_mul(s[5], s[6]));
    const Df t2 = df_sub(df_mul(s[3], s[7]), df_mul(s[4], s[6]));
    return df_add(df_sub(df_mul(s[0], t0), df_mul(s[1], t1)), df_mul(s[2], t2));
}

float edge_cost(Ptr<float, Access.Read> pos, Ptr<float, Access.Read> dc, Ptr<float, Access.Read> sh, uint rest,
                GCache a, GCache b, uint i, uint j) {
    const Df w = df_add(df_from(a.mass), df_from(b.mass));
    Df p = df_div(df_from(a.mass), df_as(w) > 0 ? w : df_from(1));
    const float pf = df_as(p);
    p = pf < 1e-12 ? df_from(1e-12) : (pf > 1 - 1e-12 ? df_from(1 - 1e-12) : p);
    const Df q = df_sub(df_from(1), p);
    const Df lp = df_log(p);
    const Df lq = df_log(q);
    Df d[3];
    Df e[3];
    Df s[9];
    [ForceUnroll] for (uint c = 0; c < 3; ++c) {
        const Df m = df_add(df_mul(p, df_from(pos[i * 3 + c])), df_mul(q, df_from(pos[j * 3 + c])));
        d[c] = df_sub(df_from(pos[i * 3 + c]), m);
        e[c] = df_sub(df_from(pos[j * 3 + c]), m);
    }
    [ForceUnroll] for (uint c = 0; c < 9; ++c) {
        const uint row = c / 3;
        const uint col = c - row * 3;
        s[c] = df_add(df_add(df_add(df_mul(p, df_from(a.sigma[c])), df_mul(q, df_from(b.sigma[c]))), df_mul(p, df_mul(d[row], d[col]))),
                      df_mul(q, df_mul(e[row], e[col])));
    }
    const Df s13 = df_mul(df_from(0.5), df_add(s[1], s[3]));
    const Df s26 = df_mul(df_from(0.5), df_add(s[2], s[6]));
    const Df s57 = df_mul(df_from(0.5), df_add(s[5], s[7]));
    s[1] = s13;
    s[3] = s13;
    s[2] = s26;
    s[6] = s26;
    s[5] = s57;
    s[7] = s57;
    s[0] = df_add(s[0], K_JITTER);
    s[4] = df_add(s[4], K_JITTER);
    s[8] = df_add(s[8], K_JITTER);
    const Df crossA = df_logadd(df_add(lp, a.selfLog), df_add(lq, df_logpdf(a.sample0, a.sample1, a.sample2, pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2], b)));
    const Df crossB = df_logadd(df_add(lp, df_logpdf(b.sample0, b.sample1, b.sample2, pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], a)), df_add(lq, b.selfLog));
    Df cost = df_add(df_add(df_mul(p, crossA), df_mul(q, crossB)), df_mul(df_from(0.5), df_add(df_add(df_mul(df_from(3), K_LOG2PI), df_log(df_floor(df_det(s), 1e-30))), df_from(3))));
    for (uint c = 0; c < 3 + rest * 3; ++c) {
        float left = c < 3 ? dc[i * 3 + c] : sh[i * rest * 3 + c - 3];
        float right = c < 3 ? dc[j * 3 + c] : sh[j * rest * 3 + c - 3];
        const float delta = left - right;
        cost = df_add(cost, df_from(delta * delta));
    }
    return df_as(cost);
}

void store_cache(Ptr<float, Access.ReadWrite> cache, uint i, GCache c) {
    Ptr<float, Access.ReadWrite> o = cache + i * 32;
    [ForceUnroll] for (uint a = 0; a < 9; ++a) o[a] = c.r[a];
    [ForceUnroll] for (uint a = 0; a < 3; ++a) o[9 + a] = c.invv[a];
    [ForceUnroll] for (uint a = 0; a < 9; ++a) o[12 + a] = c.sigma[a];
    o[21] = c.logdet;
    o[22] = c.mass;
    o[23] = c.sample0.hi; o[24] = c.sample0.lo;
    o[25] = c.sample1.hi; o[26] = c.sample1.lo;
    o[27] = c.sample2.hi; o[28] = c.sample2.lo;
    o[29] = c.selfLog.hi; o[30] = c.selfLog.lo;
}

void load_cache(Ptr<float, Access.ReadWrite> cache, uint i, out GCache c) {
    Ptr<float, Access.ReadWrite> o = cache + i * 32;
    [ForceUnroll] for (uint a = 0; a < 9; ++a) c.r[a] = o[a];
    [ForceUnroll] for (uint a = 0; a < 3; ++a) c.invv[a] = o[9 + a];
    [ForceUnroll] for (uint a = 0; a < 9; ++a) c.sigma[a] = o[12 + a];
    c.logdet = o[21];
    c.mass = o[22];
    c.sample0.hi = o[23]; c.sample0.lo = o[24];
    c.sample1.hi = o[25]; c.sample1.lo = o[26];
    c.sample2.hi = o[27]; c.sample2.lo = o[28];
    c.selfLog.hi = o[29]; c.selfLog.lo = o[30];
}

[SpecializationConstant]
const uint kPhase = 0;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Ptr<float, Access.Read> pos = (Ptr<float, Access.Read>)p.pos;
    Ptr<float, Access.Read> rot = (Ptr<float, Access.Read>)p.rot;
    Ptr<float, Access.Read> scale = (Ptr<float, Access.Read>)p.scale;
    Ptr<float, Access.Read> opacity = (Ptr<float, Access.Read>)p.opacity;
    Ptr<float, Access.Read> dc = (Ptr<float, Access.Read>)p.dc;
    Ptr<float, Access.Read> sh = (Ptr<float, Access.Read>)p.sh;
    Ptr<uint, Access.Read> neighbors = (Ptr<uint, Access.Read>)p.neighbors;
    Ptr<float, Access.ReadWrite> cache = (Ptr<float, Access.ReadWrite>)p.cache;
    Ptr<uint, Access.ReadWrite> idx = (Ptr<uint, Access.ReadWrite>)p.idx;
    Ptr<float, Access.ReadWrite> cost = (Ptr<float, Access.ReadWrite>)p.cost;
    if (kPhase == 0) {
        for (uint i = gid.x * 256u + tid.x; i < p.n; i += groups.x * 256u) {
            GCache built;
            cache_one(pos, rot, scale, opacity, i, built);
            store_cache(cache, i, built);
        }
        return;
    }
    for (uint i = gid.x * 256u + tid.x; i < p.n; i += groups.x * 256u) {
        GCache mine;
        load_cache(cache, i, mine);
        const float inf = asfloat(0x7f800000u);
        float best[4] = {inf, inf, inf, inf};
        uint chosen[4] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        for (uint a = 0; a < 16; ++a) {
            const uint j = neighbors[i * 16 + a];
            if (j == 0xffffffffu) continue;
            GCache other;
            load_cache(cache, j, other);
            const float value = edge_cost(pos, dc, sh, p.rest, mine, other, i, j);
            if (!isfinite(value) || value > best[3] || (value == best[3] && j >= chosen[3])) continue;
            int at = 3;
            while (at > 0 && (value < best[at - 1] || (value == best[at - 1] && j < chosen[at - 1]))) {
                best[at] = best[at - 1];
                chosen[at] = chosen[at - 1];
                --at;
            }
            best[at] = value;
            chosen[at] = j;
        }
        [ForceUnroll] for (uint a = 0; a < 4; ++a) {
            idx[i * 4 + a] = chosen[a];
            cost[i * 4 + a] = chosen[a] == 0xffffffffu ? asfloat(0x7f800000u) : best[a];
        }
    }
}
