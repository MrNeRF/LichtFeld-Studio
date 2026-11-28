#include "native_entry.slangh"
// Moment match for one kept row. Groups hold at most four members.
// Float-float sums track the double host merge; stored channels are float32.
#include "df64.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 24;

struct Parameters {
    uint64_t table;
    uint n;
    uint rest;
    uint pad0;
    uint pad1;
};

static const Df K_4PI = {12.566370964050293, -3.4969110629390343e-07};
static const Df K_P = {1.6074999570846558, 4.2915345943583816e-08};
static const Df K_INVP = {0.6220839619636536, 1.9373826987134635e-08};

Df df_abs(Df a) {
    if (a.hi < 0 || (a.hi == 0 && a.lo < 0)) {
        a.hi = -a.hi;
        a.lo = -a.lo;
    }
    return a;
}
Df df_max(Df a, Df b) { return df_gt(a, b) ? a : b; }
Df df_min(Df a, Df b) { return df_gt(a, b) ? b : a; }
float df_store(Df a) { return a.hi + a.lo; }

Df df_area(Df x, Df y, Df z) {
    const Df xy = df_pow(df_mul(x, y), K_P);
    const Df xz = df_pow(df_mul(x, z), K_P);
    const Df yz = df_pow(df_mul(y, z), K_P);
    const Df mean = df_div(df_add(df_add(xy, xz), yz), df_from(3));
    return df_mul(K_4PI, df_pow(mean, K_INVP));
}

// Row i of `width` floats at its 64-bit byte address, its values as small indices: NVIDIA's Vulkan computes an access
// chain's byte offset in 32 bits without VK_EXT_shader_64bit_indexing, and the 3-float rows (positions, log scales,
// colours) pass 4 GiB at 357.9M points, the quaternions at 268.4M.
Ptr<float, Access.ReadWrite> float_row(uint64_t base, uint i, uint width) {
    return (Ptr<float, Access.ReadWrite>)(base + uint64_t(i) * width * 4u);
}

Df splat_mass(uint64_t scale, Ptr<float, Access.Read> opacity, uint i, float eps) {
    Df axes[3];
    const Ptr<float, Access.ReadWrite> scale_i = float_row(scale, i, 3u);
    [ForceUnroll] for (uint a = 0; a < 3; ++a)
        axes[a] = df_max(df_exp(df_from(scale_i[a])), df_from(1e-12));
    const Df sig = df_sigmoid(opacity[i]);
    return df_add(df_mul(sig, df_area(axes[0], axes[1], axes[2])), df_from(eps));
}

void rotation(uint64_t rotations, uint i, out Df r[9]) {
    const Ptr<float, Access.ReadWrite> rot = float_row(rotations, i, 4u);
    const Df invn = df_div(df_from(1), df_max(df_sqrt(df_add(df_add(df_mul(df_from(rot[0]), df_from(rot[0])), df_mul(df_from(rot[1]), df_from(rot[1]))), df_add(df_mul(df_from(rot[2]), df_from(rot[2])), df_mul(df_from(rot[3]), df_from(rot[3]))))), df_from(1e-12)));
    const Df w = df_mul(df_from(rot[0]), invn);
    const Df x = df_mul(df_from(rot[1]), invn);
    const Df y = df_mul(df_from(rot[2]), invn);
    const Df z = df_mul(df_from(rot[3]), invn);
    const Df two = df_from(2);
    r[0] = df_sub(df_from(1), df_mul(two, df_add(df_mul(y, y), df_mul(z, z))));
    r[1] = df_mul(two, df_sub(df_mul(x, y), df_mul(w, z)));
    r[2] = df_mul(two, df_add(df_mul(x, z), df_mul(w, y)));
    r[3] = df_mul(two, df_add(df_mul(x, y), df_mul(w, z)));
    r[4] = df_sub(df_from(1), df_mul(two, df_add(df_mul(x, x), df_mul(z, z))));
    r[5] = df_mul(two, df_sub(df_mul(y, z), df_mul(w, x)));
    r[6] = df_mul(two, df_sub(df_mul(x, z), df_mul(w, y)));
    r[7] = df_mul(two, df_add(df_mul(y, z), df_mul(w, x)));
    r[8] = df_sub(df_from(1), df_mul(two, df_add(df_mul(x, x), df_mul(y, y))));
}

void covariance(Df r[9], Df variance[3], out Df s[9]) {
    for (uint a = 0; a < 3; ++a) {
        for (uint b = a; b < 3; ++b) {
            Df acc = df_from(0);
            [ForceUnroll] for (uint k = 0; k < 3; ++k)
                acc = df_add(acc, df_mul(df_mul(r[a * 3 + k], r[b * 3 + k]), variance[k]));
            s[a * 3 + b] = acc;
            s[b * 3 + a] = acc;
        }
    }
}

Df df_det(Df a[9]) {
    const Df t0 = df_sub(df_mul(a[4], a[8]), df_mul(a[5], a[7]));
    const Df t1 = df_sub(df_mul(a[3], a[8]), df_mul(a[5], a[6]));
    const Df t2 = df_sub(df_mul(a[3], a[7]), df_mul(a[4], a[6]));
    return df_add(df_sub(df_mul(a[0], t0), df_mul(a[1], t1)), df_mul(a[2], t2));
}

void decompose(inout Df a[9], out Df scales[3], out Df quaternion[4]) {
    Df v[9];
    v[0] = df_from(1); v[1] = df_from(0); v[2] = df_from(0);
    v[3] = df_from(0); v[4] = df_from(1); v[5] = df_from(0);
    v[6] = df_from(0); v[7] = df_from(0); v[8] = df_from(1);
    for (int iter = 0; iter < 24; ++iter) {
        int p = 0;
        int q = 1;
        Df maxAbs = df_abs(a[1]);
        if (df_gt(df_abs(a[2]), maxAbs)) { p = 0; q = 2; maxAbs = df_abs(a[2]); }
        if (df_gt(df_abs(a[5]), maxAbs)) { p = 1; q = 2; maxAbs = df_abs(a[5]); }
        if (df_store(maxAbs) < 1e-12) break;
        const int pp = 3 * p + p;
        const int qq = 3 * q + q;
        const int pq = 3 * p + q;
        const Df app = a[pp];
        const Df aqq = a[qq];
        const Df apq = a[pq];
        const Df tau = df_div(df_sub(aqq, app), df_mul(df_from(2), apq));
        const float sign = df_store(tau) > 0 ? 1 : (df_store(tau) < 0 ? -1 : 0);
        const Df t = df_div(df_from(sign), df_add(df_abs(tau), df_sqrt(df_add(df_from(1), df_mul(tau, tau)))));
        const Df c = df_div(df_from(1), df_sqrt(df_add(df_from(1), df_mul(t, t))));
        const Df s = df_mul(t, c);
        for (int k = 0; k < 3; ++k) {
            if (k == p || k == q) continue;
            const int kp = 3 * k + p;
            const int kq = 3 * k + q;
            const Df akp = a[kp];
            const Df akq = a[kq];
            a[kp] = df_sub(df_mul(c, akp), df_mul(s, akq));
            a[3 * p + k] = a[kp];
            a[kq] = df_add(df_mul(s, akp), df_mul(c, akq));
            a[3 * q + k] = a[kq];
        }
        a[pp] = df_add(df_sub(df_mul(df_mul(c, c), app), df_mul(df_mul(df_from(2), s), df_mul(c, apq))), df_mul(df_mul(s, s), aqq));
        a[qq] = df_add(df_add(df_mul(df_mul(s, s), app), df_mul(df_mul(df_from(2), s), df_mul(c, apq))), df_mul(df_mul(c, c), aqq));
        a[pq] = df_from(0);
        a[3 * q + p] = df_from(0);
        for (int k = 0; k < 3; ++k) {
            const int kp = 3 * k + p;
            const int kq = 3 * k + q;
            const Df vkp = v[kp];
            const Df vkq = v[kq];
            v[kp] = df_sub(df_mul(c, vkp), df_mul(s, vkq));
            v[kq] = df_add(df_mul(s, vkp), df_mul(c, vkq));
        }
    }
    int order[3];
    order[0] = 0; order[1] = 1; order[2] = 2;
    for (int i = 1; i < 3; ++i) {
        int key = order[i];
        int j = i;
        while (j > 0 && df_gt(a[4 * key], a[4 * order[j - 1]])) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = key;
    }
    Df r[9];
    [ForceUnroll] for (uint c = 0; c < 3; ++c) {
        scales[c] = df_sqrt(df_max(a[4 * order[c]], df_from(1e-18)));
        [ForceUnroll] for (uint row = 0; row < 3; ++row)
            r[row * 3 + c] = v[row * 3 + order[c]];
    }
    if (df_gt(df_from(0), df_det(r))) {
        r[2] = df_sub(df_from(0), r[2]);
        r[5] = df_sub(df_from(0), r[5]);
        r[8] = df_sub(df_from(0), r[8]);
    }
    Df w, x, y, z;
    const Df tr = df_add(df_add(r[0], r[4]), r[8]);
    if (df_gt(tr, df_from(0))) {
        const Df s = df_mul(df_sqrt(df_add(tr, df_from(1))), df_from(2));
        w = df_mul(df_from(0.25), s);
        x = df_div(df_sub(r[7], r[5]), s);
        y = df_div(df_sub(r[2], r[6]), s);
        z = df_div(df_sub(r[3], r[1]), s);
    } else if (df_gt(r[0], r[4]) && df_gt(r[0], r[8])) {
        const Df s = df_mul(df_sqrt(df_add(df_sub(df_sub(df_from(1), r[4]), r[8]), r[0])), df_from(2));
        w = df_div(df_sub(r[7], r[5]), s);
        x = df_mul(df_from(0.25), s);
        y = df_div(df_add(r[1], r[3]), s);
        z = df_div(df_add(r[2], r[6]), s);
    } else if (df_gt(r[4], r[8])) {
        const Df s = df_mul(df_sqrt(df_add(df_sub(df_sub(df_from(1), r[0]), r[8]), r[4])), df_from(2));
        w = df_div(df_sub(r[2], r[6]), s);
        x = df_div(df_add(r[1], r[3]), s);
        y = df_mul(df_from(0.25), s);
        z = df_div(df_add(r[5], r[7]), s);
    } else {
        const Df s = df_mul(df_sqrt(df_add(df_sub(df_sub(df_from(1), r[0]), r[4]), r[8])), df_from(2));
        w = df_div(df_sub(r[3], r[1]), s);
        x = df_div(df_add(r[2], r[6]), s);
        y = df_div(df_add(r[5], r[7]), s);
        z = df_mul(df_from(0.25), s);
    }
    const Df inv = df_div(df_from(1), df_max(df_sqrt(df_add(df_add(df_mul(w, w), df_mul(x, x)), df_add(df_mul(y, y), df_mul(z, z)))), df_from(1e-12)));
    quaternion[0] = df_mul(w, inv);
    quaternion[1] = df_mul(x, inv);
    quaternion[2] = df_mul(y, inv);
    quaternion[3] = df_mul(z, inv);
}

// SH rows (rest * 3 floats) start at a 64-bit byte offset: they pass 2^32 floats at 95.4M points at SH 3 (179M at
// SH 2), and the address is integer math because NVIDIA's Vulkan computes an access chain's byte offset in 32 bits
// without VK_EXT_shader_64bit_indexing (the rows pass 4 GiB at 23.9M points). Values are small indices from it.
uint64_t sh_row(uint64_t sh, uint i, uint rest) { return sh + uint64_t(i) * (rest * 3) * 4; }

void copy_one(uint64_t pos, uint64_t rot, uint64_t scale,
              Ptr<float, Access.Read> opacity, uint64_t dc, uint64_t sh,
              uint64_t opos, uint64_t orot, uint64_t oscale,
              Ptr<float, Access.ReadWrite> oopacity, uint64_t odc, uint64_t osh,
              uint rest, uint i, uint row) {
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        float_row(opos, row, 3u)[a] = float_row(pos, i, 3u)[a];
        float_row(oscale, row, 3u)[a] = float_row(scale, i, 3u)[a];
        float_row(odc, row, 3u)[a] = float_row(dc, i, 3u)[a];
    }
    [ForceUnroll] for (uint a = 0; a < 4; ++a) float_row(orot, row, 4u)[a] = float_row(rot, i, 4u)[a];
    oopacity[row] = opacity[i];
    Ptr<float, Access.Read> from = (Ptr<float, Access.Read>)sh_row(sh, i, rest);
    Ptr<float, Access.ReadWrite> to = (Ptr<float, Access.ReadWrite>)sh_row(osh, row, rest);
    for (uint a = 0; a < rest * 3; ++a) to[a] = from[a];
}

void merge_one(uint64_t pos, uint64_t rot, uint64_t scale,
               Ptr<float, Access.Read> opacity, uint64_t dc, uint64_t sh,
               Ptr<uint, Access.Read> members, uint64_t opos, uint64_t orot,
               uint64_t oscale, Ptr<float, Access.ReadWrite> oopacity, uint64_t odc,
               uint64_t osh, uint rest, uint begin, uint count, uint row) {
    Df weights[4];
    Df massSum = df_from(0);
    for (uint m = 0; m < count; ++m) {
        weights[m] = splat_mass(scale, opacity, members[begin + m], 1e-30);
        massSum = df_add(massSum, weights[m]);
    }
    Df mean[3];
    mean[0] = df_from(0); mean[1] = df_from(0); mean[2] = df_from(0);
    for (uint m = 0; m < count; ++m) {
        weights[m] = df_div(weights[m], massSum);
        const uint id = members[begin + m];
        [ForceUnroll] for (uint a = 0; a < 3; ++a)
            mean[a] = df_add(mean[a], df_mul(weights[m], df_from(float_row(pos, id, 3u)[a])));
    }
    Df sig[9];
    for (uint a = 0; a < 9; ++a) sig[a] = df_from(0);
    for (uint m = 0; m < count; ++m) {
        const uint id = members[begin + m];
        Df r[9];
        rotation(rot, id, r);
        Df variance[3];
        Df delta[3];
        [ForceUnroll] for (uint a = 0; a < 3; ++a) {
            const Df sc = df_max(df_exp(df_from(float_row(scale, id, 3u)[a])), df_from(1e-12));
            variance[a] = df_mul(sc, sc);
            delta[a] = df_sub(df_from(float_row(pos, id, 3u)[a]), mean[a]);
        }
        Df si[9];
        covariance(r, variance, si);
        for (uint a = 0; a < 9; ++a) {
            const uint rr = a / 3;
            const uint cc = a - rr * 3;
            sig[a] = df_add(sig[a], df_mul(weights[m], df_add(df_mul(delta[rr], delta[cc]), si[a])));
        }
    }
    sig[0] = df_add(sig[0], df_from(1e-8));
    sig[4] = df_add(sig[4], df_from(1e-8));
    sig[8] = df_add(sig[8], df_from(1e-8));
    Df scales[3];
    Df quat[4];
    decompose(sig, scales, quat);
    [ForceUnroll] for (uint a = 0; a < 3; ++a) {
        float_row(opos, row, 3u)[a] = df_store(mean[a]);
        float_row(oscale, row, 3u)[a] = df_store(df_log(scales[a]));
    }
    [ForceUnroll] for (uint a = 0; a < 4; ++a) float_row(orot, row, 4u)[a] = df_store(quat[a]);
    const Df area = df_max(df_area(scales[0], scales[1], scales[2]), df_from(1e-30));
    Df alpha = df_div(massSum, area);
    // The reference clamps in double; both bounds need float-float constants.
    const Df lowest = {1.0000000116860974e-07, -1.1686097468332243e-15};
    const Df highest = {0.99999988079071045, 1.9209290158528347e-08};
    alpha = df_max(lowest, df_min(highest, alpha));
    oopacity[row] = df_store(df_log(df_div(alpha, df_sub(df_from(1), alpha))));
    for (uint c = 0; c < 3 + rest * 3; ++c) {
        Df acc = df_from(0);
        for (uint m = 0; m < count; ++m) {
            const uint id = members[begin + m];
            const float value = c < 3 ? float_row(dc, id, 3u)[c] : ((Ptr<float, Access.Read>)sh_row(sh, id, rest))[c - 3];
            acc = df_add(acc, df_mul(weights[m], df_from(value)));
        }
        if (c < 3) float_row(odc, row, 3u)[c] = df_store(acc);
        else ((Ptr<float, Access.ReadWrite>)sh_row(osh, row, rest))[c - 3] = df_store(acc);
    }
}

[numthreads(256, 1, 1)]
void LFS_KERNEL_ENTRY(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p LFS_PHASE_ARGUMENT) {
    const uint3 groups = dispatchGroups();
    Ptr<uint64_t, Access.Read> table = (Ptr<uint64_t, Access.Read>)p.table;
    const uint64_t pos = table[0];
    const uint64_t rot = table[1];
    const uint64_t scale = table[2];
    Ptr<float, Access.Read> opacity = (Ptr<float, Access.Read>)table[3];
    const uint64_t dc = table[4];
    const uint64_t sh = table[5];
    Ptr<int, Access.Read> member = (Ptr<int, Access.Read>)table[6];
    Ptr<uint, Access.Read> minimum = (Ptr<uint, Access.Read>)table[7];
    Ptr<uint, Access.Read> members = (Ptr<uint, Access.Read>)table[8];
    Ptr<uint, Access.Read> offsets = (Ptr<uint, Access.Read>)table[9];
    Ptr<uint, Access.Read> rows = (Ptr<uint, Access.Read>)table[10];
    const uint64_t opos = table[11];
    const uint64_t orot = table[12];
    const uint64_t oscale = table[13];
    Ptr<float, Access.ReadWrite> oopacity = (Ptr<float, Access.ReadWrite>)table[14];
    const uint64_t odc = table[15];
    const uint64_t osh = table[16];
    for (uint i = gid.x * 256u + tid.x; i < p.n; i += groups.x * 256u) {
        const int g = member[i];
        if (g < 0) {
            copy_one(pos, rot, scale, opacity, dc, sh, opos, orot, oscale, oopacity, odc, osh, p.rest, i, rows[i]);
        } else if (i == minimum[g]) {
            const uint begin = offsets[g];
            const uint count = offsets[g + 1] - begin;
            merge_one(pos, rot, scale, opacity, dc, sh, members, opos, orot, oscale, oopacity, odc, osh, p.rest, begin, count, rows[i]);
        }
    }
}
