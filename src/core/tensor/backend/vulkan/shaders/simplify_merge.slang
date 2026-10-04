// One thread per group: the moment-matching merge of splat_simplify.cpp (merge_voxel_groups), with the
// same order of accumulation over the members; mirrors cuda/kernels/simplify_merge.cu.
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 104;

static const float kEpsCov = 1e-8f;
static const float kMinScale = 1e-12f;
static const float kMinQuatNorm = 1e-12f;
static const float kMinEval = 1e-18f;
static const float kFloatEpsilon = 1.1920929e-07f;
static const int kJacobiIterations = 32;

struct Parameters {
    uint64_t means;
    uint64_t scales;
    uint64_t rotation;
    uint64_t opacity;
    uint64_t appearance;
    uint64_t offsets;
    uint64_t members;
    uint64_t outMeans;
    uint64_t outScales;
    uint64_t outRotation;
    uint64_t outOpacity;
    uint64_t outAppearance;
    uint groups;
    uint appDim;
};

float readFloat(uint64_t address, uint64_t index) {
    Ptr<float, Access.Read> data = (Ptr<float, Access.Read>)address;
    return data[index];
}

void writeFloat(uint64_t address, uint64_t index, float value) {
    Ptr<float, Access.ReadWrite> data = (Ptr<float, Access.ReadWrite>)address;
    data[index] = value;
}

int readInt(uint64_t address, uint64_t index) {
    Ptr<int, Access.Read> data = (Ptr<int, Access.Read>)address;
    return data[index];
}

void quatToRotmat(float qw, float qx, float qy, float qz, out float R[9]) {
    const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const float wx = qw * qx, wy = qw * qy, wz = qw * qz;
    const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    R[0] = 1.0f - 2.0f * (yy + zz);
    R[1] = 2.0f * (xy - wz);
    R[2] = 2.0f * (xz + wy);
    R[3] = 2.0f * (xy + wz);
    R[4] = 1.0f - 2.0f * (xx + zz);
    R[5] = 2.0f * (yz - wx);
    R[6] = 2.0f * (xz - wy);
    R[7] = 2.0f * (yz + wx);
    R[8] = 1.0f - 2.0f * (xx + yy);
}

void sigmaFromRotVar(float R[9], float vx, float vy, float vz, out float result[9]) {
    float variance[3] = { vx, vy, vz };
    float scaled[9];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            scaled[row * 3 + col] = R[row * 3 + col] * variance[col];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            float sum = 0.0f;
            sum = fma(scaled[row * 3 + 0], R[col * 3 + 0], sum);
            sum = fma(scaled[row * 3 + 1], R[col * 3 + 1], sum);
            sum = fma(scaled[row * 3 + 2], R[col * 3 + 2], sum);
            result[row * 3 + col] = sum;
        }
}

float det3(float A[9]) {
    return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
           A[2] * (A[3] * A[7] - A[4] * A[6]);
}

// Jacobi eigendecomposition, eigenvalues descending (ties by column), right-handed vectors.
void eigenSymmetric(float input[9], out float values[3], out float vectors[9]) {
    float A[9] = input;
    float V[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    for (int iter = 0; iter < kJacobiIterations; ++iter) {
        int p = 0, q = 1;
        float maxAbs = abs(A[1]);
        if (abs(A[2]) > maxAbs) {
            p = 0;
            q = 2;
            maxAbs = abs(A[2]);
        }
        if (abs(A[5]) > maxAbs) {
            p = 1;
            q = 2;
            maxAbs = abs(A[5]);
        }
        if (maxAbs < 1e-12f)
            break;
        const int pp = 3 * p + p, qq = 3 * q + q, pq = 3 * p + q;
        const float app = A[pp], aqq = A[qq], apq = A[pq];
        const float tau = (aqq - app) / (2.0f * apq);
        const float sign = asfloat((asuint(tau) & 0x80000000u) | asuint(1.0f));
        const float t = sign / (abs(tau) + sqrt(1.0f + tau * tau));
        const float c = 1.0f / sqrt(1.0f + t * t);
        const float s = t * c;
        for (int k = 0; k < 3; ++k) {
            if (k == p || k == q)
                continue;
            const int kp = 3 * k + p, kq = 3 * k + q;
            const float akp = A[kp], akq = A[kq];
            A[kp] = c * akp - s * akq;
            A[3 * p + k] = A[kp];
            A[kq] = s * akp + c * akq;
            A[3 * q + k] = A[kq];
        }
        A[pp] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        A[qq] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
        A[pq] = 0.0f;
        A[3 * q + p] = 0.0f;
        for (int k = 0; k < 3; ++k) {
            const int kp = 3 * k + p, kq = 3 * k + q;
            const float vkp = V[kp], vkq = V[kq];
            V[kp] = c * vkp - s * vkq;
            V[kq] = s * vkp + c * vkq;
        }
    }
    float raw[3] = { A[0], A[4], A[8] };
    int order[3] = { 0, 1, 2 };
    for (int i = 1; i < 3; ++i)
        for (int j = i; j > 0; --j) {
            const int lhs = order[j], rhs = order[j - 1];
            if (!(raw[lhs] > raw[rhs] || (raw[lhs] == raw[rhs] && lhs < rhs)))
                break;
            order[j] = rhs;
            order[j - 1] = lhs;
        }
    for (int col = 0; col < 3; ++col) {
        values[col] = raw[order[col]];
        for (int row = 0; row < 3; ++row)
            vectors[row * 3 + col] = V[row * 3 + order[col]];
    }
    if (det3(vectors) < 0.0f) {
        vectors[2] *= -1.0f;
        vectors[5] *= -1.0f;
        vectors[8] *= -1.0f;
    }
}

float4 rotmatToQuat(float R[9]) {
    const float tr = R[0] + R[4] + R[8];
    float qw, qx, qy, qz;
    if (tr > 0.0f) {
        const float S = sqrt(tr + 1.0f) * 2.0f;
        qw = 0.25f * S;
        qx = (R[7] - R[5]) / S;
        qy = (R[2] - R[6]) / S;
        qz = (R[3] - R[1]) / S;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        const float S = sqrt(1.0f + R[0] - R[4] - R[8]) * 2.0f;
        qw = (R[7] - R[5]) / S;
        qx = 0.25f * S;
        qy = (R[1] + R[3]) / S;
        qz = (R[2] + R[6]) / S;
    } else if (R[4] > R[8]) {
        const float S = sqrt(1.0f + R[4] - R[0] - R[8]) * 2.0f;
        qw = (R[2] - R[6]) / S;
        qx = (R[1] + R[3]) / S;
        qy = 0.25f * S;
        qz = (R[5] + R[7]) / S;
    } else {
        const float S = sqrt(1.0f + R[8] - R[0] - R[4]) * 2.0f;
        qw = (R[3] - R[1]) / S;
        qx = (R[2] + R[6]) / S;
        qy = (R[5] + R[7]) / S;
        qz = 0.25f * S;
    }
    const float invN = 1.0f / max(sqrt(qw * qw + qx * qx + qy * qy + qz * qz), kMinQuatNorm);
    return float4(qw * invN, qx * invN, qy * invN, qz * invN);
}

float memberWeight(Parameters p, int row, inout bool collapsed) {
    const uint64_t r3 = uint64_t(row) * 3;
    const float sx = readFloat(p.scales, r3), sy = readFloat(p.scales, r3 + 1), sz = readFloat(p.scales, r3 + 2);
    collapsed = collapsed || sx == 0.0f || sy == 0.0f || sz == 0.0f;
    const float volume = max(sx, kMinScale) * max(sy, kMinScale) * max(sz, kMinScale);
    return volume * readFloat(p.opacity, uint64_t(row));
}

void mergeGroup(Parameters p, uint g) {
    const int begin = readInt(p.offsets, g), end = readInt(p.offsets, g + 1);
    const uint64_t o = uint64_t(g);
    if (end - begin == 1) {
        const uint64_t row = uint64_t(readInt(p.members, uint64_t(begin)));
        for (int a = 0; a < 3; ++a) {
            writeFloat(p.outMeans, o * 3 + a, readFloat(p.means, row * 3 + a));
            writeFloat(p.outScales, o * 3 + a, readFloat(p.scales, row * 3 + a));
        }
        for (int a = 0; a < 4; ++a)
            writeFloat(p.outRotation, o * 4 + a, readFloat(p.rotation, row * 4 + a));
        writeFloat(p.outOpacity, o, readFloat(p.opacity, row));
        for (uint k = 0; k < p.appDim; ++k)
            writeFloat(p.outAppearance, o * p.appDim + k, readFloat(p.appearance, row * p.appDim + k));
        return;
    }
    // Weights by volume and opacity; collapsed members get normalized relative weights.
    float total = 0.0f;
    float largest = -3.40282347e+38f;
    bool hasCollapsed = false;
    for (int m = begin; m < end; ++m) {
        const float w = memberWeight(p, readInt(p.members, uint64_t(m)), hasCollapsed);
        total += w;
        largest = max(largest, w);
    }
    const bool relative = total < 1e-30f && hasCollapsed;
    if (total < 1e-30f && !hasCollapsed) {
        total = 1e-30f;
    } else if (relative) {
        total = 0.0f;
        for (int m = begin; m < end; ++m) {
            bool ignored = false;
            const float w = memberWeight(p, readInt(p.members, uint64_t(m)), ignored);
            total += largest > 0.0f ? w / largest : 1.0f;
        }
    }

    const uint64_t origin = uint64_t(readInt(p.members, uint64_t(begin))) * 3;
    const float ox = hasCollapsed ? readFloat(p.means, origin) : 0.0f;
    const float oy = hasCollapsed ? readFloat(p.means, origin + 1) : 0.0f;
    const float oz = hasCollapsed ? readFloat(p.means, origin + 2) : 0.0f;
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (int m = begin; m < end; ++m) {
        const int row = readInt(p.members, uint64_t(m));
        bool ignored = false;
        float w = memberWeight(p, row, ignored);
        if (relative)
            w = largest > 0.0f ? w / largest : 1.0f;
        w = w / total;
        const uint64_t r3 = uint64_t(row) * 3;
        cx += w * (readFloat(p.means, r3) - ox);
        cy += w * (readFloat(p.means, r3 + 1) - oy);
        cz += w * (readFloat(p.means, r3 + 2) - oz);
    }
    cx += ox;
    cy += oy;
    cz += oz;
    writeFloat(p.outMeans, o * 3, cx);
    writeFloat(p.outMeans, o * 3 + 1, cy);
    writeFloat(p.outMeans, o * 3 + 2, cz);

    float sigma[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int m = begin; m < end; ++m) {
        const int row = readInt(p.members, uint64_t(m));
        bool ignored = false;
        float w = memberWeight(p, row, ignored);
        if (relative)
            w = largest > 0.0f ? w / largest : 1.0f;
        w = w / total;
        const uint64_t r3 = uint64_t(row) * 3, r4 = uint64_t(row) * 4;
        const float sx = readFloat(p.scales, r3), sy = readFloat(p.scales, r3 + 1), sz = readFloat(p.scales, r3 + 2);
        float R[9];
        quatToRotmat(readFloat(p.rotation, r4), readFloat(p.rotation, r4 + 1), readFloat(p.rotation, r4 + 2),
                     readFloat(p.rotation, r4 + 3), R);
        float sig[9];
        sigmaFromRotVar(R, sx * sx, sy * sy, sz * sz, sig);
        const float dx = readFloat(p.means, r3) - cx;
        const float dy = readFloat(p.means, r3 + 1) - cy;
        const float dz = readFloat(p.means, r3 + 2) - cz;
        sig[0] += dx * dx;
        sig[1] += dx * dy;
        sig[2] += dx * dz;
        sig[3] += dy * dx;
        sig[4] += dy * dy;
        sig[5] += dy * dz;
        sig[6] += dz * dx;
        sig[7] += dz * dy;
        sig[8] += dz * dz;
        for (int a = 0; a < 9; ++a)
            sigma[a] += w * sig[a];
    }
    sigma[1] = 0.5f * (sigma[1] + sigma[3]);
    sigma[3] = sigma[1];
    sigma[2] = 0.5f * (sigma[2] + sigma[6]);
    sigma[6] = sigma[2];
    sigma[5] = 0.5f * (sigma[5] + sigma[7]);
    sigma[7] = sigma[5];
    if (!hasCollapsed) {
        sigma[0] += kEpsCov;
        sigma[4] += kEpsCov;
        sigma[8] += kEpsCov;
    }
    float values[3], vectors[9];
    eigenSymmetric(sigma, values, vectors);
    const float zeroTolerance = hasCollapsed ? max(max(max(values[0], values[1]), values[2]), 0.0f) * (8.0f * kFloatEpsilon)
                                             : 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const float variance = max(values[axis], hasCollapsed ? 0.0f : kMinEval);
        float scale = 0.0f;
        if (variance > zeroTolerance) {
            const float raw = log(max(sqrt(variance), kMinScale));
            scale = max(exp(clamp(raw, -30.0f, 30.0f)), kMinScale);
        }
        writeFloat(p.outScales, o * 3 + axis, scale);
    }
    const float4 q = rotmatToQuat(vectors);
    writeFloat(p.outRotation, o * 4, q.x);
    writeFloat(p.outRotation, o * 4 + 1, q.y);
    writeFloat(p.outRotation, o * 4 + 2, q.z);
    writeFloat(p.outRotation, o * 4 + 3, q.w);

    float transmitted = 1.0f;
    for (int m = begin; m < end; ++m)
        transmitted *= 1.0f - readFloat(p.opacity, uint64_t(readInt(p.members, uint64_t(m))));
    writeFloat(p.outOpacity, o, clamp(1.0f - transmitted, 0.0f, 1.0f));

    for (uint k = 0; k < p.appDim; ++k) {
        float sum = 0.0f;
        for (int m = begin; m < end; ++m) {
            const int row = readInt(p.members, uint64_t(m));
            bool ignored = false;
            float w = memberWeight(p, row, ignored);
            if (relative)
                w = largest > 0.0f ? w / largest : 1.0f;
            w = w / total;
            sum += w * readFloat(p.appearance, uint64_t(row) * p.appDim + k);
        }
        writeFloat(p.outAppearance, o * p.appDim + k, sum);
    }
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)] void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters p) {
    const uint3 numWorkgroups = spirv_asm {
result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    for (uint g = dispatchThreadId.x; g < p.groups; g += stride)
        mergeGroup(p, g);
}
