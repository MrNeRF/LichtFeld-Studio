static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 72;
struct Parameters { uint64_t scales, rotations, outScales, outRotations; float linear[9]; uint count; };

// The dispatch has one Y workgroup; the runtime zero retains rounded intermediates.
static uint gOpaqueZero = 0;
float opaque(float x) { return asfloat(asuint(x) + gOpaqueZero); }
float2 twoSum(float a, float b) {
    const float total = opaque(a+b);
    const float part = opaque(total-a);
    return float2(total, opaque(a-opaque(total-part)) + opaque(b-part));
}
float2 addPair(float2 a, float2 b) {
    const float2 sum = twoSum(a.x,b.x);
    return twoSum(sum.x, (a.y+b.y)+sum.y);
}
float expDifference(float a, float b) {
    if (a == -1.0f/0.0f) return 0;
    const float2 delta = twoSum(a,-b);
    if (delta.x < -104.0f) return 0;
    const int k = int(round(delta.x*1.4426950408889634f));
    const float r = fma(-float(k),0.693147182464599609375f,delta.x)
                  + (delta.y + float(k)*1.904654323148236e-9f);
    // Range reduction bounds abs(r) by ln(2)/2.
    float poly = 1.0f/40320.0f;
    poly = fma(poly,r,1.0f/5040.0f);
    poly = fma(poly,r,1.0f/720.0f);
    poly = fma(poly,r,1.0f/120.0f);
    poly = fma(poly,r,1.0f/24.0f);
    poly = fma(poly,r,1.0f/6.0f);
    poly = fma(poly,r,0.5f);
    poly = fma(poly,r,1.0f);
    poly = fma(poly,r,1.0f);
    return ldexp(poly,k);
}
float2 logPair(float value) {
    if (value == 0) return float2(-1.0f/0.0f,0);
    const uint bits = asuint(value);
    int exponent = int((bits>>23)&255u)-127;
    float m = asfloat((bits&0x7fffffu)|0x3f800000u);
    if (m > 1.41421356237f) { m *= 0.5f; ++exponent; }
    const float z = (m-1)/(m+1), z2 = z*z;
    float poly = 1.0f/13.0f;
    poly = fma(poly,z2,1.0f/11.0f);
    poly = fma(poly,z2,1.0f/9.0f);
    poly = fma(poly,z2,1.0f/7.0f);
    poly = fma(poly,z2,1.0f/5.0f);
    poly = fma(poly,z2,1.0f/3.0f);
    const float fraction = 2*z*fma(poly,z2,1.0f);
    const float high = opaque(float(exponent)*0.693147182464599609375f);
    const float low = fma(float(exponent),0.693147182464599609375f,-high)
                    + float(exponent)*(-1.904654323148236e-9f) + fraction;
    return float2(high,low);
}
float restoreLogScale(float length, float2 linearLog, float largestLog) {
    if (length == 0) return -1.0f/0.0f;
    return addPair(addPair(logPair(length),linearLog),float2(largestLog,0)).x;
}

float scaledLength(float3 v) {
    const float scale = max(abs(v.x), max(abs(v.y), abs(v.z)));
    return scale == 0 ? 0 : scale * length(v / scale);
}

bool orthogonalize(inout float3 a, inout float3 b) {
    const float aa = dot(a, a), bb = dot(b, b), ab = dot(a, b);
    if (abs(ab) <= 2e-7f * sqrt(aa) * sqrt(bb)) return false;
    const float delta = 0.5f * (bb - aa);
    const float scale = max(abs(delta), abs(ab));
    const float d = delta / scale, e = ab / scale;
    const float h = sqrt(d*d + e*e);
    const float t = e / (d + (delta < 0 ? -h : h));
    const float c = rsqrt(1 + t*t), s = t*c;
    const float3 old = a;
    a = c*old - s*b;
    b = s*old + c*b;
    return true;
}

void sortAxes(inout float3 a, inout float3 b, inout float la, inout float lb) {
    if (la < lb) {
        const float3 v = a; a = b; b = v;
        const float l = la; la = lb; lb = l;
    }
}

void affine(Parameters p, float log_scale[3], float quaternion[4], out float result_scale[3], out float result_quaternion[4]) {
    float4 q = float4(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    const float qm = max(max(abs(q.x), abs(q.y)), max(abs(q.z), abs(q.w)));
    q = qm > 0 ? normalize(q / qm) : float4(1, 0, 0, 0);
    const float w=q.x, x=q.y, y=q.z, z=q.w;
    // Difference of squares preserves zero diagonal entries at equal quaternion components.
    const float3 r0 = float3((w*w+x*x)-(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y));
    const float3 r1 = float3(2*(x*y-w*z), (w*w+y*y)-(x*x+z*z), 2*(y*z+w*x));
    const float3 r2 = float3(2*(x*z+w*y), 2*(y*z-w*x), (w*w+z*z)-(x*x+y*y));
    const float largest_log = max(log_scale[0], max(log_scale[1], log_scale[2]));
    float linear_scale = 0;
    for (int k=0; k<9; ++k) linear_scale = max(linear_scale, abs(p.linear[k]));
    if (linear_scale == 0 || largest_log == -1.0f/0.0f) {
        for (int j=0; j<3; ++j) result_scale[j] = -1.0f/0.0f;
        result_quaternion[0] = 1;
        for (int j=1; j<4; ++j) result_quaternion[j] = 0;
        return;
    }
    // Common scale factors leave singular vectors unchanged and bound matrix entries.
    const float3 a0 = float3(p.linear[0], p.linear[1], p.linear[2]) / linear_scale;
    const float3 a1 = float3(p.linear[3], p.linear[4], p.linear[5]) / linear_scale;
    const float3 a2 = float3(p.linear[6], p.linear[7], p.linear[8]) / linear_scale;
    float3 b0 = float3(dot(a0,r0), dot(a1,r0), dot(a2,r0)) * expDifference(log_scale[0],largest_log);
    float3 b1 = float3(dot(a0,r1), dot(a1,r1), dot(a2,r1)) * expDifference(log_scale[1],largest_log);
    float3 b2 = float3(dot(a0,r2), dot(a1,r2), dot(a2,r2)) * expDifference(log_scale[2],largest_log);
    for (int sweep=0; sweep<8; ++sweep) {
        bool changed = orthogonalize(b0,b1);
        changed = orthogonalize(b0,b2) || changed;
        changed = orthogonalize(b1,b2) || changed;
        if (!changed) break;
    }
    float l0=scaledLength(b0), l1=scaledLength(b1), l2=scaledLength(b2);
    sortAxes(b0,b1,l0,l1); sortAxes(b0,b2,l0,l2); sortAxes(b1,b2,l1,l2);
    const float2 linearLog = logPair(linear_scale);
    result_scale[0] = restoreLogScale(l0,linearLog,largest_log);
    result_scale[1] = restoreLogScale(l1,linearLog,largest_log);
    result_scale[2] = restoreLogScale(l2,linearLog,largest_log);
    float3 u = l0 > 0 ? b0/l0 : float3(1,0,0);
    float3 v = l1 > 0 ? b1/l1 : float3(0,0,0);
    // The null-space basis is arbitrary; the completed frame must remain orthonormal.
    v -= u*dot(u,v);
    if (dot(v,v) < 1e-12f) {
        const float3 axis = abs(u.x) <= abs(u.y) && abs(u.x) <= abs(u.z) ? float3(1,0,0)
                          : abs(u.y) <= abs(u.z) ? float3(0,1,0) : float3(0,0,1);
        v = axis-u*dot(u,axis);
    }
    v = normalize(v);
    const float3 t = cross(u,v);
    // A right-handed frame represents reflections without changing the covariance.
    const float trace = u.x+v.y+t.z;
    if (trace > 0) {
        const float s = 2*sqrt(1+trace);
        q = float4(s/4, (v.z-t.y)/s, (t.x-u.z)/s, (u.y-v.x)/s);
    } else if (u.x > v.y && u.x > t.z) {
        const float s = 2*sqrt(1+u.x-v.y-t.z);
        q = float4((v.z-t.y)/s, s/4, (u.y+v.x)/s, (t.x+u.z)/s);
    } else if (v.y > t.z) {
        const float s = 2*sqrt(1+v.y-u.x-t.z);
        q = float4((t.x-u.z)/s, (u.y+v.x)/s, s/4, (v.z+t.y)/s);
    } else {
        const float s = 2*sqrt(1+t.z-u.x-v.y);
        q = float4((u.y-v.x)/s, (t.x+u.z)/s, (v.z+t.y)/s, s/4);
    }
    q = normalize(q);
    if (q.x < 0) q = -q;
    for (int i=0; i<4; ++i) result_quaternion[i] = q[i];
}
[numthreads(LFS_LOCAL_SIZE_X,1,1)]
void main(uint3 tid:SV_DispatchThreadID,uniform Parameters p) {
    uint3 groups=spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    gOpaqueZero = groups.y-1u;
    for(uint i=tid.x;i<p.count;i+=groups.x*LFS_LOCAL_SIZE_X) {
        float s[3],q[4],os[3],oq[4];
        for(uint k=0;k<3;++k)s[k]=((Ptr<float,Access.Read>)p.scales)[i*3+k];
        for(uint k=0;k<4;++k)q[k]=((Ptr<float,Access.Read>)p.rotations)[i*4+k];
        affine(p,s,q,os,oq);
        for(uint k=0;k<3;++k)((Ptr<float,Access.ReadWrite>)p.outScales)[i*3+k]=os[k];
        for(uint k=0;k<4;++k)((Ptr<float,Access.ReadWrite>)p.outRotations)[i*4+k]=oq[k];
    }
}
