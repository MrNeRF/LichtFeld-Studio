[SpecializationConstant] const uint operation = 0;

struct Parameters {
    uint64_t source;
    uint64_t gradient;
    uint64_t loss;
    uint64_t scratch;
    uint64_t z;
    uint64_t u;
    uint count;
    uint partialCount;
    float weight;
    float gradLoss;
    uint kind;
    uint accumulate;
};

groupshared float reduction[256];

float activate(float x, uint kind) {
    return kind == 0 ? exp(x) : 1.0f / (1.0f + exp(-x));
}

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p) {
    uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Ptr<float, Access.Read> source = (Ptr<float, Access.Read>)p.source;
    Ptr<float, Access.ReadWrite> gradient = (Ptr<float, Access.ReadWrite>)p.gradient;
    if (operation == 0) {
        const uint blocks = p.partialCount;
        const uint stride = 256u * blocks;
        float sum = 0.0f;
        for (uint i = groupId.x * 256u + lane; i < p.count; i += stride) {
            const float activated = activate(source[i], p.kind);
            sum += activated;
            if (p.accumulate != 0) {
                const float gradScale = p.weight / float(p.count);
                const float derivative = p.kind == 0
                                             ? gradScale * activated
                                             : gradScale * activated * (1.0f - activated);
                gradient[i] += derivative;
            }
        }
        reduction[lane] = sum;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) reduction[lane] += reduction[lane + offset];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0) ((Ptr<float, Access.ReadWrite>)p.scratch)[groupId.x] = reduction[0];
        return;
    }
    if (operation == 1) {
        float sum = 0.0f;
        Ptr<float, Access.Read> partials = (Ptr<float, Access.Read>)p.scratch;
        for (uint i = lane; i < p.partialCount; i += 256u) sum += partials[i];
        reduction[lane] = sum;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) reduction[lane] += reduction[lane + offset];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0) ((Ptr<float, Access.ReadWrite>)p.loss)[0] = p.weight * (reduction[0] / float(p.count));
        return;
    }
    for (uint i = groupId.x * 256u + lane; i < p.count; i += 256u * groups.x) {
        const float value = p.weight * (source[i] - ((Ptr<float, Access.Read>)p.z)[i] + ((Ptr<float, Access.Read>)p.u)[i]) *
                            source[i] * (1.0f - source[i]) * p.gradLoss;
        if (p.accumulate != 0) gradient[i] += value;
        else gradient[i] = value;
    }
}
