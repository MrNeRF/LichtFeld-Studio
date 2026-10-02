[SpecializationConstant] const uint operation = 0;

struct Parameters {
    uint64_t alpha;
    uint64_t mask;
    uint64_t roi;
    uint64_t output;
    uint64_t partials;
    uint count;
    uint maskBytes;
    uint hasRoi;
    uint mode;
    float power;
    float gradScale;
    float keepMin;
    float segmentMin;
};

groupshared float reduction[256];

float maskValue(Parameters p, uint i) {
    return p.maskBytes != 0 ? float(((Ptr<uint8_t, Access.Read>)p.mask)[i])
                            : ((Ptr<float, Access.Read>)p.mask)[i];
}

float normalizedMask(Parameters p, uint i) {
    return p.maskBytes != 0 ? maskValue(p, i) * (1.0f / 255.0f) : maskValue(p, i);
}

float maskAsFloat(Parameters p, uint i) {
    const float value = maskValue(p, i);
    return p.maskBytes != 0 ? (value != 0.0f ? 1.0f : 0.0f) : value;
}

float opacityBackground(Parameters p, uint i) {
    if (p.mode == 1) {
        const float value = normalizedMask(p, i);
        return value >= p.segmentMin && value <= p.keepMin ? 1.0f : 0.0f;
    }
    return 1.0f - maskAsFloat(p, i);
}

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p) {
    uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)p.output;
    if (operation == 0) {
        for (uint i = groupId.x * 256 + lane; i < p.count; i += groups.x * 256) {
            float value = p.mode == 1 ? (normalizedMask(p, i) > p.keepMin ? 1.0f : 0.0f)
                                      : maskAsFloat(p, i);
            if (p.hasRoi != 0) value *= ((Ptr<float, Access.Read>)p.roi)[i];
            output[i] = value;
        }
        return;
    }
    if (operation == 1 || operation == 2) {
        float sum = 0.0f;
        Ptr<float, Access.Read> alpha = (Ptr<float, Access.Read>)p.alpha;
        Ptr<float, Access.ReadWrite> partials = (Ptr<float, Access.ReadWrite>)p.partials;
        for (uint i = groupId.x * 256 + lane; i < p.count; i += groups.x * 256) {
            float value;
            if (operation == 1) {
                const float background = opacityBackground(p, i);
                value = background <= 0.0f ? 0.0f : ((background >= 1.0f && p.power > 0.0f)
                                                               ? 1.0f
                                                               : pow(background, p.power));
                if (p.hasRoi != 0) value *= ((Ptr<float, Access.Read>)p.roi)[i];
                output[i] = value * p.gradScale;
                sum += alpha[i] * value;
            } else {
                const float diff = alpha[i] - maskAsFloat(p, i);
                const float sign = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                const float roi = p.hasRoi != 0 ? ((Ptr<float, Access.Read>)p.roi)[i] : 1.0f;
                output[i] = sign * roi * p.gradScale;
                sum += abs(diff) * roi;
            }
        }
        reduction[lane] = sum;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) reduction[lane] += reduction[lane + offset];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0) partials[groupId.x] = reduction[0];
        return;
    }
    if (operation == 3) {
        Ptr<float, Access.Read> partials = (Ptr<float, Access.Read>)p.partials;
        float sum = 0.0f;
        for (uint i = lane; i < p.mode; i += 256) sum += partials[i];
        reduction[lane] = sum;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) reduction[lane] += reduction[lane + offset];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0) output[0] = p.power * (reduction[0] / float(p.count));
        return;
    }
    if (operation == 4) {
        for (uint i = groupId.x * 256 + lane; i < p.count; i += groups.x * 256) output[i] = 0.0f;
        if (groupId.x == 0 && lane == 0) ((Ptr<float, Access.ReadWrite>)p.alpha)[0] = 0.0f;
    }
}
