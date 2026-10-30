// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Public element-effect API, ABI v1. A user effect module imports this and
// defines one of
//     LfsElement effect(LfsElement e, LfsContext ctx);
//     LfsElement effect(LfsElement e, LfsContext ctx, Params p);
// where the optional `struct Params` lists float/float2/float3/float4 fields,
// each tagged [LfsParam(minimum, maximum, initial)]. Fields take parameter
// slots 0..15 in declaration order and the host fills `p` from those slots.
// Names starting with `_lfs` or `_Lfs` are reserved for generated code.

__exported import lfs_helpers;

public struct LfsElement {
    public float3 position;    // visualizer world space
    public float3 scale;       // positive source/object-space axes
    public float4 rotation;    // source-space quaternion in w,x,y,z order
    public float3 color;       // linear, view-dependent RGB before display tonemapping
    public float opacity;      // raw opacity before projection antialiasing and LoD weights
    public float3 normal;      // zero for splats in ABI v1
    public uint index;         // logical element id in this render snapshot; read-only
    public uint objectId;      // dense object slot, not a persisted UUID; read-only
    public uint kind;          // 0 = splat, other values reserved; read-only
};

public struct LfsContext {
    public float time;         // seconds on the scene timeline
    public float progress;     // clamped reveal progress; 1 without a reveal
    public float softness;     // reveal edge width as a fraction of radius; 0 without a reveal
    public float3 boundsMin;   // world-space bounds of the original visible geometry
    public float3 boundsMax;
    public float3 center;      // world-space reveal center
    public float radius;       // world-space reveal radius
    public float4x4 objectToWorld; // node transform, may include nonuniform scale or shear
    public float4x4 worldToView;
    public float3 cameraPosition;  // world space
    public float4 cameraIntrinsics; // fx, fy, cx, cy in full-image pixels
    public float4 distortionCoefficients;
    public uint2 imageSize;    // full image in pixels, also for banded renders
    public uint cameraModel;   // host camera model id
    public float4 values[16];  // parameter slots; scalars use .x, float2 .xy, float3 .xyz

    // Low-level slot access; slots past 15 read slot 15.
    public float4 param(uint slot) { return values[min(slot, 15u)]; }
};

// Generated code implements it for hosts, which declare
//     extern struct LfsBoundEffect : _LfsEffects;
// and run bound effect `index` of a draw through it.
public interface _LfsEffects {
    LfsElement apply(uint index, LfsElement e, LfsContext ctx);
};

// [LfsParam(minimum, maximum, initial)] on a Params field. Vector defaults
// broadcast `initial` to every component.
[__AttributeUsage(_AttributeTargets.Var)]
public struct LfsParamAttribute {
    public float minimum;
    public float maximum;
    public float initial;
};

// Spherical reveal around ctx.center: 1 within radius * progress, 0 outside,
// with an eased edge of width softness * radius just inside that sphere.
// Progress 0 hides and 1 shows every element, including ones beyond radius.
public float lfs_reveal_mask(float3 position, LfsContext ctx) {
    if (ctx.progress <= 0.0f) return 0.0f;
    if (ctx.progress >= 1.0f) return 1.0f;
    float d = length(position - ctx.center) / max(ctx.radius, 1e-6f);
    return lfs_ease((ctx.progress - d) / max(ctx.softness, 1e-6f));
}
