#version 450
layout(location = 0) in vec3 worldFar;
layout(location = 1) in vec3 worldNear;
layout(location = 0) out vec4 FragColor;
struct GridUniform {
    mat4 view_projection;
    vec4 view_position_plane;
    vec4 opacity_ortho;
    vec4 pivot_radius;
    vec4 near_origin;
    vec4 near_x;
    vec4 near_y;
    vec4 far_origin;
    vec4 far_x;
    vec4 far_y;
};
layout(std430, set = 0, binding = 0) readonly buffer GridUniforms {
    GridUniform grids[];
} grid_buffer;
layout(push_constant) uniform GridPush {
    int grid_index;
} push;

const vec4 planes[3] = vec4[3](
    vec4(1.0, 0.0, 0.0, 0.0),
    vec4(0.0, 1.0, 0.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0)
);

// Axis colours match the navigation gizmo.
const vec3 kAxisColors[3] = vec3[3](
    vec3(0.89, 0.15, 0.21),
    vec3(0.54, 0.86, 0.20),
    vec3(0.17, 0.48, 0.87)
);

const int axis0[3] = int[3](1, 0, 0);
const int axis1[3] = int[3](2, 2, 1);

const vec3 kLineColor = vec3(0.58, 0.64, 0.74);
const float kMinorAlpha = 0.32;
const float kMajorAlpha = 0.7;
const float kAxisAlpha = 1.0;
// Lines closer than this many pixels are not drawn; each decade fades in above it.
const float kMinSpacingPx = 8.0;

bool intersectPlane(inout float t, vec3 pos, vec3 dir, vec4 plane) {
    float d = dot(dir, plane.xyz);
    if (abs(d) < 1e-06) {
        return false;
    }

    float n = -(dot(pos, plane.xyz) + plane.w) / d;
    if (n < 0.0) {
        return false;
    }

    t = n;
    return true;
}

// Coverage of lines every `spacing` world units, `halfWidthPx` wide on screen at any distance.
float lineCoverage(vec2 pos, vec2 worldPerPx, float spacing, float halfWidthPx) {
    vec2 distPx = abs(fract(pos / spacing + 0.5) - 0.5) * spacing / worldPerPx;
    vec2 cov = clamp(halfWidthPx + 0.5 - distPx, 0.0, 1.0);
    return max(cov.x, cov.y);
}

// x = log10(spacing / kMinSpacingPx): a decade fades in as a minor level, then becomes major.
float levelAlpha(float x) {
    return mix(kMinorAlpha * smoothstep(0.0, 1.0, x), kMajorAlpha, smoothstep(1.0, 2.0, x));
}

vec4 over(vec4 src, vec4 dst) {
    float a = src.a + dst.a * (1.0 - src.a);
    vec3 rgb = (src.rgb * src.a + dst.rgb * dst.a * (1.0 - src.a)) / max(a, 1e-6);
    return vec4(rgb, a);
}

float calcDepth(vec3 p) {
    GridUniform u = grid_buffer.grids[push.grid_index];
    vec4 v = u.view_projection * vec4(p, 1.0);
    return v.z / v.w;
}

void main() {
    GridUniform u = grid_buffer.grids[push.grid_index];
    int plane = clamp(int(u.view_position_plane.w), 0, 2);
    vec3 p = worldNear;
    vec3 v = normalize(worldFar - worldNear);

    float t;
    if (!intersectPlane(t, p, v, planes[plane])) {
        discard;
    }

    vec3 worldPos = p + v * t;
    vec2 pos = plane == 0 ? worldPos.yz : (plane == 1 ? worldPos.xz : worldPos.xy);
    vec2 ddx = dFdx(pos);
    vec2 ddy = dFdy(pos);
    vec2 worldPerPx = max(vec2(length(vec2(ddx.x, ddy.x)), length(vec2(ddx.y, ddy.y))), vec2(1e-20));

    // Soft horizon: fade out where the view ray grazes the plane. Orthographic rays are all
    // parallel, so there is no horizon and the floor fade alone bounds the grid.
    float horizon = u.opacity_ortho.y > 0.5 ? 1.0 : smoothstep(0.0, 0.18, abs(dot(v, planes[plane].xyz)));
    float fade = horizon * clamp(u.opacity_ortho.x, 0.0, 1.0);
    // The grid is a floor under the orbit pivot that dissolves outward.
    vec3 pivot = u.pivot_radius.xyz;
    vec3 pivotOnPlane = pivot - planes[plane].xyz * dot(pivot, planes[plane].xyz);
    float radius = max(u.pivot_radius.w, 1e-6);
    float radial = 1.0 - smoothstep(0.1 * radius, radius, length(worldPos - pivotOnPlane));

    float lod = log2(max(worldPerPx.x, worldPerPx.y) * kMinSpacingPx) * 0.30103;
    float firstLevel = floor(lod) + 1.0;
    float gridAlpha = 0.0;
    for (int k = 0; k < 3; ++k) {
        float level = firstLevel + float(k);
        gridAlpha = max(gridAlpha, lineCoverage(pos, worldPerPx, pow(10.0, level), 0.5) * levelAlpha(level - lod));
    }

    // Axes reach further out than the grid lines.
    float axisRadial = sqrt(radial);
    vec2 axisCov = clamp(1.6 - abs(pos) / worldPerPx, 0.0, 1.0);
    vec4 color = vec4(kLineColor, gridAlpha * radial);
    color = over(vec4(kAxisColors[axis0[plane]], axisCov.y * kAxisAlpha * axisRadial), color);
    color = over(vec4(kAxisColors[axis1[plane]], axisCov.x * kAxisAlpha * axisRadial), color);
    color.a *= fade;
    if (color.a < 1.0 / 255.0) {
        discard;
    }
    FragColor = color;
    gl_FragDepth = calcDepth(worldPos);
}
