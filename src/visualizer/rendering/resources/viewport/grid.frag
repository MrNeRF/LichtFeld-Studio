#version 450
layout(location = 0) in vec3 worldFar;
layout(location = 1) in vec3 worldNear;
layout(location = 0) out vec4 FragColor;
struct GridUniform {
    mat4 view_projection;
    vec4 view_position_plane;
    vec4 opacity_padding;
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

const vec3 colors[3] = vec3[3](
    vec3(1.0, 0.2, 0.2),
    vec3(0.2, 1.0, 0.2),
    vec3(0.2, 0.2, 1.0)
);

const int axis0[3] = int[3](1, 0, 0);
const int axis1[3] = int[3](2, 2, 1);

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

float pristineGrid(in vec2 uv, in vec2 ddx, in vec2 ddy, vec2 lineWidth) {
    vec2 uvDeriv = vec2(length(vec2(ddx.x, ddy.x)), length(vec2(ddx.y, ddy.y)));
    bvec2 invertLine = bvec2(lineWidth.x > 0.5, lineWidth.y > 0.5);
    vec2 targetWidth = vec2(
        invertLine.x ? 1.0 - lineWidth.x : lineWidth.x,
        invertLine.y ? 1.0 - lineWidth.y : lineWidth.y
    );
    vec2 drawWidth = clamp(targetWidth, uvDeriv, vec2(0.5));
    vec2 lineAA = uvDeriv * 1.5;
    vec2 gridUV = abs(fract(uv) * 2.0 - 1.0);
    gridUV.x = invertLine.x ? gridUV.x : 1.0 - gridUV.x;
    gridUV.y = invertLine.y ? gridUV.y : 1.0 - gridUV.y;
    vec2 grid2 = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);

    grid2 *= clamp(targetWidth / drawWidth, 0.0, 1.0);
    grid2 = mix(grid2, targetWidth, clamp(uvDeriv * 2.0 - 1.0, 0.0, 1.0));
    grid2.x = invertLine.x ? 1.0 - grid2.x : grid2.x;
    grid2.y = invertLine.y ? 1.0 - grid2.y : grid2.y;

    return mix(grid2.x, 1.0, grid2.y);
}

float calcDepth(vec3 p) {
    GridUniform u = grid_buffer.grids[push.grid_index];
    vec4 v = u.view_projection * vec4(p, 1.0);
    return (v.z / v.w) * 0.5 + 0.5;
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
    float fade = (1.0 - smoothstep(400.0, 1000.0, length(worldPos - u.view_position_plane.xyz))) *
                 clamp(u.opacity_padding.x, 0.0, 1.0);
    float epsilon = 1.0 / 255.0;
    if (fade < epsilon) {
        discard;
    }

    vec2 levelPos = pos * 0.1;
    float levelSize = 2.0 / 1000.0;
    float levelAlpha = pristineGrid(levelPos, ddx * 0.1, ddy * 0.1, vec2(levelSize)) * fade;
    if (levelAlpha > epsilon) {
        vec3 color;
        vec2 loc = abs(levelPos);
        vec2 axisDeriv = vec2(length(vec2(ddx.x, ddy.x)), length(vec2(ddx.y, ddy.y))) * 0.1;
        float axisWidth = levelSize * 1.5;
        float axisX = 1.0 - smoothstep(axisWidth - axisDeriv.x, axisWidth + axisDeriv.x, loc.x);
        float axisY = 1.0 - smoothstep(axisWidth - axisDeriv.y, axisWidth + axisDeriv.y, loc.y);
        bool isAxisX = axisX > 0.01;
        bool isAxisY = axisY > 0.01;
        bool isAxis = isAxisX || isAxisY;
        if (isAxisX && isAxisY) {
            color = vec3(1.0);
        } else if (isAxisX) {
            color = colors[axis1[plane]];
        } else if (isAxisY) {
            color = colors[axis0[plane]];
        } else {
            color = vec3(0.4);
        }
        float axisAlpha = max(axisX, axisY);
        float finalAlpha = isAxis ? axisAlpha * fade : levelAlpha;
        FragColor = vec4(color, finalAlpha);
        gl_FragDepth = calcDepth(worldPos);
        return;
    }

    levelPos = pos;
    levelSize = 1.0 / 100.0;
    levelAlpha = pristineGrid(levelPos, ddx, ddy, vec2(levelSize)) * fade;
    if (levelAlpha > epsilon) {
        FragColor = vec4(vec3(0.3), levelAlpha);
        gl_FragDepth = calcDepth(worldPos);
        return;
    }

    levelPos = pos * 10.0;
    levelSize = 1.0 / 100.0;
    levelAlpha = pristineGrid(levelPos, ddx * 10.0, ddy * 10.0, vec2(levelSize)) * fade;
    if (levelAlpha > epsilon) {
        FragColor = vec4(vec3(0.3), levelAlpha);
        gl_FragDepth = calcDepth(worldPos);
        return;
    }

    discard;
}
