/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_viewport_pass.hpp"

#include "config.h"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "rendering/image_layout.hpp"
#include "window/vulkan_context.hpp"

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

namespace lfs::vis {

#ifdef LFS_VULKAN_VIEWER_ENABLED
    namespace {
        struct Vertex {
            glm::vec2 position;
            glm::vec2 uv;
        };

        struct FramebufferRect {
            std::int32_t x = 0;
            std::int32_t y = 0;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
        };

        struct VignettePush {
            glm::vec4 viewport_intensity_radius{0.0f};
            glm::vec4 softness_padding{0.0f};
        };

        struct GridUniform {
            glm::mat4 view_projection{1.0f};
            glm::vec4 view_position_plane{0.0f};
            glm::vec4 opacity_padding{0.0f};
            glm::vec4 near_origin{0.0f};
            glm::vec4 near_x{0.0f};
            glm::vec4 near_y{0.0f};
            glm::vec4 far_origin{0.0f};
            glm::vec4 far_x{0.0f};
            glm::vec4 far_y{0.0f};
        };

        struct GridPush {
            std::int32_t grid_index = 0;
        };

        struct OverlayPush {
            glm::vec4 padding{0.0f};
        };

        struct PivotPush {
            glm::vec4 center_size{0.0f};
            glm::vec4 color_opacity{0.26f, 0.59f, 0.98f, 1.0f};
        };

        struct TexturedOverlayPush {
            glm::vec4 tint_opacity{1.0f, 1.0f, 1.0f, 0.8f};
            glm::vec4 effects{0.0f};
        };

        [[nodiscard]] VkDescriptorSet descriptorSetFromId(const std::uintptr_t texture_id) {
            return reinterpret_cast<VkDescriptorSet>(texture_id);
        }

        constexpr const char* kScreenQuadVert = R"GLSL(
#version 450
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 0) out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)GLSL";

        constexpr const char* kSceneFrag = R"GLSL(
#version 450
layout(set = 0, binding = 0) uniform sampler2D sceneTexture;
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 FragColor;
void main() {
    FragColor = texture(sceneTexture, TexCoord);
}
)GLSL";

        constexpr const char* kVignetteFrag = R"GLSL(
#version 450
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 FragColor;
layout(push_constant) uniform VignettePush {
    vec4 viewport_intensity_radius;
    vec4 softness_padding;
} u;

float vignette_alpha(vec2 screen_uv) {
    vec2 viewport = max(u.viewport_intensity_radius.xy, vec2(1.0, 1.0));
    float intensity = u.viewport_intensity_radius.z;
    float radius = u.viewport_intensity_radius.w;
    float softness = u.softness_padding.x;
    float min_dim = min(viewport.x, viewport.y);
    float fade_width = (1.0 - clamp(radius, 0.0, 1.0)) * 0.5 * min_dim;
    if (fade_width <= 0.0) {
        return 0.0;
    }

    vec2 half_extent = 0.5 * viewport;
    vec2 inner_half = max(half_extent - vec2(fade_width), vec2(0.0, 0.0));
    vec2 p = abs(screen_uv * viewport - half_extent) - inner_half;
    float dist = length(max(p, vec2(0.0, 0.0)));
    float visible = clamp(1.0 - dist / fade_width, 0.0, 1.0);
    visible = mix(visible, smoothstep(0.0, 1.0, visible), clamp(softness, 0.0, 1.0));
    return clamp(intensity, 0.0, 1.0) * (1.0 - visible);
}

void main() {
    FragColor = vec4(0.0, 0.0, 0.0, vignette_alpha(TexCoord));
}
)GLSL";

        constexpr const char* kOverlayVert = R"GLSL(
#version 450
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 0) out vec4 Color;
void main() {
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
    Color = aColor;
}
)GLSL";

        constexpr const char* kOverlayFrag = R"GLSL(
#version 450
layout(location = 0) in vec4 Color;
layout(location = 0) out vec4 FragColor;
void main() {
    FragColor = Color;
}
)GLSL";

        constexpr const char* kShapeOverlayVert = R"GLSL(
#version 450
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aScreenPos;
layout(location = 2) in vec2 aP0;
layout(location = 3) in vec2 aP1;
layout(location = 4) in vec4 aColor;
layout(location = 5) in vec4 aParams;
layout(location = 0) out vec2 ScreenPos;
layout(location = 1) out vec2 P0;
layout(location = 2) out vec2 P1;
layout(location = 3) out vec4 Color;
layout(location = 4) out vec4 Params;
void main() {
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
    ScreenPos = aScreenPos;
    P0 = aP0;
    P1 = aP1;
    Color = aColor;
    Params = aParams;
}
)GLSL";

        constexpr const char* kShapeOverlayFrag = R"GLSL(
#version 450
layout(location = 0) in vec2 ScreenPos;
layout(location = 1) in vec2 P0;
layout(location = 2) in vec2 P1;
layout(location = 3) in vec4 Color;
layout(location = 4) in vec4 Params;
layout(location = 0) out vec4 FragColor;

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    float shape = Params.x;
    float thickness = max(Params.y, 1.0);
    float radius = max(Params.z, 0.0);
    float aa = max(Params.w, 0.75);
    float signed_dist = 1e6;

    if (shape < 0.5) {
        signed_dist = sdSegment(ScreenPos, P0, P1) - thickness * 0.5;
    } else if (shape < 1.5) {
        signed_dist = length(ScreenPos - P0) - radius;
    } else {
        signed_dist = abs(length(ScreenPos - P0) - radius) - thickness * 0.5;
    }

    float alpha = Color.a * smoothstep(aa, -aa, signed_dist);
    if (alpha <= 0.001) {
        discard;
    }
    FragColor = vec4(Color.rgb, alpha);
}
)GLSL";

        constexpr const char* kTexturedOverlayVert = R"GLSL(
#version 450
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 0) out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)GLSL";

        constexpr const char* kTexturedOverlayFrag = R"GLSL(
#version 450
layout(set = 0, binding = 0) uniform sampler2D overlayTexture;
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 FragColor;
layout(push_constant) uniform TexturedOverlayPush {
    vec4 tint_opacity;
    vec4 effects;
} u;

void main() {
    vec4 sampled = texture(overlayTexture, TexCoord);
    vec3 rgb = mix(sampled.rgb, u.tint_opacity.rgb, clamp(u.effects.x, 0.0, 1.0));
    rgb = mix(rgb, vec3(0.5), clamp(u.effects.y, 0.0, 1.0));
    FragColor = vec4(rgb, sampled.a * clamp(u.tint_opacity.a, 0.0, 1.0));
}
)GLSL";

        constexpr const char* kPivotVert = R"GLSL(
#version 450
layout(location = 0) out vec2 v_uv;
layout(push_constant) uniform PivotPush {
    vec4 center_size;
    vec4 color_opacity;
} u;

const vec2 CORNERS[4] = vec2[4](
    vec2(-1.0, -1.0), vec2(1.0, -1.0),
    vec2(1.0, 1.0), vec2(-1.0, 1.0)
);
const int INDICES[6] = int[6](0, 1, 2, 0, 2, 3);

void main() {
    vec2 corner = CORNERS[INDICES[gl_VertexIndex]];
    v_uv = corner;
    gl_Position = vec4(u.center_size.xy + corner * u.center_size.zw, 0.0, 1.0);
}
)GLSL";

        constexpr const char* kPivotFrag = R"GLSL(
#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 FragColor;
layout(push_constant) uniform PivotPush {
    vec4 center_size;
    vec4 color_opacity;
} u;

const float DOT_RADIUS = 0.06;
const float RING_WIDTH = 0.045;
const float MAX_RADIUS = 0.92;
const float EDGE_AA = 0.015;

float ring(float dist, float radius, float width) {
    float inner = radius - width * 0.5;
    float outer = radius + width * 0.5;
    return smoothstep(inner - EDGE_AA, inner + EDGE_AA, dist) *
           (1.0 - smoothstep(outer - EDGE_AA, outer + EDGE_AA, dist));
}

void main() {
    float dist = length(v_uv);
    float u_opacity = u.color_opacity.w;
    float progress = 1.0 - u_opacity;

    float flash_intensity = pow(max(0.0, 1.0 - progress * 4.0), 2.0);
    float flash = (1.0 - smoothstep(0.0, 0.25, dist)) * flash_intensity;

    float dot_scale = 1.0 + 0.3 * sin(progress * 6.28) * (1.0 - progress);
    float dot_radius = DOT_RADIUS * dot_scale;
    float dot_alpha = (1.0 - smoothstep(dot_radius - EDGE_AA, dot_radius + EDGE_AA, dist));
    dot_alpha *= pow(u_opacity, 0.5);

    float r1_prog = clamp(progress * 1.5, 0.0, 1.0);
    float r1_radius = DOT_RADIUS + r1_prog * (MAX_RADIUS - DOT_RADIUS);
    float r1_alpha = ring(dist, r1_radius, RING_WIDTH) * pow(1.0 - r1_prog, 1.5);

    float r2_prog = clamp((progress - 0.15) * 1.5, 0.0, 1.0);
    float r2_radius = DOT_RADIUS + r2_prog * (MAX_RADIUS * 0.85 - DOT_RADIUS);
    float r2_alpha = ring(dist, r2_radius, RING_WIDTH * 0.7) * pow(1.0 - r2_prog, 1.5) * 0.6;

    float glow = exp(-pow((dist - r1_radius * 0.95) * 4.0, 2.0)) * (1.0 - r1_prog) * 0.3;

    float alpha = max(max(dot_alpha, flash), max(r1_alpha + r2_alpha, glow));
    if (alpha < 0.01) discard;

    vec3 color = u.color_opacity.rgb + vec3(0.4) * (flash + dot_alpha * 0.3);
    FragColor = vec4(color, alpha);
}
)GLSL";

        constexpr const char* kGridVert = R"GLSL(
#version 450
layout(location = 0) in vec2 vertex_position;
layout(location = 0) out vec3 worldFar;
layout(location = 1) out vec3 worldNear;
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

void main() {
    GridUniform u = grid_buffer.grids[push.grid_index];
    gl_Position = vec4(vertex_position, 0.0, 1.0);
    vec2 p = vec2(vertex_position.x * 0.5 + 0.5, -vertex_position.y * 0.5 + 0.5);
    worldNear = (u.near_origin + u.near_x * p.x + u.near_y * p.y).xyz;
    worldFar = (u.far_origin + u.far_x * p.x + u.far_y * p.y).xyz;
}
)GLSL";

        constexpr const char* kGridFrag = R"GLSL(
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
)GLSL";

        void ensureGlslangInitialized() {
            static std::once_flag flag;
            std::call_once(flag, [] {
                glslang::InitializeProcess();
            });
        }

        [[nodiscard]] std::optional<std::vector<std::uint32_t>> compileGlsl(
            const char* source,
            const EShLanguage stage,
            const char* label) {
            ensureGlslangInitialized();
            const char* sources[] = {source};
            glslang::TShader shader(stage);
            shader.setStrings(sources, 1);
            shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
            shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
            shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
            constexpr EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
            if (!shader.parse(GetDefaultResources(), 450, false, messages)) {
                LOG_ERROR("Failed to compile Vulkan viewport shader {}: {}", label, shader.getInfoLog());
                return std::nullopt;
            }
            glslang::TProgram program;
            program.addShader(&shader);
            if (!program.link(messages)) {
                LOG_ERROR("Failed to link Vulkan viewport shader {}: {}", label, program.getInfoLog());
                return std::nullopt;
            }
            std::vector<std::uint32_t> spirv;
            glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);
            return spirv;
        }

        [[nodiscard]] std::optional<std::vector<std::uint8_t>> tensorToRgba8(
            const lfs::core::Tensor& image,
            const glm::ivec2 expected_size) {
            if (!image.is_valid() || image.ndim() != 3 || expected_size.x <= 0 || expected_size.y <= 0) {
                return std::nullopt;
            }

            const auto layout = lfs::rendering::detectImageLayout(image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                LOG_ERROR("Vulkan viewport pass received unsupported tensor shape [{}, {}, {}]",
                          image.size(0), image.size(1), image.size(2));
                return std::nullopt;
            }

            lfs::core::Tensor formatted = (layout == lfs::rendering::ImageLayout::HWC)
                                              ? image
                                              : image.permute({1, 2, 0}).contiguous();
            if (formatted.device() == lfs::core::Device::CUDA) {
                formatted = formatted.cpu();
            }
            if (formatted.dtype() != lfs::core::DataType::UInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
            }
            formatted = formatted.contiguous();

            const int height = static_cast<int>(formatted.size(0));
            const int width = static_cast<int>(formatted.size(1));
            const int channels = static_cast<int>(formatted.size(2));
            if (width != expected_size.x || height != expected_size.y || !formatted.ptr<std::uint8_t>()) {
                LOG_ERROR("Vulkan viewport pass dimension mismatch: {}x{} vs {}x{}",
                          width, height, expected_size.x, expected_size.y);
                return std::nullopt;
            }
            if (channels != 1 && channels != 3 && channels != 4) {
                LOG_ERROR("Vulkan viewport pass received unsupported channel count {}", channels);
                return std::nullopt;
            }

            const std::uint8_t* const src = formatted.ptr<std::uint8_t>();
            std::vector<std::uint8_t> rgba(static_cast<std::size_t>(width) *
                                           static_cast<std::size_t>(height) * 4u);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t src_offset =
                        (static_cast<std::size_t>(y) * width + x) * static_cast<std::size_t>(channels);
                    const std::size_t dst_offset = (static_cast<std::size_t>(y) * width + x) * 4u;
                    if (channels == 1) {
                        rgba[dst_offset + 0] = src[src_offset];
                        rgba[dst_offset + 1] = src[src_offset];
                        rgba[dst_offset + 2] = src[src_offset];
                        rgba[dst_offset + 3] = 255;
                    } else {
                        rgba[dst_offset + 0] = src[src_offset + 0];
                        rgba[dst_offset + 1] = src[src_offset + 1];
                        rgba[dst_offset + 2] = src[src_offset + 2];
                        rgba[dst_offset + 3] = channels >= 4 ? src[src_offset + 3] : 255;
                    }
                }
            }
            return rgba;
        }

        [[nodiscard]] FramebufferRect toFramebufferRect(
            const VulkanViewportPassParams& params,
            const VkExtent2D extent) {
            const float sx = params.framebuffer_scale.x > 0.0f ? params.framebuffer_scale.x : 1.0f;
            const float sy = params.framebuffer_scale.y > 0.0f ? params.framebuffer_scale.y : 1.0f;
            const int x0 = std::clamp(static_cast<int>(std::lround(params.viewport_pos.x * sx)),
                                      0, static_cast<int>(extent.width));
            const int y0 = std::clamp(static_cast<int>(std::lround(params.viewport_pos.y * sy)),
                                      0, static_cast<int>(extent.height));
            const int x1 = std::clamp(static_cast<int>(std::lround((params.viewport_pos.x + params.viewport_size.x) * sx)),
                                      0, static_cast<int>(extent.width));
            const int y1 = std::clamp(static_cast<int>(std::lround((params.viewport_pos.y + params.viewport_size.y) * sy)),
                                      0, static_cast<int>(extent.height));
            return {
                .x = x0,
                .y = y0,
                .width = static_cast<std::uint32_t>(std::max(x1 - x0, 0)),
                .height = static_cast<std::uint32_t>(std::max(y1 - y0, 0)),
            };
        }

        [[nodiscard]] FramebufferRect toFramebufferRect(
            const VulkanViewportPassParams& params,
            const VulkanViewportGridOverlay& grid,
            const VkExtent2D extent) {
            const float sx = params.framebuffer_scale.x > 0.0f ? params.framebuffer_scale.x : 1.0f;
            const float sy = params.framebuffer_scale.y > 0.0f ? params.framebuffer_scale.y : 1.0f;
            const int x0 = std::clamp(static_cast<int>(std::lround(grid.viewport_pos.x * sx)),
                                      0, static_cast<int>(extent.width));
            const int y0 = std::clamp(static_cast<int>(std::lround(grid.viewport_pos.y * sy)),
                                      0, static_cast<int>(extent.height));
            const int x1 = std::clamp(static_cast<int>(std::lround((grid.viewport_pos.x + grid.viewport_size.x) * sx)),
                                      0, static_cast<int>(extent.width));
            const int y1 = std::clamp(static_cast<int>(std::lround((grid.viewport_pos.y + grid.viewport_size.y) * sy)),
                                      0, static_cast<int>(extent.height));
            return {
                .x = x0,
                .y = y0,
                .width = static_cast<std::uint32_t>(std::max(x1 - x0, 0)),
                .height = static_cast<std::uint32_t>(std::max(y1 - y0, 0)),
            };
        }
    } // namespace
#endif

    struct VulkanViewportPass::Impl {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        std::uint32_t graphics_queue_family = 0;
        VkRenderPass render_pass = VK_NULL_HANDLE;

        VkCommandPool upload_command_pool = VK_NULL_HANDLE;
        VkBuffer quad_buffer = VK_NULL_HANDLE;
        VkDeviceMemory quad_memory = VK_NULL_HANDLE;
        bool quad_flip_y = false;
        bool quad_initialized = false;

        VkBuffer overlay_buffer = VK_NULL_HANDLE;
        VkDeviceMemory overlay_memory = VK_NULL_HANDLE;
        std::size_t overlay_capacity = 0;
        std::uint32_t overlay_vertex_count = 0;

        VkBuffer shape_overlay_buffer = VK_NULL_HANDLE;
        VkDeviceMemory shape_overlay_memory = VK_NULL_HANDLE;
        std::size_t shape_overlay_capacity = 0;
        std::uint32_t shape_overlay_vertex_count = 0;

        VkBuffer ui_shape_overlay_buffer = VK_NULL_HANDLE;
        VkDeviceMemory ui_shape_overlay_memory = VK_NULL_HANDLE;
        std::size_t ui_shape_overlay_capacity = 0;
        std::uint32_t ui_shape_overlay_vertex_count = 0;

        VkBuffer textured_overlay_buffer = VK_NULL_HANDLE;
        VkDeviceMemory textured_overlay_memory = VK_NULL_HANDLE;
        std::size_t textured_overlay_capacity = 0;
        std::uint32_t textured_overlay_vertex_count = 0;

        VkSampler scene_sampler = VK_NULL_HANDLE;
        VkDescriptorSetLayout scene_descriptor_layout = VK_NULL_HANDLE;
        VkDescriptorPool scene_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
        VkImage scene_image = VK_NULL_HANDLE;
        VkDeviceMemory scene_image_memory = VK_NULL_HANDLE;
        VkImageView scene_image_view = VK_NULL_HANDLE;
        VkImageLayout scene_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        glm::ivec2 scene_image_size{0, 0};
        const lfs::core::Tensor* uploaded_scene_tensor = nullptr;

        VkDescriptorSetLayout grid_descriptor_layout = VK_NULL_HANDLE;
        VkDescriptorPool grid_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet grid_descriptor_set = VK_NULL_HANDLE;
        VkBuffer grid_uniform_buffer = VK_NULL_HANDLE;
        VkDeviceMemory grid_uniform_memory = VK_NULL_HANDLE;
        std::size_t grid_uniform_capacity = 0;
        std::uint32_t grid_uniform_count = 0;

        VkPipelineLayout scene_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline scene_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout vignette_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline vignette_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout grid_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline grid_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout overlay_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline overlay_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout shape_overlay_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline shape_overlay_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout textured_overlay_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline textured_overlay_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pivot_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pivot_pipeline = VK_NULL_HANDLE;

        [[nodiscard]] bool init(VulkanContext& context) {
            if (device != VK_NULL_HANDLE) {
                return true;
            }
            device = context.device();
            physical_device = context.physicalDevice();
            graphics_queue = context.graphicsQueue();
            graphics_queue_family = context.graphicsQueueFamily();
            render_pass = context.renderPass();
            if (device == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE || render_pass == VK_NULL_HANDLE) {
                LOG_ERROR("Vulkan viewport pass requires an initialized Vulkan context");
                device = VK_NULL_HANDLE;
                return false;
            }

            VkCommandPoolCreateInfo command_pool_info{};
            command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            command_pool_info.queueFamilyIndex = graphics_queue_family;
            if (vkCreateCommandPool(device, &command_pool_info, nullptr, &upload_command_pool) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan viewport upload command pool");
                reset();
                return false;
            }

            if (!createSampler() || !createSceneDescriptors() || !createGridResources() ||
                !createQuadBuffer() || !createPipelines()) {
                reset();
                return false;
            }
            return true;
        }

        [[nodiscard]] std::uint32_t findMemoryType(const std::uint32_t type_filter,
                                                   const VkMemoryPropertyFlags properties) const {
            VkPhysicalDeviceMemoryProperties memory_properties{};
            vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
            for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
                if ((type_filter & (1u << i)) &&
                    (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }
            return std::numeric_limits<std::uint32_t>::max();
        }

        [[nodiscard]] bool createBuffer(const VkDeviceSize size,
                                        const VkBufferUsageFlags usage,
                                        const VkMemoryPropertyFlags properties,
                                        VkBuffer& buffer,
                                        VkDeviceMemory& memory) const {
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = size;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
                return false;
            }

            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device, buffer, &requirements);
            VkMemoryAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = requirements.size;
            alloc_info.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, properties);
            if (alloc_info.memoryTypeIndex == std::numeric_limits<std::uint32_t>::max() ||
                vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
                vkDestroyBuffer(device, buffer, nullptr);
                buffer = VK_NULL_HANDLE;
                return false;
            }
            if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
                vkDestroyBuffer(device, buffer, nullptr);
                vkFreeMemory(device, memory, nullptr);
                buffer = VK_NULL_HANDLE;
                memory = VK_NULL_HANDLE;
                return false;
            }
            return true;
        }

        [[nodiscard]] bool createSampler() {
            VkSamplerCreateInfo sampler_info{};
            sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler_info.magFilter = VK_FILTER_LINEAR;
            sampler_info.minFilter = VK_FILTER_LINEAR;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.maxLod = 1.0f;
            return vkCreateSampler(device, &sampler_info, nullptr, &scene_sampler) == VK_SUCCESS;
        }

        [[nodiscard]] bool createSceneDescriptors() {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.bindingCount = 1;
            layout_info.pBindings = &binding;
            if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &scene_descriptor_layout) != VK_SUCCESS) {
                return false;
            }

            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = 1;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(device, &pool_info, nullptr, &scene_descriptor_pool) != VK_SUCCESS) {
                return false;
            }

            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = scene_descriptor_pool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &scene_descriptor_layout;
            return vkAllocateDescriptorSets(device, &alloc_info, &scene_descriptor_set) == VK_SUCCESS;
        }

        [[nodiscard]] bool createGridResources() {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.bindingCount = 1;
            layout_info.pBindings = &binding;
            if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &grid_descriptor_layout) != VK_SUCCESS) {
                return false;
            }

            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = 1;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(device, &pool_info, nullptr, &grid_descriptor_pool) != VK_SUCCESS) {
                return false;
            }

            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = grid_descriptor_pool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &grid_descriptor_layout;
            if (vkAllocateDescriptorSets(device, &alloc_info, &grid_descriptor_set) != VK_SUCCESS) {
                return false;
            }

            return true;
        }

        [[nodiscard]] bool createQuadBuffer() {
            return createBuffer(sizeof(Vertex) * 6,
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                quad_buffer,
                                quad_memory);
        }

        [[nodiscard]] VkShaderModule createShaderModule(const std::vector<std::uint32_t>& spirv) const {
            VkShaderModuleCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            create_info.codeSize = spirv.size() * sizeof(std::uint32_t);
            create_info.pCode = spirv.data();
            VkShaderModule module = VK_NULL_HANDLE;
            if (vkCreateShaderModule(device, &create_info, nullptr, &module) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            return module;
        }

        enum class PipelineVertexLayout {
            ScreenQuad,
            ColorOverlay,
            TexturedOverlay,
            ShapeOverlay
        };

        [[nodiscard]] bool createPipeline(const char* vertex_source,
                                          const char* fragment_source,
                                          const char* label,
                                          VkDescriptorSetLayout descriptor_layout,
                                          const VkPushConstantRange* push_constant,
                                          bool enable_blend,
                                          PipelineVertexLayout vertex_layout,
                                          VkPipelineLayout& pipeline_layout,
                                          VkPipeline& pipeline) {
            const auto vertex_spv = compileGlsl(vertex_source, EShLangVertex, label);
            const auto fragment_spv = compileGlsl(fragment_source, EShLangFragment, label);
            if (!vertex_spv || !fragment_spv) {
                return false;
            }

            VkShaderModule vertex_module = createShaderModule(*vertex_spv);
            VkShaderModule fragment_module = createShaderModule(*fragment_spv);
            if (vertex_module == VK_NULL_HANDLE || fragment_module == VK_NULL_HANDLE) {
                if (vertex_module != VK_NULL_HANDLE)
                    vkDestroyShaderModule(device, vertex_module, nullptr);
                if (fragment_module != VK_NULL_HANDLE)
                    vkDestroyShaderModule(device, fragment_module, nullptr);
                return false;
            }

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertex_module;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragment_module;
            stages[1].pName = "main";

            VkVertexInputBindingDescription binding{};
            binding.binding = 0;
            binding.stride = sizeof(Vertex);
            if (vertex_layout == PipelineVertexLayout::ColorOverlay) {
                binding.stride = sizeof(VulkanViewportOverlayVertex);
            } else if (vertex_layout == PipelineVertexLayout::TexturedOverlay) {
                binding.stride = sizeof(VulkanViewportTexturedOverlayVertex);
            } else if (vertex_layout == PipelineVertexLayout::ShapeOverlay) {
                binding.stride = sizeof(VulkanViewportShapeOverlayVertex);
            }
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            std::array<VkVertexInputAttributeDescription, 6> attributes{};
            attributes[0].location = 0;
            attributes[0].binding = 0;
            attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
            if (vertex_layout == PipelineVertexLayout::ColorOverlay) {
                attributes[0].offset = offsetof(VulkanViewportOverlayVertex, position);
            } else if (vertex_layout == PipelineVertexLayout::TexturedOverlay) {
                attributes[0].offset = offsetof(VulkanViewportTexturedOverlayVertex, position);
            } else if (vertex_layout == PipelineVertexLayout::ShapeOverlay) {
                attributes[0].offset = offsetof(VulkanViewportShapeOverlayVertex, position);
            } else {
                attributes[0].offset = offsetof(Vertex, position);
            }
            attributes[1].location = 1;
            attributes[1].binding = 0;
            attributes[1].format = vertex_layout == PipelineVertexLayout::ColorOverlay
                                       ? VK_FORMAT_R32G32B32A32_SFLOAT
                                       : VK_FORMAT_R32G32_SFLOAT;
            if (vertex_layout == PipelineVertexLayout::ColorOverlay) {
                attributes[1].offset = offsetof(VulkanViewportOverlayVertex, color);
            } else if (vertex_layout == PipelineVertexLayout::TexturedOverlay) {
                attributes[1].offset = offsetof(VulkanViewportTexturedOverlayVertex, uv);
            } else if (vertex_layout == PipelineVertexLayout::ShapeOverlay) {
                attributes[1].offset = offsetof(VulkanViewportShapeOverlayVertex, screen_position);
            } else {
                attributes[1].offset = offsetof(Vertex, uv);
            }
            if (vertex_layout == PipelineVertexLayout::ShapeOverlay) {
                attributes[2].location = 2;
                attributes[2].binding = 0;
                attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
                attributes[2].offset = offsetof(VulkanViewportShapeOverlayVertex, p0);
                attributes[3].location = 3;
                attributes[3].binding = 0;
                attributes[3].format = VK_FORMAT_R32G32_SFLOAT;
                attributes[3].offset = offsetof(VulkanViewportShapeOverlayVertex, p1);
                attributes[4].location = 4;
                attributes[4].binding = 0;
                attributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
                attributes[4].offset = offsetof(VulkanViewportShapeOverlayVertex, color);
                attributes[5].location = 5;
                attributes[5].binding = 0;
                attributes[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
                attributes[5].offset = offsetof(VulkanViewportShapeOverlayVertex, params);
            }

            VkPipelineVertexInputStateCreateInfo vertex_input{};
            vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertex_input.vertexBindingDescriptionCount = 1;
            vertex_input.pVertexBindingDescriptions = &binding;
            vertex_input.vertexAttributeDescriptionCount =
                vertex_layout == PipelineVertexLayout::ShapeOverlay ? 6u : 2u;
            vertex_input.pVertexAttributeDescriptions = attributes.data();

            VkPipelineInputAssemblyStateCreateInfo input_assembly{};
            input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewport_state{};
            viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth{};
            depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth.depthTestEnable = VK_FALSE;
            depth.depthWriteEnable = VK_FALSE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blend_attachment.blendEnable = enable_blend ? VK_TRUE : VK_FALSE;
            blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
            blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo blend{};
            blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            blend.attachmentCount = 1;
            blend.pAttachments = &blend_attachment;

            std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
            dynamic.pDynamicStates = dynamic_states.data();

            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            if (descriptor_layout != VK_NULL_HANDLE) {
                layout_info.setLayoutCount = 1;
                layout_info.pSetLayouts = &descriptor_layout;
            }
            if (push_constant) {
                layout_info.pushConstantRangeCount = 1;
                layout_info.pPushConstantRanges = push_constant;
            }
            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
                vkDestroyShaderModule(device, vertex_module, nullptr);
                vkDestroyShaderModule(device, fragment_module, nullptr);
                return false;
            }

            VkGraphicsPipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipeline_info.stageCount = 2;
            pipeline_info.pStages = stages;
            pipeline_info.pVertexInputState = &vertex_input;
            pipeline_info.pInputAssemblyState = &input_assembly;
            pipeline_info.pViewportState = &viewport_state;
            pipeline_info.pRasterizationState = &raster;
            pipeline_info.pMultisampleState = &multisample;
            pipeline_info.pDepthStencilState = &depth;
            pipeline_info.pColorBlendState = &blend;
            pipeline_info.pDynamicState = &dynamic;
            pipeline_info.layout = pipeline_layout;
            pipeline_info.renderPass = render_pass;
            pipeline_info.subpass = 0;

            const bool ok =
                vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) == VK_SUCCESS;
            vkDestroyShaderModule(device, vertex_module, nullptr);
            vkDestroyShaderModule(device, fragment_module, nullptr);
            return ok;
        }

        [[nodiscard]] bool createPipelines() {
            VkPushConstantRange grid_push{};
            grid_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            grid_push.offset = 0;
            grid_push.size = sizeof(GridPush);
            VkPushConstantRange vignette_push{};
            vignette_push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            vignette_push.offset = 0;
            vignette_push.size = sizeof(VignettePush);
            VkPushConstantRange pivot_push{};
            pivot_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pivot_push.offset = 0;
            pivot_push.size = sizeof(PivotPush);
            VkPushConstantRange textured_overlay_push{};
            textured_overlay_push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            textured_overlay_push.offset = 0;
            textured_overlay_push.size = sizeof(TexturedOverlayPush);
            return createPipeline(kScreenQuadVert, kSceneFrag, "scene",
                                  scene_descriptor_layout, nullptr, false, PipelineVertexLayout::ScreenQuad,
                                  scene_pipeline_layout, scene_pipeline) &&
                   createPipeline(kScreenQuadVert, kVignetteFrag, "vignette",
                                  VK_NULL_HANDLE, &vignette_push, true, PipelineVertexLayout::ScreenQuad,
                                  vignette_pipeline_layout, vignette_pipeline) &&
                   createPipeline(kGridVert, kGridFrag, "grid",
                                  grid_descriptor_layout, &grid_push, true, PipelineVertexLayout::ScreenQuad,
                                  grid_pipeline_layout, grid_pipeline) &&
                   createPipeline(kOverlayVert, kOverlayFrag, "overlay",
                                  VK_NULL_HANDLE, nullptr, true, PipelineVertexLayout::ColorOverlay,
                                  overlay_pipeline_layout, overlay_pipeline) &&
                   createPipeline(kShapeOverlayVert, kShapeOverlayFrag, "shape_overlay",
                                  VK_NULL_HANDLE, nullptr, true, PipelineVertexLayout::ShapeOverlay,
                                  shape_overlay_pipeline_layout, shape_overlay_pipeline) &&
                   createPipeline(kTexturedOverlayVert, kTexturedOverlayFrag, "textured_overlay",
                                  scene_descriptor_layout, &textured_overlay_push, true,
                                  PipelineVertexLayout::TexturedOverlay,
                                  textured_overlay_pipeline_layout, textured_overlay_pipeline) &&
                   createPipeline(kPivotVert, kPivotFrag, "pivot",
                                  VK_NULL_HANDLE, &pivot_push, true, PipelineVertexLayout::ScreenQuad,
                                  pivot_pipeline_layout, pivot_pipeline);
        }

        [[nodiscard]] VkCommandBuffer beginUploadCommands() const {
            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = upload_command_pool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc_info.commandBufferCount = 1;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
                return VK_NULL_HANDLE;
            }
            return command_buffer;
        }

        [[nodiscard]] bool endUploadCommands(const VkCommandBuffer command_buffer) const {
            if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
                return false;
            }
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            const VkResult result = vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
            if (result == VK_SUCCESS) {
                vkQueueWaitIdle(graphics_queue);
            }
            vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
            return result == VK_SUCCESS;
        }

        void transitionSceneImage(const VkCommandBuffer command_buffer,
                                  const VkImageLayout old_layout,
                                  const VkImageLayout new_layout) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = old_layout;
            barrier.newLayout = new_layout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = scene_image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            if (old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            }
            if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            } else {
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            }

            vkCmdPipelineBarrier(command_buffer,
                                 src_stage,
                                 dst_stage,
                                 0,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr,
                                 1,
                                 &barrier);
        }

        void destroySceneImage() {
            if (scene_image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, scene_image_view, nullptr);
                scene_image_view = VK_NULL_HANDLE;
            }
            if (scene_image != VK_NULL_HANDLE) {
                vkDestroyImage(device, scene_image, nullptr);
                scene_image = VK_NULL_HANDLE;
            }
            if (scene_image_memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, scene_image_memory, nullptr);
                scene_image_memory = VK_NULL_HANDLE;
            }
            scene_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            scene_image_size = {0, 0};
            uploaded_scene_tensor = nullptr;
        }

        [[nodiscard]] bool ensureSceneImage(const glm::ivec2 size) {
            if (scene_image != VK_NULL_HANDLE && scene_image_size == size) {
                return true;
            }
            destroySceneImage();

            VkImageCreateInfo image_info{};
            image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.extent = {static_cast<std::uint32_t>(size.x), static_cast<std::uint32_t>(size.y), 1};
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateImage(device, &image_info, nullptr, &scene_image) != VK_SUCCESS) {
                return false;
            }

            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device, scene_image, &requirements);
            VkMemoryAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = requirements.size;
            alloc_info.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (alloc_info.memoryTypeIndex == std::numeric_limits<std::uint32_t>::max() ||
                vkAllocateMemory(device, &alloc_info, nullptr, &scene_image_memory) != VK_SUCCESS ||
                vkBindImageMemory(device, scene_image, scene_image_memory, 0) != VK_SUCCESS) {
                destroySceneImage();
                return false;
            }

            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = scene_image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &view_info, nullptr, &scene_image_view) != VK_SUCCESS) {
                destroySceneImage();
                return false;
            }

            scene_image_size = size;
            VkDescriptorImageInfo descriptor_info{};
            descriptor_info.sampler = scene_sampler;
            descriptor_info.imageView = scene_image_view;
            descriptor_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = scene_descriptor_set;
            write.dstBinding = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &descriptor_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
            return true;
        }

        void updateQuadBuffer(const bool flip_y) {
            if (quad_initialized && quad_flip_y == flip_y) {
                return;
            }
            const float top_v = flip_y ? 1.0f : 0.0f;
            const float bottom_v = flip_y ? 0.0f : 1.0f;
            const std::array<Vertex, 6> vertices{{
                {{-1.0f, -1.0f}, {0.0f, top_v}},
                {{1.0f, -1.0f}, {1.0f, top_v}},
                {{1.0f, 1.0f}, {1.0f, bottom_v}},
                {{-1.0f, -1.0f}, {0.0f, top_v}},
                {{1.0f, 1.0f}, {1.0f, bottom_v}},
                {{-1.0f, 1.0f}, {0.0f, bottom_v}},
            }};
            void* mapped = nullptr;
            if (vkMapMemory(device, quad_memory, 0, sizeof(vertices), 0, &mapped) == VK_SUCCESS && mapped) {
                std::memcpy(mapped, vertices.data(), sizeof(vertices));
                vkUnmapMemory(device, quad_memory);
                quad_flip_y = flip_y;
                quad_initialized = true;
            }
        }

        bool ensureOverlayBuffer(const std::size_t vertex_count) {
            if (vertex_count == 0) {
                overlay_vertex_count = 0;
                return true;
            }
            if (overlay_buffer != VK_NULL_HANDLE && overlay_capacity >= vertex_count) {
                return true;
            }
            if (overlay_buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, overlay_buffer, nullptr);
                overlay_buffer = VK_NULL_HANDLE;
            }
            if (overlay_memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, overlay_memory, nullptr);
                overlay_memory = VK_NULL_HANDLE;
            }

            std::size_t capacity = 256;
            while (capacity < vertex_count) {
                capacity *= 2;
            }
            if (!createBuffer(sizeof(VulkanViewportOverlayVertex) * capacity,
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              overlay_buffer,
                              overlay_memory)) {
                overlay_capacity = 0;
                overlay_vertex_count = 0;
                return false;
            }
            overlay_capacity = capacity;
            return true;
        }

        void updateOverlayBuffer(const VulkanViewportPassParams& params) {
            overlay_vertex_count = 0;
            if (params.overlay_triangles.empty()) {
                return;
            }
            if (!ensureOverlayBuffer(params.overlay_triangles.size())) {
                return;
            }
            void* mapped = nullptr;
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(sizeof(VulkanViewportOverlayVertex) * params.overlay_triangles.size());
            if (vkMapMemory(device, overlay_memory, 0, bytes, 0, &mapped) != VK_SUCCESS || !mapped) {
                return;
            }
            std::memcpy(mapped, params.overlay_triangles.data(), static_cast<std::size_t>(bytes));
            vkUnmapMemory(device, overlay_memory);
            overlay_vertex_count = static_cast<std::uint32_t>(
                std::min<std::size_t>(params.overlay_triangles.size(), std::numeric_limits<std::uint32_t>::max()));
        }

        bool ensureShapeOverlayBuffer(const std::size_t vertex_count,
                                      VkBuffer& buffer,
                                      VkDeviceMemory& memory,
                                      std::size_t& capacity,
                                      std::uint32_t& out_count) {
            if (vertex_count == 0) {
                out_count = 0;
                return true;
            }
            if (buffer != VK_NULL_HANDLE && capacity >= vertex_count) {
                return true;
            }
            if (buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, buffer, nullptr);
                buffer = VK_NULL_HANDLE;
            }
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, memory, nullptr);
                memory = VK_NULL_HANDLE;
            }

            std::size_t new_capacity = 256;
            while (new_capacity < vertex_count) {
                new_capacity *= 2;
            }
            if (!createBuffer(sizeof(VulkanViewportShapeOverlayVertex) * new_capacity,
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              buffer,
                              memory)) {
                capacity = 0;
                out_count = 0;
                return false;
            }
            capacity = new_capacity;
            return true;
        }

        void updateShapeOverlayBuffer(const std::vector<VulkanViewportShapeOverlayVertex>& vertices,
                                      VkBuffer& buffer,
                                      VkDeviceMemory& memory,
                                      std::size_t& capacity,
                                      std::uint32_t& out_count) {
            out_count = 0;
            if (vertices.empty()) {
                return;
            }
            if (!ensureShapeOverlayBuffer(vertices.size(), buffer, memory, capacity, out_count)) {
                return;
            }
            void* mapped = nullptr;
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(sizeof(VulkanViewportShapeOverlayVertex) * vertices.size());
            if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS || !mapped) {
                return;
            }
            std::memcpy(mapped, vertices.data(), static_cast<std::size_t>(bytes));
            vkUnmapMemory(device, memory);
            out_count = static_cast<std::uint32_t>(
                std::min<std::size_t>(vertices.size(), std::numeric_limits<std::uint32_t>::max()));
        }

        bool ensureTexturedOverlayBuffer(const std::size_t vertex_count) {
            if (vertex_count == 0) {
                textured_overlay_vertex_count = 0;
                return true;
            }
            if (textured_overlay_buffer != VK_NULL_HANDLE && textured_overlay_capacity >= vertex_count) {
                return true;
            }
            if (textured_overlay_buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, textured_overlay_buffer, nullptr);
                textured_overlay_buffer = VK_NULL_HANDLE;
            }
            if (textured_overlay_memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, textured_overlay_memory, nullptr);
                textured_overlay_memory = VK_NULL_HANDLE;
            }

            std::size_t capacity = 64;
            while (capacity < vertex_count) {
                capacity *= 2;
            }
            if (!createBuffer(sizeof(VulkanViewportTexturedOverlayVertex) * capacity,
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              textured_overlay_buffer,
                              textured_overlay_memory)) {
                textured_overlay_capacity = 0;
                textured_overlay_vertex_count = 0;
                return false;
            }
            textured_overlay_capacity = capacity;
            return true;
        }

        void updateTexturedOverlayBuffer(const VulkanViewportPassParams& params) {
            textured_overlay_vertex_count = 0;
            if (params.textured_overlays.empty()) {
                return;
            }
            const std::size_t vertex_count = params.textured_overlays.size() * 6u;
            if (!ensureTexturedOverlayBuffer(vertex_count)) {
                return;
            }

            std::vector<VulkanViewportTexturedOverlayVertex> vertices;
            vertices.reserve(vertex_count);
            for (const auto& overlay : params.textured_overlays) {
                vertices.insert(vertices.end(), overlay.vertices.begin(), overlay.vertices.end());
            }

            void* mapped = nullptr;
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(sizeof(VulkanViewportTexturedOverlayVertex) * vertices.size());
            if (vkMapMemory(device, textured_overlay_memory, 0, bytes, 0, &mapped) != VK_SUCCESS || !mapped) {
                return;
            }
            std::memcpy(mapped, vertices.data(), static_cast<std::size_t>(bytes));
            vkUnmapMemory(device, textured_overlay_memory);
            textured_overlay_vertex_count = static_cast<std::uint32_t>(
                std::min<std::size_t>(vertices.size(), std::numeric_limits<std::uint32_t>::max()));
        }

        void updateGridDescriptor(const VkDeviceSize range) const {
            if (grid_descriptor_set == VK_NULL_HANDLE || grid_uniform_buffer == VK_NULL_HANDLE) {
                return;
            }
            VkDescriptorBufferInfo buffer_info{};
            buffer_info.buffer = grid_uniform_buffer;
            buffer_info.offset = 0;
            buffer_info.range = range;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = grid_descriptor_set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &buffer_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        [[nodiscard]] bool ensureGridUniformBuffer(const std::size_t grid_count) {
            if (grid_count == 0) {
                grid_uniform_count = 0;
                return true;
            }
            if (grid_uniform_buffer != VK_NULL_HANDLE && grid_uniform_capacity >= grid_count) {
                return true;
            }
            if (grid_uniform_buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, grid_uniform_buffer, nullptr);
                grid_uniform_buffer = VK_NULL_HANDLE;
            }
            if (grid_uniform_memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, grid_uniform_memory, nullptr);
                grid_uniform_memory = VK_NULL_HANDLE;
            }

            std::size_t capacity = 1;
            while (capacity < grid_count) {
                capacity *= 2;
            }
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(sizeof(GridUniform) * capacity);
            if (!createBuffer(bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              grid_uniform_buffer,
                              grid_uniform_memory)) {
                grid_uniform_capacity = 0;
                grid_uniform_count = 0;
                return false;
            }
            grid_uniform_capacity = capacity;
            updateGridDescriptor(bytes);
            return true;
        }

        [[nodiscard]] static GridUniform makeGridUniform(const VulkanViewportGridOverlay& grid) {
            const glm::mat4 view_inv = glm::inverse(grid.view);
            const glm::vec3 cam_pos = glm::vec3(view_inv[3]);
            const glm::vec3 cam_right = glm::vec3(view_inv[0]);
            const glm::vec3 cam_up = glm::vec3(view_inv[1]);
            const glm::vec3 cam_forward = -glm::vec3(view_inv[2]);

            glm::vec3 near_origin{0.0f};
            glm::vec3 near_x{0.0f};
            glm::vec3 near_y{0.0f};
            glm::vec3 far_origin{0.0f};
            glm::vec3 far_x{0.0f};
            glm::vec3 far_y{0.0f};
            if (grid.orthographic) {
                const float half_width = 1.0f / grid.projection[0][0];
                const float half_height = 1.0f / std::abs(grid.projection[1][1]);
                const glm::vec3 right_offset = cam_right * half_width;
                const glm::vec3 up_offset = cam_up * half_height;
                constexpr float kRayNear = -1000.0f;
                constexpr float kRayFar = 1000.0f;

                const glm::vec3 near_center = cam_pos + cam_forward * kRayNear;
                near_origin = near_center - right_offset - up_offset;
                near_x = right_offset * 2.0f;
                near_y = up_offset * 2.0f;

                const glm::vec3 far_center = cam_pos + cam_forward * kRayFar;
                far_origin = far_center - right_offset - up_offset;
                far_x = right_offset * 2.0f;
                far_y = up_offset * 2.0f;
            } else {
                const float fov_y = 2.0f * std::atan(1.0f / std::abs(grid.projection[1][1]));
                const float aspect = std::abs(grid.projection[1][1] / grid.projection[0][0]);
                const float half_height = std::tan(fov_y * 0.5f);
                const float half_width = half_height * aspect;
                const glm::vec3 far_center = cam_pos + cam_forward;
                const glm::vec3 right_offset = cam_right * half_width;
                const glm::vec3 up_offset = cam_up * half_height;
                const glm::vec3 far_bl = far_center - right_offset - up_offset;
                const glm::vec3 far_br = far_center + right_offset - up_offset;
                const glm::vec3 far_tl = far_center - right_offset + up_offset;

                near_origin = cam_pos;
                far_origin = far_bl;
                far_x = far_br - far_bl;
                far_y = far_tl - far_bl;
            }

            GridUniform uniform{};
            uniform.view_projection = grid.view_projection;
            uniform.view_position_plane = glm::vec4(grid.view_position,
                                                    static_cast<float>(std::clamp(grid.plane, 0, 2)));
            uniform.opacity_padding = glm::vec4(std::clamp(grid.opacity, 0.0f, 1.0f), 0.0f, 0.0f, 0.0f);
            uniform.near_origin = glm::vec4(near_origin, 0.0f);
            uniform.near_x = glm::vec4(near_x, 0.0f);
            uniform.near_y = glm::vec4(near_y, 0.0f);
            uniform.far_origin = glm::vec4(far_origin, 0.0f);
            uniform.far_x = glm::vec4(far_x, 0.0f);
            uniform.far_y = glm::vec4(far_y, 0.0f);
            return uniform;
        }

        [[nodiscard]] static std::vector<VulkanViewportGridOverlay> collectGridOverlays(
            const VulkanViewportPassParams& params) {
            if (!params.grid_overlays.empty()) {
                return params.grid_overlays;
            }
            if (!params.grid_enabled) {
                return {};
            }
            return {VulkanViewportGridOverlay{
                .viewport_pos = params.viewport_pos,
                .viewport_size = params.viewport_size,
                .render_size = {
                    std::max(static_cast<int>(std::lround(params.viewport_size.x)), 1),
                    std::max(static_cast<int>(std::lround(params.viewport_size.y)), 1)},
                .view = params.grid_view,
                .projection = params.grid_projection,
                .view_projection = params.grid_view_projection,
                .view_position = params.grid_view_position,
                .plane = params.grid_plane,
                .opacity = params.grid_opacity,
                .orthographic = params.grid_orthographic,
            }};
        }

        void updateGridUniforms(const VulkanViewportPassParams& params) {
            const auto grids = collectGridOverlays(params);
            if (grids.empty()) {
                grid_uniform_count = 0;
                return;
            }
            if (!ensureGridUniformBuffer(grids.size())) {
                return;
            }

            std::vector<GridUniform> uniforms;
            uniforms.reserve(grids.size());
            for (const auto& grid : grids) {
                uniforms.push_back(makeGridUniform(grid));
            }
            if (uniforms.empty()) {
                grid_uniform_count = 0;
                return;
            }

            void* mapped = nullptr;
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(sizeof(GridUniform) * uniforms.size());
            if (vkMapMemory(device, grid_uniform_memory, 0, bytes, 0, &mapped) == VK_SUCCESS && mapped) {
                std::memcpy(mapped, uniforms.data(), static_cast<std::size_t>(bytes));
                vkUnmapMemory(device, grid_uniform_memory);
                grid_uniform_count = static_cast<std::uint32_t>(
                    std::min<std::size_t>(uniforms.size(), std::numeric_limits<std::uint32_t>::max()));
            }
        }

        void uploadSceneImage(const VulkanViewportPassParams& params) {
            if (!params.scene_image || params.scene_image_size.x <= 0 || params.scene_image_size.y <= 0) {
                uploaded_scene_tensor = nullptr;
                return;
            }
            if (uploaded_scene_tensor == params.scene_image.get() && scene_image_size == params.scene_image_size &&
                scene_image_view != VK_NULL_HANDLE) {
                return;
            }
            const auto rgba = tensorToRgba8(*params.scene_image, params.scene_image_size);
            if (!rgba || rgba->empty() || !ensureSceneImage(params.scene_image_size)) {
                return;
            }

            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VkDeviceMemory staging_memory = VK_NULL_HANDLE;
            const VkDeviceSize upload_size = static_cast<VkDeviceSize>(rgba->size());
            if (!createBuffer(upload_size,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              staging_buffer,
                              staging_memory)) {
                return;
            }

            void* mapped = nullptr;
            if (vkMapMemory(device, staging_memory, 0, upload_size, 0, &mapped) != VK_SUCCESS || !mapped) {
                vkDestroyBuffer(device, staging_buffer, nullptr);
                vkFreeMemory(device, staging_memory, nullptr);
                return;
            }
            std::memcpy(mapped, rgba->data(), rgba->size());
            vkUnmapMemory(device, staging_memory);

            VkCommandBuffer command_buffer = beginUploadCommands();
            if (command_buffer == VK_NULL_HANDLE) {
                vkDestroyBuffer(device, staging_buffer, nullptr);
                vkFreeMemory(device, staging_memory, nullptr);
                return;
            }
            transitionSceneImage(command_buffer, scene_image_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy copy{};
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = 0;
            copy.imageSubresource.baseArrayLayer = 0;
            copy.imageSubresource.layerCount = 1;
            copy.imageExtent = {
                static_cast<std::uint32_t>(params.scene_image_size.x),
                static_cast<std::uint32_t>(params.scene_image_size.y),
                1,
            };
            vkCmdCopyBufferToImage(command_buffer,
                                   staging_buffer,
                                   scene_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1,
                                   &copy);
            transitionSceneImage(command_buffer,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (endUploadCommands(command_buffer)) {
                scene_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                uploaded_scene_tensor = params.scene_image.get();
            }
            vkDestroyBuffer(device, staging_buffer, nullptr);
            vkFreeMemory(device, staging_memory, nullptr);
        }

        void prepare(const VulkanViewportPassParams& params) {
            updateQuadBuffer(params.scene_image_flip_y);
            updateGridUniforms(params);
            updateTexturedOverlayBuffer(params);
            updateOverlayBuffer(params);
            updateShapeOverlayBuffer(params.shape_overlay_triangles,
                                     shape_overlay_buffer,
                                     shape_overlay_memory,
                                     shape_overlay_capacity,
                                     shape_overlay_vertex_count);
            updateShapeOverlayBuffer(params.ui_shape_overlay_triangles,
                                     ui_shape_overlay_buffer,
                                     ui_shape_overlay_memory,
                                     ui_shape_overlay_capacity,
                                     ui_shape_overlay_vertex_count);
            uploadSceneImage(params);
        }

        void bindViewport(VkCommandBuffer command_buffer, const FramebufferRect& rect) const {
            VkViewport viewport{};
            viewport.x = static_cast<float>(rect.x);
            viewport.y = static_cast<float>(rect.y);
            viewport.width = static_cast<float>(rect.width);
            viewport.height = static_cast<float>(rect.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.offset = {rect.x, rect.y};
            scissor.extent = {rect.width, rect.height};
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
        }

        void bindQuad(VkCommandBuffer command_buffer) const {
            const VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &quad_buffer, &offset);
        }

        void clearViewport(VkCommandBuffer command_buffer, const FramebufferRect& rect, const glm::vec3 color) const {
            VkClearAttachment attachment{};
            attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            attachment.colorAttachment = 0;
            attachment.clearValue.color = VkClearColorValue{{color.r, color.g, color.b, 1.0f}};
            VkClearRect clear_rect{};
            clear_rect.rect.offset = {rect.x, rect.y};
            clear_rect.rect.extent = {rect.width, rect.height};
            clear_rect.baseArrayLayer = 0;
            clear_rect.layerCount = 1;
            vkCmdClearAttachments(command_buffer, 1, &attachment, 1, &clear_rect);
        }

        void recordGridOverlays(VkCommandBuffer command_buffer,
                                const VkExtent2D extent,
                                const VulkanViewportPassParams& params,
                                const FramebufferRect& main_rect) const {
            if (grid_uniform_count == 0 ||
                grid_pipeline == VK_NULL_HANDLE ||
                grid_descriptor_set == VK_NULL_HANDLE ||
                grid_uniform_buffer == VK_NULL_HANDLE) {
                return;
            }

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, grid_pipeline);
            vkCmdBindDescriptorSets(command_buffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    grid_pipeline_layout,
                                    0,
                                    1,
                                    &grid_descriptor_set,
                                    0,
                                    nullptr);
            const auto grids = collectGridOverlays(params);
            for (std::uint32_t i = 0;
                 i < std::min<std::uint32_t>(grid_uniform_count, static_cast<std::uint32_t>(grids.size()));
                 ++i) {
                const auto& grid = grids[i];
                if (grid.viewport_size.x <= 0.0f || grid.viewport_size.y <= 0.0f ||
                    grid.render_size.x <= 0 || grid.render_size.y <= 0 ||
                    grid.opacity <= 0.0f) {
                    continue;
                }
                const FramebufferRect grid_rect = toFramebufferRect(params, grid, extent);
                if (grid_rect.width == 0 || grid_rect.height == 0) {
                    continue;
                }
                bindViewport(command_buffer, grid_rect);
                GridPush push{};
                push.grid_index = static_cast<std::int32_t>(i);
                vkCmdPushConstants(command_buffer,
                                   grid_pipeline_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0,
                                   sizeof(push),
                                   &push);
                vkCmdDraw(command_buffer, 6, 1, 0, 0);
            }
            bindViewport(command_buffer, main_rect);
        }

        void recordShapeOverlays(VkCommandBuffer command_buffer,
                                 const VkBuffer buffer,
                                 const std::uint32_t vertex_count) const {
            if (vertex_count == 0 ||
                buffer == VK_NULL_HANDLE ||
                shape_overlay_pipeline == VK_NULL_HANDLE) {
                return;
            }
            const VkDeviceSize offset = 0;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shape_overlay_pipeline);
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &buffer, &offset);
            vkCmdDraw(command_buffer, vertex_count, 1, 0, 0);
            bindQuad(command_buffer);
        }

        void record(VkCommandBuffer command_buffer,
                    const VkExtent2D extent,
                    const VulkanViewportPassParams& params) {
            const FramebufferRect rect = toFramebufferRect(params, extent);
            if (rect.width == 0 || rect.height == 0 || quad_buffer == VK_NULL_HANDLE) {
                return;
            }
            bindViewport(command_buffer, rect);
            bindQuad(command_buffer);
            clearViewport(command_buffer, rect, params.background_color);

            const bool has_scene =
                params.scene_image && scene_image_view != VK_NULL_HANDLE &&
                scene_descriptor_set != VK_NULL_HANDLE && scene_pipeline != VK_NULL_HANDLE;
            if (has_scene) {
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_pipeline);
                vkCmdBindDescriptorSets(command_buffer,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        scene_pipeline_layout,
                                        0,
                                        1,
                                        &scene_descriptor_set,
                                        0,
                                        nullptr);
                vkCmdDraw(command_buffer, 6, 1, 0, 0);
            }
            if (textured_overlay_vertex_count > 0 && textured_overlay_pipeline != VK_NULL_HANDLE &&
                textured_overlay_buffer != VK_NULL_HANDLE && !params.textured_overlays.empty()) {
                const VkDeviceSize offset = 0;
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textured_overlay_pipeline);
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &textured_overlay_buffer, &offset);
                std::uint32_t first_vertex = 0;
                for (const auto& overlay : params.textured_overlays) {
                    if (overlay.texture_id == 0 || first_vertex + 6u > textured_overlay_vertex_count) {
                        first_vertex += 6u;
                        continue;
                    }
                    const VkDescriptorSet descriptor_set = descriptorSetFromId(overlay.texture_id);
                    if (descriptor_set == VK_NULL_HANDLE) {
                        first_vertex += 6u;
                        continue;
                    }
                    TexturedOverlayPush push{};
                    push.tint_opacity = overlay.tint_opacity;
                    push.effects = overlay.effects;
                    vkCmdBindDescriptorSets(command_buffer,
                                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            textured_overlay_pipeline_layout,
                                            0,
                                            1,
                                            &descriptor_set,
                                            0,
                                            nullptr);
                    vkCmdPushConstants(command_buffer,
                                       textured_overlay_pipeline_layout,
                                       VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0,
                                       sizeof(push),
                                       &push);
                    vkCmdDraw(command_buffer, 6, 1, first_vertex, 0);
                    first_vertex += 6u;
                }
                bindQuad(command_buffer);
            }

            if (overlay_vertex_count > 0 && overlay_pipeline != VK_NULL_HANDLE &&
                overlay_buffer != VK_NULL_HANDLE) {
                const VkDeviceSize offset = 0;
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline);
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &overlay_buffer, &offset);
                vkCmdDraw(command_buffer, overlay_vertex_count, 1, 0, 0);
                bindQuad(command_buffer);
            }

            recordShapeOverlays(command_buffer, shape_overlay_buffer, shape_overlay_vertex_count);

            if (!params.pivot_overlays.empty() && pivot_pipeline != VK_NULL_HANDLE) {
                bindQuad(command_buffer);
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pivot_pipeline);
                for (const auto& pivot : params.pivot_overlays) {
                    PivotPush push{};
                    push.center_size = {
                        pivot.center_ndc.x,
                        pivot.center_ndc.y,
                        pivot.size_ndc.x,
                        pivot.size_ndc.y,
                    };
                    push.color_opacity = {
                        pivot.color.r,
                        pivot.color.g,
                        pivot.color.b,
                        std::clamp(pivot.opacity, 0.0f, 1.0f),
                    };
                    vkCmdPushConstants(command_buffer,
                                       pivot_pipeline_layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0,
                                       sizeof(push),
                                       &push);
                    vkCmdDraw(command_buffer, 6, 1, 0, 0);
                }
            }

            recordGridOverlays(command_buffer, extent, params, rect);

            if (params.vignette_enabled && vignette_pipeline != VK_NULL_HANDLE) {
                VignettePush push{};
                push.viewport_intensity_radius = {
                    static_cast<float>(rect.width),
                    static_cast<float>(rect.height),
                    params.vignette_intensity,
                    params.vignette_radius,
                };
                push.softness_padding = {params.vignette_softness, 0.0f, 0.0f, 0.0f};
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vignette_pipeline);
                vkCmdPushConstants(command_buffer,
                                   vignette_pipeline_layout,
                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0,
                                   sizeof(push),
                                   &push);
                vkCmdDraw(command_buffer, 6, 1, 0, 0);
            }

            recordShapeOverlays(command_buffer, ui_shape_overlay_buffer, ui_shape_overlay_vertex_count);
        }

        void reset() {
            if (device != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(device);
                destroySceneImage();
                if (scene_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, scene_pipeline, nullptr);
                if (vignette_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, vignette_pipeline, nullptr);
                if (grid_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, grid_pipeline, nullptr);
                if (overlay_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, overlay_pipeline, nullptr);
                if (shape_overlay_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, shape_overlay_pipeline, nullptr);
                if (textured_overlay_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, textured_overlay_pipeline, nullptr);
                if (pivot_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, pivot_pipeline, nullptr);
                if (scene_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, scene_pipeline_layout, nullptr);
                if (vignette_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, vignette_pipeline_layout, nullptr);
                if (grid_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, grid_pipeline_layout, nullptr);
                if (overlay_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, overlay_pipeline_layout, nullptr);
                if (shape_overlay_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, shape_overlay_pipeline_layout, nullptr);
                if (textured_overlay_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, textured_overlay_pipeline_layout, nullptr);
                if (pivot_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, pivot_pipeline_layout, nullptr);
                if (grid_uniform_buffer != VK_NULL_HANDLE)
                    vkDestroyBuffer(device, grid_uniform_buffer, nullptr);
                if (grid_uniform_memory != VK_NULL_HANDLE)
                    vkFreeMemory(device, grid_uniform_memory, nullptr);
                if (quad_buffer != VK_NULL_HANDLE)
                    vkDestroyBuffer(device, quad_buffer, nullptr);
                if (quad_memory != VK_NULL_HANDLE)
                    vkFreeMemory(device, quad_memory, nullptr);
                if (overlay_buffer != VK_NULL_HANDLE)
                    vkDestroyBuffer(device, overlay_buffer, nullptr);
                if (overlay_memory != VK_NULL_HANDLE)
                    vkFreeMemory(device, overlay_memory, nullptr);
                if (shape_overlay_buffer != VK_NULL_HANDLE)
                    vkDestroyBuffer(device, shape_overlay_buffer, nullptr);
                if (shape_overlay_memory != VK_NULL_HANDLE)
                    vkFreeMemory(device, shape_overlay_memory, nullptr);
                if (ui_shape_overlay_buffer != VK_NULL_HANDLE)
                    vkDestroyBuffer(device, ui_shape_overlay_buffer, nullptr);
                if (ui_shape_overlay_memory != VK_NULL_HANDLE)
                    vkFreeMemory(device, ui_shape_overlay_memory, nullptr);
                if (textured_overlay_buffer != VK_NULL_HANDLE)
                    vkDestroyBuffer(device, textured_overlay_buffer, nullptr);
                if (textured_overlay_memory != VK_NULL_HANDLE)
                    vkFreeMemory(device, textured_overlay_memory, nullptr);
                if (scene_sampler != VK_NULL_HANDLE)
                    vkDestroySampler(device, scene_sampler, nullptr);
                if (scene_descriptor_pool != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(device, scene_descriptor_pool, nullptr);
                if (scene_descriptor_layout != VK_NULL_HANDLE)
                    vkDestroyDescriptorSetLayout(device, scene_descriptor_layout, nullptr);
                if (grid_descriptor_pool != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(device, grid_descriptor_pool, nullptr);
                if (grid_descriptor_layout != VK_NULL_HANDLE)
                    vkDestroyDescriptorSetLayout(device, grid_descriptor_layout, nullptr);
                if (upload_command_pool != VK_NULL_HANDLE)
                    vkDestroyCommandPool(device, upload_command_pool, nullptr);
            }
            *this = {};
        }
#else
        [[nodiscard]] bool init(VulkanContext&) { return false; }
        void prepare(const VulkanViewportPassParams&) {}
        void record(VkCommandBuffer, VkExtent2D, const VulkanViewportPassParams&) {}
        void reset() {}
#endif
    };

    VulkanViewportPass::VulkanViewportPass() = default;

    VulkanViewportPass::~VulkanViewportPass() {
        shutdown();
    }

    bool VulkanViewportPass::init(VulkanContext& context) {
        if (!impl_) {
            impl_ = std::make_unique<Impl>();
        }
        return impl_->init(context);
    }

    void VulkanViewportPass::prepare(VulkanContext& context, const VulkanViewportPassParams& params) {
        if (!impl_ && !init(context)) {
            return;
        }
        impl_->prepare(params);
    }

    void VulkanViewportPass::record(VkCommandBuffer command_buffer,
                                    VkExtent2D framebuffer_extent,
                                    const VulkanViewportPassParams& params) {
        if (impl_) {
            impl_->record(command_buffer, framebuffer_extent, params);
        }
    }

    void VulkanViewportPass::shutdown() {
        if (impl_) {
            impl_->reset();
        }
    }

} // namespace lfs::vis
