// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../training/vulkan/shaders/gsplat_common.slangh"
struct LensFace { float rotation[9]; float focal; int size; };
struct Parameters {
    Ptr<float, Access.ReadWrite> camera, view, radial, tangent, prism, a, b, c, d, e, f;
    float fx, fy, cx, cy;
    uint width, height, model, faces, pixels;
    float r0,r1,r2,r3,r4,r5,r6,r7,r8,focal;
    int size;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
static const float kBorderBand = 0.15f;
LensFace load_face() {
    LensFace face;
    face.rotation[0]=p.r0; face.rotation[1]=p.r1; face.rotation[2]=p.r2;
    face.rotation[3]=p.r3; face.rotation[4]=p.r4; face.rotation[5]=p.r5;
    face.rotation[6]=p.r6; face.rotation[7]=p.r7; face.rotation[8]=p.r8;
    face.focal=p.focal; face.size=p.size; return face;
}
        float3 to_face(const LensFace face, const float3 ray) {
            float r[9] = face.rotation;
            return {r[0] * ray.x + r[1] * ray.y + r[2] * ray.z,
                    r[3] * ray.x + r[4] * ray.y + r[5] * ray.z,
                    r[6] * ray.x + r[7] * ray.y + r[8] * ray.z};
        }

        float3 to_camera(const LensFace face, const float3 ray) {
            float r[9] = face.rotation;
            return {r[0] * ray.x + r[3] * ray.y + r[6] * ray.z,
                    r[1] * ray.x + r[4] * ray.y + r[7] * ray.z,
                    r[2] * ray.x + r[5] * ray.y + r[8] * ray.z};
        }

        bool pixel_ray(const GsplatCamera model, const float x, const float y, out float3 ray) {
            ray=0;
            const GsplatRay camera_ray = gsplat_pixel_ray(model, float2(x, y));
            const float len = length(camera_ray.dir);
            if (!camera_ray.valid || !(len > 0.0f) || !isfinite(len))
                return false;
            ray = camera_ray.dir / len;
            return true;
        }

        void lens_rays_kernel(const GsplatCamera model, const int width, const int height, Ptr<float, Access.ReadWrite> rays, uint tid) {
            const uint i = tid;
            if (i >= uint(width) * height)
                return;
            float3 ray;
            const bool valid = pixel_ray(model, float(i % width) + 0.5f,
                                         float(i / width) + 0.5f, ray);
            for (int c = 0; c < 3; ++c)
                rays[3 * i + c] = valid ? ray[c] : asfloat(0x7fc00000u);
        }

        void sample_face_kernel(const GsplatCamera model, Ptr<float, Access.ReadWrite> lens_rgb,
                                           const int width, const int height, const LensFace face,
                                           Ptr<float, Access.ReadWrite> face_rgb, uint tid) {
            const uint i = tid;
            const int size = face.size;
            if (i >= uint(size) * size)
                return;
            const float half = 0.5f * float(size);
            const float3 face_ray = float3((float(i % size) + 0.5f - half) / face.focal,
                                      (float(i / size) + 0.5f - half) / face.focal, 1.0f);
            float2 uv;
            const bool valid = gsplat_camera_to_image(model, to_camera(face, face_ray), uv);
            const float sx = uv.x - 0.5f;
            const float sy = uv.y - 0.5f;
            float rgb[3] = {0.5f, 0.5f, 0.5f};
            if (valid && sx >= 0.0f && sy >= 0.0f && sx <= width - 1.0f && sy <= height - 1.0f) {
                const int x0 = min(int(sx), width - 2 > 0 ? width - 2 : 0);
                const int y0 = min(int(sy), height - 2 > 0 ? height - 2 : 0);
                const int x1 = min(x0 + 1, width - 1);
                const int y1 = min(y0 + 1, height - 1);
                const float fx = sx - x0;
                const float fy = sy - y0;
                for (int c = 0; c < 3; ++c) {
                    const float top = lens_rgb[(uint(y0) * width + x0) * 3 + c] * (1.0f - fx) +
                                      lens_rgb[(uint(y0) * width + x1) * 3 + c] * fx;
                    const float bottom = lens_rgb[(uint(y1) * width + x0) * 3 + c] * (1.0f - fx) +
                                         lens_rgb[(uint(y1) * width + x1) * 3 + c] * fx;
                    rgb[c] = top * (1.0f - fy) + bottom * fy;
                }
            }
            for (int c = 0; c < 3; ++c)
                face_rgb[3 * i + c] = rgb[c];
        }

        void project_face_kernel(const GsplatCamera model, const int width, const int height,
                                            const LensFace face, Ptr<float, Access.ReadWrite> face_points,
                                            Ptr<float, Access.ReadWrite> face_normals,
                                            Ptr<float, Access.ReadWrite> face_mask, Ptr<float, Access.ReadWrite> distance,
                                            Ptr<float, Access.ReadWrite> normal, Ptr<float, Access.ReadWrite> weight, uint tid) {
            const uint i = tid;
            if (i >= uint(width) * height)
                return;
            distance[i] = asfloat(0x7fc00000u);
            weight[i] = 0.0f;
            for (int c = 0; c < 3; ++c)
                normal[3 * i + c] = 0.0f;

            float3 ray;
            if (!pixel_ray(model, float(i % width) + 0.5f, float(i / width) + 0.5f, ray))
                return;
            const float3 face_ray = to_face(face, ray);
            if (face_ray.z <= 1.0e-4f)
                return;
            const int size = face.size;
            const float half = 0.5f * float(size);
            const float u = face.focal * face_ray.x / face_ray.z + half;
            const float v = face.focal * face_ray.y / face_ray.z + half;
            const float edge = min(min(u, size - u), min(v, size - v)) / half;
            if (!(edge > 0.0f))
                return;

            const float sx = u - 0.5f;
            const float sy = v - 0.5f;
            const int x0 = int(floor(sx));
            const int y0 = int(floor(sy));
            const float fx = sx - x0;
            const float fy = sy - y0;
            float z_sum = 0.0f;
            float tap_sum = 0.0f;
            float3 n_sum = float3(0.0f);
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int x = x0 + dx;
                    const int y = y0 + dy;
                    if (x < 0 || y < 0 || x >= size || y >= size)
                        continue;
                    const uint t = uint(y) * size + x;
                    const float z = face_points[3 * t + 2];
                    if (!(face_mask[t] >= 0.5f) || !(z > 0.0f) || !isfinite(z))
                        continue;
                    const float w = (dx != 0 ? fx : 1.0f - fx) * (dy != 0 ? fy : 1.0f - fy) + 1.0e-6f;
                    z_sum += w * z;
                    tap_sum += w;
                    n_sum += w * float3(face_normals[3 * t], face_normals[3 * t + 1], face_normals[3 * t + 2]);
                }
            }
            if (tap_sum <= 0.0f)
                return;
            const float t = min(edge / kBorderBand, 1.0f);
            distance[i] = z_sum / tap_sum / face_ray.z;
            weight[i] = max(t * t * (3.0f - 2.0f * t), 1.0e-4f);
            const float n_len = length(n_sum);
            if (n_len > 0.0f) {
                const float3 n = to_camera(face, n_sum / n_len);
                for (int c = 0; c < 3; ++c)
                    normal[3 * i + c] = n[c];
            }
        }

        void blend_kernel(Ptr<float, Access.ReadWrite> distance, Ptr<float, Access.ReadWrite> normal,
                                     Ptr<float, Access.ReadWrite> weight, Ptr<float, Access.ReadWrite> scales,
                                     const int faces, const int pixels, Ptr<float, Access.ReadWrite> out_distance,
                                     Ptr<float, Access.ReadWrite> out_normal, uint tid) {
            const int p = int(tid);
            if (p >= pixels)
                return;
            float weight_sum = 0.0f;
            float distance_sum = 0.0f;
            float3 n = float3(0.0f);
            for (int k = 0; k < faces; ++k) {
                const uint j = uint(k) * pixels + p;
                const float d = distance[j];
                if (!(d > 0.0f) || !isfinite(d))
                    continue;
                const float w = weight[j];
                weight_sum += w;
                distance_sum += w * scales[k] * d;
                n += w * float3(normal[3 * j], normal[3 * j + 1], normal[3 * j + 2]);
            }
            const float n_len = length(n);
            out_distance[p] = weight_sum > 0.0f ? distance_sum / weight_sum : 0.0f;
            for (int c = 0; c < 3; ++c)
                out_normal[3 * uint(p) + c] = weight_sum > 0.0f && n_len > 0.0f ? n[c] / n_len : 0.0f;
        }

[shader("compute")][numthreads(256,1,1)]
void setup(uint3 id : SV_DispatchThreadID) {
    if(id.x != 0) return;
    GsplatSetupParams q;
    q.view=(uint64_t)p.view; q.radial=p.radial; q.tangential=p.tangent; q.prism=p.prism; q.camera=(uint64_t)p.camera;
    q.fx=p.fx; q.fy=p.fy; q.cx=p.cx; q.cy=p.cy;
    q.width=p.width; q.height=p.height; q.model=p.model;
    q.radial_count=6; q.tangential_count=2; q.prism_count=4;
    gsplat_camera_setup(q);
}
[shader("compute")][numthreads(256,1,1)]
void rays(uint3 id : SV_DispatchThreadID) { lens_rays_kernel(gsplat_camera_load(p.camera),p.width,p.height,p.a,id.x); }
[shader("compute")][numthreads(256,1,1)]
void sampleFace(uint3 id : SV_DispatchThreadID) { sample_face_kernel(gsplat_camera_load(p.camera),p.a,p.width,p.height,load_face(),p.b,id.x); }
[shader("compute")][numthreads(256,1,1)]
void projectFace(uint3 id : SV_DispatchThreadID) { project_face_kernel(gsplat_camera_load(p.camera),p.width,p.height,load_face(),p.a,p.b,p.c,p.d,p.e,p.f,id.x); }
[shader("compute")][numthreads(256,1,1)]
void blend(uint3 id : SV_DispatchThreadID) { blend_kernel(p.a,p.b,p.c,p.d,p.faces,p.pixels,p.e,p.f,id.x); }
