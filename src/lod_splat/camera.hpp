/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Pinhole camera + the projection math that drives LOD target density.
//
// A pixel at distance d covers a world-space footprint of
//     w(d) = d * 2*tan(fovy/2) / H = d / focal_px.
// "One splat per pixel" => target splat spacing at distance d is ~w(d),
// i.e. surface density ∝ 1/d². Splats alpha-blend, so good quality needs an
// overdraw factor q of ~2–6 contributions per pixel; that scales target
// density by q, i.e. target spacing by 1/sqrt(q). The refinement test is
//     projected_extent_px * sqrt(q) > tau   =>   refine,
// with tau = 1px the baseline and q the tunable quality knob.
#pragma once

#include "lod_math.hpp"
#include "splat.hpp"

#include <array>

namespace lfs::lod {

    struct Camera {
        Vec3 position{0, 0, 0};
        Mat3 rotation = Mat3::identity(); // camera-to-world; columns: right, up, -forward convention below
        float fovy_rad = 0.9f;            // vertical field of view
        int width = 1920;
        int height = 1080;
        float znear = 0.1f;
        float zfar = 1e4f;

        float focalPx() const { return 0.5f * (float)height / std::tan(0.5f * fovy_rad); }
        float aspect() const { return (float)width / (float)height; }

        // World-space footprint of one pixel at view depth d.
        float pixelFootprint(float d) const { return std::max(d, znear) / focalPx(); }

        // Camera looks down -Z in view space; forward = -rotation.col(2).
        Vec3 forward() const { return Vec3{-rotation.at(0, 2), -rotation.at(1, 2), -rotation.at(2, 2)}; }

        Vec3 worldToView(const Vec3& p) const {
            const Vec3 d = p - position;
            // view = R^T * (p - pos)  (rotation is camera-to-world)
            return {rotation.at(0, 0) * d.x + rotation.at(1, 0) * d.y + rotation.at(2, 0) * d.z,
                    rotation.at(0, 1) * d.x + rotation.at(1, 1) * d.y + rotation.at(2, 1) * d.z,
                    rotation.at(0, 2) * d.x + rotation.at(1, 2) * d.y + rotation.at(2, 2) * d.z};
        }

        static Camera lookAt(const Vec3& eye, const Vec3& target, const Vec3& up, float fovy, int w, int h) {
            Camera c;
            c.position = eye;
            c.fovy_rad = fovy;
            c.width = w;
            c.height = h;
            const Vec3 f = (target - eye).normalized();
            const Vec3 r = f.cross(up).normalized();
            const Vec3 u = r.cross(f);
            // columns: right, up, -forward (camera-to-world, view looks down -Z)
            c.rotation.at(0, 0) = r.x;
            c.rotation.at(1, 0) = r.y;
            c.rotation.at(2, 0) = r.z;
            c.rotation.at(0, 1) = u.x;
            c.rotation.at(1, 1) = u.y;
            c.rotation.at(2, 1) = u.z;
            c.rotation.at(0, 2) = -f.x;
            c.rotation.at(1, 2) = -f.y;
            c.rotation.at(2, 2) = -f.z;
            return c;
        }
    };

    // View frustum as 5 planes (near + 4 sides; far plane usually irrelevant for LOD).
    // Planes face inward: dot(n, p) + d >= 0 for points inside.
    struct Frustum {
        std::array<Vec3, 5> n;
        std::array<float, 5> d;

        static Frustum fromCamera(const Camera& cam) {
            Frustum fr;
            const Vec3 fwd = cam.forward();
            const Vec3 right{cam.rotation.at(0, 0), cam.rotation.at(1, 0), cam.rotation.at(2, 0)};
            const Vec3 up{cam.rotation.at(0, 1), cam.rotation.at(1, 1), cam.rotation.at(2, 1)};
            const float ty = std::tan(0.5f * cam.fovy_rad);
            const float tx = ty * cam.aspect();

            // Inward normals derived in view space (camera looks down -Z):
            // left (1,0,-tx), right (-1,0,-tx), bottom (0,1,-ty), top (0,-1,-ty),
            // mapped to world via the camera basis.
            fr.n[0] = fwd;                                    // near
            fr.n[1] = (right + fwd * tx).normalized();        // left
            fr.n[2] = (right * -1.f + fwd * tx).normalized(); // right
            fr.n[3] = (up + fwd * ty).normalized();           // bottom
            fr.n[4] = (up * -1.f + fwd * ty).normalized();    // top
            for (int i = 0; i < 5; ++i)
                fr.d[i] = -fr.n[i].dot(cam.position);
            fr.d[0] -= cam.znear;
            return fr;
        }

        // Conservative AABB test: false only if the box is fully outside a plane.
        bool intersects(const Aabb& b) const {
            for (int i = 0; i < 5; ++i) {
                // pick the box corner most along the normal
                const Vec3 p{n[i].x >= 0 ? b.mx.x : b.mn.x,
                             n[i].y >= 0 ? b.mx.y : b.mn.y,
                             n[i].z >= 0 ? b.mx.z : b.mn.z};
                if (n[i].dot(p) + d[i] < 0.f)
                    return false;
            }
            return true;
        }
    };

} // namespace lfs::lod
