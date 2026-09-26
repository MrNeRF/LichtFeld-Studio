// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 128;
[SpecializationConstant] const uint kPhase = 0;
struct Parameters {
    uint64_t input, output;
    float src_fx, src_fy, src_cx, src_cy, dst_fx, dst_fy, dst_cx, dst_cy;
    int sw, sh, dw, dh;
    float distortion[12];
    int model, num_distortion, channels, padding;
};
        // COLMAP sensor/models.h (BSD-3 licensed formulas)
        void apply_distortion_pinhole(
            const float x, const float y,
            float dist[12], const int num_dist,
            out float dx, out float dy) {

            const float r2 = x * x + y * y;
            const float r4 = r2 * r2;
            const float r6 = r4 * r2;

            const float k1 = num_dist > 0 ? dist[0] : 0.0f;
            const float k2 = num_dist > 1 ? dist[1] : 0.0f;
            const float k3 = num_dist > 2 ? dist[2] : 0.0f;
            const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

            const float p1 = num_dist > 3 ? dist[3] : 0.0f;
            const float p2 = num_dist > 4 ? dist[4] : 0.0f;

            dx = x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
            dy = y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;
        }

        void apply_distortion_fisheye(
            const float x, const float y,
            float dist[12], const int num_dist,
            out float dx, out float dy) {

            const float r = sqrt(x * x + y * y);
            if (r < 1e-8f) {
                dx = x;
                dy = y;
                return;
            }

            const float theta = atan(r);
            const float theta2 = theta * theta;
            const float theta4 = theta2 * theta2;
            const float theta6 = theta4 * theta2;
            const float theta8 = theta4 * theta4;

            const float k1 = num_dist > 0 ? dist[0] : 0.0f;
            const float k2 = num_dist > 1 ? dist[1] : 0.0f;
            const float k3 = num_dist > 2 ? dist[2] : 0.0f;
            const float k4 = num_dist > 3 ? dist[3] : 0.0f;

            const float theta_d = theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            const float scale = theta_d / r;

            dx = x * scale;
            dy = y * scale;
        }

        void apply_distortion_thin_prism_fisheye(
            const float x, const float y,
            float dist[12], const int num_dist,
            out float dx, out float dy) {

            const float r = sqrt(x * x + y * y);
            if (r < 1e-8f) {
                dx = x;
                dy = y;
                return;
            }

            const float theta = atan(r);
            const float theta2 = theta * theta;
            const float theta4 = theta2 * theta2;
            const float theta6 = theta4 * theta2;
            const float theta8 = theta4 * theta4;

            const float k1 = num_dist > 0 ? dist[0] : 0.0f;
            const float k2 = num_dist > 1 ? dist[1] : 0.0f;
            const float k3 = num_dist > 2 ? dist[2] : 0.0f;
            const float k4 = num_dist > 3 ? dist[3] : 0.0f;

            const float theta_d = theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            const float scale = theta_d / r;

            float xd = x * scale;
            float yd = y * scale;

            const float p1 = num_dist > 4 ? dist[4] : 0.0f;
            const float p2 = num_dist > 5 ? dist[5] : 0.0f;
            const float r2 = xd * xd + yd * yd;
            xd += 2.0f * p1 * xd * yd + p2 * (r2 + 2.0f * xd * xd);
            yd += p1 * (r2 + 2.0f * yd * yd) + 2.0f * p2 * xd * yd;

            const float s1 = num_dist > 6 ? dist[6] : 0.0f;
            const float s2 = num_dist > 7 ? dist[7] : 0.0f;
            const float s3 = num_dist > 8 ? dist[8] : 0.0f;
            const float s4 = num_dist > 9 ? dist[9] : 0.0f;
            const float r2d = xd * xd + yd * yd;
            const float r4d = r2d * r2d;
            xd += s1 * r2d + s2 * r4d;
            yd += s3 * r2d + s4 * r4d;

            dx = xd;
            dy = yd;
        }

        void apply_distortion(
            const float x, const float y,
            int model,
            float dist[12], const int num_dist,
            out float dx, out float dy) {

            switch (model) {
            case 0:
                apply_distortion_pinhole(x, y, dist, num_dist, dx, dy);
                break;
            case 2:
                apply_distortion_fisheye(x, y, dist, num_dist, dx, dy);
                break;
            case 4:
                apply_distortion_thin_prism_fisheye(x, y, dist, num_dist, dx, dy);
                break;
            default:
                dx = x;
                dy = y;
                break;
            }
        }


float tap(Ptr<float, Access.Read> src, int base, int sw, int sh, int x, int y) {
    return x>=0 && y>=0 && x<sw && y<sh ? src[base+y*sw+x] : 0.0f;
}
bool valid(Ptr<float, Access.Read> src, int i, int plane) {
    if (!isfinite(src[i])) return false;
    if (kPhase==1) return src[i]>0;
    const float a=src[i], b=src[plane+i], c=src[2*plane+i];
    return isfinite(b) && isfinite(c) && a*a+b*b+c*c>=0.25f;
}
[numthreads(LFS_LOCAL_SIZE_X,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = groups.x*LFS_LOCAL_SIZE_X;
    Ptr<float, Access.Read> src = (Ptr<float, Access.Read>)p.input;
    Ptr<float, Access.ReadWrite> dst = (Ptr<float, Access.ReadWrite>)p.output;
    const int sp=p.sw*p.sh, dp=p.dw*p.dh;
    for (uint idx=tid.x; idx<uint(dp); idx+=stride) {
        const int x=int(idx)%p.dw, y=int(idx)/p.dw;
        if (kPhase==0) {
            float dx,dy;
            apply_distortion((float(x)+0.5f-p.dst_cx)/p.dst_fx,(float(y)+0.5f-p.dst_cy)/p.dst_fy,
                p.model,p.distortion,p.num_distortion,dx,dy);
            const float sx=dx*p.src_fx+p.src_cx-0.5f, sy=dy*p.src_fy+p.src_cy-0.5f;
            const int x0=int(floor(sx)), y0=int(floor(sy));
            const float fx=sx-floor(sx), fy=sy-floor(sy);
            for(int c=0;c<p.channels;++c) {
                const float v00=tap(src,c*sp,p.sw,p.sh,x0,y0),v01=tap(src,c*sp,p.sw,p.sh,x0+1,y0);
                const float v10=tap(src,c*sp,p.sw,p.sh,x0,y0+1),v11=tap(src,c*sp,p.sw,p.sh,x0+1,y0+1);
                dst[c*dp+idx]=(1.0f-fy)*((1.0f-fx)*v00+fx*v01)+fy*((1.0f-fx)*v10+fx*v11);
            }
        } else {
            const float sx=max(0.0f,min(p.sw-1.0f,(float(x)+0.5f)*p.sw/p.dw-0.5f));
            const float sy=max(0.0f,min(p.sh-1.0f,(float(y)+0.5f)*p.sh/p.dh-0.5f));
            const int x0=int(sx),y0=int(sy);
            float3 value=0;
            float weight=0;
            if(valid(src,int(sy+0.5f)*p.sw+int(sx+0.5f),sp)) {
                for(int j=0;j<2;++j) for(int i=0;i<2;++i) {
                    const int index=min(y0+j,p.sh-1)*p.sw+min(x0+i,p.sw-1);
                    if(!valid(src,index,sp)) continue;
                    const float w=(i!=0?sx-x0:1.0f-(sx-x0))*(j!=0?sy-y0:1.0f-(sy-y0));
                    weight+=w;
                    for(int c=0;c<p.channels;++c) value[c]+=w*src[c*sp+index];
                }
            }
            if(kPhase==2) weight=sqrt(value.x*value.x+value.y*value.y+value.z*value.z);
            for(int c=0;c<p.channels;++c) dst[c*dp+idx]=weight>1e-8f?value[c]/weight:0.0f;
        }
    }
}
