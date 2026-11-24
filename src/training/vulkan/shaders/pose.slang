// SPDX-License-Identifier: GPL-3.0-or-later
struct Parameters {
    uint64_t start, rotations, offsets, m, v, steps, poses, centre, grad_pose, grad_centre;
    int row, draw, first_draw, draw_count;
    float lr_rotation, lr_translation;
};
struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
float readAt(uint64_t p, int i) { return ((Ptr<float, Access.Read>)p)[i]; }
void writeAt(uint64_t p, int i, float v) { ((Ptr<float, Access.ReadWrite>)p)[i] = v; }
// Rotate the top three rows of a homogeneous pose or its gradient. The stored matrix maps draw to owner.
void rotate_rows(bool transpose, float rotation[9], float input[16], out float result[16]) {
    bool identity = true;
    for (int i = 0; i < 9; ++i)
        identity = identity && rotation[i] == (i % 4 == 0 ? 1.0f : 0.0f);
    for (int i = 0; i < 16; ++i)
        result[i] = input[i];
    if (identity)
        return;
    for (int i = 0; i < 3; ++i) {
        const int first = transpose ? i : 3 * i;
        const int stride = transpose ? 3 : 1;
        for (int j = 0; j < 4; ++j)
            result[4 * i + j] = rotation[first] * input[j] + rotation[first + stride] * input[4 + j] +
                                rotation[first + 2 * stride] * input[8 + j];
    }
}

// exp([phi]x), row-major; exactly the identity at phi = 0.
void exp_so3(float phi[3], out float E[9]) {
    const float th2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    float A, B;
    if (th2 < float(1e-8)) {
        A = float(1) - th2 / float(6);
        B = float(0.5) - th2 / float(24);
    } else {
        const float th = sqrt(th2);
        A = sin(th) / th;
        B = (float(1) - cos(th)) / th2;
    }
    const float x = phi[0], y = phi[1], z = phi[2];
    E[0] = float(1) + B * (x * x - th2);
    E[1] = B * x * y - A * z;
    E[2] = B * x * z + A * y;
    E[3] = B * x * y + A * z;
    E[4] = float(1) + B * (y * y - th2);
    E[5] = B * y * z - A * x;
    E[6] = B * x * z - A * y;
    E[7] = B * y * z + A * x;
    E[8] = float(1) + B * (z * z - th2);
}

void compose(float w2c0[16], float off[6], out float w2c[16], out float c[3]) {
    const float phi[3] = {-off[0], -off[1], -off[2]};
    float E[9];
    exp_so3(phi, E);
    const float u[3] = {w2c0[3] - off[3], w2c0[7] - off[4], w2c0[11] - off[5]};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            w2c[i * 4 + j] = E[i * 3] * w2c0[j] + E[i * 3 + 1] * w2c0[4 + j] + E[i * 3 + 2] * w2c0[8 + j];
        w2c[i * 4 + 3] = E[i * 3] * u[0] + E[i * 3 + 1] * u[1] + E[i * 3 + 2] * u[2];
    }
    for (int j = 0; j < 4; ++j)
        w2c[12 + j] = w2c0[12 + j];
    for (int j = 0; j < 3; ++j)
        c[j] = -(w2c[j] * w2c[3] + w2c[4 + j] * w2c[7] + w2c[8 + j] * w2c[11]);
}

// dL/d(dr, dd) of the left perturbation W(δ) = exp(-[dr]x) W, b(δ) = exp(-[dr]x) (b - dd), c(δ) = c + W^T dd
// from g_w2c (rows 0-2 of dL/d(W | b)) and g_c = dL/dc.
void perturbation_grad(float w2c[16], float g_w2c[16], float g_c[3], out float g[6]) {
    // N_ij = W_i . (dL/dW)_j
    float N[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            N[i][j] = w2c[i * 4] * g_w2c[j * 4] + w2c[i * 4 + 1] * g_w2c[j * 4 + 1] + w2c[i * 4 + 2] * g_w2c[j * 4 + 2];
    const float b[3] = {w2c[3], w2c[7], w2c[11]};
    const float gb[3] = {g_w2c[3], g_w2c[7], g_w2c[11]};
    g[0] = N[2][1] - N[1][2] + gb[1] * b[2] - gb[2] * b[1];
    g[1] = N[0][2] - N[2][0] + gb[2] * b[0] - gb[0] * b[2];
    g[2] = N[1][0] - N[0][1] + gb[0] * b[1] - gb[1] * b[0];
    for (int i = 0; i < 3; ++i)
        g[3 + i] = -gb[i] + w2c[i * 4] * g_c[0] + w2c[i * 4 + 1] * g_c[1] + w2c[i * 4 + 2] * g_c[2];
}

// dL/d(dr, dd) of the left perturbation at the offset -> dL/d(r, d):
// dL/dr = J_l(-r)^T dL/ddr (exact SO(3) left Jacobian), dL/dd = E^T dL/ddd.
void chain(float off[6], float g[6], out float result[6]) {
    const float phi[3] = {-off[0], -off[1], -off[2]};
    const float th2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    float a, b;
    if (th2 < float(1e-6)) {
        a = float(0.5) - th2 / float(24);
        b = float(1) / float(6) - th2 / float(120);
    } else {
        const float th = sqrt(th2);
        a = (float(1) - cos(th)) / th2;
        b = (th - sin(th)) / (th2 * th);
    }
    // J_l(phi)^T g = g - a (phi x g) + b (phi x (phi x g))
    const float pg[3] = {phi[1] * g[2] - phi[2] * g[1], phi[2] * g[0] - phi[0] * g[2], phi[0] * g[1] - phi[1] * g[0]};
    const float ppg[3] = {phi[1] * pg[2] - phi[2] * pg[1], phi[2] * pg[0] - phi[0] * pg[2], phi[0] * pg[1] - phi[1] * pg[0]};
    for (int k = 0; k < 3; ++k)
        result[k] = g[k] - a * pg[k] + b * ppg[k];
    float E[9];
    exp_so3(phi, E);
    for (int k = 0; k < 3; ++k)
        result[3 + k] = E[k] * g[3] + E[3 + k] * g[4] + E[6 + k] * g[5];
}

// dL/d(r, d) at the offset from the rasterizer's gradient of the pose it drew, compose(w2c0, off).
void offset_grad(float w2c0[16], float off[6], float g_w2c[16], float g_c[3], out float result[6]) {
    float w2c[16], c[3], g[6];
    compose(w2c0, off, w2c, c);
    perturbation_grad(w2c, g_w2c, g_c, g);
    chain(off, g, result);
}

[numthreads(1, 1, 1)]
void main() {
    let p = ((Ptr<Parameters, Access.Read>)push.parameters)[0];
    float start[16], off[6], rot[9], matrix[16], centre_grad[3], owner_grad[16], gradient[6];
    for (int i = 0; i < 16; i++) {
        start[i] = readAt(p.start, 16 * p.row + i);
        matrix[i] = readAt(p.grad_pose, i);
    }
    for (int i = 0; i < 6; i++)
        off[i] = readAt(p.offsets, 6 * p.row + i);
    for (int i = 0; i < 9; i++)
        rot[i] = readAt(p.rotations, 9 * p.draw + i);
    for (int i = 0; i < 3; i++)
        centre_grad[i] = readAt(p.grad_centre, i);
    rotate_rows(false, rot, matrix, owner_grad);
    offset_grad(start, off, owner_grad, centre_grad, gradient);
    bool finite = true;
    for (int i = 0; i < 6; i++)
        finite = finite && isfinite(gradient[i]);
    if (finite) {
        float t = readAt(p.steps, p.row) + 1;
        float b1 = 1 - pow(.9f, t), b2 = 1 - pow(.999f, t);
        for (int k = 0; k < 6; k++) {
            int i = 6 * p.row + k;
            float m = .9f * readAt(p.m, i) + (1.f - .9f) * gradient[k];
            float v = .999f * readAt(p.v, i) + (1.f - .999f) * gradient[k] * gradient[k];
            off[k] -= (k < 3 ? p.lr_rotation : p.lr_translation) * (m / b1) / (sqrt(v / b2) + 1e-8f);
            writeAt(p.m, i, m);
            writeAt(p.v, i, v);
            writeAt(p.offsets, i, off[k]);
        }
        writeAt(p.steps, p.row, t);
    }
    float pose[16], centre[3];
    compose(start, off, pose, centre);
    for (int i = 0; i < 3; i++)
        writeAt(p.centre, i, centre[i]);
    for (int draw = 0; draw < p.draw_count; draw++) {
        for (int i = 0; i < 9; i++)
            rot[i] = readAt(p.rotations, 9 * (p.first_draw + draw) + i);
        float result[16];
        rotate_rows(true, rot, pose, result);
        for (int i = 0; i < 16; i++)
            writeAt(p.poses, 16 * draw + i, result[i]);
    }
}
