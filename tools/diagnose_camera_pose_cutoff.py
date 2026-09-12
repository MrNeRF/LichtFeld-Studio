#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Double-precision CPU diagnostic for the nine-splat camera gate fixture.

No native code, build, dependencies or CUDA. This is a small independent
mathematical reference, NOT a replacement for the native acceptance gate.
Reports finite differences with natural versus fixed contribution membership.
The latter isolates the smooth derivative used by rasterization backward.
Only the in-frustum, non-saturated SH3 fixture is supported.
"""

import math


def transpose(a):
    return list(map(list, zip(*a)))


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def exp_se3(twist):
    vx, vy, vz, x, y, z = twist
    theta2 = x*x + y*y + z*z
    omega = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    omega2 = multiply(omega, omega)
    if theta2 < 1e-8:
        a, b, c = 1-theta2/6, 0.5-theta2/24, 1/6-theta2/120
    else:
        theta = math.sqrt(theta2)
        a = math.sin(theta)/theta
        b = (1-math.cos(theta))/theta2
        c = (theta-math.sin(theta))/(theta2*theta)
    rotation = [[float(i == j)+a*omega[i][j]+b*omega2[i][j] for j in range(3)] for i in range(3)]
    jacobian = [[float(i == j)+b*omega[i][j]+c*omega2[i][j] for j in range(3)] for i in range(3)]
    translation = multiply(jacobian, [[vx], [vy], [vz]])
    return [rotation[i]+translation[i] for i in range(3)] + [[0, 0, 0, 1]]


def color(index, mean, center):
    direction = [m-c for m, c in zip(mean, center)]
    length = math.sqrt(sum(v*v for v in direction))
    x, y, z = [v/length for v in direction]
    xx, yy, zz = x*x, y*y, z*z
    bases = [-0.48860251190291987*y, 0.48860251190291987*z, -0.48860251190291987*x,
             1.0925484305920792*x*y, -1.0925484305920792*y*z,
             0.94617469575755997*zz-0.31539156525251999,
             -1.0925484305920792*x*z, 0.54627421529603959*(xx-yy),
             0.59004358992664352*y*(-3*xx+yy), 2.8906114426405538*x*y*z,
             0.45704579946446572*y*(1-5*zz), 0.3731763325901154*z*(5*zz-3),
             0.45704579946446572*x*(1-5*zz), 1.4453057213202769*z*(xx-yy),
             0.59004358992664352*x*(-xx+3*yy)]
    dc = [0.18+0.04*index, -0.22, 0.3]
    return [0.5+0.28209479177387814*dc[c]+sum(
        basis*0.16*math.sin(0.7*(3*k+c+1)+0.17*index)
        for k, basis in enumerate(bases)) for c in range(3)]


def project(pose):
    rotation = [row[:3] for row in pose[:3]]
    center = [-sum(pose[i][j]*pose[i][3] for i in range(3)) for j in range(3)]
    projected = []
    for index in range(9):
        mean = [(index % 3-1)*0.65, (index//3-1)*0.42, 2.8+0.31*(index % 4)]
        camera = [sum(pose[i][j]*mean[j] for j in range(3))+pose[i][3] for i in range(3)]
        x, y, depth = camera
        j = [[55/depth, 0, -55*x/(depth*depth)], [0, 55/depth, -55*y/(depth*depth)]]
        angle = 0.17*index
        cs, sn = math.cos(angle), math.sin(angle)
        q = [[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]]
        variance = [[0.14**2, 0, 0], [0, 0.055**2, 0], [0, 0, 0.09**2]]
        covariance = multiply(multiply(q, variance), transpose(q))
        jr = multiply(j, rotation)
        projected_cov = multiply(multiply(jr, covariance), transpose(jr))
        a, b, d = projected_cov[0][0]+0.3, projected_cov[0][1], projected_cov[1][1]+0.3
        determinant = a*d-b*b
        projected.append((depth, index, 55*x/depth+32, 55*y/depth+24,
                          d/determinant, -b/determinant, a/determinant, color(index, mean, center)))
    return sorted(projected)


def render_rgb(pose, fixed_membership=None):
    """Untiled RGB reference, also usable to investigate photometric solvers."""
    splats = project(pose)
    membership = set()
    image = []
    opacity = 1/(1+math.exp(-0.2))
    for y in range(48):
        for x in range(64):
            transmittance = 1.0
            pixel = [0.0, 0.0, 0.0]
            for _, index, mx, my, a, b, d, rgb in splats:
                dx, dy = mx-x-0.5, my-y-0.5
                alpha = opacity*math.exp(-0.5*(a*dx*dx+2*b*dx*dy+d*dy*dy))
                key = (index, y, x)
                if alpha >= 1/255:
                    membership.add(key)
                active = key in fixed_membership if fixed_membership is not None else alpha >= 1/255
                if active:
                    for c in range(3):
                        pixel[c] += transmittance*alpha*rgb[c]
                    transmittance *= 1-alpha
            image.extend(pixel)
    return image, membership


def objective(pose, fixed_membership=None):
    splats = project(pose)
    membership = set()
    value = 0.0
    opacity = 1/(1+math.exp(-0.2))
    for y in range(48):
        for x in range(64):
            transmittance = 1.0
            for _, index, mx, my, a, b, d, rgb in splats:
                dx, dy = mx-x-0.5, my-y-0.5
                alpha = opacity*math.exp(-0.5*(a*dx*dx+2*b*dx*dy+d*dy*dy))
                key = (index, y, x)
                if alpha >= 1/255:
                    membership.add(key)
                active = key in fixed_membership if fixed_membership is not None else alpha >= 1/255
                if not active:
                    continue
                weighted_color = sum(rgb[c]*(0.5+math.sin(0.13*x+0.21*y+0.3*c)) for c in range(3))/(3*64*48)
                value += transmittance*alpha*weighted_color
                transmittance *= 1-alpha
    return value, membership


def main():
    pose = exp_se3([0.06, -0.04, 0.08, 0.035, -0.025, 0.02])
    _, membership = objective(pose)
    print('CPU mathematical reference only; native CUDA validation is still required.')
    print('axis epsilon natural_derivative fixed_membership_derivative changed_pairs')
    for axis in range(6):
        for epsilon in (1e-3, 1e-4, 1e-5, 5e-6):
            results = []
            for sign in (1, -1):
                twist = [0.0]*6
                twist[axis] = sign*epsilon
                shifted = multiply(exp_se3(twist), pose)
                natural, current = objective(shifted)
                fixed, _ = objective(shifted, membership)
                results.append((natural, fixed, current))
            plus, minus = results
            print(axis, epsilon, f'{(plus[0]-minus[0])/(2*epsilon):.9g}',
                  f'{(plus[1]-minus[1])/(2*epsilon):.9g}',
                  len(plus[2] ^ membership)+len(minus[2] ^ membership))


if __name__ == '__main__':
    main()
