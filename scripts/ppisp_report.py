#!/usr/bin/env python3
"""
Generate PDF reports from LichtFeld Studio .ppisp binary files.

Reads the binary tensor format directly and produces per-camera PDF reports
showing exposure, vignetting, color correction, and CRF parameters.

Based on the NVIDIA PPISP reference report.py (Apache-2.0).

Usage:
    python3 ppisp_report.py output/splat_30000.ppisp
    python3 ppisp_report.py output/splat_30000.ppisp --output-dir reports/
"""

from __future__ import annotations

import argparse
import struct
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# -- Binary format constants --------------------------------------------------

PPISP_FILE_MAGIC = 0x50505349      # "PPIS"
PPISP_INFERENCE_MAGIC = 0x4C465049  # "LFPI"
TENSOR_MAGIC = 0x4C465354           # "LFST"

DTYPE_MAP = {
    0: (np.float32, 4),   # Float32
    1: (np.float16, 2),   # Float16
    2: (np.int32, 4),     # Int32
    3: (np.int64, 8),     # Int64
    4: (np.uint8, 1),     # UInt8
    5: (np.bool_, 1),     # Bool
}


# -- Binary reader ------------------------------------------------------------

def _read_tensor(f) -> torch.Tensor:
    # C++ struct has 4 bytes padding before uint64_t numel (alignment)
    raw = f.read(24)
    magic, version, dtype_id, _device, rank, numel = struct.unpack("<IIBBHxxxxQ", raw)
    assert magic == TENSOR_MAGIC, f"Bad tensor magic: 0x{magic:08X}"
    assert version == 1, f"Unsupported tensor version: {version}"

    shape = []
    for _ in range(rank):
        shape.append(struct.unpack("<Q", f.read(8))[0])

    np_dtype, elem_size = DTYPE_MAP[dtype_id]
    data = np.frombuffer(f.read(numel * elem_size), dtype=np_dtype).copy()

    if shape:
        data = data.reshape(shape)

    return torch.from_numpy(data)


def load_ppisp(path: Path) -> dict:
    with open(path, "rb") as f:
        magic, version, num_cameras, num_frames, flags = struct.unpack("<IIIII", f.read(20))
        _reserved = f.read(12)  # 3x uint32

        assert magic == PPISP_FILE_MAGIC, f"Not a .ppisp file (magic=0x{magic:08X})"
        assert version == 1, f"Unsupported version: {version}"

        inf_magic, inf_version = struct.unpack("<II", f.read(8))
        assert inf_magic == PPISP_INFERENCE_MAGIC
        assert inf_version == 1

        inf_num_cameras, inf_num_frames = struct.unpack("<ii", f.read(8))

        exposure_params = _read_tensor(f)
        vignetting_params = _read_tensor(f)
        color_params = _read_tensor(f)
        crf_params = _read_tensor(f)

    return {
        "num_cameras": inf_num_cameras,
        "num_frames": inf_num_frames,
        "exposure_params": exposure_params,
        "vignetting_params": vignetting_params.reshape(inf_num_cameras, 3, -1),
        "color_params": color_params.reshape(inf_num_frames, 8),
        "crf_params": crf_params.reshape(inf_num_cameras, 3, 4),
    }


# -- Plotting helpers (ported from NVIDIA PPISP report.py) --------------------

def _srgb_inverse_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    out = np.empty_like(x, dtype=np.float32)
    mask = x <= 0.04045
    out[mask] = x[mask] / 12.92
    inv = x[~mask]
    out[~mask] = np.power((inv + 0.055) / 1.055, 2.4, dtype=np.float32)
    return out


def _gray_bars(size: int = 256, num: int = 16) -> np.ndarray:
    w = size // num
    img = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(num):
        v = i / (num - 1)
        img[:, i * w: size if i == num - 1 else (i + 1) * w] = v
    return img


def _show_image(ax, img: np.ndarray, title: str):
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.axis("off")
    ax.set_title(title)


# -- Exposure -----------------------------------------------------------------

def _plot_exposure(fig, gs, exposure_params, frames_per_camera, cam):
    sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = fig.add_subplot(sub[0])
    ax_img = fig.add_subplot(sub[1])

    start = int(sum(frames_per_camera[:cam]))
    end = start + int(frames_per_camera[cam])
    vals = exposure_params[start:end].detach().float().cpu()
    mean_val = vals.mean().item() if vals.numel() else 0.0

    ax_plot.plot(np.arange(vals.numel()), vals.numpy(), "b-")
    ax_plot.axhline(mean_val, color="b", linestyle="--", alpha=0.5)
    ax_plot.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax_plot.set_xlabel("Frame Index")
    ax_plot.set_ylabel("Exposure Offset [EV]")
    ax_plot.set_title("Exposure Offset Over Time")
    ax_plot.grid(True, alpha=0.3)

    img = _gray_bars(256)
    scale = 2.0 ** float(mean_val)
    img[img.shape[0] // 2:, :] *= scale
    _show_image(ax_img, img, "Mean Exposure Visualization")
    size = img.shape[0]
    ax_img.text(size * 0.5, 20, "Original", ha="center", va="top", color="white",
                bbox=dict(facecolor="black", alpha=0.5))
    ax_img.text(size * 0.5, size - 20, f"{mean_val:+.2f} EV", ha="center", va="bottom",
                color="white", bbox=dict(facecolor="black", alpha=0.5))


# -- Vignetting ---------------------------------------------------------------

def _vig_weight_forward(r2, alphas):
    falloff = torch.ones_like(r2)
    r2_pow = r2
    for i in range(int(alphas.shape[-1])):
        falloff = falloff + alphas[..., i] * r2_pow
        r2_pow = r2_pow * r2
    return torch.clamp(falloff, 0.0, 1.0)


def _plot_vignetting(fig, gs, vig_params, cam):
    sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = fig.add_subplot(sub[0])
    ax_img = fig.add_subplot(sub[1])

    device = vig_params.device
    r = torch.linspace(0, np.sqrt(2) / 2.0, 200, device=device)
    r2 = r * r

    colors = [(1, 0, 0, 0.6), (0, 1, 0, 0.6), (0, 0, 1, 0.6)]
    for ch in range(3):
        alphas = vig_params[cam, ch, 2:]
        w = _vig_weight_forward(r2, alphas)
        ax_plot.plot(r.cpu().numpy(), w.cpu().numpy(),
                     color=colors[ch], linewidth=2.0, label=["Red", "Green", "Blue"][ch])
    ax_plot.set_title("Vignetting Curves (R,G,B)")
    ax_plot.set_xlabel("Radial Distance")
    ax_plot.set_ylabel("Light Transmission")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend()
    ax_plot.set_ylim(bottom=0)

    size = 256
    coords = torch.linspace(-0.5, 0.5, size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    img = torch.full((size, size, 3), 0.75, device=device)

    for ch in range(3):
        oc = vig_params[cam, ch, :2]
        alphas = vig_params[cam, ch, 2:]
        dx = xx - oc[0]
        dy = yy - oc[1]
        r2m = dx * dx + dy * dy
        w = _vig_weight_forward(r2m, alphas)
        img[..., ch] = img[..., ch] * w

    img_np = img.cpu().numpy()
    ax_img.imshow(np.clip(img_np, 0.0, 1.0))
    ax_img.axis("off")
    ax_img.set_title("Vignetting Effect Visualization")
    center = size * 0.5
    ax_img.axhline(y=center, color="gray", linestyle="--", alpha=0.5)
    ax_img.axvline(x=center, color="gray", linestyle="--", alpha=0.5)
    cross_size = 10
    cross_width = 2
    for ch, color in enumerate(colors):
        oc = vig_params[cam, ch, :2].cpu().numpy()
        cx = (float(oc[0]) + 0.5) * size
        cy = (float(oc[1]) + 0.5) * size
        ax_img.plot([cx - cross_size, cx + cross_size], [cy, cy],
                    color=color, linewidth=cross_width)
        ax_img.plot([cx, cx], [cy - cross_size, cy + cross_size],
                    color=color, linewidth=cross_width)


# -- Color Correction ---------------------------------------------------------

_PINV_BLOCKS = torch.tensor([
    [[0.0480542, -0.0043631], [-0.0043631, 0.0481283]],
    [[0.0580570, -0.0179872], [-0.0179872, 0.0431061]],
    [[0.0433336, -0.0180537], [-0.0180537, 0.0580500]],
    [[0.0128369, -0.0034654], [-0.0034654, 0.0128158]],
])


def _color_offsets_from_params(p):
    device = p.device
    dtype = p.dtype
    pinv = _PINV_BLOCKS.to(device=device, dtype=dtype)

    def _mul2(a, m):
        return torch.stack([
            a[..., 0] * m[0, 0] + a[..., 1] * m[0, 1],
            a[..., 0] * m[1, 0] + a[..., 1] * m[1, 1],
        ], dim=-1)

    xb, xr, xg, xn = p[..., 0:2], p[..., 2:4], p[..., 4:6], p[..., 6:8]
    yb = _mul2(xb, pinv[0])
    yr = _mul2(xr, pinv[1])
    yg = _mul2(xg, pinv[2])
    yn = _mul2(xn, pinv[3])
    return torch.stack([yb, yr, yg, yn], dim=1)


def _source_chroms(device):
    return torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.33, 0.33]], device=device)


def _homography_from_params(p):
    device = p.device
    dtype = p.dtype

    offsets = _color_offsets_from_params(p)
    bd = offsets[..., 0, :]
    rd = offsets[..., 1, :]
    gd = offsets[..., 2, :]
    nd = offsets[..., 3, :]

    s_b = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    s_r = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=dtype)
    s_g = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)
    s_gray = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0], device=device, dtype=dtype)

    t_b = torch.stack([s_b[0] + bd[..., 0], s_b[1] + bd[..., 1],
                        torch.ones_like(bd[..., 0])], dim=-1)
    t_r = torch.stack([s_r[0] + rd[..., 0], s_r[1] + rd[..., 1],
                        torch.ones_like(rd[..., 0])], dim=-1)
    t_g = torch.stack([s_g[0] + gd[..., 0], s_g[1] + gd[..., 1],
                        torch.ones_like(gd[..., 0])], dim=-1)
    t_gray = torch.stack([s_gray[0] + nd[..., 0], s_gray[1] + nd[..., 1],
                           torch.ones_like(nd[..., 0])], dim=-1)

    T = torch.stack([t_b, t_r, t_g], dim=-1)
    zero = torch.zeros_like(bd[..., 0])
    skew = torch.stack([
        torch.stack([zero, -t_gray[..., 2], t_gray[..., 1]], dim=-1),
        torch.stack([t_gray[..., 2], zero, -t_gray[..., 0]], dim=-1),
        torch.stack([-t_gray[..., 1], t_gray[..., 0], zero], dim=-1),
    ], dim=-2)

    M = torch.matmul(skew, T)
    r0, r1, r2 = M[..., 0, :], M[..., 1, :], M[..., 2, :]
    lam = torch.cross(r0, r1, dim=-1)
    n2 = (lam * lam).sum(dim=-1)
    mask = n2 < 1.0e-20
    lam = torch.where(mask.unsqueeze(-1), torch.cross(r0, r2, dim=-1), lam)
    n2 = (lam * lam).sum(dim=-1)
    mask = n2 < 1.0e-20
    lam = torch.where(mask.unsqueeze(-1), torch.cross(r1, r2, dim=-1), lam)

    S_inv = torch.tensor([[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], device=device, dtype=dtype)
    D = torch.zeros(*p.shape[:-1], 3, 3, device=device, dtype=dtype)
    D[..., 0, 0] = lam[..., 0]
    D[..., 1, 1] = lam[..., 1]
    D[..., 2, 2] = lam[..., 2]
    H = torch.matmul(T, torch.matmul(D, S_inv))
    s = H[..., 2:3, 2:3]
    denom = s + (s.abs() <= 1.0e-20).to(dtype)
    H = H / denom
    return H


def _apply_h_rg_loss(h, rg):
    r, g = rg[..., 0], rg[..., 1]
    ones = torch.ones_like(r)
    v = torch.stack([r, g, ones], dim=-1)
    vv = torch.matmul(h, v.unsqueeze(-1)).squeeze(-1)
    denom = vv[..., 2] + 1.0e-5
    return torch.stack([vv[..., 0] / denom, vv[..., 1] / denom], dim=-1)


def _apply_h_rgb_forward(h, rgb):
    intensity = rgb[..., 0] + rgb[..., 1] + rgb[..., 2]
    rgi = torch.stack([rgb[..., 0], rgb[..., 1], intensity], dim=-1)
    rgi_m = torch.matmul(h, rgi.unsqueeze(-1)).squeeze(-1)
    scale = intensity / (rgi_m[..., 2] + 1.0e-5)
    rgi_m = rgi_m * scale.unsqueeze(-1)
    r, g = rgi_m[..., 0], rgi_m[..., 1]
    b = rgi_m[..., 2] - r - g
    return torch.stack([r, g, b], dim=-1)


def _dlt_homography(src_rg, dst_rg):
    A = []
    for (x, y), (u, v) in zip(src_rg, dst_rg):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)
    _, _, vh = np.linalg.svd(A)
    h = vh[-1]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) < 1e-8:
        H = H / (np.sign(H[2, 2]) + 1e-8)
    else:
        H = H / H[2, 2]
    return H


def _chrom_triangle_size(size):
    height = int(size * np.sqrt(3.0) / 2.0)
    return size, height


def _chrom_barycentric_to_window(r, g, size):
    width, height = _chrom_triangle_size(size)
    top = (width * 0.5, 0.0)
    bl = (0.0, float(height))
    br = (float(width), float(height))
    b = 1.0 - r - g
    x = r * bl[0] + g * br[0] + b * top[0]
    y = r * bl[1] + g * br[1] + b * top[1]
    return x, y


@lru_cache(maxsize=4)
def _create_chromaticity_triangle(size):
    width, height = _chrom_triangle_size(size)
    img = np.ones((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            top = np.array([width * 0.5, 0.0])
            bl = np.array([0.0, height])
            br = np.array([width, height])
            v0 = bl - top
            v1 = br - top
            v2 = np.array([x, y], dtype=np.float32) - top
            d00 = (v0 * v0).sum()
            d01 = (v0 * v1).sum()
            d11 = (v1 * v1).sum()
            d20 = (v2 * v0).sum()
            d21 = (v2 * v1).sum()
            denom = d00 * d11 - d01 * d01
            if denom == 0:
                continue
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            if (u >= 0.0) and (v >= 0.0) and (w >= 0.0):
                r_val, g_val = v, w
                b_val = max(0.0, 1.0 - r_val - g_val)
                rgb = np.array([r_val, g_val, b_val], dtype=np.float32)
                m = rgb.max()
                if m > 0:
                    rgb = rgb / m
                img[y, x] = rgb * 0.85 + 0.15
    return img


def _plot_color(fig, gs_top, gs_bot, color_params, frames_per_camera, cam):
    start = int(sum(frames_per_camera[:cam]))
    n = int(frames_per_camera[cam])
    p = color_params[start: start + n]
    H = _homography_from_params(p)
    src = _source_chroms(color_params.device)
    tgt_list = []
    for i in range(4):
        rg_in = src[i].unsqueeze(0).expand(n, -1)
        tgt_i = _apply_h_rg_loss(H, rg_in)
        tgt_list.append(tgt_i)
    tgt = torch.stack(tgt_list, dim=1)
    shifts = (tgt - src)

    names = ["Blue", "Red", "Green", "Neutral"]
    cols = ["blue", "red", "green", "gray"]

    sub_top = gs_top.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    sub_bot = gs_bot.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_rc = fig.add_subplot(sub_top[0])
    ax_rgplot = fig.add_subplot(sub_top[1])
    ax_gm = fig.add_subplot(sub_bot[0])
    ax_img = fig.add_subplot(sub_bot[1])

    x = np.arange(n)
    for i in range(4):
        ax_rc.plot(x, shifts[:, i, 0].cpu().numpy(),
                   color=cols[i], label=names[i], alpha=0.8)
        ax_gm.plot(x, shifts[:, i, 1].cpu().numpy(), color=cols[i], alpha=0.8)
    ax_rc.set_title("Red-Cyan Shift Over Time")
    ax_rc.set_xlabel("Frame Index")
    ax_rc.set_ylabel("Red-Cyan Shift")
    ax_rc.grid(True, alpha=0.3)
    ax_rc.legend()

    ax_gm.set_title("Green-Magenta Shift Over Time")
    ax_gm.set_xlabel("Frame Index")
    ax_gm.set_ylabel("Green-Magenta Shift")
    ax_gm.grid(True, alpha=0.3)

    size = 256
    scale = 5.0
    tri_img = _create_chromaticity_triangle(size)
    ax_rgplot.imshow(tri_img)
    ax_rgplot.axis("off")
    ax_rgplot.set_title(f"Chromaticity Shifts Over Time, Scaled {scale:.1f}x")

    chroms_scaled = (src + shifts * scale).cpu().numpy()
    cross_size = 7
    cross_width = 2
    for i in range(4):
        pts = chroms_scaled[:, i, :]
        traj = np.array([_chrom_barycentric_to_window(
            float(r_val), float(g_val), size) for r_val, g_val in pts])
        ax_rgplot.plot(traj[:, 0], traj[:, 1], "-",
                       color="black", linewidth=1.0, alpha=0.7)
        fx, fy = traj[-1]
        ax_rgplot.plot([fx - cross_size, fx + cross_size],
                       [fy, fy], "-", color="black", linewidth=cross_width)
        ax_rgplot.plot([fx, fx], [fy - cross_size, fy + cross_size],
                       "-", color="black", linewidth=cross_width)
        sx, sy = _chrom_barycentric_to_window(
            float(src[i, 0].item()), float(src[i, 1].item()), size)
        ax_rgplot.plot(
            [sx - cross_size * 0.75, sx + cross_size * 0.75], [sy, sy],
            "-", color="black", linewidth=cross_width / 2, alpha=0.5)
        ax_rgplot.plot(
            [sx, sx], [sy - cross_size * 0.75, sy + cross_size * 0.75],
            "-", color="black", linewidth=cross_width / 2, alpha=0.5)

    mean_targets = tgt.mean(dim=0).cpu().numpy()
    src_np = src.cpu().numpy()
    H_np = _dlt_homography(src_np, mean_targets)
    H_mean = torch.from_numpy(H_np).to(color_params.device, dtype=color_params.dtype)

    size = 256
    bars = np.zeros((size, size, 3), dtype=np.float32)
    w = size // 4
    bars[:, 0:w] = [0, 0, 1]
    bars[:, w: 2 * w] = [1, 0, 0]
    bars[:, 2 * w: 3 * w] = [0, 1, 0]
    bars[:, 3 * w:] = [0.5, 0.5, 0.5]
    bottom = torch.from_numpy(bars[size // 2:].reshape(-1, 3)).to(color_params.device)
    corrected = _apply_h_rgb_forward(H_mean, bottom)
    vis = bars.copy()
    vis[size // 2:] = corrected.reshape(size // 2, size, 3).clamp(0, 1).cpu().numpy()
    _show_image(ax_img, vis, "Mean Color Correction Visualization")
    ax_img.text(size * 0.5, 20, "Original", ha="center", va="top", color="white",
                bbox=dict(facecolor="black", alpha=0.5))
    ax_img.text(size * 0.5, size - 20, f"Color Corrected, Scaled {scale:.1f}x",
                ha="center", va="bottom", color="white",
                bbox=dict(facecolor="black", alpha=0.5))


# -- CRF ----------------------------------------------------------------------

def _softplus_with_min(x, min_value):
    return torch.tensor(min_value, device=x.device, dtype=x.dtype) + torch.log1p(torch.exp(x))


def _crf_effective_from_raw(raw):
    toe = _softplus_with_min(raw[..., 0], 0.3)
    shoulder = _softplus_with_min(raw[..., 1], 0.3)
    gamma = _softplus_with_min(raw[..., 2], 0.1)
    center = torch.sigmoid(raw[..., 3])
    return toe, shoulder, gamma, center


def _apply_crf(toe, shoulder, gamma, center, x):
    x = torch.clamp(x, 0.0, 1.0)
    a = (shoulder * center) / torch.clamp(torch.lerp(toe, shoulder, center), min=1.0e-12)
    b = 1.0 - a
    left = a * torch.pow(torch.clamp(x / torch.clamp(center, min=1.0e-12), 0.0, 1.0), toe)
    right = 1.0 - b * torch.pow(
        torch.clamp((1.0 - x) / torch.clamp(1.0 - center, min=1.0e-12), 0.0, 1.0), shoulder)
    y0 = torch.where(x <= center, left, right)
    y = torch.pow(torch.clamp(y0, 0.0, 1.0), gamma)
    return torch.clamp(y, 0.0, 1.0)


def _plot_crf(fig, gs, crf_params, cam):
    sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = fig.add_subplot(sub[0])
    ax_img = fig.add_subplot(sub[1])

    cols = [(1, 0, 0, 0.6), (0, 1, 0, 0.6), (0, 0, 1, 0.6)]
    x = torch.linspace(0.0, 1.0, 256, device=crf_params.device)
    crf_cam = crf_params[cam]
    for ch in range(3):
        toe, shoulder, gamma, center = _crf_effective_from_raw(crf_cam[ch])
        y = _apply_crf(toe, shoulder, gamma, center, x)
        ax_plot.plot(x.cpu().numpy(), y.cpu().numpy(),
                     color=cols[ch], linewidth=2.0, label=["Red", "Green", "Blue"][ch])
        center_x = float(center.cpu().item())
        center_y = float(_apply_crf(toe, shoulder, gamma, center, center).cpu().item())
        ax_plot.scatter([center_x], [center_y], color=cols[ch], s=18, zorder=6, marker="o")
    ax_plot.axvline(1.0, color="black", linestyle="--", alpha=0.5)
    ax_plot.set_xlabel("Linear Input Intensity")
    ax_plot.set_ylabel("Output Intensity")
    ax_plot.set_title("Camera Response Function (R, G, B)")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend()

    img = _gray_bars(256)
    xin = torch.from_numpy(img[img.shape[0] // 2:, :, 0].copy()).to(crf_params.device)
    xin_lin = xin.flatten()
    for ch in range(3):
        toe, shoulder, gamma, center = _crf_effective_from_raw(crf_cam[ch])
        y = _apply_crf(toe, shoulder, gamma, center, xin_lin)
        img[img.shape[0] // 2:, :, ch] = y.reshape(
            img.shape[0] // 2, img.shape[1]).cpu().numpy()
    img = _srgb_inverse_oetf(img)
    _show_image(ax_img, img, "Tone Mapping Visualization")
    size = img.shape[0]
    ax_img.text(size * 0.5, 20, "Linear", ha="center", va="top", color="white",
                bbox=dict(facecolor="black", alpha=0.5))
    ax_img.text(size * 0.5, size - 20, "Tone Mapped", ha="center", va="bottom",
                color="white", bbox=dict(facecolor="black", alpha=0.5))


# -- Main report generation ---------------------------------------------------

@torch.no_grad()
def generate_report(ppisp_path: Path, output_dir: Path | None = None) -> list[Path]:
    matplotlib.use("Agg", force=True)

    data = load_ppisp(ppisp_path)
    num_cams = data["num_cameras"]
    num_frames = data["num_frames"]

    # Single camera: all frames belong to it
    frames_per_camera = [num_frames] if num_cams == 1 else [num_frames // num_cams] * num_cams

    if output_dir is None:
        output_dir = ppisp_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for cam in range(num_cams):
        num_rows = 5
        fig = plt.figure(figsize=(20, 5 * num_rows))
        gs = fig.add_gridspec(num_rows, 1, height_ratios=[1] * num_rows)

        row_idx = 0
        _plot_exposure(fig, gs[row_idx], data["exposure_params"], frames_per_camera, cam)
        row_idx += 1

        _plot_vignetting(fig, gs[row_idx], data["vignetting_params"], cam)
        row_idx += 1

        _plot_color(fig, gs[row_idx], gs[row_idx + 1],
                    data["color_params"], frames_per_camera, cam)
        row_idx += 2

        _plot_crf(fig, gs[row_idx], data["crf_params"], cam)

        plt.tight_layout()

        stem = ppisp_path.stem
        cam_label = f"camera_{cam}"
        out_path = output_dir / f"{stem}_{cam_label}_ppisp_report.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
        print(f"  Written: {out_path}")

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate PDF report from .ppisp file")
    parser.add_argument("ppisp_file", type=Path, help="Path to .ppisp file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to same as input)")
    args = parser.parse_args()

    if not args.ppisp_file.exists():
        print(f"Error: {args.ppisp_file} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.ppisp_file}...")
    data = load_ppisp(args.ppisp_file)
    print(f"  Cameras: {data['num_cameras']}, Frames: {data['num_frames']}")
    print(f"  Exposure: {data['exposure_params'].shape}")
    print(f"  Vignetting: {data['vignetting_params'].shape}")
    print(f"  Color: {data['color_params'].shape}")
    print(f"  CRF: {data['crf_params'].shape}")

    print("Generating report...")
    outputs = generate_report(args.ppisp_file, args.output_dir)
    print(f"Done. {len(outputs)} PDF(s) generated.")


if __name__ == "__main__":
    main()
