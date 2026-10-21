---
title: Initialize from a mesh
---

# Initialize from a mesh

Training can start from a triangle mesh instead of the dataset's sparse points. Pass the mesh to `--init`:

```sh
LichtFeld-Studio -d ./dataset --output-path ./output --init ./scan.obj --train
```

In the New Project dialog, choose the mesh as the init file under Advanced.

## What happens

When training starts, Mesh2Splat converts the mesh to splats: one flat splat per texel of a square raster laid over
the mesh bounds, oriented along the surface and colored from the base color texture, the vertex colors or the
material color. The converted splats replace the sparse points as the starting model. Nothing is written to disk.

Before training starts, the scene shows the mesh vertices as the point cloud.

## Requirements

- The mesh must be in the dataset's coordinates, as a photogrammetry or scanner mesh made from the same cameras is.
  It is not aligned or scaled. With `--centralize`, it is shifted by the same offset as the dataset.
- Supported formats: `.obj`, `.fbx`, `.gltf`, `.glb`, `.stl`, `.dae`, `.3ds`, and `.ply` with faces. A `.ply` without
  faces is a point cloud or splat file and is loaded as such.

## Splat count

The conversion raster is 1024 texels on the long side of the mesh bounds. Set it with `--init-mesh-resolution`
(16 to 2048), or with Mesh Init Resolution in the Training panel's Initialization section, which appears when the
init file is a mesh. Higher resolutions give more and smaller starting splats. When the conversion yields more splats
than `--max-cap`, the mesh is converted again at the resolution that fits the cap, so the whole surface stays covered.

The resolution only applies to mesh init files. With any other init, training logs a warning and ignores it.
