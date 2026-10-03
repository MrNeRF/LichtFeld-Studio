# Node graph examples

Twenty-three node graphs for everyday splat editing, built with `lichtfeld.nodes`
on the Mip-NeRF 360 `garden` and `bicycle` captures (1M Gaussians each). Each file
defines `SCENE`, `TITLE`, a `build(target)` that creates the graph and adds it as
a modifier on the scene node `target`, and a `check(target)` that returns numbers
proving the result. `common.py` holds the shared helpers, including height and
ground-plane distance fields for these tilted captures.

Run one from the Python console after loading the scene:

```python
import sys; sys.path.insert(0, "docs/docs/development/node-graph/examples")
import importlib, ex07_autumn_grass as ex
ex.build("garden"); print(ex.check("garden"))
```

All 23 evaluate in under 80 ms on 1M splats (Apple M5 Max, Metal).

| Example | Scene | What it does | Main nodes |
|---|---|---|---|
| [ex01](ex01_floaters_ground_sky.py) | bicycle | Removes floaters only below the ground and up in the sky | Compare, Boolean Math, Remove Floaters |
| [ex02](ex02_background_haze.py) | garden | Strips faint, oversized haze Gaussians | Opacity, Scale, Compare, Delete Geometry |
| [ex03](ex03_crop_showcase.py) | garden | Crops the scene to the table for a showcase | Ellipsoid Selection, Separate Geometry |
| [ex04](ex04_remove_bike.py) | bicycle | Clean plate: removes the bench and the bike | Height and distance fields, Delete Geometry |
| [ex05](ex05_extract_bike.py) | bicycle | Extracts the bench with the bike as an asset | Separate Geometry, Remove Floaters |
| [ex06](ex06_colour_grade.py) | garden | Grades the whole capture | Colour Correct |
| [ex07](ex07_autumn_grass.py) | garden | Turns the lawn autumn-coloured | HSV Range, Colour Correct |
| [ex08](ex08_repaint_bike.py) | bicycle | Repaints the bike frame, keeping its shading | HSV Range, Math, Recolour |
| [ex09](ex09_colour_pop.py) | garden | Greys everything except the centrepiece | Ellipsoid Selection (feathered), Colour Correct |
| [ex10](ex10_edge_fade.py) | garden | Fades the capture out softly at its edges | Map Range, Math, Set Opacity |
| [ex11](ex11_web_export.py) | bicycle | Web version: SH degree 0, half the Gaussians (≈8× smaller) | Set SH Degree, Decimate |
| [ex12](ex12_scale_clamp.py) | bicycle | Shortens needle-shaped Gaussians that streak off the capture path | Scale Clamp |
| [ex13](ex13_sharpen_subject.py) | garden | Sharpens only the centrepiece | Ellipsoid Selection, Sharpen |
| [ex14](ex14_level_scene.py) | garden | Levels the scene and centres the subject | Transform Geometry |
| [ex15](ex15_duplicate_vase.py) | garden | Copies the vase and places the copy on the table | Separate, Transform, Join Geometry |
| [ex16](ex16_mesh_prop.py) | garden + mesh | Adds a mesh prop as Gaussians | Object Info, Mesh to Splats, Join Geometry |
| [ex17](ex17_mesh_cutout.py) | bicycle + mesh | Keeps only what lies inside a mesh | Object Info, Inside Mesh, Separate Geometry |
| [ex18](ex18_height_fog.py) | garden | Adds low-lying fog | Map Range, Set Colour |
| [ex19](ex19_hand_selection.py) | garden | Deletes a viewport selection non-destructively | Stored Selection, Delete Geometry |
| [ex20](ex20_floater_preview.py) | garden | Shows what a clean-up would remove before applying it | Remove Floaters (preview) |
| [ex21](ex21_clip_mesh.py) | mesh | A modifier on a mesh node: cuts a model in half | Position, Separate XYZ, Compare, Delete Geometry |
| [ex22](ex22_sfm_points.py) | garden + points | Shows the SfM point cloud as splats | Object Info, Points to Splats |
| [ex23](ex23_posterize_plugin.py) | bicycle | A node written in Python | Posterize (`lfs_plugins`) |

Examples 16 and 17 expect a mesh node named `torus` / `sphere`, and example 22 a
point-cloud node named `garden_sparse` (the COLMAP `points3D` of the capture).
