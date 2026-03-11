# Fast-GS Rasterizer

This is an optimized CUDA implementation of the 3DGS rasterizer by Florian Hahlbohm.

It includes a complete overhaul of the involved maths to skip unnecessary computations.
Also, all activation functions of the Gaussians' parameters are fused into the respective kernels.

Most importantly, this implementation combines ideas and tricks from the following works:

- Underlying algorithm is based on "3D Gaussian Splatting for Real-Time Radiance Field Rendering" by Kerbl and Kopanas et al. 2023
- Overall architecture is based on "Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency" by Hahlbohm et al. 2025
- Culling and load balancing improvements are based on "StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering" by Radl and Steiner et al. 2024
- Efficient blending backward pass is based on "Taming 3DGS: High-Quality Radiance Fields with Limited Resources" by Mallick and Goel et al. 2024
- Separate depth and tile sorting is based on "Splatshop: Efficiently Editing Large Gaussian Splat Models" by Schütz et al. 2025

Please do not forget to credit these works if you use this implementation or parts of it in your own work.