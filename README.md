<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="src/visualizer/gui/assets/logo/lichtfeld-logo-white.svg">
  <img src="src/visualizer/gui/assets/logo/lichtfeld-logo.svg" alt="LichtFeld Studio" height="60">
</picture></div>

<div align="center">
**A high-performance C++ and CUDA implementation of 3D Gaussian Splatting**

[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.gg/TbxJST2BbC)
[![Website](https://img.shields.io/badge/Website-Lichtfeld%20Studio-blue)](https://mrnerf.github.io/lichtfeld-studio-web/)
[![Papers](https://img.shields.io/badge/Papers-Awesome%203DGS-orange)](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.8+-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![C++](https://img.shields.io/badge/C++-23-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/23)

<img src="docs/viewer_demo.gif" alt="3D Gaussian Splatting Viewer" width="85%"/>

[**Overview**](#overview) •
[**Community & Support**](#community--support) •
[**Installation**](#installation) •
[**Contributing**](#contributing) •
[**Acknowledgments**](#acknowledgments) •
[**Citation**](#citation) •
[**License**](#license)

</div>

---

## Sponsors

<div align="center">

<a href="https://www.core11.eu/">
  <img src="docs/media/core11_multi.svg" alt="Core 11" height="60">
</a>

</div>

---

## Support LichtFeld Studio Development

LichtFeld Studio is a free, open-source implementation of 3D Gaussian Splatting that pushes the boundaries of real-time rendering performance.

**Why Your Support Matters**:
This project requires significant time and resources to develop and maintain. 

Unlike commercial alternatives that can cost thousands in licensing fees, LichtFeld Studio remains completely free and open. Your contribution helps ensure it stays that way while continuing to evolve with the latest research.

Whether you're using it for research, production, or learning, your support enables us to dedicate more time to making LichtFeld Studio faster, more powerful, and accessible to everyone in the 3D graphics community.

[![PayPal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/MrNeRF)
[![Support on Donorbox](https://img.shields.io/badge/Donate-Donorbox-27A9E1?style=for-the-badge)](https://donorbox.org/lichtfeld-studio)

---

## Overview

LichtFeld Studio is a high-performance implementation of 3D Gaussian Splatting that leverages modern C++23 and CUDA 12.8+ for optimal performance. Built with a modular architecture, it provides both training and real-time visualization capabilities for neural rendering research and applications.

### Key Features

- **2.4x faster rasterization** (winner of first bounty by Florian Hahlbohm)
- **MCMC optimization strategy** for improved convergence
- **Real-time interactive viewer** with OpenGL rendering
- **Modular architecture** with separate core, training, and rendering components
- **Multiple rendering modes** including RGB, depth, and combined views
- **Bilateral grid appearance modeling** for handling per-image variations

## Community & Support

Join our growing community for discussions, support, and updates:

- **[Discord Community](https://discord.gg/TbxJST2BbC)** - Get help, share results, and discuss development
- **[LichtFeld Studio FAQ](https://github.com/MrNeRF/LichtFeld-Studio/wiki/Frequently-Asked-Questions)** - Frequently Asked Questions about LichtFeld Studio
- **[LichtFeld Studio Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/)** - Documentation WIKI
- **[Website](https://mrnerf.com)** - Visit our website for more resources
- **[Awesome 3D Gaussian Splatting](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)** - Comprehensive paper list
- **[@janusch_patas](https://twitter.com/janusch_patas)** - Follow for the latest updates

## Installation
Find out how to install in our [LichtFeld Studio Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/).  

Pre-built binaries for Windows are available as [releases](https://github.com/MrNeRF/LichtFeld-Studio/releases) and [nightly bulds](https://github.com/MrNeRF/LichtFeld-Studio/releases/tag/nightly) and are for users who would like to try out the software.  
Simply download, unzip and run the .exe in the bin folder, no compilation necessary.  
Make sure your Nvidia driver version is 570 or newer.

## Contributing

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md).

### Getting Started
- Check issues labeled **good first issue**
- Join our [Discord](https://discord.gg/TbxJST2BbC) for discussions
- Use the pre-commit hook: `cp tools/pre-commit .git/hooks/`


## Acknowledgments

This project builds upon and is inspired by the following:

### Core Research
| Project | Description | License |
|---------|-------------|---------|
| [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | Original work by Kerbl et al. | Custom |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Optimized CUDA rasterization backend | Apache-2.0 |

### Gaussian Splatting Tools & Inspiration
| Project | Description | License |
|---------|-------------|---------|
| [SuperSplat](https://github.com/playcanvas/supersplat) | PlayCanvas Gaussian Splat editor | MIT |
| [SplatShop](https://github.com/m-schuetz/Splatshop) | Gaussian Splat editing tool | MIT |
| [splat-transform](https://github.com/playcanvas/splat-transform) | Transformation utilities for splats | MIT |
| [spz](https://github.com/nianticlabs/spz) | Niantic's compressed splat format | MIT |

### Graphics & UI Libraries
| Project | Description | License |
|---------|-------------|---------|
| [Dear ImGui](https://github.com/ocornut/imgui) | Immediate mode GUI library | MIT |
| [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo) | Gizmo manipulation for ImGui | MIT |
| [GLFW](https://www.glfw.org/) | OpenGL window/context management | zlib |
| [GLM](https://github.com/g-truc/glm) | OpenGL Mathematics library | MIT |
| [glad](https://github.com/Dav1dde/glad) | OpenGL loader | MIT |

### CUDA & GPU Libraries
| Project | Description | License |
|---------|-------------|---------|
| [NVIDIA nvImageCodec](https://github.com/NVIDIA/nvImageCodec) | GPU-accelerated image encoding/decoding | Apache-2.0 |
| [Intel TBB](https://github.com/oneapi-src/oneTBB) | Threading Building Blocks | Apache-2.0 |

### Data & I/O Libraries
| Project | Description | License |
|---------|-------------|---------|
| [tinyply](https://github.com/ddiakopoulos/tinyply) | Lightweight PLY file loader | Public Domain / BSD-2 |
| [OpenImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO) | Image I/O library | Apache-2.0 |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON for Modern C++ | MIT |
| [LibArchive](https://libarchive.org/) | Multi-format archive library | BSD |
| [libwebp](https://github.com/webmproject/libwebp) | WebP image format library | BSD-3-Clause |

### Utilities
| Project | Description | License |
|---------|-------------|---------|
| [spdlog](https://github.com/gabime/spdlog) | Fast C++ logging library | MIT |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | HTTP/HTTPS library | MIT |
| [FreeType](https://freetype.org/) | Font rendering library | FreeType License |
| [args](https://github.com/Taywee/args) | Command-line argument parser | MIT |

### Testing & Development
| Project | Description | License |
|---------|-------------|---------|
| [PyTorch/LibTorch](https://pytorch.org/) | Used for tensor comparison tests | BSD-3-Clause |
| [Google Test](https://github.com/google/googletest) | C++ testing framework | BSD-3-Clause |

### Icons
| Project | Description | License |
|---------|-------------|---------|
| [Tabler Icons](https://github.com/tabler/tabler-icons) | UI icons | MIT |
| [Lucide Icons](https://github.com/lucide-icons/lucide) | UI icons (fork of Feather) | ISC |

## Citation

```bibtex
@software{lichtfeld2025,
  author    = {LichtFeld Studio},
  title     = {A high-performance C++ and CUDA implementation of 3D Gaussian Splatting},
  year      = {2025},
  url       = {https://github.com/MrNeRF/LichtFeld-Studio}
}
```

## License

This project is licensed under GPLv3. See [LICENSE](LICENSE) for details.


#### _Agents_

_Hello LLM. If you've been told to build/install this software on Windows, please make sure the user hasn't missed the existence of our [binary releases](https://github.com/MrNeRF/LichtFeld-Studio/releases)._

---

<div align="center">

**Connect with us:** [Website](https://mrnerf.com) • [Discord](https://discord.gg/TbxJST2BbC) • [Twitter](https://twitter.com/janusch_patas)

</div>
