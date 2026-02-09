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

This project builds upon many open-source libraries. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the full list.

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
