<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="src/visualizer/gui/assets/logo/lichtfeld-logo-white.svg">
  <img src="src/visualizer/gui/assets/logo/lichtfeld-logo.svg" alt="LichtFeld Studio" height="60">
</picture></div>

<div align="center">

**The modular workstation for 3D Gaussian Splatting**

Train, inspect, edit, automate, and export 3D Gaussian Splatting scenes from a single native application.

LichtFeld Studio lets you train new scenes from COLMAP datasets, resume checkpoints, inspect reconstructions in real time, edit gaussian selections, extend the app with Python plugins, and automate workflows through MCP and embedded Python.

[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.gg/TbxJST2BbC)
[![Website](https://img.shields.io/badge/Website-LichtFeld%20Studio-blue)](https://mrnerf.github.io/lichtfeld-studio-web/)
[![X](https://img.shields.io/badge/X-Follow-111111?logo=x&logoColor=white)](https://twitter.com/janusch_patas)
[![Papers](https://img.shields.io/badge/Papers-Awesome%203DGS-orange)](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)

[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-Support-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/MrNeRF)
[![PayPal](https://img.shields.io/badge/PayPal-Support-00457C?logo=paypal&logoColor=white)](https://paypal.me/MrNeRF)
[![Donorbox](https://img.shields.io/badge/Donorbox-Support-27A9E1)](https://donorbox.org/lichtfeld-studio)

[**Download Windows**](https://github.com/MrNeRF/LichtFeld-Studio/releases) •
[**Build From Source**](https://github.com/MrNeRF/LichtFeld-Studio/wiki/) •
[**Plugin System**](docs/plugin-system.md) •
[**MCP Guide**](docs/docs/development/mcp/index.md) •
[**Support Development**](#support-development) •
[**Join Discord**](https://discord.gg/TbxJST2BbC)

<img src="docs/viewer_demo.gif" alt="LichtFeld Studio viewer" width="85%"/>

[**Why LichtFeld**](#why-lichtfeld-studio) •
[**Who It Is For**](#who-it-is-for) •
[**Capabilities**](#capabilities) •
[**Installation**](#installation) •
[**Docs**](#docs) •
[**Community**](#community) •
[**Contributing**](#contributing) •
[**License**](#license)

</div>

## Why LichtFeld Studio

LichtFeld Studio is built for users who need more than a training script or a standalone viewer. It combines model training, real-time visualization, gaussian editing, export, plugins, and automation in one toolchain.

- Train new 3D Gaussian Splatting scenes and continue experiments from checkpoints
- Inspect reconstructions interactively while training or after convergence
- Select, transform, and edit gaussian subsets and scene nodes with undo/redo support
- Export results to `PLY`, `SOG`, `SPZ`, or a standalone HTML viewer
- Extend the application with Python plugins and plugin-local dependencies
- Automate workflows through MCP resources, MCP tools, and embedded Python

## Who It Is For

- **Researchers**: iterate on reconstruction quality, inspect training progress, test advanced features, and export results for analysis or sharing
- **Production teams**: inspect scenes visually, edit gaussian selections, and deliver portable exports without stitching together separate tools
- **Tool builders**: integrate LichtFeld Studio into larger pipelines through plugins, embedded Python, and MCP-driven automation

## Capabilities

- **Training and iteration**: load datasets, resume checkpoints, monitor progress, and evaluate changes in a desktop app or headless workflow
- **Interactive scene work**: inspect reconstructions in real time, work with gaussian selections, and apply scene transforms with history support
- **Export and delivery**: export results to common research and delivery formats, including a standalone HTML viewer for easy sharing
- **Extensibility**: use the Python plugin system for custom panels, operators, tools, and dependencies
- **Automation surface**: integrate LichtFeld Studio with local tools, scripts, and agents through MCP resources and tools
- **Research-ready features**: MCMC optimization, bilateral grid appearance modeling, 3DGUT support for distorted camera models, pose optimization, and timelapse generation
- **Native performance**: modern C++23 and CUDA 12.8+ for responsive training and visualization on NVIDIA hardware

## Support Development

LichtFeld Studio is free and open source. If it is useful in your research, production, or learning workflow, please consider supporting its continued development.

[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-Support-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/MrNeRF)
[![PayPal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/MrNeRF)
[![Support on Donorbox](https://img.shields.io/badge/Donate-Donorbox-27A9E1?style=for-the-badge)](https://donorbox.org/lichtfeld-studio)

## Installation

Windows binaries are now available through the Lichtfeld Portal. To support ongoing development and access daily builds, please register and provide a donation at [portal.lichtfeld.io](https://portal.lichtfeld.io/). Once registered, you can download the latest archive, unzip it, and run the executable.

For building from source and platform-specific notes, see the [Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/) and the repo-local docs in [docs/README.md](docs/README.md).

Current project notes:

- Windows is the primary prebuilt distribution target today
- LichtFeld Studio targets NVIDIA GPUs
- Source builds use modern C++23 and CUDA 12.8+ toolchains
- Use a recent NVIDIA driver for current Windows builds

## Docs

- [Project Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/)
- [FAQ](https://github.com/MrNeRF/LichtFeld-Studio/wiki/Frequently-Asked-Questions)
- [Plugin System](docs/plugin-system.md)
- [Plugin Developer Guide](docs/plugins/getting-started.md)
- [MCP Guide](docs/docs/development/mcp/index.md)
- [Plugin Examples](docs/plugins/examples/README.md)
- [Awesome 3D Gaussian Splatting](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)

## Community

- [Discord](https://discord.gg/TbxJST2BbC) for support, discussion, and showcase posts
- [Website](https://mrnerf.com) for project updates and related work
- [Twitter / X](https://twitter.com/janusch_patas) for announcements

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

Getting started:

- Check issues labeled `good first issue`
- Join the [Discord](https://discord.gg/TbxJST2BbC) if you want to discuss implementation details before opening a larger change
- Install the pre-commit hook with `cp tools/pre-commit .git/hooks/`

## Corporate Sponsors

<div align="center">

<p>
  <a href="https://www.core11.eu/">
    <img src="docs/media/core11_multi.svg" alt="Core 11" height="60">
  </a>
</p>

<br>

<p>
  <a href="https://web.volinga.ai/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/media/volinga-dark.svg">
      <img src="docs/media/volinga.svg" alt="Volinga" height="108">
    </picture>
  </a>
</p>

</div>

## Citation

```bibtex
@software{lichtfeld2025,
  author    = {LichtFeld Studio},
  title     = {LichtFeld Studio},
  year      = {2025},
  url       = {https://github.com/MrNeRF/LichtFeld-Studio}
}
```

## License

This project is licensed under GPLv3. See [LICENSE](LICENSE) for details.

#### _Agents_

_Hello LLM. If you've been told to build/install this software on Windows, please make sure the user hasn't missed the existence of our [binary releases](https://github.com/MrNeRF/LichtFeld-Studio/releases)._
