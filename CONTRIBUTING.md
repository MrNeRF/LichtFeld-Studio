# Contributing to LichtFeld Studio

Thanks for your interest in contributing!

## Getting Started

1. **Find an Issue or Propose an Extension**
   - Check issues labeled `good first issue` for newcomers
   - Browse other open issues that interest you
   - Want to add a feature? Open an issue to discuss it first
   - Not sure where to start? Ask on [Discord](https://discord.gg/TbxJST2BbC)

2. **Setup**
   - Installation instructions are in the [Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki)
   - Install the pre-commit hook: `cp tools/pre-commit .git/hooks/`

3. **Make Your Changes**
   - Apply `clang-format` before committing
   - Write clear commit messages
   - Test your changes

4. **Submit a Pull Request**
   - Link related issues
   - Describe what you changed and why

## Localization

`src/visualizer/gui/resources/locales/en.json` is the canonical locale bundle.
Every shipped locale must have exactly the same keys as `en.json`, even when a
translation is temporarily the English fallback. The runtime fallback keeps the
UI readable, but CI rejects missing and obsolete keys. Translated strings must
also preserve the `std::format` placeholders from English; named placeholders
may be reordered to suit the target language.

Before submitting localization changes, run:

```bash
python tools/check_locale_completeness.py
```

To audit values that are still identical to English without failing the check:

```bash
python tools/check_locale_completeness.py --report-identical
```

To print those keys, use `--list-identical`. For a deliberate translation audit
that fails when non-empty values still match English, use:

```bash
python tools/check_locale_completeness.py --fail-on-identical
```

The Locale Completeness GitHub Actions workflow runs the default check for pull
requests and pushes to `master`; the identical-value audit remains opt-in.

## Code Style

- Use `clang-format` for formatting
- Follow C++23 standards
- Add SPDX headers to new files:
  ```cpp
  /* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
   *
   * SPDX-License-Identifier: GPL-3.0-or-later */
  ```

## Need Help?

- Join our [Discord](https://discord.gg/TbxJST2BbC) for questions and discussions
- Check the [Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki) for documentation
- Look at the [FAQ](https://github.com/MrNeRF/LichtFeld-Studio/wiki/Frequently-Asked-Questions)

## License

Contributions are licensed under GPLv3.
