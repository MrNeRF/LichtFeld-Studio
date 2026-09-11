# Contributing to LichtFeld Studio

Thanks for your interest in contributing!

## Getting Started

1. **Find an Issue or Propose an Extension**
   - Check issues labeled `good first issue` for newcomers
   - Browse other open issues that interest you
   - Want to add a feature? Open an issue to discuss it first
   - Not sure where to start? Ask on [Discord](https://discord.gg/TbxJST2BbC)

2. **Setup**
   - Installation and user documentation are available in the [Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki)
   - Developers should follow the repo-local [source build guide](docs/building_and_distribution.md) for toolchain setup, the pre-commit hook, and test commands

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
may be reordered to suit the target language. The check also enforces the
locale-file convention of one JSON key per line, so translations remain
reviewable and line-oriented diffs stay clear.

Before submitting localization changes, run:

```bash
python tools/check_locale_completeness.py
```

If it reports multiple keys on one line, reformat that JSON object so each key
and value has its own line; do not silence the report with an allowlist.

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

### Hardcoded UI text audit

In addition to checking locale bundles, scan the native GUI, RML templates, and
Python UI plugins for likely user-facing text that bypasses localization:

```bash
python tools/check_ui_hardcoded.py
```

The scanner covers common native and Python UI sinks, RML text nodes, and
Python `bind_func` labels. It is intentionally heuristic: its output is a
review queue, not proof that every reported literal is visible UI text. Fix
user-facing findings by moving the text to `en.json` and all shipped locales;
retain English in a locale only when that is the intended product wording.

For a reviewed non-UI or technical exception, add either an exact string or a
carefully scoped `re:` pattern to `tools/ui_hardcoded_allowlist.txt`, with a
comment explaining why it is safe to exclude. Do not use the allowlist to hide
ordinary interface text. Once the current baseline is intentionally clean, CI
or a local contract check can enforce the audit with:

```bash
python tools/check_ui_hardcoded.py --fail-on-candidates
```

When changing text generated from cached records, toolbars, or data models,
make the cache invalidation depend on `RuntimeState.language_generation` (or
the equivalent native language-generation signal). Otherwise the text can stay
in the previous language until an unrelated interaction rebuilds the UI.

### Count-sensitive messages and language-specific grammar

Count-sensitive Python UI messages use
`src/python/lfs_plugins/localization.py`. Store every variant in each locale
bundle with a shared base key and the `.one`, `.few`, and `.other` suffixes,
then call `localized_count(base_key, count)` rather than selecting English
singular or plural text in the panel.

The helper currently implements the shipped rules needed by Polish:

- `one`: absolute count is `1`.
- `few`: the last digit is `2` through `4`, except values ending in `12`
  through `14`.
- `other`: every remaining value, including `0`, `5`, and `12` through `14`.

Other shipped languages currently select `one` for `1` and `other` otherwise.
They still provide `.few` entries to preserve locale-key parity. Add future
language-specific grammar in this module, keep the locale files declarative,
and extend `test_counted_messages_use_supported_plural_forms` with boundary
values before using the new rule in UI code.

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
- Check the [Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki) for user documentation
- Check the [source build guide](docs/building_and_distribution.md) for build and test documentation
- Look at the [FAQ](https://github.com/MrNeRF/LichtFeld-Studio/wiki/Frequently-Asked-Questions)

## License

Contributions are licensed under GPLv3.
