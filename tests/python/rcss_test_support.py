# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read matching RCSS rules without treating theme tokens as rule braces."""

import re


def rule_bodies(css: str, selector: str) -> list[str]:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    selector = " ".join(selector.split())
    return [
        body
        for selectors, body in re.findall(r"([^{}]+)\{((?:@\{[^}]+\}|[^{}])*)\}", css)
        if selector in {" ".join(value.split()) for value in selectors.split(",")}
    ]


def rule_body(css: str, selector: str) -> str:
    bodies = rule_bodies(css, selector)
    assert bodies, f"Missing RCSS selector: {selector}"
    return "\n".join(bodies)
