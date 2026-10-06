# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
# Run stubgen with INCLUDE_PRIVATE, then retain its usual private-name filter
# except for Tensor's public trailing-underscore methods (in-place operations
# and scalar conversions). Signatures still come from the live bindings.
(^|\.)_[^_.][^.]+$:

^(?!lichtfeld\.Tensor\.)[^\n]*\.[^.]{2,}(?<!_)_$:
