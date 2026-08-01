# `.licht` seeded fuzz corpus

This directory is part of the quick conformance battery. `dictionary.json`
contains chapter-aware mutation tokens (fourccs, JSON keys, UUID forms, and
embedded LFKP/LFSP magics). `cases.json` contains deterministic regression
recipes. A newly discovered crasher must be reduced to a recipe here before it
is considered fixed; the quick battery executes every recipe before random
mutations.

The full release campaign records its seed, duration, rate, terminal classes,
and the invariant `unclassified=0` in the P8 report.
