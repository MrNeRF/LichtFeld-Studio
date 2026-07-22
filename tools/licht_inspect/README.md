# Independent `.licht` inspector

This directory contains the pure-Python reference parser for the byte grammar
in `docs/licht_format_spec.md`. It has no application or C++ dependency. Python
stdlib is sufficient for the committed golden fixtures; install the optional
`zstandard` package to inspect production zstd indexes or payloads.

From the repository root:

```sh
python3 tools/licht_inspect/licht_inspect.py inspect FILE.licht
python3 tools/licht_inspect/licht_inspect.py inspect FILE.licht --json
python3 tools/licht_inspect/licht_inspect.py verify FILE.licht
python3 tools/licht_inspect/licht_inspect.py extract FILE.licht \
  --fourcc PROJ --uuid UUID -o project.json
python3 tools/licht_inspect/licht_inspect.py recovery MASTER.licht \
  MASTER.licht.autosave MASTER.licht.autosave.tmp
```

The container is always opened read-only. `extract` writes only its explicitly
named output and refuses to replace an existing file unless `--force` is used.
When the selected authority requires a newer reader version or an unknown
required-reader capability, `inspect` still dumps its structurally validated
headers/commit/index and exits 4 with `unsupported_newer`; `extract` and
`verify` refuse that terminal. Inspect output also reports whether the selected
commit is safe for a declared version-1.0 writer.

Regenerate deterministic golden fixtures with:

```sh
python3 tools/licht_inspect/make_fixtures.py
```

`make_fixtures.py` is a deliberately incomplete non-production writer. It does
not implement the durability protocol and must never be linked or copied into
application persistence code. Golden indexes use the specified `STORED` test
mode so fixture bytes do not depend on a local zstd package or library version.

Generate the full default 10,000-case randomized oracle corpus plus the golden
catalogue into an external directory with:

```sh
python3 tools/licht_inspect/oracle_corpus.py \
  --output-dir /tmp/licht-oracle
```

Oracle expectations are assigned by mutation family from the published open
and durability rules before any parser runs. The generator API yields records
whose authority fields are `mutated_file`, `expected_outcome`, and
`mutation_description`. Materialized manifests use the filename in
`mutated_file` and include its SHA-256.

Run the cheap committed-fixture and sampled-oracle test:

```sh
python3 tools/licht_inspect/run_selftest.py
```

The test covers every golden fixture class, full CRC verification, sidecar
offer/stale-base behavior, all three CLI surfaces, the reader/writer
compatibility matrix, and every randomized mutation family with fixed seeds.

## Conformance battery

`run_conformance.py` is the heavy robustness battery on top of the selftest:

```sh
python3 tools/licht_inspect/run_conformance.py --quick        # CI, <60 s
python3 tools/licht_inspect/run_conformance.py --full         # exhaustive, minutes
python3 tools/licht_inspect/run_conformance.py --fuzz-minutes 10
```

Categories: golden-oracle baseline, exhaustive truncation sweep (every append/head
boundary asserts its exact spec outcome), CRC-valid semantic corruption
constructions, randomized append-lifecycle property runs checked against an
independent state model (including reader-pinned-at-generation-N), sparse-file
64-bit/scale edges, and a seeded, time-boxed randomized malformed-input sweep
whose invariant is a classified terminal outcome for every input.
`spec_byte_verifier.py` is a second, from-the-spec reader with no imports from
`licht_inspect.py`; `--full` cross-checks the main parser against it.

## C++ writer acceptance contract (P2)

The battery is the acceptance harness for the production C++ container writer:

1. Given the fixtures' logical inputs (constants table in `make_fixtures.py`),
   the C++ writer must reproduce every golden fixture **byte-for-byte**.
2. Every file the C++ writer produces must pass `licht_inspect verify` and the
   battery's open matrix.
3. The C++ reader must agree with this parser's classification on the full
   oracle corpus and all battery categories (dual-parser agreement is necessary,
   never sufficient — the spec oracle remains the authority).
