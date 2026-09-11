#!/usr/bin/env bash
# Build the debug fixture and assert the tensor pretty printers behave, under both
# gdb and cuda-gdb. Bash (not zsh) on purpose: the flag list is passed as an array,
# because zsh does not word-split unquoted parameter expansions.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
CUDA_GDB="${CUDA_GDB:-/usr/local/cuda-12.8/bin/cuda-gdb}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FIXTURE="$REPO/tools/gdb/tensor_pp_fixture.cpp"
PRINTERS="$REPO/tools/gdb/lfs_pretty_printers.py"
BIN="$WORK/tensor_pp_fixture"

fail=0
check() { # check <label> <expected-substring> <file>
    if grep -qF -- "$2" "$3"; then
        echo "  PASS  $1"
    else
        echo "  FAIL  $1 (expected: $2)"
        fail=1
    fi
}

[ -f "$BUILD_DIR/compile_commands.json" ] || {
    echo "no compile_commands.json in $BUILD_DIR — configure with: cmake --preset build" >&2
    exit 2
}

echo "== extracting compile flags from $BUILD_DIR/compile_commands.json"
mapfile -t FLAGS < <(python3 - "$BUILD_DIR/compile_commands.json" <<'PY'
import json, shlex, sys
db = json.load(open(sys.argv[1]))
ent = next((e for e in db if e['file'].endswith('core/tensor/tensor.cpp')), None)
if ent is None:
    ent = next(e for e in db if '/core/tensor/' in e['file'] and e['file'].endswith('.cpp'))
args = shlex.split(ent.get('command') or ' '.join(ent['arguments']))
out, i = [], 0
while i < len(args):
    a = args[i]
    if a in ('-isystem', '-include', '-iquote'):
        if i + 1 < len(args):
            out += [a, args[i + 1]]
            i += 2
            continue
    # *_EXPORTS marks the library's own TUs; this fixture is a consumer.
    elif a.startswith('-I') or (a.startswith('-D') and 'EXPORTS' not in a):
        out.append(a)
    i += 1
print('\n'.join(out))
PY
)
[ "${#FLAGS[@]}" -gt 0 ] || { echo "flag extraction produced nothing" >&2; exit 2; }
echo "   ${#FLAGS[@]} flags"

echo "== compiling fixture (-g -O0)"
g++ -g -O0 -std=gnu++23 "${FLAGS[@]}" -o "$BIN" "$FIXTURE" \
    -L "$BUILD_DIR" -llfs_core -L /usr/local/cuda-12.8/lib64 -lcudart \
    -Wl,-rpath,"$BUILD_DIR" -Wl,-rpath,/usr/local/cuda-12.8/lib64 || {
    echo "fixture failed to compile" >&2
    exit 2
}

run_dbg() { # run_dbg <debugger> <outfile> <extra -ex...>
    local dbg="$1" out="$2"
    shift 2
    timeout 300 "$dbg" -batch -nx \
        -ex "source $PRINTERS" \
        -ex "break inspect_here" -ex "run" -ex "up" \
        "$@" "$BIN" >"$out" 2>&1
}

echo "== gdb"
GOUT="$WORK/gdb.txt"
run_dbg gdb "$GOUT" \
    -ex "print cpu_f32" -ex "print cpu_i32" -ex "print cpu_rank3" \
    -ex "print cuda_f32" -ex "print empty_tensor" -ex "print shape" \
    -ex "lfs-tensor cpu_rank3 4" -ex "lfs-tensor cuda_f32 4"
check "summary: cpu float32 shape+elems" "Tensor float32[2, 3] cpu elems=6 bytes=24" "$GOUT"
check "summary: int32 dtype"             "Tensor int32[4] cpu elems=4"               "$GOUT"
check "summary: rank-3 strides"          "strides = [12, 4, 1]"                      "$GOUT"
check "summary: cuda device + bytes"     "Tensor float32[3, 256, 256] cuda elems=196608 bytes=786432" "$GOUT"
check "summary: empty tensor"            "Tensor <empty> float32 cpu"                "$GOUT"
check "TensorShape printer"              "TensorShape [7, 11] rank=2 elems=77"       "$GOUT"
check "host values (full 1.5f)"          "1.5, 1.5, 1.5, 1.5"                        "$GOUT"
check "refuses CUDA read under gdb"      "refusing to read CUDA memory"              "$GOUT"

if [ -x "$CUDA_GDB" ]; then
    echo "== cuda-gdb"
    COUT="$WORK/cudagdb.txt"
    run_dbg "$CUDA_GDB" "$COUT" \
        -ex "print cuda_f32" \
        -ex "lfs-tensor cuda_f32 6" -ex "lfs-tensor cpu_rank3 4" -ex "lfs-tensor cpu_i32 4"
    check "cuda summary"                 "Tensor float32[3, 256, 256] cuda"          "$COUT"
    check "device values via @global"    "1, 1, 1, 1, 1, 1"                          "$COUT"
    check "host values still work"       "1.5, 1.5, 1.5, 1.5"                        "$COUT"
    check "int32 zeros"                  "0, 0, 0, 0"                                "$COUT"
    check "no bogus refusal in cuda-gdb" "on cuda, 196608 elems"                     "$COUT"
    if grep -qF "refusing to read CUDA memory" "$COUT"; then
        echo "  FAIL  cuda-gdb should not refuse"
        fail=1
    fi
else
    echo "== cuda-gdb not found at $CUDA_GDB — skipping device checks"
fi

echo
[ "$fail" -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit "$fail"
