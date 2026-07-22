#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
g++ -std=c++20 -O2 -Wall -Wextra -o "$ROOT/os_semantics" "$ROOT/os_semantics.cpp"
echo "built $ROOT/os_semantics"
