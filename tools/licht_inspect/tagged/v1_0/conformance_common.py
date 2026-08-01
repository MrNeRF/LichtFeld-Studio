"""Shared primitives for the spec-authoritative .licht conformance battery."""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
import signal
import time
import uuid
from typing import Iterator

try:
    from .licht_inspect import (
        FormatError,
        UnsupportedNewer,
        classify_open,
        open_container,
    )
except ImportError:  # Direct script execution.
    from licht_inspect import (
        FormatError,
        UnsupportedNewer,
        classify_open,
        open_container,
    )


DEFAULT_SEED = 0x4C494348545F434F
TERMINAL_OUTCOMES = frozenset(
    {"hard_fail", "repair_only", "unsupported_newer"}
)


class CaseTimeout(RuntimeError):
    """One parser invocation exceeded the battery's per-case deadline."""


@dataclasses.dataclass(frozen=True)
class ConformanceConfig:
    mode: str
    seed: int
    exhaustive_truncation: bool
    lifecycle_sequences: int
    lifecycle_steps: int
    scale_rows: int
    scale_generations: int
    fuzz_cases: int
    per_case_timeout_seconds: float
    independent_verifier: bool


@dataclasses.dataclass(frozen=True)
class CategoryResult:
    name: str
    cases: int
    seconds: float
    detail: str


def config_for_mode(mode: str, seed: int) -> ConformanceConfig:
    if mode == "quick":
        return ConformanceConfig(
            mode=mode,
            seed=seed,
            exhaustive_truncation=False,
            lifecycle_sequences=4,
            lifecycle_steps=18,
            scale_rows=1_000,
            scale_generations=120,
            fuzz_cases=750,
            per_case_timeout_seconds=0.5,
            independent_verifier=False,
        )
    if mode == "full":
        return ConformanceConfig(
            mode=mode,
            seed=seed,
            exhaustive_truncation=True,
            lifecycle_sequences=16,
            lifecycle_steps=48,
            scale_rows=5_000,
            scale_generations=320,
            fuzz_cases=10_000,
            per_case_timeout_seconds=1.0,
            independent_verifier=True,
        )
    raise ValueError(f"unknown conformance mode {mode!r}")


@contextlib.contextmanager
def case_deadline(seconds: float) -> Iterator[None]:
    """Bound one parser call without spawning a process per fuzz case."""

    if not hasattr(signal, "setitimer"):
        yield
        return

    def _expired(_signum: int, _frame: object) -> None:
        raise CaseTimeout(f"parser call exceeded {seconds:.3f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def classified_outcome(path: Path, timeout_seconds: float) -> tuple[str, str]:
    with case_deadline(timeout_seconds):
        outcome, detail = classify_open(path)
    if outcome not in TERMINAL_OUTCOMES and not outcome.startswith("open_gen_"):
        raise AssertionError(f"unclassified parser result {outcome!r}: {detail}")
    return outcome, detail


def assert_outcome(
    path: Path,
    expected: str,
    *,
    timeout_seconds: float,
    recovery_warning: bool | None = None,
) -> None:
    actual, detail = classified_outcome(path, timeout_seconds)
    if actual != expected:
        raise AssertionError(
            f"{path.name}: spec oracle expected {expected}, parser returned "
            f"{actual}: {detail}"
        )
    if recovery_warning is None or not actual.startswith("open_gen_"):
        return
    with case_deadline(timeout_seconds):
        container = open_container(path)
    has_warning = any("recovery warning" in item for item in container.warnings)
    if has_warning != recovery_warning:
        raise AssertionError(
            f"{path.name}: recovery_warning expected {recovery_warning}, "
            f"got {has_warning}: {container.warnings}"
        )


def inspectable_container(path: Path, timeout_seconds: float):
    """Return normal or unsupported-newer inspection state."""

    with case_deadline(timeout_seconds):
        try:
            return open_container(path)
        except UnsupportedNewer as error:
            return error.inspection


def deterministic_uuid(namespace: int, value: int) -> uuid.UUID:
    """RFC-4122-shaped identity derived only from test inputs."""

    tag = ((namespace & 0xFFFF) << 112) | (value & ((1 << 112) - 1))
    raw = bytearray(tag.to_bytes(16, "big"))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return uuid.UUID(bytes=bytes(raw))


def timed_result(name: str, started: float, cases: int, detail: str) -> CategoryResult:
    return CategoryResult(name, cases, time.monotonic() - started, detail)
