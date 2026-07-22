"""Independent read-only tooling for the LichtFeld .licht container."""

from .licht_inspect import (
    Container,
    FormatError,
    HardFailure,
    RepairRequired,
    UnsupportedNewer,
    WriteCompatibility,
    WriteRefused,
    assess_write_compatibility,
    classify_open,
    evaluate_recovery,
    open_container,
    require_safe_write,
)

__all__ = [
    "Container",
    "FormatError",
    "HardFailure",
    "RepairRequired",
    "UnsupportedNewer",
    "WriteCompatibility",
    "WriteRefused",
    "assess_write_compatibility",
    "classify_open",
    "evaluate_recovery",
    "open_container",
    "require_safe_write",
]
