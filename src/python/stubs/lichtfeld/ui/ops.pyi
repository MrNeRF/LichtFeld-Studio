"""Operator invocation"""

import lichtfeld


def invoke(arg0: str, /, **kwargs) -> lichtfeld.OperatorReturnValue:
    """Invoke an operator by id with optional properties"""

def poll(id: str) -> bool:
    """Check if operator can run"""

def cancel_modal() -> None:
    """Cancel any active modal operator"""

def has_modal() -> bool:
    """Check if a modal operator is running"""
