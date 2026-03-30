"""Compositional operations system"""

from typing import overload

from . import edit as edit, select as select, transform as transform


class Stage:
    def __or__(self, arg: Stage, /) -> Pipeline:
        """Chain two stages into a pipeline"""

    def execute(self) -> dict:
        """Execute this stage immediately"""

class Pipeline:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, name: str) -> None:
        """Create a named pipeline"""

    def add(self, arg: Stage, /) -> Pipeline:
        """Append a stage to the pipeline"""

    def __or__(self, arg: Stage, /) -> Pipeline:
        """Append a stage via pipe operator"""

    def execute(self) -> dict:
        """Execute all stages and return result dict"""

    def poll(self) -> bool:
        """Check if all stages can execute"""
