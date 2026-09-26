"""Pin a project locator and its on-disk identity across delayed writes."""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ProjectPathIdentity:
    path: Path
    canonical_path: Path
    identity: tuple | None

    @classmethod
    def capture(cls, path):
        path = Path(path).expanduser().absolute()
        canonical = path.resolve()
        try:
            with path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                # The superblock contains the project UUID and file UUID.
                identity = (stat.st_dev, stat.st_ino, stream.read(256))
        except FileNotFoundError:
            identity = None
        return cls(path, canonical, identity)

    def validate(self):
        current = self.capture(self.path)
        if (current.canonical_path, current.identity) != (self.canonical_path, self.identity):
            raise ValueError(f"The project identity or path changed at {self.path}. Refresh Projects and try again.")
