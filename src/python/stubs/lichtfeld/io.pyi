"""File I/O operations"""

import enum
import os
import pathlib

import lichtfeld
import lichtfeld.scene


class Hash128:
    def __init__(self) -> None: ...

    @staticmethod
    def from_hex(value: str) -> Hash128: ...

    def to_hex(self) -> str: ...

    def __str__(self) -> str: ...

class LocatorBase(enum.Enum):
    PROJECT = 0

    DATASET = 1

    ABSOLUTE = 2

    SEARCH_ROOT = 3

class ReferenceLocator:
    def __init__(self) -> None: ...

    @property
    def preferred(self) -> str: ...

    @preferred.setter
    def preferred(self, arg: str, /) -> None: ...

    @property
    def base(self) -> LocatorBase: ...

    @base.setter
    def base(self, arg: LocatorBase, /) -> None: ...

    @property
    def absolute_fallback(self) -> str | None: ...

    @absolute_fallback.setter
    def absolute_fallback(self, arg: str | None) -> None: ...

class FingerprintKind(enum.Enum):
    FILE = 0

    DIRECTORY = 1

class ReferenceFingerprint:
    def __init__(self) -> None: ...

    @property
    def kind(self) -> FingerprintKind: ...

    @kind.setter
    def kind(self, arg: FingerprintKind, /) -> None: ...

    @property
    def size(self) -> int: ...

    @size.setter
    def size(self, arg: int, /) -> None: ...

    @property
    def mtime_unix_ns(self) -> int: ...

    @mtime_unix_ns.setter
    def mtime_unix_ns(self, arg: int, /) -> None: ...

    @property
    def head_xxh3(self) -> Hash128: ...

    @head_xxh3.setter
    def head_xxh3(self, arg: Hash128, /) -> None: ...

    @property
    def tail_xxh3(self) -> Hash128: ...

    @tail_xxh3.setter
    def tail_xxh3(self, arg: Hash128, /) -> None: ...

    @property
    def full_xxh3(self) -> Hash128 | None: ...

    @full_xxh3.setter
    def full_xxh3(self, arg: Hash128 | None) -> None: ...

class FingerprintDisposition(enum.Enum):
    MATCH_FAST_PATH = 0

    MATCH_MTIME_REFRESHED = 1

    MISSING = 2

    CONTENT_MISMATCH = 3

    TYPE_MISMATCH = 4

class FingerprintCheck:
    @property
    def disposition(self) -> FingerprintDisposition: ...

    @property
    def observed(self) -> ReferenceFingerprint | None: ...

    @property
    def diagnostic(self) -> str: ...

    @property
    def matches(self) -> bool: ...

class ReferenceRecord:
    def __init__(self) -> None: ...

    @property
    def uuid(self) -> str: ...

    @uuid.setter
    def uuid(self, arg: str, /) -> None: ...

    @property
    def key(self) -> str: ...

    @key.setter
    def key(self, arg: str, /) -> None: ...

    @property
    def kind(self) -> str: ...

    @kind.setter
    def kind(self, arg: str, /) -> None: ...

    @property
    def locator(self) -> ReferenceLocator: ...

    @locator.setter
    def locator(self, arg: ReferenceLocator, /) -> None: ...

    @property
    def fingerprint(self) -> ReferenceFingerprint: ...

    @fingerprint.setter
    def fingerprint(self, arg: ReferenceFingerprint, /) -> None: ...

    @property
    def unresolved(self) -> bool: ...

    @unresolved.setter
    def unresolved(self, arg: bool, /) -> None: ...

class ReferencesChapter:
    def __init__(self) -> None: ...

    @staticmethod
    def parse(value: str) -> ReferencesChapter: ...

    def to_json(self) -> str: ...

    def records(self) -> list[ReferenceRecord]: ...

    def find(self, uuid: str) -> ReferenceRecord | None: ...

    def upsert(self, record: ReferenceRecord) -> None: ...

    def remove(self, uuid: str) -> bool: ...

    def verify_and_refresh(self, uuid: str, path: str | os.PathLike) -> FingerprintCheck: ...

    def relink(self, uuid: str, locator: ReferenceLocator, path: str | os.PathLike, accept_content_change: bool = False) -> None: ...

def fingerprint_path(path: str | os.PathLike, include_full_hash: bool = False) -> ReferenceFingerprint:
    """Fingerprint a file or directory for durable content identity."""

def check_fingerprint(path: str | os.PathLike, expected: ReferenceFingerprint) -> FingerprintCheck:
    """Compare a path with a previously stored content fingerprint."""

def asset_library_dir() -> pathlib.Path:
    """Return the canonical user Asset Manager storage directory."""

class ProjectContainerRole(enum.Enum):
    MASTER = 0

    AUTOSAVE_SIDECAR = 1

class ProjectOpenState(enum.Enum):
    OPEN = 0

    UNSUPPORTED_NEWER = 1

    REPAIR_ONLY = 2

    HARD_FAIL = 3

class ProjectInspection:
    @property
    def project_uuid(self) -> str: ...

    @property
    def file_uuid(self) -> str: ...

    @property
    def commit_uuid(self) -> str: ...

    @property
    def generation(self) -> int: ...

    @property
    def created_at_unix_ns(self) -> int: ...

    @property
    def saved_at_unix_ns(self) -> int: ...

    @property
    def physical_file_size(self) -> int: ...

    @property
    def role(self) -> ProjectContainerRole: ...

    @property
    def open_state(self) -> ProjectOpenState: ...

    @property
    def has_preview(self) -> bool: ...

    @property
    def fallback_preview_path(self) -> str: ...

def inspect_project(path: str | os.PathLike) -> ProjectInspection:
    """
    Inspect validated .licht container metadata without reading project payloads.
    """

class LoadResult:
    @property
    def splat_data(self) -> lichtfeld.scene.SplatData | None:
        """Loaded splat data, or None"""

    @property
    def scene_center(self) -> lichtfeld.Tensor:
        """Scene center [3] tensor"""

    @property
    def loader_used(self) -> str:
        """Name of loader that was used"""

    @property
    def load_time_ms(self) -> int:
        """Load time in milliseconds"""

    @property
    def warnings(self) -> list[str]:
        """List of warning messages from loading"""

    @property
    def cameras(self) -> lichtfeld.scene.CameraDataset | None:
        """Camera dataset, or None"""

    @property
    def point_cloud(self) -> lichtfeld.scene.PointCloud | None:
        """Point cloud, or None"""

    @property
    def is_dataset(self) -> bool:
        """Whether loaded data is a dataset with cameras"""

def load(path: str | os.PathLike, format: str | None = None, resize_factor: int | None = None, max_width: int | None = None, images_folder: str | None = None, progress: object | None = None, min_track_length: int | None = None) -> LoadResult:
    """Load a scene or splat file from path"""

def load_point_cloud(path: str | os.PathLike) -> tuple:
    """Load a PLY as point cloud, returns (means [N,3], colors [N,3]) tensors"""

def save_ply(data: lichtfeld.scene.SplatData, path: str | os.PathLike, binary: bool = True, progress: object | None = None, extra_attributes: object | None = None, include_provenance: bool = True) -> None:
    """
    Save splat data as PLY file with optional extra per-vertex float attributes. include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.
    """

def save_point_cloud_ply(point_cloud: lichtfeld.scene.PointCloud, path: str | os.PathLike, extra_attributes: object | None = None, include_provenance: bool = True) -> None:
    """
    Save a point cloud as PLY file (xyz + colors) with optional extra per-vertex float attributes. include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.
    """

def save_sog(data: lichtfeld.scene.SplatData, path: str | os.PathLike, kmeans_iterations: int = 10, use_gpu: bool = True, progress: object | None = None, include_provenance: bool = True) -> None:
    """
    Save splat data as SOG compressed file. include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.
    """

def save_spz(data: lichtfeld.scene.SplatData, path: str | os.PathLike, version: int = 4, include_provenance: bool = True) -> None:
    """
    Save splat data as SPZ compressed file.

    version: SPZ container version, 4 (zstd, default) or 3 (legacy gzip).
    include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded. Ignored for SPZ v3.
    """

def save_usd(data: lichtfeld.scene.SplatData, path: str | os.PathLike, include_provenance: bool = True) -> None:
    """
    Save splat data as OpenUSD gaussian file. include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.
    """

def save_nurec_usdz(data: lichtfeld.scene.SplatData, path: str | os.PathLike, include_provenance: bool = True) -> None:
    """
    Save splat data as NuRec USDZ compatible with PLY_to_USD / Omniverse. include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.
    """

def export_html(data: lichtfeld.scene.SplatData, path: str | os.PathLike, kmeans_iterations: int = 10, progress: object | None = None, include_provenance: bool = True) -> None:
    """
    Export splat data as self-contained HTML viewer. include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.
    """

def is_dataset_path(path: str | os.PathLike) -> bool:
    """Check if path is a dataset directory"""

def is_gaussian_splat_ply(path: str | os.PathLike) -> bool:
    """
    Check if PLY file is a 3D Gaussian splat (has opacity, scale_0, rot_0 properties)
    """

def get_supported_formats() -> list[str]:
    """Get list of supported file format names"""

def get_supported_extensions() -> list[str]:
    """Get list of supported file extensions"""

def save_image(path: str | os.PathLike, image: lichtfeld.Tensor, include_provenance: bool = True) -> None:
    """
    Save image tensor to file (PNG, JPG, TIFF, EXR). Accepts [H,W,C] or [C,H,W] float [0,1]. include_provenance (default true) writes a full Comment stamp on PNG and JPEG; when false, a minimal build stamp is still embedded.
    """
