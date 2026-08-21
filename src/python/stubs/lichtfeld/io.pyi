"""File I/O operations"""

import enum
import os

import lichtfeld
import lichtfeld.scene


class Hash128:
    def __init__(self) -> None: ...

    @staticmethod
    def from_hex(value: str) -> Hash128: ...

    def to_hex(self) -> str: ...


class LocatorBase(enum.Enum):
    PROJECT = 0
    DATASET = 1
    ABSOLUTE = 2
    SEARCH_ROOT = 3


class ReferenceLocator:
    def __init__(self) -> None: ...

    preferred: str
    base: LocatorBase
    absolute_fallback: str | None


class FingerprintKind(enum.Enum):
    FILE = 0
    DIRECTORY = 1


class ReferenceFingerprint:
    def __init__(self) -> None: ...

    kind: FingerprintKind
    size: int
    mtime_unix_ns: int
    head_xxh3: Hash128
    tail_xxh3: Hash128
    full_xxh3: Hash128 | None


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

    uuid: str
    key: str
    kind: str
    locator: ReferenceLocator
    fingerprint: ReferenceFingerprint
    unresolved: bool


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


def asset_library_dir() -> os.PathLike:
    """Return the canonical user Asset Manager storage directory."""


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
