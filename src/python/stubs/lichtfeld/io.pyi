"""File I/O operations"""

from collections.abc import Callable
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

class ProjectCommitKind(enum.Enum):
    EXPLICIT = 1

    AUTOSAVE = 2

    RECOVERED = 3

    COMPACTION = 4

    CONTENTS = 5

class ProjectRowKind(enum.Enum):
    LIVE = 0

    TOMBSTONE = 1

    SIDECAR_BASE_REFERENCE = 2

class ProjectCompression(enum.Enum):
    STORED = 0

    ZSTD_FRAMED = 1

    BYTE_SHUFFLE_ZSTD_FRAMED = 2

class ProjectVersion:
    def __init__(self) -> None: ...

    @property
    def major(self) -> int: ...

    @property
    def minor(self) -> int: ...

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

class ProjectOpenClassification:
    @property
    def state(self) -> ProjectOpenState: ...

    @property
    def generation(self) -> int: ...

    @property
    def diagnostic(self) -> str: ...

class ProjectStorageStats:
    @property
    def physical_bytes(self) -> int: ...

    @property
    def estimated_live_bytes(self) -> int: ...

    @property
    def dead_bytes(self) -> int: ...

    @property
    def dead_ratio(self) -> float: ...

class ProjectLicense:
    def __init__(self) -> None: ...

    @property
    def identifier(self) -> str: ...

    @identifier.setter
    def identifier(self, arg: str, /) -> None: ...

    @property
    def notice(self) -> str: ...

    @notice.setter
    def notice(self, arg: str, /) -> None: ...

class ProjectInspectorCard:
    @property
    def path(self) -> pathlib.Path: ...

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
    def validation_scope(self) -> str: ...

    @property
    def has_preview(self) -> bool: ...

    @property
    def preview_bytes(self) -> int: ...

    @property
    def title(self) -> str | None: ...

    @property
    def min_reader_version(self) -> ProjectVersion: ...

    @property
    def min_safe_writer_version(self) -> ProjectVersion: ...

    @property
    def commit_kind(self) -> ProjectCommitKind: ...

    @property
    def diagnostic(self) -> str: ...

class ProjectThumbnailSourceAvailability:
    @property
    def first_dataset_image(self) -> bool: ...

    @property
    def first_embedded_image(self) -> bool: ...

class ProjectInspectorSave:
    @property
    def sequence(self) -> int: ...

    @property
    def generation(self) -> int: ...

    @property
    def kind(self) -> ProjectCommitKind: ...

    @property
    def saved_at_unix_ns(self) -> int: ...

    @property
    def bytes_added(self) -> int: ...

    @property
    def holds_checkpoint(self) -> bool: ...

    @property
    def checkpoint_iteration(self) -> int | None: ...

    @property
    def planned_iterations(self) -> int | None: ...

    @property
    def strategy(self) -> str: ...

    @property
    def gaussians(self) -> int | None: ...

    @property
    def operation(self) -> str: ...

    @property
    def source_save_generation(self) -> int: ...

    @property
    def source_saved_at_unix_ns(self) -> int: ...

    @property
    def source_save_kind(self) -> ProjectCommitKind: ...

class ProjectInspectorChapter:
    @property
    def fourcc(self) -> str: ...

    @property
    def instance_uuid(self) -> str: ...

    @property
    def row_kind(self) -> ProjectRowKind: ...

    @property
    def compression(self) -> ProjectCompression: ...

    @property
    def stored_bytes(self) -> int: ...

    @property
    def uncompressed_bytes(self) -> int: ...

    @property
    def source_generation(self) -> int: ...

class ProjectInspectorCheckpoint:
    @property
    def instance_uuid(self) -> str: ...

    @property
    def source_generation(self) -> int: ...

    @property
    def iteration(self) -> int: ...

    @property
    def gaussians(self) -> int: ...

    @property
    def sh_degree(self) -> int: ...

    @property
    def binds_scene_graph(self) -> bool: ...

    @property
    def header_reachable(self) -> bool: ...

    @property
    def retained(self) -> bool: ...

class ProjectInspectorSceneGraph:
    @property
    def node_counts_by_type(self) -> dict[str, int]: ...

    @property
    def dataset_node_name(self) -> str: ...

    @property
    def training_node_id(self) -> str | None: ...

class ProjectInspectorParameters:
    @property
    def active_strategy(self) -> str: ...

    @property
    def planned_iterations(self) -> int | None: ...

    @property
    def embedded_dataset_present(self) -> bool: ...

    @property
    def embedded_dataset_complete(self) -> bool: ...

    @property
    def embedded_images(self) -> int: ...

    @property
    def embedded_normals(self) -> int: ...

    @property
    def embedded_sparse(self) -> int: ...

class ProjectInspectorReference:
    @property
    def key(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def path(self) -> pathlib.Path: ...

    @property
    def reachable(self) -> bool: ...

class ProjectMetricHistorySample:
    @property
    def iteration(self) -> int: ...

    @property
    def value(self) -> float: ...

class ProjectLastEvaluationMetrics:
    @property
    def iteration(self) -> int: ...

    @property
    def psnr(self) -> float: ...

    @property
    def ssim(self) -> float: ...

class ProjectInspectorMetrics:
    @property
    def loss_samples(self) -> int: ...

    @property
    def psnr_samples(self) -> int: ...

    @property
    def last_loss(self) -> ProjectMetricHistorySample | None: ...

    @property
    def last_psnr(self) -> ProjectMetricHistorySample | None: ...

    @property
    def last_evaluation(self) -> ProjectLastEvaluationMetrics | None: ...

class ProjectInspectorDetails:
    @property
    def card(self) -> ProjectInspectorCard: ...

    @property
    def storage(self) -> ProjectStorageStats: ...

    @property
    def save_history(self) -> list[ProjectInspectorSave]: ...

    @property
    def chapters(self) -> list[ProjectInspectorChapter]: ...

    @property
    def manifest(self) -> dict[str, str]: ...

    @property
    def license(self) -> ProjectLicense | None: ...

    @property
    def scene_graph(self) -> ProjectInspectorSceneGraph: ...

    @property
    def parameters(self) -> ProjectInspectorParameters: ...

    @property
    def retained_checkpoints(self) -> list[ProjectInspectorCheckpoint]: ...

    @property
    def references(self) -> list[ProjectInspectorReference]: ...

    @property
    def metrics(self) -> ProjectInspectorMetrics: ...

    @property
    def autosave_sidecar_present(self) -> bool: ...

    @property
    def chapters_requiring_full_read(self) -> list[str]: ...

class ProjectVerificationStatus(enum.Enum):
    VERIFIED = 0

    CANCELED = 1

    FAILED = 2

class ProjectVerificationResult:
    @property
    def status(self) -> ProjectVerificationStatus: ...

    @property
    def verified_chunks(self) -> int: ...

    @property
    def first_mismatch(self) -> str | None: ...

class ProjectReduceCheckpoint:
    @property
    def instance_uuid(self) -> str: ...

    @property
    def iteration(self) -> int: ...

    @property
    def bytes(self) -> int: ...

    @property
    def scng_bound(self) -> bool: ...

class ProjectReduceDatasetPayload:
    @property
    def chunk_uuid(self) -> str: ...

    @property
    def rel_path(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def bytes(self) -> int: ...

    @property
    def external_replacement_validated(self) -> bool: ...

class ProjectReduceProjection:
    @property
    def enabled(self) -> bool: ...

    @property
    def allowed(self) -> bool: ...

    @property
    def reclaimable_bytes(self) -> int: ...

    @property
    def projected_size(self) -> int: ...

class ProjectReducePlan:
    @property
    def path(self) -> pathlib.Path: ...

    @property
    def input_commit_uuid(self) -> str: ...

    @property
    def physical_size(self) -> int: ...

    @property
    def tombstone_bytes(self) -> int: ...

    @property
    def superseded_rows(self) -> int: ...

    @property
    def retained_checkpoints(self) -> list[ProjectReduceCheckpoint]: ...

    @property
    def embedded_dataset(self) -> list[ProjectReduceDatasetPayload]: ...

    @property
    def drop_checkpoints(self) -> ProjectReduceProjection: ...

    @property
    def drop_embedded_dataset(self) -> ProjectReduceProjection: ...

    @property
    def compact(self) -> ProjectReduceProjection: ...

class ProjectReduceResult:
    @property
    def card(self) -> ProjectInspectorCard: ...

    @property
    def checkpoints_removed(self) -> int: ...

    @property
    def dataset_images_removed(self) -> int: ...

    @property
    def dataset_normals_removed(self) -> int: ...

    @property
    def dataset_sparse_removed(self) -> int: ...

    @property
    def bytes_reclaimed(self) -> int: ...

    @property
    def recovery_copy(self) -> pathlib.Path: ...

class DatasetEmbedResult:
    @property
    def card(self) -> ProjectInspectorCard: ...

    @property
    def images_embedded(self) -> int: ...

    @property
    def normals_embedded(self) -> int: ...

    @property
    def sparse_embedded(self) -> int: ...

    @property
    def bytes_embedded(self) -> int: ...

class DatasetReferenceResult:
    @property
    def card(self) -> ProjectInspectorCard: ...

    @property
    def reference_uuid(self) -> str: ...

    @property
    def content_replaced(self) -> bool: ...

class ProjectExportFormat(enum.Enum):
    PLY = 0

    SOG = 1

    SSOG = 2

    SPZ = 3

class ProjectExportResult:
    @property
    def destination(self) -> pathlib.Path: ...

    @property
    def format(self) -> ProjectExportFormat: ...

    @property
    def gaussian_count(self) -> int: ...

    @property
    def bytes_written(self) -> int: ...

class ProjectRepairResult:
    @property
    def card(self) -> ProjectInspectorCard: ...

    @property
    def saves_recovered(self) -> int: ...

def classify_project(path: str | os.PathLike) -> ProjectOpenClassification:
    """Classify a .licht path without throwing for damaged heads."""

def project_storage_stats(path: str | os.PathLike) -> ProjectStorageStats: ...

def run_project_operation(path: str | os.PathLike, project_uuid: str, commit_uuid: str, operation: Callable) -> None: ...

def backup_project_file(path: str | os.PathLike) -> pathlib.Path: ...

def restore_project_backup(path: str | os.PathLike, backup: str | os.PathLike, project_uuid: str, commit_uuid: str) -> None: ...

def inspect_project_card(path: str | os.PathLike) -> ProjectInspectorCard: ...

def inspect_project_details(path: str | os.PathLike, checkpoint_byte_budget: int = 8388608) -> ProjectInspectorDetails: ...

def read_preview(path: str | os.PathLike) -> bytes: ...

def restore_save(path: str | os.PathLike, generation: int, destination: str | os.PathLike) -> ProjectInspectorCard: ...

def rebind_checkpoint(path: str | os.PathLike, checkpoint_instance_uuid: str) -> ProjectInspectorCard: ...

def compact_project_file(path: str | os.PathLike, progress: object | None = None, cancel: object | None = None) -> ProjectInspectorCard: ...

def plan_reduce_size(path: str | os.PathLike) -> ProjectReducePlan: ...

def undo_contents_removal(path: str | os.PathLike, id: str) -> ProjectInspectorCard: ...

def reduce_size(path: str | os.PathLike, options: dict = {}, progress: object | None = None, cancel: object | None = None) -> ProjectReduceResult: ...

def embed_dataset_file(path: str | os.PathLike, progress: object | None = None, cancel: object | None = None) -> DatasetEmbedResult: ...

def set_dataset_reference(path: str | os.PathLike, dataset_dir: str | os.PathLike, accept_content_change: bool = False) -> DatasetReferenceResult: ...

def export_project_as(path: str | os.PathLike, format: str, destination: str | os.PathLike, progress: object | None = None, cancel: object | None = None) -> ProjectExportResult: ...

def repair_project(path: str | os.PathLike, destination: str | os.PathLike, expected_project: str = '') -> ProjectRepairResult: ...

def verify_project_file(path: str | os.PathLike, progress: object | None = None, cancel: object | None = None) -> ProjectVerificationResult: ...

def set_project_preview(path: str | os.PathLike, png_bytes: bytes) -> ProjectInspectorCard: ...

def inspect_project_thumbnail_sources(path: str | os.PathLike) -> ProjectThumbnailSourceAvailability: ...

def preview_from_first_dataset_image(path: str | os.PathLike) -> ProjectInspectorCard: ...

def preview_from_first_embedded_image(path: str | os.PathLike) -> ProjectInspectorCard: ...

def preview_from_image_file(project_path: str | os.PathLike, image_path: str | os.PathLike) -> ProjectInspectorCard: ...

def set_project_license(path: str | os.PathLike, identifier: str, notice: str = '') -> ProjectInspectorCard: ...

def clear_project_license(path: str | os.PathLike) -> ProjectInspectorCard: ...

def set_project_title(path: str | os.PathLike, title: str) -> ProjectInspectorCard: ...

def inspect_project(path: str | os.PathLike, resolve_preview_fallback: bool = True) -> ProjectInspection:
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

def save_ssog(splat: lichtfeld.scene.SplatData, path: str | os.PathLike, lod_levels: int = 4, lod_ratio: float = 0.5, chunk_count_k: int = 512, chunk_extent: float = 16.0, chunk_min_k: int = 8, kmeans_iterations: int = 10, use_gpu: bool = True, progress: object | None = None, include_provenance: bool = True) -> None:
    """
    Save splat data as a PlayCanvas multi-LOD SSOG (.ssog, lod-meta.json). include_provenance (default true) writes a full provenance stamp; when false, a minimal build stamp is still embedded.
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

def is_ssog_path(path: str | os.PathLike) -> bool:
    """Check for an SSOG bundle, manifest or directory."""

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
