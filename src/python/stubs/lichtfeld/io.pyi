import os

import lichtfeld
import lichtfeld.scene


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

def load(path: str | os.PathLike, format: str | None = None, resize_factor: int | None = None, max_width: int | None = None, images_folder: str | None = None, progress: object | None = None) -> LoadResult:
    """Load a scene or splat file from path"""

def load_point_cloud(path: str | os.PathLike) -> tuple:
    """Load a PLY as point cloud, returns (means [N,3], colors [N,3]) tensors"""

def save_ply(data: lichtfeld.scene.SplatData, path: str | os.PathLike, binary: bool = True, progress: object | None = None) -> None:
    """Save splat data as PLY file"""

def save_sog(data: lichtfeld.scene.SplatData, path: str | os.PathLike, kmeans_iterations: int = 10, use_gpu: bool = True, progress: object | None = None) -> None:
    """Save splat data as SOG compressed file"""

def save_spz(data: lichtfeld.scene.SplatData, path: str | os.PathLike) -> None:
    """Save splat data as SPZ compressed file"""

def export_html(data: lichtfeld.scene.SplatData, path: str | os.PathLike, kmeans_iterations: int = 10, progress: object | None = None) -> None:
    """Export splat data as self-contained HTML viewer"""

def is_dataset_path(path: str | os.PathLike) -> bool:
    """Check if path is a dataset directory"""

def get_supported_formats() -> list[str]:
    """Get list of supported file format names"""

def get_supported_extensions() -> list[str]:
    """Get list of supported file extensions"""

def save_image(path: str | os.PathLike, image: lichtfeld.Tensor) -> None:
    """
    Save image tensor to file (PNG, JPG, TIFF, EXR). Accepts [H,W,C] or [C,H,W] float [0,1].
    """
