from collections.abc import Sequence
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import lichtfeld


class SplatData:
    @property
    def means_raw(self) -> lichtfeld.Tensor:
        """Raw means tensor [N, 3] (view)"""

    @property
    def sh0_raw(self) -> lichtfeld.Tensor:
        """Raw SH0 tensor [N, 1, 3] (view)"""

    @property
    def shN_raw(self) -> lichtfeld.Tensor:
        """Raw SHN tensor [N, (degree+1)^2-1, 3] (view)"""

    @property
    def scaling_raw(self) -> lichtfeld.Tensor:
        """Raw scaling tensor [N, 3] (log-space, view)"""

    @property
    def rotation_raw(self) -> lichtfeld.Tensor:
        """Raw rotation tensor [N, 4] (quaternions, view)"""

    @property
    def opacity_raw(self) -> lichtfeld.Tensor:
        """Raw opacity tensor [N, 1] (logit-space, view)"""

    def get_means(self) -> lichtfeld.Tensor:
        """Get means (same as means_raw for now)"""

    def get_opacity(self) -> lichtfeld.Tensor:
        """Get opacity with sigmoid applied"""

    def get_rotation(self) -> lichtfeld.Tensor:
        """Get normalized rotation quaternions"""

    def get_scaling(self) -> lichtfeld.Tensor:
        """Get scaling with exp applied"""

    def get_shs(self) -> lichtfeld.Tensor:
        """Get concatenated SH coefficients (sh0 + shN)"""

    def get_colors_rgb(self) -> lichtfeld.Tensor:
        """Get RGB colors [N, 3] in [0,1] range (handles SH0 encoding)"""

    def set_colors_rgb(self, colors: lichtfeld.Tensor) -> None:
        """
        Set RGB colors from [N, 3] tensor in [0,1] range (handles SH0 encoding)
        """

    @property
    def active_sh_degree(self) -> int:
        """Current active SH degree"""

    @property
    def max_sh_degree(self) -> int:
        """Maximum SH degree"""

    @property
    def scene_scale(self) -> float:
        """Scene scale factor"""

    @property
    def num_points(self) -> int:
        """Number of Gaussians"""

    @property
    def deleted(self) -> lichtfeld.Tensor:
        """Deletion mask tensor [N] (bool)"""

    def has_deleted_mask(self) -> bool:
        """Check if deletion mask exists"""

    def visible_count(self) -> int:
        """Number of visible (non-deleted) Gaussians"""

    def soft_delete(self, mask: lichtfeld.Tensor) -> lichtfeld.Tensor:
        """Mark Gaussians as deleted by mask, returns previous state for undo"""

    def undelete(self, mask: lichtfeld.Tensor) -> None:
        """Restore deleted Gaussians by mask"""

    def clear_deleted(self) -> None:
        """Clear all soft-deleted flags"""

    def apply_deleted(self) -> int:
        """Permanently remove deleted Gaussians, returns count removed"""

    def increment_sh_degree(self) -> None:
        """Increment active SH degree by 1"""

    def set_active_sh_degree(self, degree: int) -> None:
        """Set active SH degree"""

    def set_max_sh_degree(self, degree: int) -> None:
        """Set maximum SH degree"""

    def reserve_capacity(self, capacity: int) -> None:
        """Reserve capacity for Gaussians (for densification)"""

class NodeType(enum.Enum):
    SPLAT = 0

    POINTCLOUD = 1

    GROUP = 2

    CROPBOX = 3

    ELLIPSOID = 4

    DATASET = 5

    CAMERA_GROUP = 6

    CAMERA = 7

    IMAGE_GROUP = 8

    IMAGE = 9

class SelectionGroup:
    @property
    def id(self) -> int:
        """Group identifier"""

    @property
    def name(self) -> str:
        """Group display name"""

    @property
    def color(self) -> tuple[float, float, float]:
        """Group color as (r, g, b) tuple"""

    @property
    def count(self) -> int:
        """Number of selected Gaussians in this group"""

    @property
    def locked(self) -> bool:
        """Whether the group is locked from editing"""

class CropBox:
    @property
    def __property_group__(self) -> str:
        """Property group name for introspection"""

    def get(self, name: str) -> object:
        """Get property value by name"""

    def set(self, name: str, value: object) -> None:
        """Set property value by name"""

    def prop_info(self, name: str) -> dict:
        """Get metadata for a property"""

    def __getattr__(self, arg: str, /) -> object: ...

    def __setattr__(self, arg0: str, arg1: object, /) -> None: ...

    def __dir__(self) -> list:
        """List available attributes"""

class Ellipsoid:
    @property
    def __property_group__(self) -> str:
        """Property group name for introspection"""

    def get(self, name: str) -> object:
        """Get property value by name"""

    def set(self, name: str, value: object) -> None:
        """Set property value by name"""

    def prop_info(self, name: str) -> dict:
        """Get metadata for a property"""

    def __getattr__(self, arg: str, /) -> object: ...

    def __setattr__(self, arg0: str, arg1: object, /) -> None: ...

    def __dir__(self) -> list:
        """List available attributes"""

class PointCloud:
    @property
    def means(self) -> lichtfeld.Tensor:
        """Position tensor [N, 3]"""

    @property
    def colors(self) -> lichtfeld.Tensor:
        """Color tensor [N, 3]"""

    @property
    def normals(self) -> lichtfeld.Tensor | None:
        """Normal tensor [N, 3]"""

    @property
    def sh0(self) -> lichtfeld.Tensor | None:
        """Base SH coefficients [N, 1, 3]"""

    @property
    def shN(self) -> lichtfeld.Tensor | None:
        """Higher-order SH coefficients [N, K, 3]"""

    @property
    def opacity(self) -> lichtfeld.Tensor | None:
        """Opacity tensor [N, 1]"""

    @property
    def scaling(self) -> lichtfeld.Tensor | None:
        """Scaling tensor [N, 3]"""

    @property
    def rotation(self) -> lichtfeld.Tensor | None:
        """Rotation quaternion tensor [N, 4]"""

    @property
    def size(self) -> int:
        """Number of points"""

    def is_gaussian(self) -> bool:
        """Check if point cloud has Gaussian attributes"""

    @property
    def attribute_names(self) -> list[str]:
        """List of valid attribute names"""

    def normalize_colors(self) -> None:
        """Normalize color values to [0, 1] range"""

    def filter(self, keep_mask: lichtfeld.Tensor) -> int:
        """Filter points by boolean mask, returns number of points removed"""

    def filter_indices(self, indices: lichtfeld.Tensor) -> int:
        """
        Keep only points at specified indices, returns number of points removed
        """

    def set_data(self, points: lichtfeld.Tensor, colors: lichtfeld.Tensor) -> None:
        """Replace point cloud data with new points and colors tensors"""

    def set_colors(self, colors: lichtfeld.Tensor) -> None:
        """Update colors without re-uploading positions [N, 3]"""

    def set_means(self, points: lichtfeld.Tensor) -> None:
        """Update positions without re-uploading colors [N, 3]"""

class SceneNode:
    @property
    def __property_group__(self) -> str:
        """Property group name for introspection"""

    def get(self, name: str) -> object:
        """Get property value by name"""

    def set(self, name: str, value: object) -> None:
        """Set property value by name"""

    @property
    def id(self) -> int:
        """Unique node identifier"""

    @property
    def parent_id(self) -> int:
        """Parent node identifier (-1 for root)"""

    @property
    def children(self) -> list[int]:
        """List of child node IDs"""

    @property
    def type(self) -> NodeType:
        """Node type (SPLAT, GROUP, CAMERA, etc.)"""

    @property
    def world_transform(self) -> tuple:
        """World-space transform as 4x4 row-major tuple"""

    def set_local_transform(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(4, 4))], /) -> None:
        """Set local transform from a [4, 4] ndarray"""

    @property
    def gaussian_count(self) -> int:
        """Number of Gaussians owned by this node"""

    @property
    def centroid(self) -> tuple[float, float, float]:
        """Centroid position (x, y, z)"""

    def splat_data(self) -> SplatData | None:
        """Get SplatData for SPLAT nodes (None otherwise)"""

    def point_cloud(self) -> PointCloud | None:
        """Get PointCloud for POINTCLOUD nodes (None otherwise)"""

    def cropbox(self) -> CropBox | None:
        """Get CropBox for CROPBOX nodes (None otherwise)"""

    def ellipsoid(self) -> Ellipsoid | None:
        """Get Ellipsoid for ELLIPSOID nodes (None otherwise)"""

    @property
    def camera_uid(self) -> int:
        """Camera unique identifier"""

    @property
    def image_path(self) -> str:
        """Path to the camera image file"""

    @property
    def mask_path(self) -> str:
        """Path to the camera mask file"""

    @property
    def has_camera(self) -> bool:
        """Whether this node has camera data"""

    @property
    def has_mask(self) -> bool:
        """Whether this camera node has a mask file"""

    def load_mask(self, resize_factor: int = 1, max_width: int = 3840, invert: bool = False, threshold: float = 0.5) -> lichtfeld.Tensor | None:
        """
        Load mask as tensor [1, H, W] on CUDA (None if not a camera node or no mask)
        """

    @property
    def camera_R(self) -> lichtfeld.Tensor | None:
        """Camera rotation matrix [3, 3]"""

    @property
    def camera_T(self) -> lichtfeld.Tensor | None:
        """Camera translation vector [3, 1]"""

    @property
    def camera_focal_x(self) -> float | None:
        """Camera focal length in pixels (x)"""

    @property
    def camera_focal_y(self) -> float | None:
        """Camera focal length in pixels (y)"""

    @property
    def camera_width(self) -> int | None:
        """Camera image width in pixels"""

    @property
    def camera_height(self) -> int | None:
        """Camera image height in pixels"""

    def prop_info(self, name: str) -> dict:
        """Get metadata for a property"""

    def __getattr__(self, arg: str, /) -> object: ...

    def __setattr__(self, arg0: str, arg1: object, /) -> None: ...

    def __dir__(self) -> list: ...

class NodeCollectionIterator:
    def __next__(self) -> SceneNode:
        """Advance to the next node"""

class NodeCollection:
    def __len__(self) -> int:
        """Return the number of nodes"""

    def __getitem__(self, index: int) -> SceneNode:
        """Get node by index"""

    def __iter__(self) -> NodeCollectionIterator:
        """Return an iterator over all nodes"""

class Scene:
    def is_valid(self) -> bool:
        """Check if scene reference is still valid (thread-safe)"""

    @property
    def generation(self) -> int:
        """Generation counter when scene was acquired"""

    def add_group(self, name: str, parent: int = -1) -> int:
        """Add an empty group node, returns node ID"""

    def add_splat(self, name: str, means: lichtfeld.Tensor, sh0: lichtfeld.Tensor, shN: lichtfeld.Tensor, scaling: lichtfeld.Tensor, rotation: lichtfeld.Tensor, opacity: lichtfeld.Tensor, sh_degree: int = 0, scene_scale: float = 1.0, parent: int = -1) -> int:
        """
        Add a new splat node from tensor data.

        Args:
            name: Node name in scene graph
            means: Position tensor [N, 3]
            sh0: Base SH color [N, 1, 3]
            shN: Higher SH coefficients [N, K, 3] or empty
            scaling: Log-scale factors [N, 3]
            rotation: Quaternions [N, 4] (wxyz)
            opacity: Logit opacity [N, 1]
            sh_degree: SH degree (0 for RGB only)
            scene_scale: Scene scale factor
            parent: Parent node ID (-1 for root)

        Returns:
            Node ID of created splat
        """

    def add_point_cloud(self, name: str, points: lichtfeld.Tensor, colors: lichtfeld.Tensor, parent: int = -1) -> int:
        """Add a point cloud node from tensor data [N,3] positions and colors"""

    def add_camera_group(self, name: str, parent: int, camera_count: int) -> int:
        """Add a camera group node"""

    def add_camera(self, name: str, parent: int, R: lichtfeld.Tensor, T: lichtfeld.Tensor, focal_x: float, focal_y: float, width: int, height: int, image_path: str = '', uid: int = -1) -> int:
        """
        Add a camera node with intrinsic and extrinsic parameters.

        Args:
            name: Camera node name
            parent: Parent node ID (typically a camera group)
            R: Rotation matrix [3,3] (world-to-camera)
            T: Translation vector [3,1]
            focal_x: Focal length in pixels (x)
            focal_y: Focal length in pixels (y)
            width: Image width in pixels
            height: Image height in pixels
            image_path: Optional path to camera image
            uid: Optional unique identifier (-1 for auto-assigned)

        Returns:
            Node ID of created camera
        """

    def remove_node(self, name: str, keep_children: bool = False) -> None:
        """Remove a node by name, optionally keeping its children"""

    def rename_node(self, old_name: str, new_name: str) -> bool:
        """Rename a node, returns true on success"""

    def clear(self) -> None:
        """Remove all nodes from the scene"""

    def reparent(self, node_id: int, new_parent_id: int) -> None:
        """Move a node under a new parent"""

    def root_nodes(self) -> list[int]:
        """Get all root-level nodes"""

    def get_node_by_id(self, id: int) -> SceneNode | None:
        """Find a node by its integer ID (None if not found)"""

    def get_node(self, name: str) -> SceneNode | None:
        """Find a node by name (None if not found)"""

    def get_nodes(self, type: NodeType | None = None) -> list[SceneNode]:
        """Get nodes, optionally filtered by NodeType"""

    def get_visible_nodes(self) -> list[SceneNode]:
        """Get all visible nodes in the scene"""

    def is_node_effectively_visible(self, id: int) -> bool:
        """Check if a node is visible considering parent visibility"""

    def get_world_transform(self, node_id: int) -> tuple:
        """Get world-space transform as 4x4 row-major tuple"""

    @overload
    def set_node_transform(self, name: str, transform: Annotated[NDArray[numpy.float32], dict(shape=(4, 4))]) -> None:
        """Set node local transform from a [4, 4] ndarray"""

    @overload
    def set_node_transform(self, name: str, transform: lichtfeld.Tensor) -> None:
        """Set node local transform from a [4, 4] Tensor"""

    def combined_model(self) -> SplatData | None:
        """Get the merged SplatData for all visible splats (None if empty)"""

    def training_model(self) -> SplatData | None:
        """Get the SplatData used for training (None if unavailable)"""

    def set_training_model_node(self, name: str) -> None:
        """Set which node provides the training model"""

    @property
    def training_model_node_name(self) -> str:
        """Name of the node providing the training model"""

    @overload
    def get_node_bounds(self, id: int) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """
        Get axis-aligned bounding box as ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """

    @overload
    def get_node_bounds(self, name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Get axis-aligned bounding box by node name"""

    @overload
    def get_node_bounds_center(self, id: int) -> tuple[float, float, float]:
        """Get center of the node bounding box as (x, y, z)"""

    @overload
    def get_node_bounds_center(self, name: str) -> tuple[float, float, float]:
        """Get center of the node bounding box by name"""

    def get_cropbox_for_splat(self, splat_id: int) -> int:
        """Get the crop box node ID associated with a splat (-1 if none)"""

    def get_or_create_cropbox_for_splat(self, splat_id: int) -> int:
        """Get or create a crop box for a splat, returns cropbox node ID"""

    def get_cropbox_data(self, cropbox_id: int) -> CropBox | None:
        """Get CropBox data for a cropbox node (None if invalid)"""

    def set_cropbox_data(self, cropbox_id: int, data: CropBox) -> None:
        """Set CropBox data for a cropbox node"""

    @property
    def selection_mask(self) -> lichtfeld.Tensor | None:
        """Current selection mask tensor [N] uint8 (None if no selection)"""

    def set_selection(self, indices: Sequence[int]) -> None:
        """Set selection from index tensor"""

    def set_selection_mask(self, mask: lichtfeld.Tensor) -> None:
        """Set selection from boolean mask tensor [N]"""

    def clear_selection(self) -> None:
        """Clear all selected Gaussians"""

    def has_selection(self) -> bool:
        """Check if any Gaussians are selected"""

    def add_selection_group(self, name: str, color: tuple[float, float, float]) -> int:
        """Add a named selection group with (r, g, b) color, returns group ID"""

    def remove_selection_group(self, id: int) -> None:
        """Remove a selection group by ID"""

    def rename_selection_group(self, id: int, name: str) -> None:
        """Rename a selection group"""

    def set_selection_group_color(self, id: int, color: tuple[float, float, float]) -> None:
        """Set a selection group color as (r, g, b) tuple"""

    def set_selection_group_locked(self, id: int, locked: bool) -> None:
        """Lock or unlock a selection group"""

    def is_selection_group_locked(self, id: int) -> bool:
        """Check if a selection group is locked"""

    @property
    def active_selection_group(self) -> int:
        """Currently active selection group ID"""

    @active_selection_group.setter
    def active_selection_group(self, arg: int, /) -> None: ...

    def selection_groups(self) -> list[SelectionGroup]:
        """Get all selection groups"""

    def update_selection_group_counts(self) -> None:
        """Recompute selection counts for all groups"""

    def clear_selection_group(self, id: int) -> None:
        """Clear all selections in a group"""

    def reset_selection_state(self) -> None:
        """Reset all selection state to defaults"""

    def set_camera_training_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a camera for training by name"""

    @property
    def active_camera_count(self) -> int:
        """Number of cameras enabled for training"""

    def get_active_cameras(self) -> list[SceneNode]:
        """Get camera nodes enabled for training"""

    def has_training_data(self) -> bool:
        """Check if training dataset is loaded"""

    @property
    def scene_center(self) -> lichtfeld.Tensor:
        """Scene center position as a [3] tensor"""

    @property
    def node_count(self) -> int:
        """Total number of nodes in the scene"""

    @property
    def total_gaussian_count(self) -> int:
        """Total number of Gaussians across all nodes"""

    def has_nodes(self) -> bool:
        """Check if the scene contains any nodes"""

    def apply_deleted(self) -> int:
        """Permanently remove soft-deleted Gaussians from all nodes"""

    def invalidate_cache(self) -> None:
        """Invalidate the combined model cache"""

    def notify_changed(self) -> None:
        """Notify the renderer that scene data has changed"""

    def duplicate_node(self, name: str) -> str:
        """Duplicate a node by name, returns new node ID"""

    def merge_group(self, group_name: str) -> str:
        """Merge all splats in a group into a single node, returns merged node ID"""

    @property
    def nodes(self) -> NodeCollection:
        """Iterable collection of all scene nodes"""

class Camera:
    @property
    def focal_x(self) -> float:
        """Focal length X in pixels"""

    @property
    def focal_y(self) -> float:
        """Focal length Y in pixels"""

    @property
    def center_x(self) -> float:
        """Principal point X in pixels"""

    @property
    def center_y(self) -> float:
        """Principal point Y in pixels"""

    @property
    def fov_x(self) -> float:
        """Horizontal field of view in radians"""

    @property
    def fov_y(self) -> float:
        """Vertical field of view in radians"""

    @property
    def image_width(self) -> int:
        """Image width in pixels"""

    @property
    def image_height(self) -> int:
        """Image height in pixels"""

    @property
    def camera_width(self) -> int:
        """Camera sensor width"""

    @property
    def camera_height(self) -> int:
        """Camera sensor height"""

    @property
    def image_name(self) -> str:
        """Image filename"""

    @property
    def image_path(self) -> str:
        """Full path to image file"""

    @property
    def mask_path(self) -> str:
        """Full path to mask file"""

    @property
    def has_mask(self) -> bool:
        """Whether a mask file exists"""

    @property
    def uid(self) -> int:
        """Unique camera identifier"""

    @property
    def R(self) -> lichtfeld.Tensor:
        """Rotation matrix [3, 3]"""

    @property
    def T(self) -> lichtfeld.Tensor:
        """Translation vector [3]"""

    @property
    def K(self) -> lichtfeld.Tensor:
        """Intrinsic matrix [3, 3]"""

    @property
    def world_view_transform(self) -> lichtfeld.Tensor:
        """World-to-view transform [4, 4]"""

    @property
    def cam_position(self) -> lichtfeld.Tensor:
        """Camera position in world space [3]"""

    def load_image(self, resize_factor: int = 1, max_width: int = 3840) -> lichtfeld.Tensor:
        """Load image as tensor [C, H, W] on CUDA"""

    def load_mask(self, resize_factor: int = 1, max_width: int = 3840, invert: bool = False, threshold: float = 0.5) -> lichtfeld.Tensor:
        """Load mask as tensor [1, H, W] on CUDA"""

class CameraDataset:
    def __len__(self) -> int:
        """Number of cameras"""

    def __getitem__(self, index: int) -> Camera:
        """Get camera by index"""

    def get_camera_by_filename(self, filename: str) -> Camera | None:
        """Find camera by image filename"""

    def cameras(self) -> list[Camera]:
        """Get all cameras as a list"""

    def set_resize_factor(self, factor: int) -> None:
        """Set image resize factor for all cameras"""

    def set_max_width(self, width: int) -> None:
        """Set maximum image width for all cameras"""
