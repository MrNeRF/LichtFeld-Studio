from collections.abc import Callable, Sequence
import enum
from typing import overload

from numpy.typing import NDArray
import typing_extensions

from . import (
    animation as animation,
    app as app,
    build_info as build_info,
    io as io,
    keymap as keymap,
    log as log,
    mcp as mcp,
    ops as ops,
    packages as packages,
    pipeline as pipeline,
    plugins as plugins,
    scene as scene,
    scripts as scripts,
    selection as selection,
    ui as ui,
    undo as undo
)


class RenderMode(enum.Enum):
    SPLATS = 0

    POINTS = 1

    RINGS = 2

    CENTERS = 3

class OperatorResult(enum.Enum):
    FINISHED = 0

    CANCELLED = 1

    RUNNING = 2

class Hook(enum.Enum):
    training_start = 0

    iteration_start = 1

    pre_optimizer_step = 2

    post_step = 3

    training_end = 4

class Session:
    def __init__(self) -> None: ...

    def optimizer(self) -> Optimizer:
        """Get optimizer view"""

    def model(self) -> Model:
        """Get model view"""

    def pause(self) -> None:
        """Pause training"""

    def resume(self) -> None:
        """Resume training"""

    def request_stop(self) -> None:
        """Request training stop"""

class ScopedHandler:
    def __init__(self) -> None: ...

    def on_training_start(self, callback: Callable) -> None:
        """Register training start callback"""

    def on_iteration_start(self, callback: Callable) -> None:
        """Register iteration start callback"""

    def on_pre_optimizer_step(self, callback: Callable) -> None:
        """Register pre-optimizer callback"""

    def on_post_step(self, callback: Callable) -> None:
        """Register post-step callback"""

    def on_training_end(self, callback: Callable) -> None:
        """Register training end callback"""

    def clear(self) -> None:
        """Unregister all callbacks"""

class Context:
    def __init__(self) -> None: ...

    @property
    def iteration(self) -> int:
        """Current iteration count"""

    @property
    def max_iterations(self) -> int:
        """Maximum iteration count"""

    @property
    def loss(self) -> float:
        """Current training loss value"""

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians in the model"""

    @property
    def is_refining(self) -> bool:
        """Whether training is in refinement phase"""

    @property
    def is_training(self) -> bool:
        """Whether training is currently running"""

    @property
    def is_paused(self) -> bool:
        """Whether training is paused"""

    @property
    def phase(self) -> str:
        """Current training phase name"""

    @property
    def strategy(self) -> str:
        """Active training strategy name"""

    def refresh(self) -> None:
        """Update cached snapshot from current state"""

class Gaussians:
    def __init__(self) -> None: ...

    @property
    def count(self) -> int:
        """Total number of Gaussians"""

    @property
    def sh_degree(self) -> int:
        """Current active SH degree"""

    @property
    def max_sh_degree(self) -> int:
        """Maximum SH degree"""

class Optimizer:
    def __init__(self) -> None: ...

    def scale_lr(self, factor: float) -> None:
        """Scale learning rate by factor"""

    def set_lr(self, value: float) -> None:
        """Set learning rate"""

    def get_lr(self) -> float:
        """Get current learning rate"""

class Model:
    def __init__(self) -> None: ...

    def clamp(self, attr: str, min: float | None = None, max: float | None = None) -> None:
        """Clamp attribute values"""

    def scale(self, attr: str, factor: float) -> None:
        """Scale attribute by factor"""

    def set(self, attr: str, value: float) -> None:
        """Set attribute value"""

def context() -> Context:
    """Get current training context"""

def gaussians() -> Gaussians:
    """Get Gaussians info"""

def session() -> Session:
    """Get training session"""

def trainer_state() -> str:
    """Get trainer state"""

def finish_reason() -> str | None:
    """Get finish reason if training finished"""

def trainer_error() -> str | None:
    """Get trainer error message"""

def start_training() -> None:
    """Start training with current parameters"""

def pause_training() -> None:
    """Pause the current training run"""

def resume_training() -> None:
    """Resume a paused training run"""

def stop_training() -> None:
    """Stop the current training run"""

def reset_training() -> None:
    """Reset training state to initial"""

def save_checkpoint() -> None:
    """Save a training checkpoint to disk"""

def clear_scene() -> None:
    """Remove all nodes from the scene"""

def switch_to_edit_mode() -> None:
    """Switch from training to edit mode"""

def load_file(path: str, is_dataset: bool = False, output_path: str = '', init_path: str = '') -> None:
    """Load a file (PLY, checkpoint) or dataset into the scene."""

def load_config_file(path: str) -> None:
    """Load a JSON configuration file."""

def load_checkpoint_for_training(checkpoint_path: str, dataset_path: str, output_path: str) -> None:
    """
    Load a checkpoint for training with specified dataset and output paths.
    """

def request_exit() -> None:
    """Request application exit (shows confirmation if needed)."""

def force_exit() -> None:
    """Force immediate application exit (bypasses confirmation)."""

def export_scene(format: int, path: str, node_names: Sequence[str], sh_degree: int) -> None:
    """Export scene nodes to file. Format: 0=PLY, 1=SOG, 2=SPZ, 3=HTML."""

def save_config_file(path: str) -> None:
    """Save current training configuration to a JSON file."""

def has_trainer() -> bool:
    """Check if a trainer instance exists"""

def loss_buffer() -> list[float]:
    """Get the recent loss history as a list of floats"""

def trainer_elapsed_seconds() -> float:
    """Get elapsed training time in seconds"""

def trainer_eta_seconds() -> float:
    """Get estimated remaining time in seconds (-1 if unavailable)"""

def trainer_strategy_type() -> str:
    """Get training strategy type (mcmc, default, etc.)"""

def trainer_is_gut_enabled() -> bool:
    """Check if GUT is enabled"""

def trainer_max_gaussians() -> int:
    """Get maximum number of gaussians"""

def trainer_num_splats() -> int:
    """Get current number of splats"""

def trainer_current_iteration() -> int:
    """Get current iteration"""

def trainer_total_iterations() -> int:
    """Get total iterations"""

def trainer_current_loss() -> float:
    """Get current loss"""

def set_node_visibility(name: str, visible: bool) -> None:
    """Set visibility of a scene node by name"""

def remove_node(name: str, keep_children: bool = False) -> None:
    """Remove a scene node by name"""

def select_node(name: str) -> None:
    """Select a scene node by name"""

def deselect_all() -> None:
    """Deselect all scene nodes"""

def reparent_node(name: str, new_parent: str) -> None:
    """Move a node under a new parent node"""

def rename_node(old_name: str, new_name: str) -> None:
    """Rename a scene node"""

def add_group(name: str, parent: str = '') -> None:
    """Add a group node to the scene"""

def get_selected_node_transform() -> list[float] | None:
    """Get transform matrix (16 floats, column-major) of selected node"""

def set_selected_node_transform(matrix: Sequence[float]) -> None:
    """Set transform matrix (16 floats, column-major) of selected node"""

def get_selection_center() -> list[float] | None:
    """Get center of current selection (local space)"""

def get_selection_world_center() -> list[float] | None:
    """Get center of current selection (world space)"""

def has_scene() -> bool:
    """Check if a scene is loaded"""

def has_selection() -> bool:
    """Check if a node is selected"""

def get_selected_node_names() -> list[str]:
    """Get names of all selected nodes"""

def get_selected_node_name() -> str:
    """Get name of primary selected node"""

def can_transform_selection() -> bool:
    """Check if selected node can be transformed"""

def get_num_gaussians() -> int:
    """Get total number of gaussians in scene"""

def get_node_transform(name: str) -> list[float] | None:
    """Get node transform matrix (16 floats, column-major)"""

def set_node_transform(name: str, matrix: Sequence[float]) -> None:
    """Set node transform matrix (16 floats, column-major)"""

def capture_selection_transforms() -> dict:
    """Capture transforms of all selected nodes"""

def decompose_transform(matrix: Sequence[float]) -> dict:
    """Decompose transform into translation, rotation, scale"""

def compose_transform(translation: Sequence[float], euler_deg: Sequence[float], scale: Sequence[float]) -> list[float]:
    """
    Compose transform matrix from translation, euler angles (degrees), and scale
    """

def load_icon(name: str) -> int:
    """
    Load an icon texture from assets/icon/{name}.png, returns OpenGL texture ID
    """

def free_icon(texture_id: int) -> None:
    """Free an icon texture"""

def reset_camera() -> None:
    """Reset camera to default position and orientation"""

def toggle_fullscreen() -> None:
    """Toggle fullscreen mode"""

def is_fullscreen() -> bool:
    """Check if the window is in fullscreen mode"""

def toggle_ui() -> None:
    """Toggle UI overlay visibility"""

def get_render_mode() -> RenderMode:
    """Get current render mode (Splats, Points, Rings, Centers)"""

def set_render_mode(mode: RenderMode) -> None:
    """Set the render mode (Splats, Points, Rings, Centers)"""

def is_orthographic() -> bool:
    """Check if orthographic projection is active"""

def set_orthographic(ortho: bool) -> None:
    """Enable or disable orthographic projection"""

def on_training_start(callback: Callable) -> Callable:
    """Decorator for training start handler"""

def on_iteration_start(callback: Callable) -> Callable:
    """Decorator for iteration start handler"""

def on_post_step(callback: Callable) -> Callable:
    """Decorator for post-step handler"""

def on_pre_optimizer_step(callback: Callable) -> Callable:
    """Decorator for pre-optimizer handler"""

def on_training_end(callback: Callable) -> Callable:
    """Decorator for training end handler"""

class Tensor:
    def __init__(self) -> None: ...

    @property
    def shape(self) -> tuple:
        """Tensor shape as tuple"""

    @property
    def ndim(self) -> int:
        """Number of dimensions"""

    @property
    def numel(self) -> int:
        """Total number of elements"""

    @property
    def device(self) -> str:
        """Device: 'cpu' or 'cuda'"""

    @property
    def dtype(self) -> str:
        """Data type"""

    @property
    def is_contiguous(self) -> bool:
        """Whether memory is contiguous"""

    @property
    def is_cuda(self) -> bool:
        """Whether tensor is on CUDA"""

    def clone(self) -> Tensor:
        """Deep copy of tensor"""

    def cpu(self) -> Tensor:
        """Move tensor to CPU"""

    def cuda(self) -> Tensor:
        """Move tensor to CUDA"""

    def contiguous(self) -> Tensor:
        """Make tensor contiguous"""

    def sync(self) -> None:
        """Synchronize CUDA stream"""

    def size(self, dim: int) -> int:
        """Size of dimension"""

    def item(self) -> float:
        """Extract scalar value"""

    def numpy(self, copy: bool = True) -> object:
        """Convert to NumPy array"""

    @staticmethod
    def from_numpy(arr: NDArray, copy: bool = True) -> Tensor:
        """Create tensor from NumPy array"""

    @staticmethod
    def zeros(shape: Sequence[int], device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create tensor filled with zeros"""

    @staticmethod
    def ones(shape: Sequence[int], device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create tensor filled with ones"""

    @staticmethod
    def full(shape: Sequence[int], value: float, device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create tensor filled with value"""

    @overload
    @staticmethod
    def arange(end: float) -> Tensor:
        """Create 1D tensor with values from 0 to end"""

    @overload
    @staticmethod
    def arange(start: float, end: float, step: float = 1.0, device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create 1D tensor with values from start to end"""

    @staticmethod
    def linspace(start: float, end: float, steps: int, device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create 1D tensor with evenly spaced values"""

    @overload
    @staticmethod
    def eye(n: int, device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create identity matrix"""

    @overload
    @staticmethod
    def eye(m: int, n: int, device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create m x n matrix with ones on diagonal"""

    def __dlpack__(self, stream: object | None = None) -> typing_extensions.CapsuleType:
        """Export as DLPack capsule"""

    def __dlpack_device__(self) -> tuple:
        """Get DLPack device tuple"""

    @staticmethod
    def from_dlpack(obj: object) -> Tensor:
        """Create tensor from DLPack capsule or object"""

    def __getitem__(self, arg: object, /) -> Tensor:
        """Get item/slice"""

    def __setitem__(self, arg0: object, arg1: object, /) -> None:
        """Set item/slice"""

    @overload
    def __add__(self, arg: Tensor, /) -> Tensor:
        """Add tensor"""

    @overload
    def __add__(self, arg: float, /) -> Tensor:
        """Add scalar"""

    def __radd__(self, arg: float, /) -> Tensor:
        """Reverse add scalar"""

    @overload
    def __iadd__(self, arg: Tensor, /) -> Tensor:
        """In-place add tensor"""

    @overload
    def __iadd__(self, arg: float, /) -> Tensor:
        """In-place add scalar"""

    @overload
    def __sub__(self, arg: Tensor, /) -> Tensor:
        """Subtract tensor"""

    @overload
    def __sub__(self, arg: float, /) -> Tensor:
        """Subtract scalar"""

    def __rsub__(self, arg: float, /) -> Tensor:
        """Reverse subtract scalar"""

    @overload
    def __isub__(self, arg: Tensor, /) -> Tensor:
        """In-place subtract tensor"""

    @overload
    def __isub__(self, arg: float, /) -> Tensor:
        """In-place subtract scalar"""

    @overload
    def __mul__(self, arg: Tensor, /) -> Tensor:
        """Multiply tensor"""

    @overload
    def __mul__(self, arg: float, /) -> Tensor:
        """Multiply scalar"""

    def __rmul__(self, arg: float, /) -> Tensor:
        """Reverse multiply scalar"""

    @overload
    def __imul__(self, arg: Tensor, /) -> Tensor:
        """In-place multiply tensor"""

    @overload
    def __imul__(self, arg: float, /) -> Tensor:
        """In-place multiply scalar"""

    @overload
    def __truediv__(self, arg: Tensor, /) -> Tensor:
        """Divide tensor"""

    @overload
    def __truediv__(self, arg: float, /) -> Tensor:
        """Divide scalar"""

    def __rtruediv__(self, arg: float, /) -> Tensor:
        """Reverse divide scalar"""

    @overload
    def __itruediv__(self, arg: Tensor, /) -> Tensor:
        """In-place divide tensor"""

    @overload
    def __itruediv__(self, arg: float, /) -> Tensor:
        """In-place divide scalar"""

    def __neg__(self) -> Tensor:
        """Negate"""

    def __abs__(self) -> Tensor:
        """Absolute value"""

    def sigmoid(self) -> Tensor:
        """Sigmoid activation"""

    def exp(self) -> Tensor:
        """Exponential"""

    def log(self) -> Tensor:
        """Natural logarithm"""

    def sqrt(self) -> Tensor:
        """Square root"""

    def relu(self) -> Tensor:
        """ReLU activation"""

    def sin(self) -> Tensor:
        """Sine"""

    def cos(self) -> Tensor:
        """Cosine"""

    def tan(self) -> Tensor:
        """Tangent"""

    def tanh(self) -> Tensor:
        """Hyperbolic tangent"""

    def floor(self) -> Tensor:
        """Floor"""

    def ceil(self) -> Tensor:
        """Ceiling"""

    def round(self) -> Tensor:
        """Round to nearest"""

    def abs(self) -> Tensor:
        """Absolute value"""

    @overload
    def __eq__(self, arg: Tensor, /) -> Tensor:
        """Equal tensor"""

    @overload
    def __eq__(self, arg: float, /) -> Tensor:
        """Equal scalar"""

    @overload
    def __ne__(self, arg: Tensor, /) -> Tensor:
        """Not equal tensor"""

    @overload
    def __ne__(self, arg: float, /) -> Tensor:
        """Not equal scalar"""

    @overload
    def __lt__(self, arg: Tensor, /) -> Tensor:
        """Less than tensor"""

    @overload
    def __lt__(self, arg: float, /) -> Tensor:
        """Less than scalar"""

    @overload
    def __le__(self, arg: Tensor, /) -> Tensor:
        """Less equal tensor"""

    @overload
    def __le__(self, arg: float, /) -> Tensor:
        """Less equal scalar"""

    @overload
    def __gt__(self, arg: Tensor, /) -> Tensor:
        """Greater than tensor"""

    @overload
    def __gt__(self, arg: float, /) -> Tensor:
        """Greater than scalar"""

    @overload
    def __ge__(self, arg: Tensor, /) -> Tensor:
        """Greater equal tensor"""

    @overload
    def __ge__(self, arg: float, /) -> Tensor:
        """Greater equal scalar"""

    def __and__(self, arg: Tensor, /) -> Tensor:
        """Logical AND"""

    def __or__(self, arg: Tensor, /) -> Tensor:
        """Logical OR"""

    def __invert__(self) -> Tensor:
        """Logical NOT"""

    def sum(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Sum reduction"""

    def mean(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Mean reduction"""

    def max(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Max reduction"""

    def min(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Min reduction"""

    def sum_scalar(self) -> float:
        """Sum all elements to scalar"""

    def mean_scalar(self) -> float:
        """Mean of all elements as scalar"""

    def max_scalar(self) -> float:
        """Max of all elements as scalar"""

    def min_scalar(self) -> float:
        """Min of all elements as scalar"""

    def reshape(self, shape: Sequence[int]) -> Tensor:
        """Reshape tensor"""

    def view(self, shape: Sequence[int]) -> Tensor:
        """View tensor with new shape"""

    def squeeze(self, dim: int | None = None) -> Tensor:
        """Remove size-1 dimensions"""

    def unsqueeze(self, dim: int) -> Tensor:
        """Add size-1 dimension"""

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """Transpose dimensions"""

    def permute(self, dims: Sequence[int]) -> Tensor:
        """Permute dimensions"""

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        """Flatten dimensions"""

    def expand(self, sizes: Sequence[int]) -> Tensor:
        """Expand tensor to larger size"""

    def repeat(self, repeats: Sequence[int]) -> Tensor:
        """Repeat tensor along dimensions"""

    def t(self) -> Tensor:
        """Transpose 2D tensor"""

    def log2(self) -> Tensor:
        """Base-2 logarithm"""

    def log10(self) -> Tensor:
        """Base-10 logarithm"""

    def log1p(self) -> Tensor:
        """Log(1 + x)"""

    def exp2(self) -> Tensor:
        """2^x"""

    def rsqrt(self) -> Tensor:
        """Reciprocal square root"""

    def square(self) -> Tensor:
        """Element-wise square"""

    def asin(self) -> Tensor:
        """Arc sine"""

    def acos(self) -> Tensor:
        """Arc cosine"""

    def atan(self) -> Tensor:
        """Arc tangent"""

    def sinh(self) -> Tensor:
        """Hyperbolic sine"""

    def cosh(self) -> Tensor:
        """Hyperbolic cosine"""

    def trunc(self) -> Tensor:
        """Truncate to integer"""

    def sign(self) -> Tensor:
        """Sign of elements"""

    def reciprocal(self) -> Tensor:
        """1/x"""

    def gelu(self) -> Tensor:
        """GELU activation"""

    def swish(self) -> Tensor:
        """Swish activation"""

    def isnan(self) -> Tensor:
        """Check for NaN"""

    def isinf(self) -> Tensor:
        """Check for infinity"""

    def isfinite(self) -> Tensor:
        """Check for finite values"""

    @overload
    def pow(self, exponent: float) -> Tensor:
        """Power with scalar exponent"""

    @overload
    def pow(self, exponent: Tensor) -> Tensor:
        """Power with tensor exponent"""

    def __pow__(self, arg: float, /) -> Tensor:
        """Power operator"""

    def prod(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Product reduction"""

    def std(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Standard deviation"""

    def var(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Variance"""

    def argmax(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Index of maximum"""

    def argmin(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Index of minimum"""

    def all(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Check if all true"""

    def any(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Check if any true"""

    def norm(self, p: float = 2.0) -> Tensor:
        """Lp norm"""

    def norm_scalar(self, p: float = 2.0) -> float:
        """Lp norm as scalar"""

    def index_select(self, dim: int, indices: Tensor) -> Tensor:
        """Select along dimension by indices"""

    def gather(self, dim: int, indices: Tensor) -> Tensor:
        """Gather values along dimension"""

    def masked_select(self, mask: Tensor) -> Tensor:
        """Select elements where mask is true"""

    def masked_fill(self, mask: Tensor, value: float) -> Tensor:
        """Fill elements where mask is true"""

    def nonzero(self) -> Tensor:
        """Indices of non-zero elements"""

    def matmul(self, other: Tensor) -> Tensor:
        """Matrix multiplication"""

    def mm(self, other: Tensor) -> Tensor:
        """Matrix multiplication (2D)"""

    def bmm(self, other: Tensor) -> Tensor:
        """Batched matrix multiplication"""

    def dot(self, other: Tensor) -> Tensor:
        """Dot product"""

    def __matmul__(self, arg: Tensor, /) -> Tensor:
        """Matrix multiplication operator"""

    def clamp(self, min: float, max: float) -> Tensor:
        """Clamp values to range"""

    def maximum(self, other: Tensor) -> Tensor:
        """Element-wise maximum"""

    def minimum(self, other: Tensor) -> Tensor:
        """Element-wise minimum"""

    def to(self, dtype: str) -> Tensor:
        """Convert to specified dtype"""

    @staticmethod
    def rand(shape: Sequence[int], device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create tensor with uniform random values [0, 1)"""

    @staticmethod
    def randn(shape: Sequence[int], device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create tensor with normal random values"""

    @staticmethod
    def empty(shape: Sequence[int], device: str = 'cuda', dtype: str = 'float32') -> Tensor:
        """Create uninitialized tensor"""

    @staticmethod
    def randint(low: int, high: int, shape: Sequence[int], device: str = 'cuda') -> Tensor:
        """Create tensor with random integers"""

    @staticmethod
    def zeros_like(other: Tensor) -> Tensor:
        """Create zeros tensor like other"""

    @staticmethod
    def ones_like(other: Tensor) -> Tensor:
        """Create ones tensor like other"""

    @staticmethod
    def rand_like(other: Tensor) -> Tensor:
        """Create random tensor like other"""

    @staticmethod
    def randn_like(other: Tensor) -> Tensor:
        """Create normal random tensor like other"""

    @staticmethod
    def empty_like(other: Tensor) -> Tensor:
        """Create uninitialized tensor like other"""

    @staticmethod
    def full_like(other: Tensor, value: float) -> Tensor:
        """Create filled tensor like other"""

    @staticmethod
    def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
        """Concatenate tensors"""

    @staticmethod
    def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
        """Stack tensors"""

    @staticmethod
    def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
        """Conditional select"""

    def __repr__(self) -> str:
        """String representation"""

    def __array__(self, dtype: object | None = None) -> object:
        """Return numpy array view (zero-copy for CPU contiguous tensors)"""

class ViewInfo:
    @property
    def rotation(self) -> Tensor: ...

    @property
    def translation(self) -> Tensor: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def fov_x(self) -> float: ...

    @property
    def fov_y(self) -> float: ...

class ViewportRender:
    @property
    def image(self) -> Tensor: ...

    @property
    def screen_positions(self) -> Tensor | None: ...

def get_viewport_render() -> ViewportRender | None:
    """
    Get the current viewport's rendered image and screen positions (None if not available)
    """

def capture_viewport() -> ViewportRender | None:
    """
    Capture viewport render for async processing (clones data, safe to use from background threads)
    """

def render_view(rotation: Tensor, translation: Tensor, width: int, height: int, fov: float = 60.0, bg_color: Tensor | None = None) -> Tensor | None:
    """
    Render scene from arbitrary camera parameters.

    Args:
        rotation: [3, 3] camera rotation matrix
        translation: [3] camera position
        width: Render width in pixels
        height: Render height in pixels
        fov: Field of view in degrees (default: 60)
        bg_color: Optional [3] RGB background color

    Returns:
        Tensor [H, W, 3] RGB image on CUDA, or None if scene not available
    """

def compute_screen_positions(rotation: Tensor, translation: Tensor, width: int, height: int, fov: float = 60.0) -> Tensor | None:
    """
    Compute screen positions of all Gaussians for a given camera view.

    Args:
        rotation: [3, 3] camera rotation matrix
        translation: [3] camera position
        width: Viewport width in pixels
        height: Viewport height in pixels
        fov: Field of view in degrees (default: 60)

    Returns:
        Tensor [N, 2] with (x, y) pixel coordinates for each Gaussian
    """

def get_current_view() -> ViewInfo | None:
    """Get current viewport camera info (None if not available)"""

def get_render_scene() -> scene.Scene | None:
    """Get the current render scene (None if not available)"""

class RenderSettings:
    @property
    def __property_group__(self) -> str: ...

    def get(self, name: str) -> object:
        """Get property value by name"""

    def set(self, name: str, value: object) -> None:
        """Set property value by name"""

    def prop_info(self, name: str) -> dict: ...

    def get_all_properties(self) -> dict:
        """Get all property descriptors as Python Property objects"""

    def __getattr__(self, arg: str, /) -> object: ...

    def __setattr__(self, arg0: str, arg1: object, /) -> None: ...

    def __dir__(self) -> list: ...

def get_render_settings() -> RenderSettings | None: ...

def register_class(cls: object) -> None:
    """Register a class (Panel, Operator, or Menu)"""

def unregister_class(cls: object) -> None:
    """Unregister a class (Panel, Operator, or Menu)"""

class GizmoEventType(enum.Enum):
    PRESS = 0

    RELEASE = 1

    MOVE = 2

    DRAG = 3

class GizmoResult(enum.Enum):
    PASS_THROUGH = 0

    RUNNING_MODAL = 1

    FINISHED = 2

    CANCELLED = 3

class GizmoContext:
    def __init__(self) -> None: ...

    @property
    def has_selection(self) -> bool:
        """Whether any gaussians are selected"""

    @property
    def selection_center(self) -> tuple[float, float, float]:
        """Selection center in world space (x, y, z)"""

    @property
    def selection_center_screen(self) -> tuple[float, float]:
        """Selection center in screen space (x, y)"""

    @property
    def camera_position(self) -> tuple[float, float, float]:
        """Camera position in world space (x, y, z)"""

    @property
    def camera_forward(self) -> tuple[float, float, float]:
        """Camera forward direction (x, y, z)"""

    def world_to_screen(self, pos: tuple[float, float, float]) -> tuple[float, float] | None:
        """Project world position to screen coordinates"""

    def screen_to_world_ray(self, pos: tuple[float, float]) -> tuple[float, float, float] | None:
        """Get world-space ray direction from screen point"""

    def draw_line(self, start: tuple[float, float], end: tuple[float, float], color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a 2D line"""

    def draw_circle(self, center: tuple[float, float], radius: float, color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a 2D circle outline"""

    def draw_rect(self, min: tuple[float, float], max: tuple[float, float], color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a 2D rectangle outline"""

    def draw_filled_rect(self, min: tuple[float, float], max: tuple[float, float], color: tuple[float, float, float, float]) -> None:
        """Draw a filled 2D rectangle"""

    def draw_filled_circle(self, center: tuple[float, float], radius: float, color: tuple[float, float, float, float]) -> None:
        """Draw a filled 2D circle"""

    def draw_line_3d(self, start: tuple[float, float, float], end: tuple[float, float, float], color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a 3D line"""

def register_gizmo(gizmo_class: object) -> None:
    """Register a gizmo class for viewport overlay drawing"""

def unregister_gizmo(id: str) -> None:
    """Unregister a gizmo by ID"""

def unregister_all_gizmos() -> None:
    """Unregister all gizmos"""

def get_gizmo_ids() -> list[str]:
    """Get all registered gizmo IDs"""

def has_gizmos() -> bool:
    """Check if any gizmos are registered"""

class OperatorReturnValue:
    @property
    def status(self) -> str:
        """Result status string"""

    @property
    def finished(self) -> bool:
        """Whether operator completed successfully"""

    @property
    def cancelled(self) -> bool:
        """Whether operator was cancelled"""

    @property
    def running_modal(self) -> bool:
        """Whether operator is running as modal"""

    @property
    def pass_through(self) -> bool:
        """Whether event should pass through"""

    def __getattr__(self, arg: str, /) -> object:
        """Access return data by attribute name"""

    def __bool__(self) -> bool:
        """True if operator finished successfully"""

def register_uilist(list_class: object) -> None:
    """Register a UIList class for custom list rendering"""

def unregister_uilist(id: str) -> None:
    """Unregister a UIList by ID"""

def unregister_all_uilists() -> None:
    """Unregister all UIList classes"""

def get_uilist_ids() -> list[str]:
    """Get all registered UIList IDs"""

class DrawHandlerTiming(enum.Enum):
    PRE_VIEW = 0

    POST_VIEW = 1

    POST_UI = 2

class ViewportDrawContext:
    def __init__(self) -> None: ...

    def world_to_screen(self, pos: tuple[float, float, float]) -> tuple[float, float] | None:
        """Project a (x, y, z) world position to (sx, sy) screen coordinates"""

    def screen_to_world_ray(self, screen_pos: tuple[float, float]) -> tuple[float, float, float]:
        """
        Convert (sx, sy) screen position to a normalized world-space ray direction
        """

    @property
    def camera_position(self) -> tuple[float, float, float]:
        """Camera position as (x, y, z)"""

    @property
    def camera_forward(self) -> tuple[float, float, float]:
        """Camera forward direction as (x, y, z)"""

    @property
    def viewport_size(self) -> tuple[float, float]:
        """Viewport dimensions as (width, height)"""

    def draw_line_2d(self, start: tuple[float, float], end: tuple[float, float], color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a 2D line from start to end in screen coordinates"""

    def draw_circle_2d(self, center: tuple[float, float], radius: float, color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a 2D circle outline in screen coordinates"""

    def draw_rect_2d(self, min: tuple[float, float], max: tuple[float, float], color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a 2D rectangle outline in screen coordinates"""

    def draw_filled_rect_2d(self, min: tuple[float, float], max: tuple[float, float], color: tuple[float, float, float, float]) -> None:
        """Draw a filled 2D rectangle in screen coordinates"""

    def draw_filled_circle_2d(self, center: tuple[float, float], radius: float, color: tuple[float, float, float, float]) -> None:
        """Draw a filled 2D circle in screen coordinates"""

    def draw_text_2d(self, pos: tuple[float, float], text: str, color: tuple[float, float, float, float]) -> None:
        """Draw text at a 2D screen position"""

    def draw_line_3d(self, start: tuple[float, float, float], end: tuple[float, float, float], color: tuple[float, float, float, float], thickness: float = 1.0) -> None:
        """Draw a 3D line between two world-space points"""

    def draw_point_3d(self, pos: tuple[float, float, float], color: tuple[float, float, float, float], size: float = 4.0) -> None:
        """Draw a point at a 3D world-space position"""

def draw_handler(timing: str = 'POST_VIEW') -> object:
    """
    Decorator to register a viewport draw handler (PRE_VIEW, POST_VIEW, POST_UI)
    """

def add_draw_handler(id: str, callback: object, timing: str = 'POST_VIEW') -> None:
    """Add a viewport draw handler with explicit id"""

def remove_draw_handler(id: str) -> None:
    """Remove a viewport draw handler"""

def clear_draw_handlers() -> None:
    """Clear all viewport draw handlers"""

def get_draw_handler_ids() -> list[str]:
    """Get list of registered draw handler ids"""

def has_draw_handlers() -> bool:
    """Check if any draw handlers are registered"""

class MaskMode(enum.Enum):
    NONE = 0

    SEGMENT = 1

    IGNORE = 2

    ALPHA_CONSISTENT = 3

class BackgroundMode(enum.Enum):
    SOLID_COLOR = 0

    MODULATION = 1

    IMAGE = 2

    RANDOM = 3

class OptimizationParams:
    def __init__(self) -> None: ...

    @property
    def __property_group__(self) -> str:
        """Property group identifier"""

    def get(self, name: str) -> object:
        """Get property value by name"""

    def set(self, name: str, value: object) -> None:
        """Set property value by name"""

    def __getattr__(self, name: str) -> object:
        """Get property value by attribute name"""

    def prop_info(self, prop_id: str) -> dict:
        """Get metadata for a property"""

    def reset(self, prop_id: str) -> None:
        """Reset property to default value"""

    def properties(self) -> list:
        """List all properties with their current values"""

    def get_all_properties(self) -> dict:
        """Get all property descriptors as Python Property objects"""

    def has_params(self) -> bool:
        """Check if ParameterManager is available"""

    @property
    def iterations(self) -> int:
        """Maximum training iterations"""

    @iterations.setter
    def iterations(self, arg: int, /) -> None: ...

    @property
    def means_lr(self) -> float:
        """Learning rate for gaussian positions"""

    @means_lr.setter
    def means_lr(self, arg: float, /) -> None: ...

    @property
    def shs_lr(self) -> float:
        """Learning rate for spherical harmonics"""

    @shs_lr.setter
    def shs_lr(self, arg: float, /) -> None: ...

    @property
    def opacity_lr(self) -> float:
        """Learning rate for opacity"""

    @opacity_lr.setter
    def opacity_lr(self, arg: float, /) -> None: ...

    @property
    def scaling_lr(self) -> float:
        """Learning rate for gaussian scales"""

    @scaling_lr.setter
    def scaling_lr(self, arg: float, /) -> None: ...

    @property
    def rotation_lr(self) -> float:
        """Learning rate for rotations"""

    @rotation_lr.setter
    def rotation_lr(self, arg: float, /) -> None: ...

    @property
    def lambda_dssim(self) -> float:
        """Weight for structural similarity loss"""

    @lambda_dssim.setter
    def lambda_dssim(self, arg: float, /) -> None: ...

    @property
    def sh_degree(self) -> int:
        """Spherical harmonics degree (0-3)"""

    @sh_degree.setter
    def sh_degree(self, arg: int, /) -> None: ...

    @property
    def max_cap(self) -> int:
        """Maximum number of gaussians"""

    @max_cap.setter
    def max_cap(self, arg: int, /) -> None: ...

    @property
    def strategy(self) -> str:
        """Active optimization strategy name"""

    def set_strategy(self, strategy: str) -> None:
        """Set active strategy ('mcmc' or 'adc')"""

    @property
    def headless(self) -> bool:
        """Whether running without visualization"""

    @property
    def tile_mode(self) -> int:
        """Tile mode (1, 2, or 4)"""

    @tile_mode.setter
    def tile_mode(self, arg: int, /) -> None: ...

    @property
    def steps_scaler(self) -> float:
        """Scale factor for training step counts"""

    @steps_scaler.setter
    def steps_scaler(self, arg: float, /) -> None: ...

    @property
    def gut(self) -> bool:
        """Enable Gaussian Unscented Transform"""

    @gut.setter
    def gut(self, arg: bool, /) -> None: ...

    @property
    def use_bilateral_grid(self) -> bool:
        """Enable bilateral grid color correction"""

    @use_bilateral_grid.setter
    def use_bilateral_grid(self, arg: bool, /) -> None: ...

    @property
    def enable_sparsity(self) -> bool:
        """Enable sparsity optimization"""

    @enable_sparsity.setter
    def enable_sparsity(self, arg: bool, /) -> None: ...

    @property
    def mip_filter(self) -> bool:
        """Enable mip filtering (anti-aliasing)"""

    @mip_filter.setter
    def mip_filter(self, arg: bool, /) -> None: ...

    @property
    def ppisp(self) -> bool:
        """Enable per-pixel image signal processing"""

    @ppisp.setter
    def ppisp(self, arg: bool, /) -> None: ...

    @property
    def bg_mode(self) -> BackgroundMode:
        """Background rendering mode"""

    @bg_mode.setter
    def bg_mode(self, arg: BackgroundMode, /) -> None: ...

    @property
    def bg_color(self) -> tuple[float, float, float]:
        """Background color as (r, g, b) tuple"""

    @bg_color.setter
    def bg_color(self, arg: tuple[float, float, float], /) -> None: ...

    @property
    def bg_image_path(self) -> str:
        """Path to background image"""

    @bg_image_path.setter
    def bg_image_path(self, arg: str, /) -> None: ...

    @property
    def random(self) -> bool:
        """Use random initialization instead of SfM"""

    @random.setter
    def random(self, arg: bool, /) -> None: ...

    @property
    def mask_mode(self) -> MaskMode:
        """Attention mask behavior during training"""

    @mask_mode.setter
    def mask_mode(self, arg: MaskMode, /) -> None: ...

    @property
    def invert_masks(self) -> bool:
        """Swap object and background in masks"""

    @invert_masks.setter
    def invert_masks(self, arg: bool, /) -> None: ...

    @property
    def use_alpha_as_mask(self) -> bool:
        """Use alpha channel from RGBA images as mask source"""

    @use_alpha_as_mask.setter
    def use_alpha_as_mask(self, arg: bool, /) -> None: ...

    @property
    def undistort(self) -> bool:
        """Undistort images on-the-fly before training"""

    @undistort.setter
    def undistort(self, arg: bool, /) -> None: ...

    @property
    def save_steps(self) -> list[int]:
        """List of iterations at which to save checkpoints"""

    def add_save_step(self, step: int) -> None:
        """Add a save step (ignored if duplicate)"""

    def remove_save_step(self, step: int) -> None:
        """Remove a save step"""

    def clear_save_steps(self) -> None:
        """Clear all save steps"""

def optimization_params() -> OptimizationParams:
    """Get the optimization parameters object"""

class DatasetParams:
    def __init__(self) -> None: ...

    @property
    def __property_group__(self) -> str:
        """Property group identifier"""

    def get(self, name: str) -> object:
        """Get property value by name"""

    def set(self, name: str, value: object) -> None:
        """Set property value by name"""

    def prop_info(self, prop_id: str) -> dict:
        """Get metadata for a property"""

    def properties(self) -> list:
        """List all properties with their current values"""

    def get_all_properties(self) -> dict:
        """Get all property descriptors as Python Property objects"""

    def has_params(self) -> bool:
        """Check if TrainerManager is available"""

    def can_edit(self) -> bool:
        """Check if dataset params can be edited (before training starts)"""

    @property
    def data_path(self) -> str:
        """Path to training data directory"""

    @property
    def output_path(self) -> str:
        """Path for output files"""

    @property
    def images(self) -> str:
        """Subfolder name containing images"""

    @property
    def test_every(self) -> int:
        """Use every Nth image for testing"""

    @property
    def resize_factor(self) -> int:
        """Image resize factor (-1 = auto)"""

    @resize_factor.setter
    def resize_factor(self, arg: int, /) -> None: ...

    @property
    def max_width(self) -> int:
        """Maximum image width in pixels"""

    @max_width.setter
    def max_width(self, arg: int, /) -> None: ...

    @property
    def use_cpu_cache(self) -> bool:
        """Cache images in CPU memory"""

    @use_cpu_cache.setter
    def use_cpu_cache(self, arg: bool, /) -> None: ...

    @property
    def use_fs_cache(self) -> bool:
        """Use filesystem cache for images"""

    @use_fs_cache.setter
    def use_fs_cache(self, arg: bool, /) -> None: ...

def dataset_params() -> DatasetParams:
    """Get the dataset parameters object"""

def on_property_change(property_path: str, callback: Callable) -> int:
    """
    Register a callback for property changes. Returns subscription ID.
    Usage: lf.on_property_change('optimization.means_lr', lambda old, new: print(f'{old} -> {new}'))
    """

def unsubscribe_property_change(subscription_id: int) -> None:
    """Unsubscribe from property change notifications"""

def property_callback(property_path: str) -> object:
    """
    Decorator for property change handlers.
    Usage: @lf.property_callback('optimization.means_lr')
           def on_lr_change(old_val, new_val): ...
    """

def get_scene() -> scene.Scene | None:
    """Get the current scene (None if not available)"""

def get_scene_generation() -> int:
    """Get current scene generation counter (for validity checking)"""

def run(path: str) -> None:
    """Execute a Python script file"""

def list_scene() -> None:
    """Print the scene graph tree"""

def on_frame(callback: Callable) -> None:
    """Register a callback to be called each frame with delta time (seconds)"""

def stop_animation() -> None:
    """Stop any running animation (clears frame callback)"""

def mat4(rows: Sequence[Sequence[float]]) -> Tensor:
    """Create a 4x4 matrix tensor from nested list [[r0], [r1], [r2], [r3]]"""

def help() -> None:
    """Show help for lichtfeld module"""

class DatasetInfo:
    """Information about a dataset directory"""

    @property
    def base_path(self) -> str:
        """Root directory of the dataset"""

    @property
    def images_path(self) -> str:
        """Path to the images directory"""

    @property
    def sparse_path(self) -> str:
        """Path to the COLMAP sparse reconstruction"""

    @property
    def masks_path(self) -> str:
        """Path to the masks directory"""

    @property
    def has_masks(self) -> bool:
        """Whether the dataset includes masks"""

    @property
    def image_count(self) -> int:
        """Number of images in the dataset"""

    @property
    def mask_count(self) -> int:
        """Number of masks in the dataset"""

    def __repr__(self) -> str: ...

def detect_dataset_info(path: str) -> DatasetInfo:
    """Detect dataset information from a directory path"""

class CheckpointHeader:
    """Information from a checkpoint file header"""

    @property
    def iteration(self) -> int: ...

    @property
    def num_gaussians(self) -> int: ...

    @property
    def sh_degree(self) -> int: ...

    def __repr__(self) -> str: ...

def read_checkpoint_header(path: str) -> CheckpointHeader | None:
    """Read checkpoint header information (None if failed)"""

class CheckpointParams:
    """Training parameters from a checkpoint"""

    @property
    def dataset_path(self) -> str: ...

    @property
    def output_path(self) -> str: ...

    def __repr__(self) -> str: ...

def read_checkpoint_params(path: str) -> CheckpointParams | None:
    """Read training parameters from a checkpoint (None if failed)"""

__all__: tuple = ('context', 'gaussians', 'session', 'get_scene', 'Tensor', 'Hook', 'ScopedHandler', 'on_training_start', 'on_iteration_start', 'on_post_step', 'on_pre_optimizer_step', 'on_training_end', 'on_frame', 'stop_animation', 'run', 'list_scene', 'mat4', 'help', 'scene', 'io', 'packages', 'mcp')
