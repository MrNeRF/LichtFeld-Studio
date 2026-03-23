# Typed property system for operators

from typing import Any, Callable, Optional, Type, TypeVar, Iterator, List
import logging

T = TypeVar("T", bound="PropertyGroup")
LOG = logging.getLogger(__name__)


class PropSubtype:
    """Property subtypes that affect UI widget selection (Blender-compatible)."""

    NONE = ""
    FILE_PATH = "FILE_PATH"
    DIR_PATH = "DIR_PATH"
    FILE_NAME = "FILE_NAME"
    COLOR = "COLOR"
    COLOR_GAMMA = "COLOR_GAMMA"
    TRANSLATION = "TRANSLATION"
    DIRECTION = "DIRECTION"
    VELOCITY = "VELOCITY"
    ACCELERATION = "ACCELERATION"
    XYZ = "XYZ"
    EULER = "EULER"
    QUATERNION = "QUATERNION"
    AXISANGLE = "AXISANGLE"
    ANGLE = "ANGLE"
    FACTOR = "FACTOR"
    PERCENTAGE = "PERCENTAGE"
    TIME = "TIME"
    DISTANCE = "DISTANCE"
    POWER = "POWER"
    TEMPERATURE = "TEMPERATURE"
    PIXEL = "PIXEL"
    UNSIGNED = "UNSIGNED"
    LAYER = "LAYER"
    LAYER_MEMBER = "LAYER_MEMBER"


class Property:
    """Base class for typed operator properties.

    Implements the descriptor protocol for seamless attribute access on PropertyGroup.
    """

    def __init__(
        self,
        default: Any = None,
        name: str = "",
        description: str = "",
        subtype: str = "",
        update: Optional[Callable[["PropertyGroup", Any], None]] = None,
    ):
        self.default = default
        self.name = name
        self.description = description
        self.subtype = subtype
        self.update = update
        self._attr_name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self._attr_name = name

    def __get__(self, obj: Any, objtype: type = None) -> Any:
        """Get the property value from the instance."""
        if obj is None:
            return self
        if not hasattr(obj, "_property_values"):
            return self.default
        return obj._property_values.get(self._attr_name, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        """Set the property value on the instance."""
        if not hasattr(obj, "_property_values"):
            object.__setattr__(obj, "_property_values", {})
        validated = self.validate(value)
        obj._property_values[self._attr_name] = validated
        if self.update:
            try:
                self.update(obj, None)
            except Exception as e:
                LOG.warning(f"Property update callback error for '{self._attr_name}': {e}")

    def validate(self, value: Any) -> Any:
        """Validate and potentially coerce value. Override in subclasses."""
        return value


class FloatProperty(Property):
    """Float property with optional min/max bounds.

    Subtypes:
        ANGLE: Radians input, displayed as degrees
        FACTOR: 0-1 range slider
        PERCENTAGE: 0-100 percentage slider
        TIME: Time value
        DISTANCE: Distance value
        POWER: Power value
    """

    def __init__(
        self,
        default: float = 0.0,
        min: float = float("-inf"),
        max: float = float("inf"),
        step: float = 0.1,
        precision: int = 3,
        subtype: str = "",
        **kw,
    ):
        super().__init__(default, subtype=subtype, **kw)
        self.min = min
        self.max = max
        self.step = step
        self.precision = precision

        if subtype == PropSubtype.FACTOR and min == float("-inf") and max == float("inf"):
            self.min = 0.0
            self.max = 1.0
        elif subtype == PropSubtype.PERCENTAGE and min == float("-inf") and max == float("inf"):
            self.min = 0.0
            self.max = 100.0

    def validate(self, value: Any) -> float:
        v = float(value)
        return max(self.min, min(self.max, v))


class IntProperty(Property):
    """Integer property with optional min/max bounds."""

    def __init__(
        self,
        default: int = 0,
        min: int = -(2**31),
        max: int = 2**31 - 1,
        step: int = 1,
        **kw,
    ):
        super().__init__(default, **kw)
        self.min = min
        self.max = max
        self.step = step

    def validate(self, value: Any) -> int:
        v = int(value)
        return max(self.min, min(self.max, v))


class BoolProperty(Property):
    """Boolean property."""

    def __init__(self, default: bool = False, **kw):
        super().__init__(default, **kw)

    def validate(self, value: Any) -> bool:
        return bool(value)


class EnumProperty(Property):
    """Enumeration property with fixed choices."""

    def __init__(
        self,
        items: Optional[list[tuple[str, str, str]]] = None,
        default: Optional[str] = None,
        **kw,
    ):
        # items is a list of (identifier, label, description) tuples
        self.items = items or []
        if default is None and self.items:
            default = self.items[0][0]
        super().__init__(default, **kw)

    def validate(self, value: Any) -> str:
        s = str(value)
        valid_ids = [item[0] for item in self.items]
        if s in valid_ids:
            return s
        return self.default or ""


class StringProperty(Property):
    """String property with optional max length.

    Subtypes:
        FILE_PATH: File path (uses file picker)
        DIR_PATH: Directory path (uses folder picker)
        FILE_NAME: File name only
    """

    def __init__(
        self, default: str = "", maxlen: int = 0, subtype: str = "", **kw
    ):
        super().__init__(default, subtype=subtype, **kw)
        self.maxlen = maxlen

    def validate(self, value: Any) -> str:
        s = str(value)
        if self.maxlen > 0:
            s = s[: self.maxlen]
        return s


class FloatVectorProperty(Property):
    """Vector of floats (2D, 3D, 4D).

    Subtypes:
        COLOR: RGB/RGBA color (uses color picker)
        COLOR_GAMMA: RGB/RGBA color with gamma correction
        TRANSLATION: 3D translation vector
        DIRECTION: 3D direction vector (normalized)
        VELOCITY: 3D velocity vector
        ACCELERATION: 3D acceleration vector
        XYZ: Generic XYZ values
        EULER: Euler rotation angles
        QUATERNION: Quaternion rotation (size=4)
    """

    def __init__(
        self,
        default: tuple = (0.0, 0.0, 0.0),
        size: int = 3,
        min: float = float("-inf"),
        max: float = float("inf"),
        subtype: str = "",
        **kw,
    ):
        super().__init__(default, subtype=subtype, **kw)
        self.size = size
        self.min = min
        self.max = max

        if subtype in (PropSubtype.COLOR, PropSubtype.COLOR_GAMMA):
            if min == float("-inf"):
                self.min = 0.0
            if max == float("inf"):
                self.max = 1.0

    def validate(self, value: Any) -> tuple:
        if not hasattr(value, "__iter__"):
            value = (float(value),) * self.size
        result = []
        for i, v in enumerate(value):
            if i >= self.size:
                break
            fv = float(v)
            fv = max(self.min, min(self.max, fv))
            result.append(fv)
        while len(result) < self.size:
            result.append(0.0)
        return tuple(result)


class IntVectorProperty(Property):
    """Vector of integers."""

    def __init__(
        self,
        default: tuple = (0, 0, 0),
        size: int = 3,
        min: int = -(2**31),
        max: int = 2**31 - 1,
        **kw,
    ):
        super().__init__(default, **kw)
        self.size = size
        self.min = min
        self.max = max

    def validate(self, value: Any) -> tuple:
        if not hasattr(value, "__iter__"):
            value = (int(value),) * self.size
        result = []
        for i, v in enumerate(value):
            if i >= self.size:
                break
            iv = int(v)
            iv = max(self.min, min(self.max, iv))
            result.append(iv)
        while len(result) < self.size:
            result.append(0)
        return tuple(result)


class PropertyGroup:
    """Base class for custom property structs (bpy.types.PropertyGroup equivalent).

    Use PropertyGroup to define structured data that can be stored on
    operators, panels, or the scene. Properties are defined as class
    attributes using Property subclasses.

    Example:
        class MaterialSettings(PropertyGroup):
            color = FloatVectorProperty(default=(1, 1, 1), size=3)
            roughness = FloatProperty(default=0.5, min=0, max=1)
            metallic = FloatProperty(default=0.0, min=0, max=1)
    """

    _property_values: dict
    _runtime_properties: dict
    _instances: dict = {}  # Class name -> singleton instance
    _value_cache: dict = {}  # Class name -> cached values (for hot-reload)
    _runtime_cache: dict = {}  # Class name -> cached runtime properties (for hot-reload)

    def __init_subclass__(cls, **kwargs):
        """Track subclasses for instance management."""
        super().__init_subclass__(**kwargs)
        cls._instances[cls.__name__] = None

    def __init__(self) -> None:
        self._property_values = {}
        self._runtime_properties = {}
        self._init_properties()
        self._restore_values()

    @classmethod
    def get_instance(cls: Type[T]) -> T:
        """Get or create singleton instance of this PropertyGroup.

        This ensures Python is the source of truth - only one instance exists.
        """
        if cls._instances.get(cls.__name__) is None:
            cls._instances[cls.__name__] = cls()
        return cls._instances[cls.__name__]

    def _save_values(self) -> None:
        """Save current values before hot-reload."""
        PropertyGroup._value_cache[self.__class__.__name__] = dict(self._property_values)
        PropertyGroup._runtime_cache[self.__class__.__name__] = dict(self._runtime_properties)

    def _restore_values(self) -> None:
        """Restore values after hot-reload (if cached)."""
        cached = PropertyGroup._value_cache.get(self.__class__.__name__)
        if cached:
            descriptors = self._get_property_descriptors()
            for key, value in cached.items():
                if key in self._property_values:
                    prop = descriptors.get(key)
                    if prop is not None:
                        try:
                            self._property_values[key] = prop.validate(value)
                        except (ValueError, TypeError):
                            pass
                    else:
                        self._property_values[key] = value
        runtime_cached = PropertyGroup._runtime_cache.get(self.__class__.__name__)
        if runtime_cached:
            self._runtime_properties = dict(runtime_cached)
            for name, prop in self._runtime_properties.items():
                if name not in self._property_values:
                    self._property_values[name] = prop.default

    @classmethod
    def clear_cache(cls, class_name: str = "") -> None:
        """Clear hot-reload value cache, optionally for a single class."""
        if class_name:
            cls._value_cache.pop(class_name, None)
            cls._runtime_cache.pop(class_name, None)
        else:
            cls._value_cache.clear()
            cls._runtime_cache.clear()

    def _init_properties(self) -> None:
        """Initialize property values from class-level Property descriptors."""
        for name, prop in self._get_property_descriptors().items():
            self._property_values[name] = prop.default

    @classmethod
    def _get_property_descriptors(cls) -> dict[str, Property]:
        """Get all Property descriptors defined on this class."""
        result = {}
        for klass in cls.__mro__:
            if klass is PropertyGroup or klass is object:
                continue
            for name, value in vars(klass).items():
                if isinstance(value, Property) and name not in result:
                    result[name] = value
        return result

    def add_property(self, name: str, prop: Property) -> None:
        """Add a property at runtime."""
        self._runtime_properties[name] = prop
        if name not in self._property_values:
            self._property_values[name] = prop.default

    def remove_property(self, name: str) -> None:
        """Remove a runtime property."""
        self._runtime_properties.pop(name, None)
        self._property_values.pop(name, None)

    def get_all_properties(self) -> dict[str, Property]:
        """Get both class-level and runtime properties."""
        result = self._get_property_descriptors()
        result.update(self._runtime_properties)
        return result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if hasattr(self, '_runtime_properties') and name in self._runtime_properties:
            return self._property_values.get(name, self._runtime_properties[name].default)
        props = self._get_property_descriptors()
        if name in props:
            return self._property_values.get(name, props[name].default)
        raise AttributeError(f"'{type(self).__name__}' has no property '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        prop = None
        if hasattr(self, '_runtime_properties') and name in self._runtime_properties:
            prop = self._runtime_properties[name]
            self._property_values[name] = prop.validate(value)
        else:
            props = self._get_property_descriptors()
            if name in props:
                prop = props[name]
                self._property_values[name] = prop.validate(value)
            else:
                object.__setattr__(self, name, value)
                return

        if prop and prop.update:
            try:
                prop.update(self, None)
            except Exception as e:
                LOG.warning(f"Property update callback error for '{name}': {e}")

    def get(self, prop_id: str) -> Any:
        """Get property value by ID."""
        return getattr(self, prop_id)

    def set(self, prop_id: str, value: Any) -> None:
        """Set property value by ID."""
        setattr(self, prop_id, value)


class CollectionProperty(Property):
    """Property that stores a list of PropertyGroup items.

    CollectionProperty is for storing custom plugin data (presets, settings lists),
    NOT for scene selection - use ctx.selected_nodes for that.

    Example:
        class BrushPreset(PropertyGroup):
            name = StringProperty(default="Untitled")
            radius = FloatProperty(default=10.0, min=1, max=100)

        class BrushSettings(PropertyGroup):
            presets = CollectionProperty(type=BrushPreset)
            active_index = IntProperty()
    """

    def __init__(self, type: Type[T], **kw) -> None:
        super().__init__(default=None, **kw)
        self.item_type = type
        self._items: List[T] = []

    def add(self) -> T:
        """Add a new item to the collection and return it."""
        item = self.item_type()
        self._items.append(item)
        return item

    def remove(self, index: int) -> None:
        """Remove item at given index."""
        if 0 <= index < len(self._items):
            del self._items[index]

    def clear(self) -> None:
        """Remove all items from the collection."""
        self._items.clear()

    def move(self, from_idx: int, to_idx: int) -> None:
        """Move item from one index to another."""
        if 0 <= from_idx < len(self._items) and 0 <= to_idx < len(self._items):
            item = self._items.pop(from_idx)
            self._items.insert(to_idx, item)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def validate(self, value: Any) -> List[T]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return list(value)


class PointerProperty(Property):
    """Property that references another object (bpy.props.PointerProperty equivalent).

    PointerProperty creates a reference to another type. On first access,
    the referenced object is auto-created if it doesn't exist.

    Example:
        class SceneSettings(PropertyGroup):
            quality = EnumProperty(items=[('LOW', 'Low', ''), ('HIGH', 'High', '')])

        # On a panel or operator:
        settings = PointerProperty(type=SceneSettings)
    """

    def __init__(self, type: Type[T], **kw) -> None:
        super().__init__(default=None, **kw)
        self.pointer_type = type
        self._instance: Optional[T] = None

    def get_instance(self) -> T:
        """Get or create the referenced instance."""
        if self._instance is None:
            self._instance = self.pointer_type()
        return self._instance

    def validate(self, value: Any) -> Any:
        if value is None:
            self._instance = None
        elif isinstance(value, self.pointer_type):
            self._instance = value
        return self._instance


class TensorProperty(Property):
    """Property that stores a GPU tensor with shape/dtype/device validation.

    TensorProperty is designed for storing tensor data with automatic
    validation of shape patterns, data type, and device placement.

    Shape patterns use -1 for variable dimensions:
        (-1, 3)  = any N x 3 tensor
        (-1,)    = any 1D tensor
        (100, 3) = exactly 100 x 3 tensor

    Example:
        class GaussianSettings(PropertyGroup):
            positions = TensorProperty(shape=(-1, 3), dtype="float32", device="cuda")
            mask = TensorProperty(shape=(-1,), dtype="bool", device="cuda")

        settings = GaussianSettings.get_instance()
        settings.positions = lf.Tensor.zeros([100, 3], device="cuda")
    """

    def __init__(
        self,
        shape: tuple = (),
        dtype: str = "float32",
        device: str = "cuda",
        **kw,
    ) -> None:
        super().__init__(default=None, **kw)
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def validate(self, value: Any) -> Any:
        if value is None:
            return None

        try:
            import lichtfeld as lf
        except ImportError:
            return value

        if not isinstance(value, lf.Tensor):
            raise TypeError(f"Expected lf.Tensor, got {type(value).__name__}")

        if self.dtype and str(value.dtype) != self.dtype:
            raise ValueError(f"Expected dtype '{self.dtype}', got '{value.dtype}'")

        if self.device and str(value.device) != self.device:
            raise ValueError(f"Expected device '{self.device}', got '{value.device}'")

        if self.shape:
            value_shape = tuple(value.shape)
            if len(value_shape) != len(self.shape):
                raise ValueError(
                    f"Expected {len(self.shape)}D tensor, got {len(value_shape)}D"
                )
            for i, (expected, actual) in enumerate(zip(self.shape, value_shape)):
                if expected != -1 and expected != actual:
                    raise ValueError(
                        f"Shape mismatch at dim {i}: expected {expected}, got {actual}"
                    )

        return value
