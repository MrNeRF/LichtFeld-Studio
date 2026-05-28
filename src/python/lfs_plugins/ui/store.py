# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Python facade for LichtFeld runtime state."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from threading import Lock
from typing import Generic, TypeAlias, TypeVar

from .signals import ComputedSignal, Signal

T = TypeVar("T")
DirtySpec: TypeAlias = str | Iterable[str] | None

logger = logging.getLogger(__name__)


def _native_store():
    try:
        import lichtfeld as lf

        return getattr(lf.ui, "store", None)
    except Exception:
        return None


def native_value(field: str, fallback: T) -> T:
    native = _native_store()
    if native is None:
        return fallback
    try:
        return getattr(RuntimeState, field).value
    except Exception:
        return fallback


class StateSignal(Generic[T]):
    """Signal-shaped wrapper around one native runtime-state field."""

    __slots__ = ("_field", "_fallback", "_lock", "_subscribers", "_next_id")

    def __init__(self, field: str, initial_value: T) -> None:
        self._field = field
        self._fallback = initial_value
        self._lock = Lock()
        self._subscribers: dict[int, Callable[[T], None]] = {}
        self._next_id = 0

    @property
    def value(self) -> T:
        native = _native_store()
        if native is not None:
            return native.get(self._field)
        return self._fallback

    @value.setter
    def value(self, new_value: T) -> None:
        native = _native_store()
        if native is not None:
            self._fallback = new_value
            native.set(self._field, new_value)
            return

        if self._fallback == new_value:
            return

        self._fallback = new_value
        if _batch_context.is_batching:
            _batch_context.pending_notifications.add(self)
            return

        self._notify()

    def _notify(self) -> None:
        with self._lock:
            callbacks = list(self._subscribers.values())
        for callback in callbacks:
            try:
                callback(self._fallback)
            except Exception as e:
                logger.error(
                    "Runtime state signal '%s' callback error: %s",
                    self._field,
                    e,
                )

    def subscribe(self, callback: Callable[[T], None]) -> Callable[[], None]:
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subscribers[sub_id] = callback

        native = _native_store()
        native_token = None
        if native is not None:
            native_token = native.subscribe(self._field, callback)

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(sub_id, None)
            if native is not None and native_token is not None:
                native.unsubscribe(native_token)

        return unsubscribe

    def subscribe_as(self, owner: str, callback: Callable[[T], None]) -> Callable[[], None]:
        from .subscription_registry import SubscriptionRegistry

        unsub = self.subscribe(callback)
        return SubscriptionRegistry.instance().register(owner, unsub)

    def peek(self) -> T:
        return self.value


def invalidate_panel(handle: object | None, dirty: DirtySpec = None) -> None:
    """Invalidate an RML data model handle using the most surgical available path."""
    if handle is None:
        return

    if dirty is None:
        request_update = getattr(handle, "request_update", None)
        if callable(request_update):
            request_update()
            return

        dirty_all = getattr(handle, "dirty_all", None)
        if callable(dirty_all):
            dirty_all()
        return

    if dirty == "*":
        dirty_all = getattr(handle, "dirty_all", None)
        if callable(dirty_all):
            dirty_all()
        return

    dirty_field = getattr(handle, "dirty", None)
    if not callable(dirty_field):
        return

    if isinstance(dirty, str):
        dirty_field(dirty)
        return

    for field in dirty:
        dirty_field(field)


class PanelStateBinding:
    """Own runtime-state subscriptions for a dirty-policy RML panel.

    Common panel code should use this instead of hand-written subscription
    lists. The binding keeps lifetime and data-model invalidation together.
    """

    __slots__ = ("_handle", "_unsubscribers")

    def __init__(self, handle: object | None = None) -> None:
        self._handle = handle
        self._unsubscribers: list[Callable[[], None]] = []

    @property
    def active(self) -> bool:
        return bool(self._unsubscribers)

    def set_handle(self, handle: object | None) -> PanelStateBinding:
        self._handle = handle
        return self

    def watch(
        self,
        *signals: StateSignal[object] | Signal[object] | ComputedSignal[object],
        refresh: Callable[[], None] | None = None,
        dirty: DirtySpec = None,
        immediate: bool = False,
    ) -> PanelStateBinding:
        """Refresh and invalidate the panel when any runtime-state signal changes."""

        def on_change(_value: object) -> None:
            if refresh is not None:
                refresh()
            invalidate_panel(self._handle, dirty)

        for signal in signals:
            self._unsubscribers.append(signal.subscribe(on_change))

        if immediate:
            on_change(None)

        return self

    def close(self) -> None:
        unsubscribers = self._unsubscribers
        self._unsubscribers = []
        for unsubscribe in unsubscribers:
            try:
                unsubscribe()
            except Exception:
                logger.exception("Panel state unsubscribe failed")


class RuntimeState:
    """Single public runtime-state surface for plugins.

    Native fields are backed by the C++ runtime-state bridge. A small set of
    Python-only compatibility signals remains here until those values have
    native producers.
    """

    iteration = StateSignal[int]("iteration", 0)
    total_iterations = StateSignal[int]("total_iterations", 0)
    loss = StateSignal[float]("loss", 0.0)
    num_gaussians = StateSignal[int]("num_gaussians", 0)
    max_gaussians = StateSignal[int]("max_gaussians", 0)
    training_running = StateSignal[bool]("training_running", False)
    training_state = StateSignal[str]("training_state", "idle")
    trainer_loaded = StateSignal[bool]("trainer_loaded", False)
    eval_psnr = StateSignal[float | None]("eval_psnr", None)
    eval_ssim = StateSignal[float | None]("eval_ssim", None)
    scene_generation = StateSignal[int]("scene_generation", 0)
    selection_generation = StateSignal[int]("selection_generation", 0)
    fps = StateSignal[float]("fps", 0.0)
    mode_text = StateSignal[str]("mode_text", "")
    active_tool = StateSignal[str]("active_tool", "")
    active_submode = StateSignal[str]("active_submode", "")
    transform_space = StateSignal[int]("transform_space", 0)
    pivot_mode = StateSignal[int]("pivot_mode", 0)
    import_overlay_state = StateSignal[dict[str, object]]("import_overlay_state", {})
    video_export_overlay_state = StateSignal[dict[str, object]](
        "video_export_overlay_state",
        {},
    )
    export_progress_state = StateSignal[dict[str, object]]("export_progress_state", {})
    mesh2splat_state = StateSignal[dict[str, object]]("mesh2splat_state", {})
    splat_simplify_state = StateSignal[dict[str, object]]("splat_simplify_state", {})
    scripts_generation = StateSignal[int]("scripts_generation", 0)
    language_generation = StateSignal[int]("language_generation", 0)

    # Compatibility names from the old AppState surface.
    is_training = training_running
    trainer_state = training_state
    has_trainer = trainer_loaded
    max_iterations = total_iterations
    psnr = ComputedSignal(
        lambda: (
            0.0
            if RuntimeState.eval_psnr.value is None
            else RuntimeState.eval_psnr.value
        ),
        [eval_psnr],
    )

    # Python-only compatibility values. These should move to native producers
    # before the legacy AppState alias is removed.
    has_scene = Signal(False, "has_scene")
    scene_path = Signal("", "scene_path")
    has_selection = Signal(False, "has_selection")
    selection_count = Signal(0, "selection_count")
    viewport_width = Signal(0, "viewport_width")
    viewport_height = Signal(0, "viewport_height")
    is_headless = Signal(False, "is_headless")

    training_progress = ComputedSignal(
        lambda: RuntimeState.iteration.value / RuntimeState.total_iterations.value
        if RuntimeState.total_iterations.value > 0
        else 0.0,
        [iteration, total_iterations],
    )
    can_start_training = ComputedSignal(
        lambda: RuntimeState.trainer_loaded.value
        and RuntimeState.training_state.value in ("idle", "ready"),
        [trainer_loaded, training_state],
    )

    @classmethod
    def reset(cls) -> None:
        """Reset Python fallback state. Primarily used by tests and shutdown."""
        cls.training_running.value = False
        cls.training_state.value = "idle"
        cls.trainer_loaded.value = False
        cls.iteration.value = 0
        cls.total_iterations.value = 0
        cls.loss.value = 0.0
        cls.num_gaussians.value = 0
        cls.max_gaussians.value = 0
        cls.eval_psnr.value = None
        cls.eval_ssim.value = None
        cls.scene_generation.value = 0
        cls.selection_generation.value = 0
        cls.fps.value = 0.0
        cls.mode_text.value = ""
        cls.active_tool.value = ""
        cls.active_submode.value = ""
        cls.transform_space.value = 0
        cls.pivot_mode.value = 0
        cls.import_overlay_state.value = {}
        cls.video_export_overlay_state.value = {}
        cls.export_progress_state.value = {}
        cls.mesh2splat_state.value = {}
        cls.splat_simplify_state.value = {}
        cls.scripts_generation.value = 0
        cls.language_generation.value = 0
        cls.has_scene.value = False
        cls.scene_path.value = ""
        cls.has_selection.value = False
        cls.selection_count.value = 0
        cls.viewport_width.value = 0
        cls.viewport_height.value = 0
        cls.is_headless.value = False

    @classmethod
    def bind_native_store(cls) -> None:
        """Compatibility no-op: native-backed fields are already live."""

    @classmethod
    def unbind_native_store(cls) -> None:
        """Compatibility no-op kept for older plugins."""


class _BatchContext:
    __slots__ = ("depth", "pending_notifications")

    def __init__(self) -> None:
        self.depth = 0
        self.pending_notifications: set[StateSignal[object]] = set()

    @property
    def is_batching(self) -> bool:
        return self.depth > 0

    def begin(self) -> None:
        self.depth += 1

    def end(self) -> None:
        self.depth = max(0, self.depth - 1)
        if self.depth != 0:
            return
        pending = list(self.pending_notifications)
        self.pending_notifications.clear()
        for signal in pending:
            signal._notify()


_batch_context = _BatchContext()


@contextmanager
def batch_updates():
    native = _native_store()
    if native is not None:
        native.begin_batch()
        try:
            yield
        finally:
            native.end_batch()
        return

    _batch_context.begin()
    try:
        yield
    finally:
        _batch_context.end()


# Compatibility aliases kept for existing plugins. New code should import
# RuntimeState, StateSignal, and PanelStateBinding from lfs_plugins.ui.
AppStore = RuntimeState
NativeAppStore = RuntimeState
AppState = RuntimeState
StoreSignal = StateSignal
PanelStoreBinding = PanelStateBinding

__all__ = [
    "RuntimeState",
    "StateSignal",
    "PanelStateBinding",
    "batch_updates",
    "invalidate_panel",
    "native_value",
    # Compatibility aliases.
    "AppState",
    "AppStore",
    "NativeAppStore",
    "StoreSignal",
    "PanelStoreBinding",
]
