# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Python facade for the C++ reactive app store."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")

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
        return getattr(AppStore, field).value
    except Exception:
        return fallback


class StoreSignal(Generic[T]):
    """Signal-shaped wrapper around one C++ app-store field."""

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
                logger.error("Store signal '%s' callback error: %s", self._field, e)

    def subscribe(self, callback: Callable[[T], None]) -> Callable[[], None]:
        native = _native_store()
        if native is not None:
            token = native.subscribe(self._field, callback)

            def unsubscribe() -> None:
                native.unsubscribe(token)

            return unsubscribe

        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subscribers[sub_id] = callback

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(sub_id, None)

        return unsubscribe

    def subscribe_as(self, owner: str, callback: Callable[[T], None]) -> Callable[[], None]:
        from .subscription_registry import SubscriptionRegistry

        unsub = self.subscribe(callback)
        return SubscriptionRegistry.instance().register(owner, unsub)

    def peek(self) -> T:
        return self.value


class AppStore:
    iteration = StoreSignal[int]("iteration", 0)
    total_iterations = StoreSignal[int]("total_iterations", 0)
    loss = StoreSignal[float]("loss", 0.0)
    num_gaussians = StoreSignal[int]("num_gaussians", 0)
    max_gaussians = StoreSignal[int]("max_gaussians", 0)
    training_running = StoreSignal[bool]("training_running", False)
    training_state = StoreSignal[str]("training_state", "idle")
    trainer_loaded = StoreSignal[bool]("trainer_loaded", False)
    eval_psnr = StoreSignal[float | None]("eval_psnr", None)
    eval_ssim = StoreSignal[float | None]("eval_ssim", None)
    scene_generation = StoreSignal[int]("scene_generation", 0)
    selection_generation = StoreSignal[int]("selection_generation", 0)
    fps = StoreSignal[float]("fps", 0.0)
    mode_text = StoreSignal[str]("mode_text", "")
    active_tool = StoreSignal[str]("active_tool", "")
    active_submode = StoreSignal[str]("active_submode", "")
    transform_space = StoreSignal[int]("transform_space", 0)
    pivot_mode = StoreSignal[int]("pivot_mode", 0)
    import_overlay_state = StoreSignal[dict[str, object]]("import_overlay_state", {})
    video_export_overlay_state = StoreSignal[dict[str, object]]("video_export_overlay_state", {})
    mesh2splat_state = StoreSignal[dict[str, object]]("mesh2splat_state", {})
    splat_simplify_state = StoreSignal[dict[str, object]]("splat_simplify_state", {})


class _BatchContext:
    __slots__ = ("depth", "pending_notifications")

    def __init__(self) -> None:
        self.depth = 0
        self.pending_notifications: set[StoreSignal[object]] = set()

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
