# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Python facade for the C++ reactive app store."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


def _native_store():
    try:
        import lichtfeld as lf

        return getattr(lf.ui, "store", None)
    except Exception:
        return None


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
        self._fallback = new_value
        native = _native_store()
        if native is not None:
            native.set(self._field, new_value)
            return

        with self._lock:
            callbacks = list(self._subscribers.values())
        for callback in callbacks:
            callback(new_value)

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
    fps = StoreSignal[float]("fps", 0.0)
    mode_text = StoreSignal[str]("mode_text", "")


@contextmanager
def batch_updates():
    native = _native_store()
    if native is None:
        yield
        return

    native.begin_batch()
    try:
        yield
    finally:
        native.end_batch()
