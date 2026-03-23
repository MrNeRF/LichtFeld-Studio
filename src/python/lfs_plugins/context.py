# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin context - data objects passed to capability handlers."""

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SceneContext:
    """Scene data passed to capability handlers."""

    scene: Any  # PyScene object from lf.rendering.get_render_scene()

    def set_selection_mask(self, mask: Any) -> None:
        """Apply a selection mask to the scene."""
        if self.scene is not None:
            self.scene.set_selection_mask(mask)


@dataclass
class ViewContext:
    """Viewport data passed to capability handlers."""

    image: Any  # PyTensor [H, W, 3]
    screen_positions: Optional[Any]  # PyTensor [N, 2] or None
    width: int
    height: int
    fov: float
    rotation: Any  # PyTensor [3, 3]
    translation: Any  # PyTensor [3]


class CapabilityBroker:
    """Broker for inter-capability invocation."""

    _local = threading.local()

    def __init__(self, registry: "CapabilityRegistry"):
        self._registry = registry

    def invoke(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke another capability."""
        stack = getattr(CapabilityBroker._local, 'call_stack', None)
        if stack is None:
            stack = set()
            CapabilityBroker._local.call_stack = stack
        if name in stack:
            return {"success": False, "error": f"Circular call: {name}"}
        stack.add(name)
        try:
            return self._registry.invoke(name, args or {})
        finally:
            stack.discard(name)

    def has(self, name: str) -> bool:
        return self._registry.has(name)

    def list_all(self) -> List[str]:
        return [cap.name for cap in self._registry.list_all()]


@dataclass
class PluginContext:
    """Context passed to capability handlers."""

    scene: Optional[SceneContext]
    view: Optional[ViewContext]
    capabilities: CapabilityBroker

    @classmethod
    def build(cls, registry: "CapabilityRegistry", include_view: bool = True) -> "PluginContext":
        """Build context from current application state."""
        try:
            import lichtfeld as lf
        except ModuleNotFoundError as exc:
            if exc.name != "lichtfeld":
                raise
            return cls(
                scene=None,
                view=None,
                capabilities=CapabilityBroker(registry),
            )

        # Get scene
        scene_ctx = None
        try:
            py_scene = lf.get_render_scene()
            if py_scene is not None:
                scene_ctx = SceneContext(scene=py_scene)
        except Exception as e:
            lf.log.error(f"Failed to get scene: {e}")

        # Get view if requested
        view_ctx = None
        if include_view:
            try:
                viewport = lf.capture_viewport()
                view_info = lf.get_current_view()
                if viewport is not None and view_info is not None:
                    view_ctx = ViewContext(
                        image=viewport.image,
                        screen_positions=viewport.screen_positions,
                        width=view_info.width,
                        height=view_info.height,
                        fov=view_info.fov_x,
                        rotation=view_info.rotation,
                        translation=view_info.translation,
                    )
            except Exception:
                pass

        return cls(
            scene=scene_ctx,
            view=view_ctx,
            capabilities=CapabilityBroker(registry),
        )


from .capabilities import CapabilityRegistry  # noqa: E402
