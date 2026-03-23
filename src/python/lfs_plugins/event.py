import lichtfeld.ui as ui

_KEY_NAMES = {
    ui.key.SPACE: "SPACE",
    ui.key.ESCAPE: "ESC",
    ui.key.ENTER: "ENTER",
    ui.key.KP_ENTER: "ENTER",
    ui.key.TAB: "TAB",
    ui.key.BACKSPACE: "BACKSPACE",
    ui.key.INSERT: "INSERT",
    ui.key.DELETE: "DELETE",
    ui.key.RIGHT: "RIGHT",
    ui.key.LEFT: "LEFT",
    ui.key.DOWN: "DOWN",
    ui.key.UP: "UP",
}

for i in range(26):
    _KEY_NAMES[ui.key.A + i] = chr(ord("A") + i)

for i in range(10):
    _KEY_NAMES[ui.key.NUM_0 + i] = str(i)

for i in range(10):
    kp_key = getattr(ui.key, f"KP_{i}", None)
    if kp_key is not None:
        _KEY_NAMES[kp_key] = str(i)

for i in range(12):
    _KEY_NAMES[ui.key.F1 + i] = f"F{i+1}"


class Event:
    TYPE_MOUSE_BUTTON = ui.ModalEventType.MouseButton
    TYPE_MOUSE_MOVE = ui.ModalEventType.MouseMove
    TYPE_SCROLL = ui.ModalEventType.Scroll
    TYPE_KEY = ui.ModalEventType.Key

    def __init__(self, raw_event):
        self._raw = raw_event

    @property
    def type_str(self) -> str:
        t = self._raw.type
        if t == self.TYPE_MOUSE_BUTTON:
            btn = self._raw.button
            if btn == ui.mouse.LEFT:
                return "LEFTMOUSE"
            if btn == ui.mouse.RIGHT:
                return "RIGHTMOUSE"
            if btn == ui.mouse.MIDDLE:
                return "MIDDLEMOUSE"
            return "MOUSE"
        if t == self.TYPE_MOUSE_MOVE:
            return "MOUSEMOVE"
        if t == self.TYPE_SCROLL:
            if self._raw.scroll_y > 0:
                return "WHEELUPMOUSE"
            if self._raw.scroll_y < 0:
                return "WHEELDOWNMOUSE"
            return "WHEEL"
        if t == self.TYPE_KEY:
            return _KEY_NAMES.get(self._raw.key, f"KEY_{self._raw.key}")
        return "NONE"

    @property
    def value_str(self) -> str:
        t = self._raw.type
        if t == self.TYPE_MOUSE_BUTTON or t == self.TYPE_KEY:
            return "PRESS" if self._raw.action == ui.action.PRESS else "RELEASE"
        if t == self.TYPE_SCROLL:
            return "PRESS"
        return "NONE"

    @property
    def is_mouse_button(self) -> bool:
        return self._raw.type == self.TYPE_MOUSE_BUTTON

    @property
    def is_mouse_move(self) -> bool:
        return self._raw.type == self.TYPE_MOUSE_MOVE

    @property
    def is_scroll(self) -> bool:
        return self._raw.type == self.TYPE_SCROLL

    @property
    def is_key(self) -> bool:
        return self._raw.type == self.TYPE_KEY

    @property
    def mouse_x(self) -> float:
        return self._raw.x

    @property
    def mouse_y(self) -> float:
        return self._raw.y

    @property
    def delta_x(self) -> float:
        return self._raw.delta_x

    @property
    def delta_y(self) -> float:
        return self._raw.delta_y

    @property
    def scroll_x(self) -> float:
        return self._raw.scroll_x

    @property
    def scroll_y(self) -> float:
        return self._raw.scroll_y

    @property
    def button(self) -> int:
        return self._raw.button

    @property
    def key(self) -> int:
        return self._raw.key

    @property
    def action(self) -> int:
        return self._raw.action

    @property
    def mods(self) -> int:
        return self._raw.mods

    @property
    def shift(self) -> bool:
        return bool(self._raw.mods & ui.mod.SHIFT)

    @property
    def ctrl(self) -> bool:
        return bool(self._raw.mods & ui.mod.CONTROL)

    @property
    def alt(self) -> bool:
        return bool(self._raw.mods & ui.mod.ALT)

    @property
    def over_gui(self) -> bool:
        return getattr(self._raw, "over_gui", False)


_last_event: Event = None


def set_last_event(event: Event):
    global _last_event
    _last_event = event


def get_last_event() -> Event:
    return _last_event
