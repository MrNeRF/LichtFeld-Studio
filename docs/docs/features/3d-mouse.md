# 3D mouse navigation

LichtFeld Studio supports 3Dconnexion SpaceMouse devices on Linux and Windows.
Connect the device before or after starting the application. Open **Preferences >
Input > 3D mouse** to see its name and backend, enable navigation, adjust translation
and rotation speeds, or allow roll by turning off **Keep horizon level**.

In **Orbit** and **Trackball**, the scene follows the cap: push right to move the
scene right, pull toward you to bring it closer, and tilt or twist to orbit the
current mouse-orbit pivot. Pan and dolly scale with distance to that pivot.
**Fly** and **Drone** move and turn the camera instead, using the navigation speed
preference. **Walk** also follows the camera, but retains collision, floor support,
and its horizontal walking plane; lifting the cap does not leave the floor.

Navigation uses the focused viewport in independent split view. Input is ignored
while the application is unfocused, a mouse button is held, or a modal tool or text
field is active. A small dead zone and quadratic response reduce accidental
cross-axis movement without disabling simultaneous six-axis input. Releasing the
cap stops movement immediately; missing reports time out after 250 ms.

On the Compact, the **right button** frames the selection (or the scene when there
is no selection); the **left button** returns to the viewport's saved Home view.
These relocation actions leave Walk mode before moving the camera.
Only buttons 1 and 2 have default actions on larger models. Device LEDs, displays,
and vendor radial menus are not configured by LichtFeld Studio.

## Linux setup

The preferred backend uses SDL3 HID. Many distributions restrict access to
`/dev/hidraw*`. To grant the active local user access, create
`/etc/udev/rules.d/70-3dconnexion.rules` with:

```udev
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", TAG+="uaccess"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="046d", TAG+="uaccess"
```

Then run `sudo udevadm control --reload-rules` and unplug and reconnect the device.
The `046d` rule includes other Logitech HID devices; add an
`ATTRS{idProduct}=="c626"` condition if you want to limit it to a particular model.
The `uaccess` tag grants access to the active local login session, not arbitrary
remote users. No world-writable permissions or membership in the `input` group
are required.

Alternatively, install your distribution's **spacenavd** package and start its
service (typically `sudo systemctl enable --now spacenavd`). If HID cannot be
opened, LichtFeld connects to `/var/run/spnav.sock` automatically. It uses the
daemon's native socket protocol without requiring libspnav. The daemon's axis
mapping and sensitivity settings still apply; use its default mapping initially.
Older devices with nonstandard HID layouts can be used through the daemon.

## Windows and 3DxWare

SDL3 HID is tried first. If the device cannot be opened, LichtFeld uses Windows
Raw Input on the application's main window. Exactly one backend supplies motion.
Both native APIs can operate without installing 3DxWare. The vendor describes
HID and Raw Input as independent of its driver; SDL opens HID handles with shared
read/write access. A driver configuration or another application's exclusive
handle can still require the fallback. [Vendor explanation](https://forum.3dconnexion.com/viewtopic.php?t=43031),
[SDL Windows HID implementation](https://github.com/libsdl-org/SDL/blob/main/src/hidapi/windows/hid.c).

3DxWare may treat an unfamiliar application as a keyboard/mouse/joystick target
and send extra keys or wheel events. These are separate from six-axis navigation.
For a quick check, use **Stop 3DxWare** in the Windows Start menu, then restart
LichtFeld. To retain the driver, create an application-specific profile for
`LichtFeld-Studio.exe` in `%APPDATA%\3Dconnexion\3DxWare\Cfg`, using your installed
driver's profile/template format. Set its `Transport` to `HID` (or `RawInput` if
that is the backend shown in Preferences), leave axis mappings at their native
defaults, and remove keyboard/mouse/scroll output mappings for this application.
Restart 3DxWare after editing. Do not remove the driver's global default profiles.
[Profile location](https://3dconnexion.com/us/support/faq/where-can-i-find-my-3dxware-configuration-files/),
[transport and axis configuration](https://forum.3dconnexion.com/viewtopic.php?t=43031).

## Protocol references

Supported USB IDs come from [spacenavd's device table](https://github.com/FreeSpacenav/spacenavd/blob/master/src/dev.c).
Common models have a nominal range of ±350; the HID report descriptor supplies
the logical range when available. The daemon backend uses its ±500 normalization
and honors daemon-side sensitivity. Split/combined motion reports and button
arrays follow the formats used by [Blender's Windows backend](https://github.com/blender/blender/blob/main/intern/ghost/intern/GHOST_SystemWin32.cc).
The socket format is eight native-endian signed 32-bit words, as defined by
[libspnav](https://github.com/FreeSpacenav/libspnav/blob/master/src/spnav.c)
and [spacenavd](https://github.com/FreeSpacenav/spacenavd/blob/master/src/proto_unix.c).
