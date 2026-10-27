# Gamepad navigation

LichtFeld uses SDL3's semantic gamepad mapping on Windows, Linux and macOS.
Recognized controllers share the same logical sticks, triggers and buttons;
button names and physical labels may differ between manufacturers. Raw joysticks
without an SDL gamepad mapping are listed in the input test but cannot navigate.
This feature does not install vendor drivers or create SDL hardware mappings.

Open **Preferences > Input > Joypad** to enable navigation, choose a connected
controller and adjust the dead zone, rotation speed and movement multiplier.
Only one controller drives the camera. The sole usable controller is selected
automatically; multiple devices can be selected explicitly. Connection IDs are
temporary and are never stored in a profile. Disconnecting or changing the
controller revokes its viewport assignment. Reconnecting requires activation
again, even when a profile is restored for the same model.

Click the intended viewport once, then release the controls briefly to activate
navigation. Its assignment survives interaction with other application panels.
Modals, native file dialogs, Preferences, controller tests, text editing in the
Sequencer, keyboard camera movement and viewport mouse gestures suspend the
relevant controller operations. Resume with neutral controls after an
interruption. Scene and split-layout changes require a new viewport click.

The controls follow the current camera navigation mode. The default left shoulder
button cycles navigation modes, skipping unavailable modes, including Walk when
no usable collision volume exists. The right shoulder button cycles splats,
points, rings and center markers. Walk uses the existing collision settings and
ground support; this feature does not add jumping or a new physics system.

## Assignments and profiles

Expand **Assignments** to edit common controls or override a camera mode.
Clearing an assignment differs from inheriting its common value. The Sequencer
has its own context: while open, X/West adds or replaces a keyframe from the
assigned viewport, Y/North deletes the selected keyframe, and D-pad left/right
select adjacent keyframes and move that same viewport. Delete requires an
actual selection and protects the first keyframe. These edits use normal undo.
Sequencer bindings take precedence over camera bindings for the same control.

Named profiles store tuning and assignments independently of keyboard keymaps.
They do not store global enablement, a connected device ID or permission to move
a viewport. Optional model associations restore a profile using SDL's model GUID;
identical controllers share that association. When a model is remembered,
explicit profile selection, creation and import update its association. Disable
the remember checkbox to remove it. Import/export transfers one layout, not the
whole library or device associations.

Slider edits are coalesced after a short idle period and flushed when leaving
the page. A concurrent device or profile change discards an unfinished gesture.
Resetting the Joypad page resets its active layout; the general reset also
removes named profiles and model associations and disables navigation. Safe mode
continues to prevent writes to saved preferences.

## Input test

**Test inputs** displays the selected controller's sticks, triggers and buttons
without moving the camera. Navigation and diagnostic selection are independent.
An assignment capture is cancelled if its navigation device, connection
generation or configuration changes. Short button presses remain buffered until
their consumer reads them; diagnostic reads do not consume navigation commands.

## Integration notes

The viewer-thread service owns SDL handles and returns value snapshots.
Navigation resolves semantic actions before updating cameras; it does not
synthesize keyboard events. Discrete Sequencer actions do not require a
camera-compatible projection or sticks. Walk physics retains an explicit
viewport owner across neutral input and focus changes. Effective input, rather
than raw stick noise inside the neutral threshold, wakes the viewer.

Hardware checks should cover at least Xbox-style and PlayStation-style layouts,
USB/Bluetooth reconnection, two identical controllers, split views, modal
interruptions and a drifting stick. Virtual SDL tests cover routing and lifetime
rules but cannot certify vendor drivers or physical calibration.
