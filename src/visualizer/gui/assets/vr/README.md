Bundled VR hand models

These models are drawn as the user's hands while viewing in VR.

Source
- WebXR Input Profiles, by the W3C Immersive Web Working Group:
  https://github.com/immersive-web/webxr-input-profiles
  (`packages/assets/profiles/generic-hand`)

License
- MIT, Copyright (c) 2019 Amazon:
  https://github.com/immersive-web/webxr-input-profiles/blob/main/packages/assets/LICENSE.md
- MIT is GPL-compatible per GNU: https://www.gnu.org/licenses/license-list.html#Expat

Included files
- `hand_left.glb`
  Source: `packages/assets/profiles/generic-hand/left.glb`
- `hand_right.glb`
  Source: `packages/assets/profiles/generic-hand/right.glb`

Notes
- The files are unmodified. Each holds one skinned mesh with one bone per hand
  joint, named after the WebXR hand joints, which are the joints of OpenXR's
  `XR_EXT_hand_tracking`.
