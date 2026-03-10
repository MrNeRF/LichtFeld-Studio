"""Animation system API"""



class AnimationTrack:
    """A single property animation track with keyframes"""

    @property
    def id(self) -> int:
        """Track ID"""

    @property
    def target_path(self) -> str:
        """Target property path (e.g., 'node:Model.transform')"""

    @property
    def keyframe_count(self) -> int:
        """Number of keyframes"""

    def add_keyframe(self, time: float, value: object, easing: str = 'ease_in_out') -> None:
        """Add a keyframe at the specified time"""

    def remove_keyframe(self, index: int) -> None:
        """Remove keyframe at index"""

    def evaluate(self, time: float) -> object:
        """Evaluate the track at the given time"""

    def keyframes(self) -> list:
        """Get all keyframes as a list of dicts"""

class AnimationClip:
    """Multi-track animation container"""

    @property
    def name(self) -> str:
        """Clip name"""

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def track_count(self) -> int:
        """Number of tracks"""

    @property
    def duration(self) -> float:
        """Total duration of the clip"""

    def add_track(self, value_type: str, target_path: str) -> AnimationTrack:
        """
        Add a new track. value_type: 'bool', 'int', 'float', 'vec2', 'vec3', 'vec4', 'quat', 'mat4'
        """

    def remove_track(self, id: int) -> None:
        """Remove track by ID"""

    def get_track(self, id: int) -> AnimationTrack | None:
        """Get track by ID"""

    def get_track_by_path(self, path: str) -> AnimationTrack | None:
        """Get track by target path"""

    def tracks(self) -> list:
        """Get all tracks"""

    def evaluate(self, time: float) -> dict:
        """Evaluate all tracks at the given time, returns dict of path -> value"""

class Timeline:
    """Animation timeline with camera keyframes and multi-track clips"""

    @property
    def has_animation_clip(self) -> bool:
        """True if an animation clip exists"""

    @property
    def keyframe_count(self) -> int:
        """Number of camera keyframes"""

    @property
    def camera_duration(self) -> float:
        """Duration of camera animation"""

    @property
    def total_duration(self) -> float:
        """Total duration including all clips"""

    def animation_clip(self) -> AnimationClip:
        """Get or create the animation clip"""

    def evaluate_clip(self, time: float) -> dict:
        """Evaluate the animation clip at the given time"""
