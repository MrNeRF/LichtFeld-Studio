"""Shared Media Ingest API"""

import enum
import os
import pathlib


class SelectionMode(enum.Enum):
    FPS = 0

    Interval = 1

class ResizeMode(enum.Enum):
    Original = 0

    Scale = 1

    Custom = 2

class SharpnessMethod(enum.Enum):
    Laplacian = 0

    Tenengrad = 1

    Combined = 2

class FrameFileFormat(enum.Enum):
    PNG = 0

    JPEG = 1

class StreamKind(enum.Enum):
    Video = 0

    Audio = 1

    Subtitle = 2

    Data = 3

    Attachment = 4

    Unknown = 5

class OrientationSource(enum.Enum):
    Unspecified = 0

    RotateTag = 1

    DisplayMatrix = 2

class ProbeDepth(enum.Enum):
    Headers = 0

    StreamInfo = 1

class FramePixelFormat(enum.Enum):
    RGB8 = 0

class TimestampOrigin(enum.Enum):
    Missing = 0

    BestEffort = 1

    Presentation = 2

class Selection:
    def __init__(self) -> None: ...

    @property
    def mode(self) -> SelectionMode: ...

    @mode.setter
    def mode(self, arg: SelectionMode, /) -> None: ...

    @property
    def fps(self) -> float: ...

    @fps.setter
    def fps(self, arg: float, /) -> None: ...

    @property
    def interval(self) -> int: ...

    @interval.setter
    def interval(self, arg: int, /) -> None: ...

class Geometry:
    def __init__(self) -> None: ...

    @property
    def mode(self) -> ResizeMode: ...

    @mode.setter
    def mode(self, arg: ResizeMode, /) -> None: ...

    @property
    def scale(self) -> float: ...

    @scale.setter
    def scale(self, arg: float, /) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def clockwise_rotation(self) -> int: ...

    @clockwise_rotation.setter
    def clockwise_rotation(self, arg: int, /) -> None: ...

class Sharpness:
    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    @property
    def method(self) -> SharpnessMethod: ...

    @method.setter
    def method(self, arg: SharpnessMethod, /) -> None: ...

    @property
    def threshold(self) -> float: ...

    @threshold.setter
    def threshold(self, arg: float, /) -> None: ...

    @property
    def window_candidates(self) -> int: ...

    @window_candidates.setter
    def window_candidates(self, arg: int, /) -> None: ...

    @property
    def window(self) -> bool: ...

    @window.setter
    def window(self, arg: bool, /) -> None: ...

class IngestRequest:
    def __init__(self) -> None: ...

    @property
    def input(self) -> pathlib.Path: ...

    @input.setter
    def input(self, arg: str | os.PathLike, /) -> None: ...

    @property
    def selection(self) -> Selection: ...

    @selection.setter
    def selection(self, arg: Selection, /) -> None: ...

    @property
    def geometry(self) -> Geometry: ...

    @geometry.setter
    def geometry(self, arg: Geometry, /) -> None: ...

    @property
    def sharpness(self) -> Sharpness: ...

    @sharpness.setter
    def sharpness(self, arg: Sharpness, /) -> None: ...

    @property
    def start_seconds(self) -> float: ...

    @start_seconds.setter
    def start_seconds(self, arg: float, /) -> None: ...

    @property
    def end_seconds(self) -> float: ...

    @end_seconds.setter
    def end_seconds(self, arg: float, /) -> None: ...

    @property
    def convert_hdr_to_sdr(self) -> bool: ...

    @convert_hdr_to_sdr.setter
    def convert_hdr_to_sdr(self, arg: bool, /) -> None: ...

    @property
    def allow_hardware_decode(self) -> bool: ...

    @allow_hardware_decode.setter
    def allow_hardware_decode(self, arg: bool, /) -> None: ...

class FileFrameSinkOptions:
    def __init__(self) -> None: ...

    @property
    def output_directory(self) -> pathlib.Path: ...

    @output_directory.setter
    def output_directory(self, arg: str | os.PathLike, /) -> None: ...

    @property
    def filename_pattern(self) -> str: ...

    @filename_pattern.setter
    def filename_pattern(self, arg: str, /) -> None: ...

    @property
    def format(self) -> FrameFileFormat: ...

    @format.setter
    def format(self, arg: FrameFileFormat, /) -> None: ...

    @property
    def jpeg_quality(self) -> int: ...

    @jpeg_quality.setter
    def jpeg_quality(self, arg: int, /) -> None: ...

class FileExtraction:
    def __init__(self) -> None: ...

    @property
    def files(self) -> FileFrameSinkOptions: ...

    @files.setter
    def files(self, arg: FileFrameSinkOptions, /) -> None: ...

    @property
    def write_metadata(self) -> bool: ...

    @write_metadata.setter
    def write_metadata(self, arg: bool, /) -> None: ...

class Rational:
    @property
    def numerator(self) -> int: ...

    @property
    def denominator(self) -> int: ...

class Timestamp:
    @property
    def ticks(self) -> int: ...

    @property
    def time_base(self) -> Rational: ...

class ColorDescription:
    @property
    def primaries(self) -> str | None: ...

    @property
    def transfer(self) -> str | None: ...

    @property
    def matrix(self) -> str | None: ...

    @property
    def range(self) -> str | None: ...

    @property
    def component_depth(self) -> int | None: ...

class Orientation:
    @property
    def source(self) -> OrientationSource: ...

    @property
    def rotate_tag(self) -> str | None: ...

    @property
    def display_matrix(self) -> list[int] | None: ...

    @property
    def clockwise_degrees(self) -> float | None: ...

    @property
    def reflected(self) -> bool | None: ...

class StreamDescription:
    @property
    def index(self) -> int: ...

    @property
    def kind(self) -> StreamKind: ...

    @property
    def codec(self) -> str: ...

    @property
    def width(self) -> int | None: ...

    @property
    def height(self) -> int | None: ...

    @property
    def pixel_format(self) -> str | None: ...

    @property
    def time_base(self) -> Rational | None: ...

    @property
    def nominal_frame_rate(self) -> Rational | None: ...

    @property
    def average_frame_rate(self) -> Rational | None: ...

    @property
    def sample_aspect_ratio(self) -> Rational | None: ...

    @property
    def start(self) -> Timestamp | None: ...

    @property
    def duration(self) -> Timestamp | None: ...

    @property
    def declared_frame_count(self) -> int | None: ...

    @property
    def attached_picture(self) -> bool: ...

    @property
    def default_disposition(self) -> bool: ...

    @property
    def color(self) -> ColorDescription: ...

    @property
    def orientation(self) -> Orientation: ...

class MediaDescription:
    @property
    def container(self) -> str: ...

    @property
    def stream_info_probed(self) -> bool: ...

    @property
    def start(self) -> Timestamp | None: ...

    @property
    def duration(self) -> Timestamp | None: ...

    @property
    def streams(self) -> list[StreamDescription]: ...

    @property
    def selected_video_stream(self) -> int | None: ...

class FrameLayout:
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def row_stride(self) -> int: ...

    @property
    def format(self) -> FramePixelFormat: ...

class FrameInfo:
    @property
    def source_timestamp(self) -> Timestamp | None: ...

    @property
    def timestamp_origin(self) -> TimestampOrigin: ...

    @property
    def decode_index(self) -> int: ...

    @property
    def delivery_index(self) -> int: ...

    @property
    def relative_seconds(self) -> float: ...

    @property
    def legacy_source_frame(self) -> int: ...

    @property
    def sharpness_score(self) -> float: ...

class IngestProgress:
    @property
    def processed(self) -> int: ...

    @property
    def estimated(self) -> int: ...

    @property
    def discarded(self) -> int: ...

class IngestReport:
    @property
    def frames_accepted(self) -> int: ...

    @property
    def discarded(self) -> int: ...

class IngestCapabilities:
    @property
    def software_decode(self) -> bool: ...

    @property
    def rgb8(self) -> bool: ...

    @property
    def png(self) -> bool: ...

    @property
    def jpeg(self) -> bool: ...

    @property
    def hardware_decode(self) -> bool: ...

    @property
    def hdr_to_sdr(self) -> bool: ...

class CodecBuildInfo:
    @property
    def ffmpeg_version(self) -> str: ...

    @property
    def ffmpeg_license(self) -> str: ...

    @property
    def ffmpeg_configuration(self) -> str: ...

class FrameSurface:
    @property
    def layout(self) -> FrameLayout: ...

    @property
    def info(self) -> FrameInfo: ...

    @property
    def pixels(self) -> bytes: ...

class MediaIngest:
    @staticmethod
    def capabilities() -> IngestCapabilities: ...

    @staticmethod
    def codec_build_info() -> CodecBuildInfo: ...

    @staticmethod
    def probe(path: str | os.PathLike, headers_only: bool = False, timeout_ms: int = 10000) -> MediaDescription: ...

    @staticmethod
    def extract_files(request: IngestRequest, files: FileExtraction, progress: object | None = None, cancelled: object | None = None) -> IngestReport: ...

    @staticmethod
    def extract(request: IngestRequest, payload_budget: int = 268435456, frame_limit: int = 100000, progress: object | None = None, cancelled: object | None = None) -> tuple: ...
