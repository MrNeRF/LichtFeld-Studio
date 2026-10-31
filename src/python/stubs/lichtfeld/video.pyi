"""Video probing and frame extraction"""



def probe(path: str) -> dict:
    """
    Read container metadata without decoding: width, height, fps, duration, total_frames, rotation, spherical, is_hdr. Raises RuntimeError when the file cannot be opened.
    """

def extract_frames(video_path: str, output_dir: str, fps: float = 2.0, jpg_quality: int = 95, scale: float = 1.0, sharpness_enabled: bool = True, sharpness_threshold: float = 10.0, rotation: int = 0, start_time: float = 0.0, end_time: float = -1.0, convert_hdr_to_sdr: bool = True, filename_pattern: str = 'frame_%06d', on_progress: object | None = None, should_cancel: object | None = None) -> dict:
    """
    Extract JPEG frames at a fixed rate. Blocks with the GIL released; on_progress(current, total, discarded) and should_cancel() are called from the extraction thread. Returns {outcome, error} with outcome one of completed, cancelled, failed.
    """
