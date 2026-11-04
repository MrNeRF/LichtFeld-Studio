# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit dataset/model/viewer frames and bounded immutable camera metadata."""
import hashlib
import json
from pathlib import Path

SCHEMA = 1
MAX_CAMERAS = 10000
MAX_BYTES = 32 * 1024 * 1024
FLIP = ((1,0,0,0),(0,-1,0,0),(0,0,-1,0),(0,0,0,1))
IDENTITY = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
# Studio navigation uses +Y vertical; Blender's source world uses +Z.
BLENDER_TO_VISUALIZER = ((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1))


def matrix(value):
    import numpy as np
    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (4,4) or not np.isfinite(result).all() or
            np.max(np.abs(result)) > 1e10 or not np.allclose(result[3], [0,0,0,1], atol=1e-7) or
            abs(np.linalg.det(result[:3,:3])) < 1e-12):
        raise ValueError('Invalid session coordinate transform')
    return result


def validate(value):
    import math
    if not isinstance(value,dict) or type(value.get('schema')) is not int or value.get('schema') != SCHEMA:
        raise ValueError('Unsupported viewer geometry')
    matrix(value.get('data_to_source', FLIP))
    matrix(value.get('model_to_data', IDENTITY))
    cameras=value.get('cameras', [])
    if not isinstance(cameras,list) or len(cameras)>MAX_CAMERAS:
        raise ValueError('Invalid session camera count')
    for item in cameras:
        if not isinstance(item,dict) or not isinstance(item.get('name'),str) or len(item['name'])>1024:
            raise ValueError('Invalid session camera name')
        matrix(item.get('camera_to_data'))
        for key in ('width','height'):
            if type(item.get(key)) is not int or not 0<item[key]<=65536:
                raise ValueError('Invalid session camera size')
        for key in ('fx','fy','cx','cy'):
            if type(item.get(key)) not in (int,float) or not math.isfinite(item[key]) or abs(item[key])>1e9:
                raise ValueError('Invalid session camera intrinsics')
        if item['fx']<=0 or item['fy']<=0:
            raise ValueError('Invalid session camera focal length')
    return value


def from_capture(frames):
    import numpy as np
    flip=np.asarray(FLIP)
    value={'schema':SCHEMA,'source_frame':'blender-world','data_frame':'lichtfeld-data',
           'data_to_source':FLIP,'model_to_data':IDENTITY,'cameras':[]}
    for frame in frames:
        value['cameras'].append({'name':Path(frame['file_path']).name,
            'camera_to_data':(flip @ matrix(frame['transform_matrix']) @ flip).tolist(),
            'width':frame['w'],'height':frame['h'],
            **{new:frame[old] for new,old in (('fx','fl_x'),('fy','fl_y'),('cx','cx'),('cy','cy'))}})
    return validate(value)


def source_model_transform(value=None):
    import numpy as np
    value=value or {}
    return matrix(value.get('data_to_source',FLIP)) @ matrix(value.get('model_to_data',IDENTITY))


def viewer_data_transform(value=None):
    """Presentation in native node/data space; never change the captured dataset."""
    import numpy as np
    value=value or {}
    if value.get('source_frame')=='blender-world':
        # The renderer applies FLIP at its display boundary. Compose it once
        # here with the declared data->source frame and source->Studio basis.
        return matrix(FLIP) @ matrix(BLENDER_TO_VISUALIZER) @ matrix(value.get('data_to_source',FLIP))
    return np.eye(4)


def viewer_model_transform(value=None):
    value=value or {}
    return viewer_data_transform(value) @ matrix(value.get('model_to_data',IDENTITY))


def visualizer_model_transform(value=None):
    return matrix(FLIP) @ viewer_model_transform(value)


def publish(publisher, value):
    # Import the host's transport without requiring either application runtime.
    if __package__:
        try:
            from .transport import atomic_json
        except ModuleNotFoundError:
            from .session_transport import atomic_json
    else:
        from session_transport import atomic_json
    value=validate(value)
    path=publisher.root/'view_geometry.json'
    payload=json.dumps(value,allow_nan=False).encode('utf-8')
    if len(payload)>MAX_BYTES:
        raise ValueError('Session camera metadata exceeds the size limit')
    atomic_json(path,value)
    actual=path.read_bytes()
    publisher.update(geometry={'path':path.name,'bytes':len(actual),'sha256':hashlib.sha256(actual).hexdigest()})


def load(root, descriptor):
    if not descriptor:return None
    if not isinstance(descriptor,dict) or descriptor.get('path')!='view_geometry.json':
        raise ValueError('Invalid session geometry reference')
    size=descriptor.get('bytes')
    if type(size) is not int or not 0<size<=MAX_BYTES:
        raise ValueError('Invalid session geometry size')
    root=Path(root).resolve();path=root/descriptor['path']
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Invalid session geometry path')
    with path.open('rb') as stream:
        data=stream.read(size+1)
    if len(data)!=size or hashlib.sha256(data).hexdigest()!=descriptor.get('sha256'):
        raise ValueError('Session geometry integrity check failed')
    return validate(json.loads(data))
