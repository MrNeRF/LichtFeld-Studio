# SPDX-License-Identifier: GPL-3.0-or-later
"""Pause an owned fixture trainer and verify discovery updates without another step."""
import json
import os
from pathlib import Path
import threading
import time
import lichtfeld as lf
from lfs_plugins.training_runtime import producer
from lfs_plugins.training_runtime.transport import read_json

_pause_hooks = lf.ScopedHandler()
_pause_requested = False


def step(hook):
    global _pause_requested
    if _pause_requested or hook['iteration'] < 20:
        return
    _pause_requested = True
    lf.session().pause()
    def check():
        result = {}
        try:
            deadline = time.monotonic()+15
            while time.monotonic()<deadline:
                stream = producer._stream
                if stream and stream.publisher:
                    status = read_json(stream.publisher.root/'session.json', {})
                    if status.get('state') == 'paused':
                        result = {'ok':True, 'iteration':status['iteration'], 'state':status['state']}
                        break
                time.sleep(.05)
            else:
                result = {'ok':False, 'error':'Paused status did not reach discovery metadata'}
        except Exception as error:
            result = {'ok':False, 'error':str(error)}
        finally:
            (Path(os.environ['LFS_TEST_AREA'])/'pause-status-proof.json').write_text(json.dumps(result), encoding='utf-8')
            lf.session().resume()
    threading.Thread(target=check, daemon=True, name='pause-status-probe').start()


_pause_hooks.on_post_step(step)
