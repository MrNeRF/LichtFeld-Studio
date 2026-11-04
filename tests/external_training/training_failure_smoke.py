# SPDX-License-Identifier: GPL-3.0-or-later
"""Run ordinary headless training with an unavailable observer registry.

Requires a real dataset and matching training configuration. Use a new scratch area.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
p=argparse.ArgumentParser();p.add_argument('backend',type=Path);p.add_argument('dataset',type=Path)
p.add_argument('config',type=Path);p.add_argument('area',type=Path);args=p.parse_args()
area=args.area.resolve();area.mkdir(parents=True,exist_ok=True)
blocked=area/'registry-is-a-file';blocked.write_text('Deliberately unavailable registry')
env={**os.environ,'LFS_HOME':str(area/'runtime'),'LFS_TRAINING_SESSIONS_DIR':str(blocked)}
with (area/'trainer.log').open('wb') as log:
 result=subprocess.run([str(args.backend.resolve()),'--headless','-d',str(args.dataset.resolve()),
  '-o',str(area/'output'),'--config',str(args.config.resolve()),'--test-every','4',
  '--centralize','off','--no-download','--export','ply'],env=env,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0, f'Training failed: {result.returncode}; see {area / "trainer.log"}'
assert (area/'output/project.licht').is_file(), 'Ordinary project save missing'
assert list((area/'output').glob('splat_*.ply')), 'Ordinary export missing'
assert blocked.is_file(), 'Optional observer unexpectedly changed the blocked registry'
(area/'proof.json').write_text(json.dumps({'passed':True,'exit_code':result.returncode,'normal_project_saved':True}))
print('TRAINING_CONTINUES_WITH_UNAVAILABLE_REGISTRY_AND_SAVES_RESULT',flush=True)
