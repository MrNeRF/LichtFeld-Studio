# SPDX-License-Identifier: GPL-3.0-or-later
"""Qualify TrainingEnd against a real successful run and a failed terminal save."""
import argparse
import json
import os
from pathlib import Path
import subprocess
p=argparse.ArgumentParser();p.add_argument('backend',type=Path);p.add_argument('dataset',type=Path)
p.add_argument('config',type=Path);p.add_argument('area',type=Path);args=p.parse_args()
area=args.area.resolve();area.mkdir(parents=True,exist_ok=True)
config=json.loads(args.config.read_text(encoding='utf-8'))
config['optimization'].update(iterations=32,start_refine=33,stop_refine=34,save_steps=[32],eval_steps=[32])
config_path=area/'training.json';config_path.write_text(json.dumps(config))
proof=[]
for failure in (False,True):
 root=area/('save-failure' if failure else 'success');root.mkdir(exist_ok=True)
 target=root/'requested.licht'
 if failure:
  blocked=root/'not-a-directory';blocked.write_text('Intentionally block the final save')
  target=blocked/'requested.licht'
 env={**os.environ,'LFS_HOME':str(root/'runtime'),'LFS_TRAINING_SESSIONS_DIR':str(root/'sessions'),
      'LFS_TRAINING_PUBLISHER':'automatic'}
 with (root/'trainer.log').open('wb') as log:
  run=subprocess.run([str(args.backend.resolve()),'--headless','-d',str(args.dataset.resolve()),
   '-o',str(root/'output'),'--config',str(config_path),'--test-every','4','--centralize','off',
   '--no-download','--save-project-at-iter','32','--save-project-path',str(target)],
   env=env,stdout=log,stderr=subprocess.STDOUT)
 records=[json.loads(path.read_text(encoding='utf-8')) for path in (root/'sessions').glob('*/session.json')]
 assert len(records)==1,records
 record=records[0];expected='failed' if failure else 'completed'
 assert record['state']==expected,(expected,record,run.returncode)
 assert record['iteration']==32,record
 assert (run.returncode!=0)==failure,run.returncode
 proof.append({'case':expected,'iteration':record['iteration'],'exit_code':run.returncode})
print('ACTUAL_TERMINAL_SAVE_OUTCOME_VERIFIED',proof,flush=True)
(area/'proof.json').write_text(json.dumps(proof))
