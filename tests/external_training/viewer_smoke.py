# SPDX-License-Identifier: GPL-3.0-or-later
"""Real GUI contracts. Run only with a disposable GUI and publisher_fixture.py."""
import argparse
import base64
import json
from pathlib import Path
import textwrap
import time
from mcp_test_client import MCP
parser=argparse.ArgumentParser();parser.add_argument('area',type=Path);args=parser.parse_args()
area=args.area.resolve();sessions=json.loads((area/'ready.json').read_text())
c=MCP();c.rpc('initialize',{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'external-viewer-tests','version':'1'}})
for uri in ('lichtfeld://runtime/catalog','lichtfeld://runtime/state','lichtfeld://ui/state','lichtfeld://scene/state','lichtfeld://selection/current'):c.resource(uri)

def run(code,marker):
 wrapper='''import lichtfeld as lf, threading
import lfs_plugins.training_runtime.panel as p
done=threading.Event();errors=[]
def action():
 try:
CODE
 except Exception as error: errors.append(repr(error))
 finally: done.set()
lf.ui.schedule_on_ui_thread(action)
lf.ui.request_redraw()
assert done.wait(10), 'UI action timed out'
assert not errors, errors
print(MARKER)
'''.replace('CODE',textwrap.indent(code,'  ')).replace('MARKER',repr(marker))
 result=c.call('editor_run',{'code':wrapper,'wait_for_completion':True,'timeout_ms':2000,'show_console':False})
 assert marker in json.dumps(result),result
 print(marker,flush=True)

def wait(code,marker):
 deadline=time.monotonic()+20
 while time.monotonic()<deadline:
  try:run(code,marker);return
  except AssertionError:time.sleep(.25)
 raise AssertionError(marker)

run('lf.ui.set_main_panel_active_tab("lfs.training_sessions")','PANEL_OPEN')
wait(f'''panel=lf.ui.get_panel_object("lfs.training_sessions")
assert panel.doc is not None
assert any(x["id"]=={sessions['first']!r} for x in panel.records)
assert panel.doc.get_element_by_id("et-show-past") is None''','ACTIVE_ONLY_PANEL')
run(f'''panel=lf.ui.get_panel_object("lfs.training_sessions")
lf.get_scene().clear()
panel.connect(next(x for x in panel.records if x["id"]=={sessions['first']!r}))''','CONNECT_FIRST')
wait('assert p._viewer.snapshot_in_scene()','FIRST_SNAPSHOT')
run('''try:
 lf.get_scene().remove_external_viewer(p._viewer.name,"wrong-uuid")
except RuntimeError: pass
else: raise AssertionError("Wrong identity removed a preview")
assert p._viewer.snapshot_in_scene()''','REMOVE_IDENTITY_GUARD')
run('''panel=lf.ui.get_panel_object("lfs.training_sessions")
p._viewer.copy_busy=True
panel.render()
assert panel.doc.get_element_by_id("et-copy").has_attribute("disabled")
p._viewer.copy_busy=False
old=p._viewer.display_arrays
panel.dispatch("disconnect",None)
panel.dispatch("reconnect",None)
assert p._viewer.display_arrays is old''','RECONNECT_PRESERVES_SNAPSHOT')
run(f'''panel=lf.ui.get_panel_object("lfs.training_sessions")
oldname=p._viewer.name
panel.connect(next(x for x in panel.records if x["id"]=={sessions['second']!r}))
assert lf.get_scene().get_node(oldname) is None''','SWITCH_REMOVES_OLD_PREVIEW')
wait('assert p._viewer.snapshot_in_scene()','SECOND_SNAPSHOT')
(area/'finish').touch()
wait('assert p._viewer.reader.status.get("state")=="completed"','TRAINING_END_RECEIVED')
run('''panel=lf.ui.get_panel_object("lfs.training_sessions")
if not panel.browser_open: panel.dispatch("browse-toggle",None)''','BROWSE_FINISHED_TRAINERS')
wait('''panel=lf.ui.get_panel_object("lfs.training_sessions")
assert not panel.records
assert p._viewer.snapshot_in_scene()
panel.render()
assert "Training ended" in panel.doc.get_element_by_id("et-current-state").get_inner_rml()
assert panel.doc.get_element_by_id("et-reconnect").is_class_set("hidden")''','END_RETAINS_PREVIEW_NO_HISTORY')
run('''panel=lf.ui.get_panel_object("lfs.training_sessions")
name=p._viewer.name
panel.dispatch("discard",None)
assert p._viewer is None
assert lf.get_scene().get_node(name) is None''','REMOVE_PREVIEW')
time.sleep(.8)
for item in c.call('render_capture_window').get('content',[]):
 if item.get('type')=='image':(area/'active-only-panel.png').write_bytes(base64.b64decode(item['data']))
(area/'proof.json').write_text(json.dumps({'passed':True,'sessions':sessions}))
