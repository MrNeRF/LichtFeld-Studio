# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest


@pytest.fixture
def editor(monkeypatch):
    empty = {"positive": "none", "negative": "none", "inverted": False}
    actions = [{"id": "move_x", "continuous": True, "modes": ["orbit", "walk"]},
               {"id": "next_rendering", "continuous": False, "modes": ["orbit", "walk"]}]
    catalog = {"modes": ["orbit", "walk"], "actions": actions,
               "controls": [{"id": "axis.left_x", "axis": True, "signed": True, "index": 0},
                            {"id": "button.south", "axis": False, "signed": False, "index": 0},
                            {"id": "button.north", "axis": False, "signed": False, "index": 1}]}
    profile = {"bindings": {"common": {a["id"]: copy.deepcopy(empty) for a in actions}, "modes": {}}, "move_speed": 2}
    defaults = copy.deepcopy(profile)
    state = {"selected": 7, "generation": 1, "axes": [0]*6, "raw_buttons": [False]*2}
    saves, choices, selected, records = [], [], [], {}
    def save(serialized):
        value = json.loads(serialized)
        saves.append(value)
        profile.clear()
        profile.update(value)
        return True
    ui = SimpleNamespace(tr=lambda x:x, get_gamepad_profile_json=lambda:json.dumps(profile),
        get_gamepad_default_profile_json=lambda:json.dumps(defaults),
        get_gamepad_binding_catalog_json=lambda:json.dumps(catalog), set_gamepad_profile_json=save,
        get_gamepad_test_state=lambda:state, set_gamepad_test_device=lambda value:selected.append(value) or True,
        request_redraw=lambda **kwargs:None, confirm_dialog=lambda *args: choices.append(args))
    monkeypatch.setitem(sys.modules, 'lichtfeld', SimpleNamespace(ui=ui))
    spec=importlib.util.spec_from_file_location('bindings_under_test', Path(__file__).resolve().parents[2]/'src/python/lfs_plugins/gamepad_bindings.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    handle=SimpleNamespace(dirty=lambda *_:None,update_record_list=lambda name,value:records.update({name:value}))
    model=SimpleNamespace(get_handle=lambda:handle,bind_record_list=lambda *_:None,bind=lambda *_:None,bind_func=lambda *_:None,bind_event=lambda *_:None)
    owner=SimpleNamespace(navigation={"selected":7},state=state)
    subject=module.GamepadBindingsSection(owner,model)
    now=[100.0]; monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    return SimpleNamespace(subject=subject,module=module,profile=profile,state=state,saves=saves,choices=choices,selected=selected,records=records,now=now,ui=ui)


def test_explicit_clear_differs_from_inheritance(editor):
    e=editor;e.profile['bindings']['common']['move_x']['positive']='axis.left_x'
    e.subject.scope='orbit';e.subject.edit('move_x','clear')
    assert e.module.effective(e.profile,'orbit','move_x')['positive']=='none'
    assert e.module.effective(e.profile,'walk','move_x')['positive']=='axis.left_x'
    e.subject.edit('move_x','inherit')
    assert 'move_x' not in e.profile['bindings']['modes']['orbit']
    assert e.module.effective(e.profile,'orbit','move_x')['positive']=='axis.left_x'


def test_conflict_cancel_preserves_profile_and_replace_is_mode_scoped(editor):
    e=editor;e.profile['bindings']['common']['next_rendering']['positive']='button.south'
    before=copy.deepcopy(e.profile);e.subject.scope='orbit'
    e.subject.change('move_x',{'positive':'button.south','negative':'none','inverted':False})
    assert e.profile==before and not e.saves
    e.choices[-1][-1]('common.cancel'); assert e.profile==before
    e.choices[-1][-1]('gamepad.binding.replace')
    assert e.module.effective(e.profile,'orbit','next_rendering')['positive']=='none'
    assert e.module.effective(e.profile,'walk','next_rendering')['positive']=='button.south'


def test_capture_requires_stable_neutral_and_ignores_initial_held_input(editor):
    e=editor;e.state['raw_buttons'][0]=True;e.subject.edit('move_x','assign')
    e.now[0]+=1;e.subject.sync();assert not e.saves
    e.state['raw_buttons'][0]=False;e.subject.sync();e.now[0]+=.13;e.subject.sync()
    e.state['axes'][0]=.9;e.subject.sync()
    assert e.profile['bindings']['common']['move_x']['positive']=='axis.left_x'
    assert e.subject.capture is None and e.selected[-1]==0


def test_discrete_capture_rejects_stick_and_accepts_button(editor):
    e=editor;e.subject.edit('next_rendering','assign');e.now[0]+=.13;e.subject.sync()
    e.state['axes'][0]=.9;e.subject.sync();assert not e.saves
    e.state['raw_buttons'][0]=True;e.subject.sync()
    assert e.profile['bindings']['common']['next_rendering']['positive']=='button.south'


def test_disconnect_and_generation_change_cancel_capture(editor):
    e=editor;e.subject.edit('move_x','assign');e.state['generation']=2;e.subject.sync()
    assert e.subject.capture is None and not e.saves


def test_button_pair_waits_for_release_and_can_skip_opposite(editor):
    e=editor;e.subject.edit('move_x','assign');e.now[0]+=.13;e.subject.sync()
    e.state['raw_buttons'][0]=True;e.subject.sync()
    assert e.subject.capture['phase']=='negative' and not e.saves
    e.now[0]+=1;e.subject.sync();assert not e.saves
    e.subject.finish_capture()
    assert e.profile['bindings']['common']['move_x']['positive']=='button.south'
    assert e.profile['bindings']['common']['move_x']['negative']=='none'


def test_reset_scope_preserves_tuning_and_other_modes(editor):
    e=editor;e.profile['bindings']['common']['move_x']['positive']='axis.left_x'
    e.subject.scope='orbit';e.subject.reset();e.choices[-1][-1]('gamepad.binding.reset')
    assert e.profile['move_speed']==2
    assert e.module.effective(e.profile,'orbit','move_x')['positive']=='none'
    assert e.module.effective(e.profile,'walk','move_x')['positive']=='axis.left_x'
    e.subject.scope='common';e.subject.reset();e.choices[-1][-1]('gamepad.binding.reset')
    assert e.profile['bindings']['modes']=={} and e.profile['move_speed']==2


def test_save_failure_is_visible_without_mutating_profile(editor):
    e=editor;e.ui.set_gamepad_profile_json=lambda _:False
    e.subject.edit('move_x','invert')
    assert e.subject.message=='gamepad.binding.save_error'
    assert not e.profile['bindings']['common']['move_x']['inverted']

def test_same_scope_model_write_does_not_cancel_capture(editor):
    e=editor;e.subject.edit('move_x','assign');e.subject.set_scope('common')
    assert e.subject.capture is not None


def test_controller_switch_cancels_capture_without_saving(editor):
    e=editor;e.subject.edit('move_x','assign');e.state['selected']=8;e.subject.sync()
    assert e.subject.capture is None and not e.saves and e.selected[-1]==0


def test_opposite_button_capture_requires_neutral_between_inputs(editor):
    e=editor;e.subject.edit('move_x','assign');e.now[0]+=.13;e.subject.sync()
    e.state['raw_buttons'][0]=True;e.subject.sync()
    e.state['raw_buttons'][0]=False;e.subject.sync();e.now[0]+=.13;e.subject.sync()
    e.state['raw_buttons'][1]=True;e.subject.sync()
    assert e.profile['bindings']['common']['move_x']=={
        'positive':'button.south','negative':'button.north','inverted':False}


def test_all_binding_labels_exist_in_every_locale(editor):
    locales=Path(__file__).resolve().parents[2]/'src/visualizer/gui/resources/locales'
    for path in locales.glob('*.json'):
        strings=json.loads(path.read_text(encoding='utf-8'))
        for action in editor.subject.catalog['actions']:
            assert strings['gamepad.binding.'+action['id']]
        for key in ('common','title','context','reset_mode','reset_all','capture_negative','save_error'):
            assert strings['gamepad.binding.'+key]
        assert strings['preferences.back_to_gamepad']!='Torna al joypad'

def test_short_button_press_can_be_captured_after_release(editor):
    e=editor;e.subject.edit('next_rendering','assign');e.now[0]+=.13;e.subject.sync()
    e.state['raw_pressed']=[True,False]
    e.subject.sync()
    assert e.profile['bindings']['common']['next_rendering']['positive']=='button.south'


def test_skipping_is_available_only_for_opposite_direction(editor):
    e=editor;e.subject.edit('next_rendering','assign');e.subject.finish_capture()
    assert e.subject.capture is not None and not e.saves

def test_assignment_options_expand_only_one_row_without_changing_profile(editor):
    e=editor;before=copy.deepcopy(e.profile)
    e.subject.edit('move_x','options')
    assert [r['id'] for r in e.records['gamepad_binding_rows'] if r['expanded']]==['move_x']
    e.subject.edit('next_rendering','options')
    assert [r['id'] for r in e.records['gamepad_binding_rows'] if r['expanded']]==['next_rendering']
    e.subject.edit('next_rendering','options')
    assert not any(r['expanded'] for r in e.records['gamepad_binding_rows'])
    assert e.profile==before and not e.saves

def test_reassigning_stick_keeps_explicit_inversion_option(editor):
    e=editor;e.profile['bindings']['common']['move_x']['inverted']=True
    e.subject.edit('move_x','assign');e.now[0]+=.13;e.subject.sync()
    e.state['axes'][0]=.9;e.subject.sync()
    assert e.profile['bindings']['common']['move_x']['inverted'] is True

@pytest.fixture
def shoulders(editor):
    e=editor
    for name in ('precision','fast','next_navigation'):
        e.profile['bindings']['common'][name]=dict(e.module.EMPTY)
        e.subject.catalog['actions'].append({'id':name,'continuous':False,'modes':['orbit','walk']})
    e.profile['bindings']['common']['precision']['positive']='button.left_shoulder'
    e.profile['bindings']['common']['fast']['positive']='button.right_shoulder'
    e.profile['enabled']=True
    return e


def test_common_shoulder_preset_replaces_conflicts_without_resetting_tuning(shoulders):
    e=shoulders
    e.profile['bindings']['common']['move_x']['positive']='axis.left_x'
    e.profile['bindings']['modes']['orbit']={'next_navigation':{**e.module.EMPTY,'positive':'button.north'}}
    candidate=e.module.shoulder_navigation_preset(e.profile,e.subject.catalog,'common')
    assert candidate['bindings']['common']['next_navigation']['positive']=='button.left_shoulder'
    assert candidate['bindings']['common']['next_rendering']['positive']=='button.right_shoulder'
    assert candidate['bindings']['common']['precision']['positive']=='none'
    assert candidate['bindings']['common']['fast']['positive']=='none'
    assert candidate['bindings']['common']['move_x']['positive']=='axis.left_x'
    assert 'next_navigation' not in candidate['bindings']['modes']['orbit']
    assert candidate['move_speed']==2 and candidate['enabled'] is True
    assert not e.module.conflicts(candidate,e.subject.catalog)
    assert e.profile['bindings']['common']['precision']['positive']=='button.left_shoulder'


def test_mode_shoulder_preset_changes_only_that_context(shoulders):
    e=shoulders
    candidate=e.module.shoulder_navigation_preset(e.profile,e.subject.catalog,'orbit')
    assert e.module.effective(candidate,'orbit','next_navigation')['positive']=='button.left_shoulder'
    assert e.module.effective(candidate,'orbit','precision')['positive']=='none'
    assert e.module.effective(candidate,'walk','precision')['positive']=='button.left_shoulder'
    assert candidate['bindings']['common']==e.profile['bindings']['common']
    assert not e.module.conflicts(candidate,e.subject.catalog)


def test_shoulder_preset_preserves_modifiers_assigned_to_other_buttons(shoulders):
    e=shoulders;e.profile['bindings']['common']['precision']['positive']='button.south'
    candidate=e.module.shoulder_navigation_preset(e.profile,e.subject.catalog,'common')
    assert candidate['bindings']['common']['precision']['positive']=='button.south'
    assert not e.module.conflicts(candidate,e.subject.catalog)


def test_shoulder_preset_confirmation_is_atomic_and_cancels_capture(shoulders):
    e=shoulders;e.subject.edit('move_x','assign');assert e.subject.capture
    before=copy.deepcopy(e.profile)
    e.subject.apply_shoulder_preset()
    assert e.subject.capture is None and e.selected[-1]==0
    e.choices[-1][-1]('common.cancel')
    assert e.profile==before and not e.saves
    e.subject.apply_shoulder_preset();e.choices[-1][-1]('gamepad.binding.apply_preset')
    assert e.profile['bindings']['common']['next_navigation']['positive']=='button.left_shoulder'
    assert len(e.saves)==1


def test_switching_profile_cancels_capture_and_discards_queued_modal_edits(editor):
    e = editor
    token = ["profile-a"]
    e.ui.get_gamepad_profiles_json = lambda: token[0]
    e.subject.edit('move_x', 'assign')
    assert e.subject.capture is not None
    token[0] = "profile-b"
    e.subject.sample()
    assert e.subject.capture is None and not e.saves
    assert e.subject.message == "gamepad.profile.changed_error"
    e.profile['bindings']['common']['next_rendering']['positive'] = 'button.south'
    e.subject.change('move_x', {'positive': 'button.south', 'negative': 'none', 'inverted': False})
    token[0] = "profile-c"
    e.choices[-1][-1]('gamepad.binding.replace')
    assert not e.saves
