"""Tests for the Projects Inspector cache, models, and action contract."""

from types import SimpleNamespace
import threading
import time

import pytest

from lfs_plugins.project_inspector import (
    InspectionFactsPipeline,
    details_rows,
    dialog_model,
    inspection_cache_key,
    operation_actions,
    thumbnail_source_options,
)


def _entry(**overrides):
    value = {
        "id": "project",
        "path": "/tmp/project.licht",
        "file_size_bytes": 100,
        "path_mtime_ns": 20,
        "commit_uuid": "commit-a",
        "status": "AVAILABLE",
    }
    value.update(overrides)
    return value


def test_cache_key_uses_stat_identity_and_commit():
    entry = _entry(stat_identity={"size": 101, "mtime_ns": 22, "st_dev": 3, "st_ino": 4})
    assert inspection_cache_key(entry) == (101, 22, 3, 4, "commit-a")
    changed = dict(entry, commit_uuid="commit-b")
    assert inspection_cache_key(changed) != inspection_cache_key(entry)


def test_pipeline_inspects_visible_cards_and_selected_details_once():
    calls = []
    results = []
    card = SimpleNamespace(has_preview=True, physical_file_size=100)
    details = SimpleNamespace(retained_checkpoints=[SimpleNamespace(iteration=30)])
    pipeline = InspectionFactsPipeline(
        lambda path: calls.append(("card", path)) or card,
        lambda path: calls.append(("details", path)) or details,
        lambda asset_id, kind, result, error: results.append((asset_id, kind, result, error)),
        max_background_details=0,
    )
    pipeline.refresh([_entry()], "project")
    assert _wait(lambda: len(results) == 2)
    pipeline.refresh([_entry()], "project")
    time.sleep(0.05)
    assert len(results) == 2
    assert calls == [("card", "/tmp/project.licht"), ("details", "/tmp/project.licht")]
    pipeline.close()


def test_pipeline_cancellation_drops_stale_result():
    started = threading.Event()
    release = threading.Event()
    results = []

    def inspect_card(_path):
        started.set()
        release.wait(1)
        return object()

    pipeline = InspectionFactsPipeline(inspect_card, lambda _path: object(), lambda *args: results.append(args))
    pipeline.refresh([_entry()], "")
    assert started.wait(1)
    pipeline.cancel()
    release.set()
    time.sleep(0.05)
    assert results == []
    pipeline.close()


@pytest.mark.parametrize("state", ["HARD_FAIL", "REPAIR_ONLY", "UNSUPPORTED_NEWER"])
@pytest.mark.parametrize("selected_id", ["project", ""])
def test_pipeline_delivers_unavailable_card_without_reading_details(state, selected_id):
    calls, results = [], []
    card = SimpleNamespace(open_state=SimpleNamespace(name=state), diagnostic="Unavailable project")
    pipeline = InspectionFactsPipeline(
        lambda path: card,
        lambda path: calls.append(path),
        lambda *args: results.append(args),
    )
    try:
        pipeline.refresh([_entry()], selected_id)
        pipeline._thread.join(2)
        assert not pipeline._thread.is_alive()
        assert calls == []
        assert results == [("project", "card", card, None)]
        assert pipeline.cached("project").error == ""

        # A replaced or restored file must become inspectable again.
        card = SimpleNamespace(open_state=SimpleNamespace(name="OPEN"))
        pipeline.refresh([_entry(commit_uuid="restored")], "project")
        pipeline._thread.join(2)
        assert calls == ["/tmp/project.licht"]
        assert [row[1] for row in results] == ["card", "card", "details"]
    finally:
        pipeline.close()


def test_details_model_hides_metrics_without_samples_and_formats_embedded_dataset():
    details = SimpleNamespace(
        card=SimpleNamespace(saved_at_unix_ns=0, title="Bicycle"),
        storage=SimpleNamespace(physical_bytes=1000, dead_bytes=499, dead_ratio=0.499),
        save_history=[SimpleNamespace(sequence=3, saved_at_unix_ns=0)],
        references=[],
        parameters=SimpleNamespace(
            active_strategy="MRNF", embedded_dataset_present=True,
            embedded_dataset_complete=True, embedded_images=194,
            embedded_normals=194, embedded_sparse=3,
        ),
        scene_graph=SimpleNamespace(dataset_node_name="images"),
        retained_checkpoints=[SimpleNamespace(iteration=30000, gaussians=1000000, sh_degree=3, binds_scene_graph=True)],
        metrics=SimpleNamespace(loss_samples=0, psnr_samples=0),
        license=None,
        autosave_sidecar_present=False,
    )
    model = details_rows(_entry(), details, format_size=lambda n: f"{n} B", format_time=lambda n: "")
    assert model["dataset"] == "embedded, 194 images, 194 normals and 3 sparse, complete"
    assert model["reclaimable_percent"] == "49.9%"
    assert not model["has_metrics"]
    assert model["title"] == "Bicycle"


def test_context_actions_open_inspector_and_file_operations():
    actions = {row["action"] for row in operation_actions(_entry())}
    assert actions == {"inspector", "export_as", "update_thumbnail", "rename"}


def test_thumbnail_source_options_only_offer_sources_available_for_target(tmp_path):
    target = tmp_path / "project.licht"
    entry = _entry(path=str(target))

    options = thumbnail_source_options(
        entry,
        viewport_available=True,
        dataset_available=True,
        embedded_available=True,
    )
    assert options == ["viewport", "first_dataset", "first_embedded", "image_file"]
    assert dialog_model(
        "update_thumbnail",
        entry=entry,
        viewport_available=True,
        dataset_available=True,
        embedded_available=True,
    )["source"] == "first_dataset"

    # A PLY-only project, a mismatched active scene, or undecodable embedded
    # content all arrive as false native availability flags.
    ply_only = _entry(path=str(target))
    options = thumbnail_source_options(ply_only)
    assert options == ["image_file"]
    assert dialog_model("update_thumbnail", entry=ply_only)["source"] == "image_file"

    from lfs_plugins.project_dialog import form_content

    body, _buttons = form_content(
        "update_thumbnail",
        {"sources": ["image_file"], "source": "image_file"},
        tr=lambda key: key,
        confirm_label="Update",
        busy=False,
    )
    assert 'value="image_file" selected' in body
    assert "projects.dialog.first_dataset_image" not in body
    assert "projects.dialog.first_embedded_image" not in body
    assert "projects.dialog.current_viewport" not in body


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _contents_details(**changes):
    result = dict(
        card=SimpleNamespace(has_preview=False), storage=SimpleNamespace(dead_ratio=0, dead_bytes=0),
        save_history=[], retained_checkpoints=[], references=[], chapters=[], manifest={}, license=None,
        parameters=SimpleNamespace(embedded_dataset_present=False, active_strategy='mcmc'),
        scene_graph=SimpleNamespace(training_node_id='model', dataset_node_name=''),
        metrics=SimpleNamespace(loss_samples=0, psnr_samples=0),
    )
    result.update(changes)
    return SimpleNamespace(**result)


def _contents(details, plan=None, **kwargs):
    import json
    from pathlib import Path
    from lfs_plugins.project_inspector import contents_rows
    translations = json.loads((Path(__file__).parents[2] / 'src/visualizer/gui/resources/locales/en.json').read_text())
    return contents_rows(_entry(), details, plan, tr=lambda key: translations[key],
                         format_size=lambda size: f'{size} B', format_time=lambda timestamp: '2026-08-27' if timestamp else '', **kwargs)


def test_empty_contents_has_only_the_two_add_rows():
    rows = _contents(_contents_details())
    assert [(r['id'], r['label'], r['action']) for r in rows] == [
        ('thumbnail', 'Add thumbnail', 'thumbnail'), ('license', 'Add license', 'license')]
    assert not any(r['removable'] for r in rows)


def test_contents_saves_have_dates_and_only_older_saves_have_actions():
    rows = _contents(_contents_details(save_history=[
        SimpleNamespace(generation=1, bytes_added=38, saved_at_unix_ns=10),
        SimpleNamespace(generation=2, bytes_added=412, saved_at_unix_ns=20)]))
    current, older = rows[:2]
    assert current['label'] == 'Save 2 of 2, 2026-08-27'
    assert current['current'] and not older['current']
    assert 'A saved state of the project.' in current['tooltip']
    assert not current['removable'] and not current['has_action']
    assert older['action'] == 'restore' and older['removable'] and older['generation'] == 1
    assert older['size'] == '38 B'

    edited = _contents(_contents_details(save_history=[
        SimpleNamespace(generation=1, kind='AUTOSAVE', saved_at_unix_ns=10,
                        checkpoint_iteration=12000, planned_iterations=30000, strategy='mrnf', gaussians=1000000),
        SimpleNamespace(generation=2, kind='CONTENTS', saved_at_unix_ns=20,
                        source_save_generation=1, operation='dataset_removed')]))
    saves = [r for r in edited if r['kind']=='save']
    assert len(saves)==1 and saves[0]['current'] and saves[0]['autosave']
    assert saves[0]['label']=='Save 1 of 1, 2026-08-27, step 12,000 of 30,000, mrnf, 1.0 M gaussians'
    assert saves[0]['detail']=='edited 2026-08-27: dataset removed'
    assert saves[0]['generation']==2


def test_contents_checkpoints_only_list_retained_payloads_and_match_sizes_by_identity():
    rows = _contents(_contents_details(retained_checkpoints=[
        SimpleNamespace(instance_uuid='old', iteration=10, retained=False),
        SimpleNamespace(instance_uuid='a', iteration=20, retained=True, header_reachable=True),
        SimpleNamespace(instance_uuid='b', iteration=30, retained=True, header_reachable=False)]),
        SimpleNamespace(retained_checkpoints=[SimpleNamespace(instance_uuid='b', bytes=200), SimpleNamespace(instance_uuid='a', bytes=100)]))
    cps = [r for r in rows if r['kind'] == 'checkpoint']
    assert [(r['id'], r['bytes'], r['action']) for r in cps] == [('checkpoint:a',100,'resume'),('checkpoint:b',200,'')]
    assert all(r['removable'] for r in cps)
    assert cps[0]['label'] == 'Checkpoint, iteration 20, mcmc'


@pytest.mark.parametrize("order", [
    [2200, 3000, 3300, 7000, 7183],
    [3000, 2200, 7183, 7000, 3300],
    [7183, 7000, 3300, 3000, 2200],
])
def test_checkpoint_display_order_is_independent_of_storage_order(order):
    checkpoints = [SimpleNamespace(instance_uuid=str(step), iteration=step)
                   for step in order]
    checkpoints.extend([
        SimpleNamespace(instance_uuid="removed", iteration=1, retained=False),
        SimpleNamespace(instance_uuid="3000-again", iteration=3000),
    ])
    rows = _contents(_contents_details(retained_checkpoints=checkpoints))
    actual = [(row["iteration"], row["checkpoint_uuid"])
              for row in rows if row["kind"] == "checkpoint"]
    assert actual == [(2200, "2200"), (3000, "3000"), (3000, "3000-again"),
                      (3300, "3300"), (7000, "7000"), (7183, "7183")]
    assert [cp.iteration for cp in checkpoints[:5]] == order


def test_contents_embedded_dataset_counts_images_without_counting_normals_as_images():
    details = _contents_details(parameters=SimpleNamespace(embedded_dataset_present=True, embedded_images=194, embedded_normals=194, embedded_sparse=3))
    plan = SimpleNamespace(embedded_dataset=[SimpleNamespace(bytes=100), SimpleNamespace(bytes=50)], drop_embedded_dataset=SimpleNamespace(allowed=False))
    dataset = next(r for r in _contents(details,plan) if r['kind']=='dataset')
    assert dataset['label'] == 'Dataset, 194 images embedded' and dataset['bytes']==150
    assert dataset['removable'] and dataset['remove_disabled']
    plan.drop_embedded_dataset.allowed=True
    assert not next(r for r in _contents(details,plan) if r['kind']=='dataset')['remove_disabled']


def test_external_dataset_always_offers_locate_and_only_embeds_reachable_inputs():
    for reachable in (False,True):
        rows = _contents(_contents_details(references=[SimpleNamespace(kind='dataset',path='/dataset',reachable=reachable)]))
        dataset = next(r for r in rows if r['kind']=='external')
        assert dataset['action']=='embed' and dataset['secondary']=='locate'
        assert dataset['action_disabled'] is not reachable
        assert '/dataset' in dataset['label']
        assert not dataset['removable']


def test_contents_thumbnail_metrics_and_license_use_native_parts():
    rows=_contents(_contents_details(card=SimpleNamespace(has_preview=True), chapters=[SimpleNamespace(fourcc='THMB',stored_bytes=40),SimpleNamespace(fourcc='METR',stored_bytes=8)],
        metrics=SimpleNamespace(loss_samples=8,psnr_samples=4),license=SimpleNamespace(identifier='CC-BY-4.0',notice='Credit: Owner')))
    parts={r['id']:r for r in rows}
    assert parts['thumbnail']['bytes']==40 and parts['thumbnail']['action']=='thumbnail'
    assert parts['metrics']['label']=='Metrics, 12 samples' and parts['metrics']['bytes']==8
    assert parts['license']['label']=='License, CC BY 4.0' and parts['license']['action_label']=='Change'
    assert all(parts[id]['removable'] for id in ('thumbnail','metrics','license'))


def test_pending_contents_survive_new_models_and_do_not_restore_removed_saves():
    import json
    details=_contents_details(save_history=[SimpleNamespace(generation=1,bytes_added=38),SimpleNamespace(generation=2)],manifest={
        'contents_removals':json.dumps({'rows':[{'id':'save:1','kind':'save','generation':1,'bytes':400},{'id':'checkpoint:a','kind':'checkpoint','iteration':30,'bytes':210}]})})
    rows=_contents(details)
    assert not any(r['id']=='save:1' and r['action']=='restore' for r in rows)
    pending=[r for r in rows if r['pending']]
    assert len(pending)==1 and pending[0]['bytes']==38
    assert pending[0]['undo'] and not pending[0]['removable'] and not pending[0]['has_action']
    assert pending[0]['detail']=='removed, freed by Compact'
    assert not any(r['kind']=='checkpoint' for r in rows)
    assert next(r for r in rows if r['id']=='compact')['has_detail']


def test_compact_threshold_and_busy_state():
    for ratio, expected in ((.009,False),(.01,True),(.49,True)):
        rows=_contents(_contents_details(storage=SimpleNamespace(dead_ratio=ratio,dead_bytes=210)),busy=True)
        assert any(r['id']=='compact' for r in rows) is expected
        assert all(r['disabled'] for r in rows)


def test_standard_license_mapping_and_credit():
    from lfs_plugins.project_inspector import LICENSES,license_value,license_fields
    assert [identifier for identifier,_ in LICENSES] == ['CC0-1.0','CC-BY-4.0','CC-BY-SA-4.0','CC-BY-NC-4.0','CC-BY-NC-SA-4.0','CC-BY-ND-4.0','CC-BY-NC-ND-4.0','LicenseRef-Proprietary','custom']
    for identifier,_ in LICENSES[:-1]:
        actual,notice=license_value(dict(license_choice=identifier,attribution='Owner'))
        assert actual==identifier
        assert notice==('' if identifier in {'CC0-1.0','LicenseRef-Proprietary'} else 'Credit: Owner')
        fields=license_fields(SimpleNamespace(identifier=actual,notice=notice))
        assert fields['license_choice']==identifier
        assert license_value(fields)==(actual,notice)


def test_custom_license_mapping_and_multiline_notice_round_trip():
    import pytest
    from lfs_plugins.project_inspector import license_fields,license_value
    identifier,notice=license_value(dict(license_choice='custom',license_name='My Scan License',license_text='Ask first.\nhttps://example.org/license',attribution='Studio'))
    assert identifier=='LicenseRef-MyScanLicense'
    assert notice=='Ask first.\nhttps://example.org/license\nCredit: Studio'
    assert license_value(license_fields(SimpleNamespace(identifier=identifier,notice=notice)))==(identifier,notice)
    for fields in ({'license_name':'','license_text':'terms'},{'license_name':'Name','license_text':''},{'license_name':'!!!','license_text':'terms'}):
        with pytest.raises(ValueError):license_value(dict(license_choice='custom',**fields))


def test_removed_checkpoint_does_not_describe_current_training():
    details = _contents_details(retained_checkpoints=[
        SimpleNamespace(iteration=30000, retained=False, binds_scene_graph=True),
    ])
    model = details_rows(_entry(), details, format_size=str, format_time=str)
    assert model["iteration"] == ""
    assert not model["resumable"]


def test_license_removal_size_counts_utf8_notice_bytes():
    license_obj = SimpleNamespace(identifier="CC-BY-4.0", notice="Credit: Renée")
    rows = _contents(_contents_details(license=license_obj))
    row = next(row for row in rows if row["id"] == "license")
    assert row["bytes"] == len((license_obj.identifier + license_obj.notice).encode("utf-8"))


def test_license_form_uses_chooser_and_escapes_custom_text():
    from lfs_plugins.project_dialog import form_content
    body,_=form_content('license',dict(license_choice='custom',license_name='<scan>',license_text='<terms>'),tr=lambda key:key,confirm_label='Save',busy=False)
    assert 'name="license_choice"' in body and 'name="license_name"' in body and 'name="license_text"' in body
    assert 'name="identifier"' not in body and 'name="notice"' not in body
    assert '&lt;scan&gt;' in body and '&lt;terms&gt;' in body
    for choice in ('CC0-1.0','LicenseRef-Proprietary'):
        body,_=form_content('license',dict(license_choice=choice),tr=lambda key:key,confirm_label='Save',busy=False)
        assert 'name="attribution"' not in body and 'name="license_text"' not in body


def test_custom_license_textarea_round_trips_special_characters():
    from html.parser import HTMLParser
    from lfs_plugins.project_dialog import form_content
    from lfs_plugins.project_inspector import license_value

    notice = 'Tom & Jerry\'s "cut" <1> > https://example.org/?a=1&b=2'
    body, _ = form_content(
        'license',
        dict(license_choice='custom', license_name='Notice', license_text=notice),
        tr=lambda key: key,
        confirm_label='Save',
        busy=False,
    )

    class TextareaParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_textarea = False
            self.attributes = {}
            self.value = []

        def handle_starttag(self, tag, attrs):
            self.in_textarea = tag == 'textarea'
            if self.in_textarea:
                self.attributes = dict(attrs)

        def handle_endtag(self, tag):
            if tag == 'textarea':
                self.in_textarea = False

        def handle_data(self, value):
            if self.in_textarea:
                self.value.append(value)

    parser = TextareaParser()
    parser.feed(body)
    assert parser.attributes['value'] == notice
    assert ''.join(parser.value) == ''
    identifier, saved_notice = license_value(
        dict(license_choice='custom', license_name='Notice', license_text=parser.attributes['value'])
    )
    assert identifier == 'LicenseRef-Notice'
    assert saved_notice == notice


def test_queued_inspection_is_discarded_and_retried_after_cancel():
    callbacks, results = [], []
    pipeline = InspectionFactsPipeline(lambda _path: object(), lambda _path: object(),
        lambda *args: results.append(args), scheduler=callbacks.append)
    pipeline.refresh([_entry()], 'project')
    pipeline._thread.join(2)
    pipeline.cancel()
    for callback in callbacks:
        callback()
    assert results == []
    callbacks.clear()
    pipeline.refresh([_entry()], 'project')
    pipeline._thread.join(2)
    for callback in callbacks:
        callback()
    assert [row[1] for row in results] == ['card', 'details']


def test_card_failure_is_logged_and_delivered_without_reading_details(caplog):
    results = []
    details_calls = []
    def fail(path):
        raise OSError('unreadable marker')
    pipeline = InspectionFactsPipeline(fail, details_calls.append, lambda *args: results.append(args))
    pipeline.refresh([_entry(path='/项目.licht')], 'project')
    pipeline._thread.join(2)
    assert len(results) == 1 and isinstance(results[0][3], OSError)
    assert details_calls == []
    assert 'Inspect project card failed path=/项目.licht' in caplog.text
    assert 'Inspect project details failed' not in caplog.text
    pipeline.close()


def test_details_failure_after_readable_card_is_logged_and_delivered(caplog):
    results = []
    def fail(path):
        raise OSError('unreadable marker')
    pipeline = InspectionFactsPipeline(
        lambda path: SimpleNamespace(open_state="OPEN"), fail, lambda *args: results.append(args))
    pipeline.refresh([_entry(path='/项目.licht')], 'project')
    pipeline._thread.join(2)
    assert len(results) == 2 and isinstance(results[1][3], OSError)
    assert 'Inspect project details failed path=/项目.licht' in caplog.text
    pipeline.close()


def test_inspection_scheduler_failure_can_be_retried(caplog):
    def fail(_callback):
        raise RuntimeError('scheduler marker')
    pipeline = InspectionFactsPipeline(lambda _path: object(), lambda _path: object(),
        lambda *_args: None, scheduler=fail)
    pipeline.refresh([_entry()], 'project')
    pipeline._thread.join(2)
    assert not pipeline._thread.is_alive()
    assert pipeline.cached('project').card is None
    assert pipeline.cached('project').error == 'scheduler marker'
    assert 'Schedule project inspection failed project=project' in caplog.text


def test_checkpoint_order_uses_numeric_iterations_before_uuid_ties():
    checkpoints = [
        SimpleNamespace(instance_uuid="00000000-0000-0000-0000-000000000001", iteration=100),
        SimpleNamespace(instance_uuid="00000000-0000-0000-0000-000000000004", iteration=10),
        SimpleNamespace(instance_uuid="00000000-0000-0000-0000-000000000003", iteration=10),
        SimpleNamespace(instance_uuid="00000000-0000-0000-0000-000000000002", iteration=9),
    ]
    rows = _contents(_contents_details(retained_checkpoints=checkpoints))
    actual = [row["checkpoint_uuid"] for row in rows if row["kind"] == "checkpoint"]
    assert actual == [checkpoints[i].instance_uuid for i in (3, 2, 1, 0)]
