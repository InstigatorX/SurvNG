import pytest

from survng.app.config import AppConfig, CameraConfig, NativeBudgetConfig, effective_native_budget
from survng.native_budget import NativeBudget, overlaps
from survng.native_spatial import spatial_plan


def camera():
    return CameraConfig(id='front', name='Front', stream_url='rtsp://unused.invalid/live', zones=[{
        'name':'drive', 'points':[{'x':.5,'y':.5},{'x':1,'y':.5},{'x':1,'y':1},{'x':.5,'y':1}]}])


def plan():
    config = AppConfig(cameras=[camera()])
    config.detector.native.budget = NativeBudgetConfig(enabled=True, cooldown_seconds=1, approach_padding=.1)
    return spatial_plan(config.cameras[0], config.detector)


def test_defaults_and_camera_inheritance():
    config = AppConfig(cameras=[camera()])
    c = config.cameras[0]
    assert not effective_native_budget(c, config.detector).enabled
    config.detector.native.budget.enabled = True
    config.detector.native.budget.motion_threshold = .2
    c.native_budget.idle_fps = 2
    c.native_budget.motion_threshold = .3
    assert effective_native_budget(c, config.detector).motion_threshold == .3
    c.native_budget.motion_threshold = None
    assert effective_native_budget(c, config.detector).motion_threshold == .2
    c.native_budget.enabled = False
    assert not effective_native_budget(c, config.detector).enabled
    c.native_budget.enabled = None
    assert effective_native_budget(c, config.detector).enabled


@pytest.mark.parametrize('enabled', [True, False])
def test_budget_reports_effective_motion_setting(enabled):
    p = plan()
    p['budget']['motion_enabled'] = enabled
    from survng.app.local_observability import _camera_snapshot
    status = NativeBudget(p).status()
    assert status['motion_enabled'] is enabled
    snapshot = _camera_snapshot({'live_pipeline': {'native_budget': status}})
    assert snapshot['inference_budget']['motion_enabled'] is enabled


def test_idle_is_periodic_and_motion_wakes_immediately():
    b = NativeBudget(plan())
    admitted = [t/5 for t in range(35) if b.select(t/5)]
    assert len(admitted) < 20
    assert b.mode == 'idle'
    assert b.selected_full_frame
    assert b.select(7.01, [(0.42,.6,.46,.8)])
    assert b.mode == 'active'
    assert not b.selected_full_frame
    assert b.counts['motion_wakes'] == 1
    assert b.select(7.21)
    b.select(9)
    assert b.mode == 'idle'


def test_outside_motion_does_not_wake_but_ignore_does_not_mask_incident():
    p = plan()
    p['zones'].append({'name':'ignore', 'behavior':'ignore', 'points':p['zones'][0]['points']})
    b = NativeBudget(p)
    b.select(0)
    b.select(3, [(0,.1,.1,.2)])
    assert b.mode == 'idle'
    b.select(3.2, [(.7,.6,.8,.8)])
    assert b.mode == 'active'


def rectangle_zone(left, top, right, bottom, **settings):
    return {'name': 'exclusion', 'behavior': 'ignore', 'exclude_from_ema': True,
            'points': [{'x': left, 'y': top}, {'x': right, 'y': top},
                       {'x': right, 'y': bottom}, {'x': left, 'y': bottom}], **settings}


@pytest.mark.parametrize('behavior', ['ignore', 'incident', 'none'])
def test_motion_exclusion_overrides_padding_and_preserves_object_wakes(behavior):
    p = plan()
    p['zones'].append(rectangle_zone(0, 0, 1, .65, behavior=behavior))
    b = NativeBudget(p)
    b.select(0)
    # Includes the approach margin and part of the incident polygon itself.
    b.select(3, [(.42, .42, .6, .6)])
    assert b.mode == 'idle'
    assert b.counts['excluded_motion_regions'] == 1
    assert b.counts['motion_wakes'] == 0
    b.objects([{'label': 'person', 'confidence': .99,
                'box': {'x1': 50, 'y1': 40, 'x2': 60, 'y2': 60}}], 100, 100, 3)
    b.select(3.1)
    assert b.mode == 'active'
    assert b.counts['object_wakes'] == 1
    # Excluded motion cannot extend an existing active hold.
    hold = b.active_until
    b.select(3.2, [(.5, .5, .6, .6)])
    assert b.active_until == hold


def test_crossing_motion_only_wakes_for_remaining_eligible_area():
    p = plan()
    p['zones'].append(rectangle_zone(.4, 0, 1, 1))
    b = NativeBudget(p)
    b.select(0)
    # Unexcluded portion is outside the incident zone AND its approach margin.
    b.select(3, [(.2, .6, .6, .8)])
    assert b.mode == 'idle'
    assert b.counts['excluded_motion_regions'] == 1
    p['zones'][-1] = rectangle_zone(0, 0, 1, .7)
    b = NativeBudget(p)
    b.select(0)
    b.select(3, [(.6, .6, .8, .8)])
    assert b.mode == 'active'
    assert b.counts['excluded_motion_regions'] == 0


def test_overlapping_exclusions_without_incident_zones():
    p = plan()
    p['zones'] = [rectangle_zone(0, 0, .6, 1), rectangle_zone(.4, 0, 1, 1)]
    b = NativeBudget(p)
    b.select(0)
    b.select(3, [(.1, .1, .9, .9)])
    assert b.mode == 'idle'
    assert b.counts['excluded_motion_regions'] == 1


def test_disabled_exclusion_and_mixed_motion_regions():
    p = plan()
    p['zones'].append(rectangle_zone(0, 0, 1, 1, enabled=False))
    b = NativeBudget(p)
    assert b.motion_relevant([(.6, .6, .8, .8)])
    p['zones'][-1] = rectangle_zone(0, 0, 1, .7)
    b = NativeBudget(p)
    assert b.motion_relevant([(.6, .8, .9, .9), (.6, .5, .7, .6)])
    assert b.counts['excluded_motion_regions'] == 1


def test_exclusion_uses_concave_polygon_not_bounding_box():
    p = plan()
    p['zones'] = [{'name': 'L', 'behavior': 'none', 'exclude_from_ema': True,
                   'points': [{'x': x, 'y': y} for x, y in
                              [(0, 0), (1, 0), (1, .4), (.4, .4), (.4, 1), (0, 1)]]}]
    b = NativeBudget(p)
    assert b.motion_relevant([(.5, .5, .8, .8)])
    assert not b.motion_relevant([(.1, .1, .3, .9)])


def test_diagonal_exclusion_union_and_narrow_unexcluded_gap():
    p = plan()
    p['zones'] = [
        {'name': 'lower', 'behavior': 'none', 'exclude_from_ema': True,
         'points': [{'x': x, 'y': y} for x, y in [(0, 0), (1, 0), (0, 1)]]},
        {'name': 'upper', 'behavior': 'none', 'exclude_from_ema': True,
         'points': [{'x': x, 'y': y} for x, y in [(1, 0), (1, 1), (0, 1)]]},
    ]
    assert not NativeBudget(p).motion_relevant([(.1, .1, .9, .9)])
    # A thin uncovered strip must survive; this is not a coarse pixel mask.
    p['zones'][1]['points'][0]['y'] = .0001
    p['zones'][1]['points'][2]['x'] = .0001
    assert NativeBudget(p).motion_relevant([(.49, .49, .51, .51)])


def test_incident_polygon_concavity_is_preserved_with_exclusions():
    p = plan()
    p['budget']['approach_padding'] = 0
    p['zones'][0]['points'] = [{'x': x, 'y': y} for x, y in [(0, 0), (1, 0), (0, 1)]]
    p['zones'].append(rectangle_zone(0, 0, .2, .2))
    b = NativeBudget(p)
    assert not b.motion_relevant([(.8, .8, .9, .9)])
    assert b.motion_relevant([(.3, .3, .4, .4)])


def test_existing_exclusion_flag_survives_config_and_spatial_plan():
    config = AppConfig(cameras=[camera()])
    config.cameras[0].zones[0].exclude_from_ema = True
    restored = AppConfig.model_validate(config.model_dump())
    p = spatial_plan(restored.cameras[0], restored.detector)
    assert p['zones'][0]['exclude_from_ema'] is True
    assert not NativeBudget(p).motion_relevant([(.6, .6, .8, .8)])


def test_fresh_object_relevance_class_confidence_and_cooldown():
    p = plan()
    p['zones'][0]['object_classes'] = ['person']
    p['zones'][0]['confidence_threshold'] = .7
    b = NativeBudget(p)
    b.select(0); b.select(3)
    obj = {'label':'car','confidence':.9,'box':{'x1':60,'y1':40,'x2':80,'y2':70}}
    b.objects([obj],100,100,3)
    assert b.active_until < 3
    obj['label']='person'; obj['confidence']=.6
    b.objects([obj],100,100,3)
    assert b.active_until < 3
    obj['confidence']=.9
    b.objects([obj],100,100,3)
    b.select(3.01)
    assert b.mode == 'active'
    b.objects([],100,100,3.2)
    b.select(3.8)
    assert b.mode == 'active'
    b.select(5)
    assert b.mode == 'idle'


def test_confirmation_hold_even_with_zero_cooldown_and_reset():
    p = plan(); p['budget']['cooldown_seconds']=0; p['budget']['confirmation_hold_seconds']=2
    b=NativeBudget(p)
    b.select(0); b.select(1.9)
    assert b.mode == 'active'
    b.select(3)
    assert b.mode == 'idle'
    assert b.select(0)
    assert b.mode == 'active'


def test_motion_overlap_uses_polygon_not_its_bounding_rectangle():
    triangle=[{'x':0,'y':0},{'x':1,'y':0},{'x':0,'y':1}]
    assert not overlaps((.8,.8,.9,.9),triangle,0)
    assert overlaps((.4,.4,.6,.6),triangle,0)
    assert overlaps((.1,-.2,.2,.2),triangle,0)
    assert overlaps((-.5,.49,1.5,.51),triangle,0)
    assert overlaps((1,0,1,0),triangle,0)


@pytest.mark.parametrize('field,value', [('motion_threshold',1.1),('min_persistence',0),('block_size',8),('idle_fps',0),('pixel_diff_threshold',256)])
def test_invalid_native_properties(field,value):
    with pytest.raises(ValueError):
        NativeBudgetConfig(**{field:value})


def test_invalid_effective_rates_and_idle_batch_rejected():
    config=AppConfig(cameras=[camera()]); config.cameras[0].native_budget.idle_fps=6
    with pytest.raises(ValueError,match='idle FPS'):
        AppConfig.model_validate(config.model_dump())
    config.cameras[0].native_budget.idle_fps=.5
    config.cameras[0].native_budget.enabled=True
    config.detector.native.batch_size=2
    with pytest.raises(ValueError,match='adaptive idle rate/batch'):
        AppConfig.model_validate(config.model_dump())


def test_intentional_idle_gap_is_healthy():
    from unittest.mock import Mock
    from survng.app.native_activity import NativeActivity
    config=AppConfig(cameras=[camera()])
    config.cameras[0].native_budget.enabled=True
    config.cameras[0].native_budget.idle_fps=.5
    activity=NativeActivity(config.cameras[0],config.detector,Mock(),Mock(),Mock())
    activity.health='healthy'; activity.last_fresh=100
    activity.tick(now=102.5)
    assert activity.health=='healthy'
    activity.tick(now=104)
    assert activity.health=='metadata_stale'


def test_changed_confirmation_rebuilds_adaptive_controller():
    from survng.app.config_application import manager_owned_config
    config = AppConfig(cameras=[camera()])
    changed = config.model_copy(deep=True)
    changed.detector.event_confirmation_frames += 1
    assert manager_owned_config(config) == manager_owned_config(changed)
    config.cameras[0].native_budget.enabled = True
    changed.cameras[0].native_budget.enabled = True
    assert manager_owned_config(config) != manager_owned_config(changed)


def test_budget_status_is_allowlisted():
    from survng.app.local_observability import _camera_snapshot
    result = _camera_snapshot({'live_pipeline': {'native_budget': {
        'mode': 'idle', 'target_fps': 1, 'skipped_frames': 12, 'excluded_motion_regions': 7, 'secret': 'hidden'}}})
    assert {k: v for k, v in result['inference_budget'].items() if v is not None} == {'mode': 'idle', 'target_fps': 1, 'skipped_frames': 12, 'excluded_motion_regions': 7}
