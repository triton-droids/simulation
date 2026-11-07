import math
import numpy as np
import pytest
from robocub.soccer.field.geometry import Field,geometry


def test_baseline_dimensions_and_line_outer_edges():
    f=Field.load();s={g['name']:g for g in geometry(f)}
    assert (f.length,f.width,f.goal_width,f.goal_height,f.goal_depth)==(14,9,2.5,1.7,1)
    assert s['touch_1']['pos'][1]+s['touch_1']['size'][1]/2==4.5
    assert s['end_1']['pos'][0]+f.line_width/2==7
    assert s['goal_area_front_1']['pos'][0]-f.line_width/2==6
    assert s['penalty_area_front_1']['pos'][0]-f.line_width/2==4
    assert s['post_1_1']['pos'][1]-f.post_width/2==1.25
    assert s['crossbar_1']['pos'][2]-f.post_width/2==1.7
    assert s['penalty_mark_1']['pos'][0]==5
    assert len([k for k in s if k.startswith('center_circle_')])==96
    assert len([k for k in s if k.startswith('corner_')])==64
    assert not any(g['collision'] for g in s.values() if 'circle' in g['name'] or 'corner' in g['name'])


def test_spawn_bounds_and_geometry_fixed():
    f=Field();assert f.legal(0,0);assert not f.legal(7,0);assert not f.legal(float('nan'),0)
    assert f.spacing>f.length+2*f.goal_depth+2*f.border
    assert Field.load('large_22x14').length==22

@pytest.mark.parametrize('kwargs',[{'width':-1},{'length':float('nan')},{'goal_area_width':8,'penalty_area_width':6},{'line_width':1}])
def test_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):Field(**kwargs)
