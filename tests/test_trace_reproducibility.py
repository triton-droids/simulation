import numpy as np
import pytest
from robocub.utils.trace_reproducibility import compare_traces

def test_equal_and_divergent_traces():
    a={'height':np.ones((4,)), 'valid':np.ones(4,dtype=bool)}
    assert compare_traces(a,a)['exact']
    b={k:v.copy() for k,v in a.items()};b['height'][2]+=.1;b['valid'][3]=False
    r=compare_traces(a,b)
    assert not r['allclose'] and r['fields']['height']['first_difference_step']==3
    assert r['fields']['valid']['first_difference_step']==4

def test_nonfinite_and_schema_fail_closed():
    a={'x':np.array([np.nan,1.])}
    assert not compare_traces(a,a)['allclose']
    with pytest.raises(ValueError):compare_traces(a,{'y':np.ones(2)})
    with pytest.raises(ValueError):compare_traces(a,{'x':np.ones(3)})
