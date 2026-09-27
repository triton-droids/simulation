import numpy as np
from source.scripts.audit_g1_target_clipping import clipping_stats


def test_distinguishes_raw_action_from_asymmetric_target_clipping():
    result = clipping_stats(np.array([[0.], [-.5], [-2.]]), np.array([-.363]), .5, np.array([[-.4,.4]]))
    np.testing.assert_allclose(result['raw_saturation_fraction'], [1/3])
    np.testing.assert_allclose(result['target_clipped_fraction'], [2/3])
    np.testing.assert_allclose(result['target_excess_max_rad'], [.463])
