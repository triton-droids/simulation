"""Compare diagnostic traces without changing validation gates."""
import numpy as np


def compare_traces(a, b):
    if set(a) != set(b):
        raise ValueError('Trace fields differ')
    rows = {}
    for key in a:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape:
            raise ValueError('Trace shapes differ: ' + key)
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        close = np.isclose(x, y, rtol=1e-5, atol=1e-6)
        bad = np.flatnonzero(~close.reshape(len(x), -1).all(axis=1))
        rows[key] = dict(finite=finite, exact=bool(finite and np.array_equal(x, y)),
                         allclose=bool(finite and close.all()),
                         first_difference_step=int(bad[0] + 1) if len(bad) else None,
                         max_abs=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if finite else None)
    return dict(exact=all(r['exact'] for r in rows.values()),
                allclose=all(r['allclose'] for r in rows.values()), fields=rows)
