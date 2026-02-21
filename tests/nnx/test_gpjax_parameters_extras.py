"""Adapted from [gpjax.tests.test_parameters.py](https://github.com/thomaspinder/GPJax/blob/2718ee823d6f7d9aa0332d9c88578f1d2ccffa7f/tests/test_parameters.py)"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from gpjax.parameters import transform
from hypothesis import (
    given,
    settings,
)
from hypothesis import (
    strategies as st,
)
from hypothesis.extra.numpy import arrays
from jax.experimental import checkify

from kalman_filter_jax.nnx.gpjax_parameters_extras import (
    DEFAULT_BIJECTION,
    PSDMatrix,
)


def valid_shapes(min_dims=0, max_dims=2):
    return st.integers(min_dims, max_dims).flatmap(
        lambda d: st.lists(st.integers(1, 5), min_size=d, max_size=d).map(tuple)
    )

def real_arrays(shape_strategy=valid_shapes(), min_value=None, max_value=None):  # noqa: B008
    return arrays(
        dtype=np.float64,
        shape=shape_strategy,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    ).map(jnp.array)

def psd_matrices(n_min=1, n_max=5):
    return st.integers(n_min, n_max).flatmap(
        lambda n: arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(min_value=0.1, max_value=10, width=64),
        ).map(lambda x: jnp.dot(jnp.array(x), jnp.array(x).T) + jnp.eye(n) * 0.1)
    )

def non_psd_matrices(n_min=2, n_max=5):
    return st.integers(n_min, n_max).flatmap(
        lambda n: arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(min_value=-10, max_value=10, width=64),
        ).map(lambda x: (jnp.array(x) + jnp.array(x).T) / 2.0)
         .filter(lambda x: jnp.any(jnp.linalg.eigvalsh(x) <= 0))
    )

@given(value=psd_matrices())
def test_psd_parameter_valid(value):
    p = PSDMatrix(value)
    assert jnp.allclose(p[...], value)
    assert p.tag == "psd"

@given(value=non_psd_matrices())
def test_psd_parameter_invalid(value):
    # should catch non-positive definite matrices during init
    with pytest.raises(checkify.JaxRuntimeError):
        PSDMatrix(value)

@settings(deadline=None)
@given(value=psd_matrices(n_min=1, n_max=4))
def test_psd_transform_roundtrip(value):
    params = nnx.State({"p": PSDMatrix(value)})

    # 1. unconstrain (matrix -> vector)
    unconstrained = transform(params, DEFAULT_BIJECTION, inverse=True)
    vec = unconstrained["p"][...]

    # check that it flattened to N(N+1)/2 size
    n = value.shape[0]
    assert vec.shape == (n * (n + 1) // 2,)

    # 2. re-constrain (vector -> matrix)
    reconstructed = transform(unconstrained, DEFAULT_BIJECTION, inverse=False)

    assert jnp.allclose(reconstructed["p"][...], value, atol=1e-5)
    assert jnp.all(jnp.linalg.eigvalsh(reconstructed["p"][...]) > 0)
