import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from gpjax.parameters import Parameter, transform

from kalman_filter_jax.nnx.gpjax_parameters_extras import DEFAULT_BIJECTION
from kalman_filter_jax.nnx.params import TrainableCovariance


@pytest.mark.parametrize("dim", [2, 3, 5])
def test_trainable_covariance_properties(dim):
    key = jax.random.PRNGKey(2)

    # valid PSD matrix for the given dimension
    M = jax.random.normal(key, (dim, dim))
    init_val = M @ M.T + jnp.eye(dim) * 0.1

    cov_module = TrainableCovariance(init_val)

    # test covariance matrix properties
    cov_matrix = cov_module(jnp.array(0.0))
    assert cov_matrix.shape == (dim, dim), "Covariances not square"
    assert jnp.allclose(cov_matrix, cov_matrix.T, atol=1e-6), "Covariances asymmetric"
    assert jnp.all(jnp.linalg.eigvalsh(cov_matrix) > 0), "Covariances not PSD"

    _, params, _static_state = nnx.split(cov_module, Parameter, ...)

    assert len(params) > 0
    assert "matrix" in params
    assert params.matrix.tag == "psd"
    assert params.matrix.tag in DEFAULT_BIJECTION

    # unconstrain: matrix -> N(N+1)/2 vector
    unconstrained_state = transform(
        params=params,
        params_bijection=DEFAULT_BIJECTION,
        inverse=True,
    )
    unconstrained_val = unconstrained_state.matrix[...]

    assert unconstrained_val.ndim == 1
    assert unconstrained_val.shape[0] == (dim * (dim + 1)) // 2

    # constrain: vector -> matrix
    reconstructed_state = transform(
        params=unconstrained_state,
        params_bijection=DEFAULT_BIJECTION,
        inverse=False,
    )
    final_matrix = reconstructed_state.matrix[...]

    assert jnp.allclose(final_matrix, init_val, atol=1e-5)
    assert jnp.allclose(final_matrix, final_matrix.T)
    assert jnp.all(jnp.linalg.eigvalsh(final_matrix) > 0)

    # test gradient flow on unconstrained state
    # mimics what happens inside a training loss function
    # takes in unconstrained vector
    def loss_fn(state: nnx.State):
        constrained_state = transform(
            params=state,
            params_bijection=DEFAULT_BIJECTION,
            inverse=False,
        )
        # extract the matrix and do something (e.g. log-Likelihood, trace)
        # trace as simple scalar objective
        return jnp.trace(constrained_state.matrix[...])

    # compute gradients with respect to the unconstrained vector
    grads = jax.grad(loss_fn)(unconstrained_state)

    assert jnp.all(jnp.isfinite(grads.matrix[...])), "gradients contained NaNs/Infs"
    assert grads.matrix[...].shape == unconstrained_val.shape
