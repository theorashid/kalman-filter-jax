import jax
import jax.numpy as jnp
import pytest
from jax import Array

from kalman_filter_jax.basic.cuthbert import kalman_filter
from kalman_filter_jax.basic.kalman_filter import (
    KalmanParams,
    PosteriorFilter,
)

jax.config.update("jax_enable_x64", val=True)


def test_kalman_filter_recovers_noiseless_trajectory(linear_motion_data) -> None:
    params, emissions, true_states = linear_motion_data

    posterior = jax.jit(kalman_filter)(params, emissions)

    assert jnp.allclose(
        posterior.filtered_means,
        true_states,
        atol=1e-4,
    ), "The Kalman Filter failed to recover the deterministic trajectory."


@pytest.mark.parametrize(
    "data_fixture",
    ["linear_motion_data", "noisy_linear_motion_data"],
)
def test_log_likelihood_is_finite(request, data_fixture) -> None:
    params, emissions, _ = request.getfixturevalue(data_fixture)
    posterior = kalman_filter(params, emissions)

    assert jnp.isfinite(posterior.marginal_log_likelihood)


@pytest.mark.parametrize(
    "data_fixture",
    ["linear_motion_data", "noisy_linear_motion_data"],
)
def test_covariance_properties(request, data_fixture) -> None:
    """Checks that filtered covariances remain symmetric and positive-definite."""
    params, emissions, _ = request.getfixturevalue(data_fixture)
    posterior: PosteriorFilter = kalman_filter(params, emissions)

    covs = posterior.filtered_covariances

    is_symmetric: Array = jnp.allclose(covs, jnp.matrix_transpose(covs), atol=1e-6)
    assert is_symmetric, "Filtered covariances are not symmetric."

    eigenvalues = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvalues > 0), "Filtered covariances are not positive-definite."


def test_batch_filter() -> None:
    """Test if we can vmap the filter."""
    batch_size, timesteps, state_dim, obs_dim = 5, 10, 2, 1
    key = jax.random.PRNGKey(2)

    batched_params = KalmanParams(
        initial_mean=jax.random.normal(key, (batch_size, state_dim)),
        initial_covariance=jnp.tile(jnp.eye(state_dim), (batch_size, 1, 1)),
        dynamics_weights=jnp.tile(jnp.eye(state_dim), (batch_size, 1, 1)),
        dynamics_covariance=jnp.tile(jnp.eye(state_dim) * 0.1, (batch_size, 1, 1)),
        emission_weights=jnp.tile(
            jnp.zeros((obs_dim, state_dim)), (batch_size, 1, 1)
        ),
        emission_covariance=jnp.tile(jnp.eye(obs_dim), (batch_size, 1, 1)),
    )

    emissions = jax.random.normal(key, (batch_size, timesteps, obs_dim))

    batch_filter = jax.vmap(kalman_filter, in_axes=(0, 0))
    result = batch_filter(batched_params, emissions)

    assert result.filtered_means.shape == (batch_size, timesteps, state_dim)


def test_parallel_matches_sequential(noisy_linear_motion_data) -> None:
    params, emissions, _ = noisy_linear_motion_data

    seq = kalman_filter(params, emissions, parallel=False)
    par = kalman_filter(params, emissions, parallel=True)

    assert jnp.allclose(seq.filtered_means, par.filtered_means, atol=1e-3)
    assert jnp.allclose(
        seq.marginal_log_likelihood, par.marginal_log_likelihood, atol=1e-2
    )
