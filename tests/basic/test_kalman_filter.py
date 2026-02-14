import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from kalman_filter_jax.basic.kalman_filter import (
    KalmanParams,
    PosteriorFilter,
    kalman_filter,
)


def test_kalman_filter_recovers_noiseless_trajectory(linear_motion_data) -> None:
    # TODO: add a test with noise?
    params, emissions, true_states = linear_motion_data

    # TODO: right place to jit?
    posterior = jax.jit(kalman_filter)(params, emissions)

    assert jnp.allclose(
        posterior.filtered_means,
        true_states,
        atol=1e-4
    ), "The Kalman Filter failed to recover the deterministic trajectory."


def test_log_likelihood_is_finite(linear_motion_data) -> None:
    params, emissions, _ = linear_motion_data
    posterior = kalman_filter(params, emissions)

    assert jnp.isfinite(posterior.marginal_log_likelihood)


def test_covariance_properties(
    linear_motion_data: tuple[
        KalmanParams,
        Float[Array, "time obs"],
        Float[Array, "time state"]
    ]
) -> None:
    """
    Checks that filtered covariances remain symmetric and positive-definite.
    """
    params, emissions, _ = linear_motion_data
    posterior: PosteriorFilter = kalman_filter(params, emissions)

    covs: Float[Array, "time state state"] = posterior.filtered_covariances

    # 1. Symmetry: P should equal P.T for every timestep
    # We transpose the last two axes (-1, -2) to check symmetry
    is_symmetric: Array = jnp.allclose(covs, jnp.matrix_transpose(covs), atol=1e-6)
    assert is_symmetric, "Filtered covariances are not symmetric."

    # 2. Positive Definiteness: All eigenvalues should be > 0
    eigenvalues: Float[Array, "time state"] = jnp.linalg.eigvalsh(covs)
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
        emission_weights=jnp.tile(jnp.zeros((obs_dim, state_dim)), (batch_size, 1, 1)),
        emission_covariance=jnp.tile(jnp.eye(obs_dim), (batch_size, 1, 1))
    )

    emissions = jax.random.normal(key, (batch_size, timesteps, obs_dim))

    # (0, 0) maps 1st argument (params) and 2nd argument (emissions) over axis 0 (batch)
    # consider jax.jit(jax.vmap(...)) here
    batch_filter = jax.vmap(kalman_filter, in_axes=(0, 0))
    result = batch_filter(batched_params, emissions)

    assert result.filtered_means.shape == (batch_size, timesteps, state_dim)
