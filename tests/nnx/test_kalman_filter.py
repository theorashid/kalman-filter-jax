import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from kalman_filter_jax.nnx.kalman_filter import KalmanFilter
from kalman_filter_jax.nnx.params import (
    ConstantCovariance,
    ConstantWeights,
)


def test_kalman_filter_recovers_noiseless_trajectory(linear_motion_data) -> None:
    params, emissions, true_states = linear_motion_data

    model = KalmanFilter(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_weights=ConstantWeights(params.dynamics_weights),
        dynamics_covariance=ConstantCovariance(params.dynamics_covariance),
        emission_weights=ConstantWeights(params.emission_weights),
        emission_covariance=ConstantCovariance(params.emission_covariance)
    )

    posterior = nnx.jit(model)(emissions)

    assert jnp.allclose(
        posterior.filtered_means,
        true_states,
        atol=1e-4
    ), "The Kalman Filter failed to recover the deterministic trajectory."

@pytest.mark.parametrize(
    "data_fixture",
    ["linear_motion_data", "noisy_linear_motion_data"]
)
def test_log_likelihood_is_finite(request, data_fixture) -> None:
    params, emissions, _ = request.getfixturevalue(data_fixture)

    model = KalmanFilter(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_weights=ConstantWeights(params.dynamics_weights),
        dynamics_covariance=ConstantCovariance(params.dynamics_covariance),
        emission_weights=ConstantWeights(params.emission_weights),
        emission_covariance=ConstantCovariance(params.emission_covariance)
    )

    posterior = model(emissions)

    assert jnp.isfinite(posterior.marginal_log_likelihood)

@pytest.mark.parametrize("cov_type", ["constant"])
@pytest.mark.parametrize(
    "data_fixture",
    ["linear_motion_data", "noisy_linear_motion_data"]
)
def test_covariance_properties(request, data_fixture, cov_type) -> None:
    """
    Checks that filtered covariances remain symmetric and positive-definite.
    """
    params, emissions, _ = request.getfixturevalue(data_fixture)
    # state_dim = params.initial_mean.shape[0]
    # obs_dim = params.emission_weights.shape[0]

    # key = jax.random.key(1)
    # k1, k2 = jax.random.split(key, 2)

    # Factory for the dynamic covariance components
    if cov_type == "constant":
        q_cov = ConstantCovariance(params.dynamics_covariance)
        r_cov = ConstantCovariance(params.emission_covariance)
    # elif cov_type == "trainable":

    model = KalmanFilter(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_weights=ConstantWeights(params.dynamics_weights),
        dynamics_covariance=q_cov,
        emission_weights=ConstantWeights(params.emission_weights),
        emission_covariance=r_cov,
    )

    posterior = nnx.jit(model)(emissions)
    covs = posterior.filtered_covariances

    # Check symmetry: P == P.T
    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)

    # Check positive-definiteness (all eigenvalues > 0)
    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0), "Filtered covariances are not positive-definite."

def test_shared_model_batch_data() -> None:
    """Scenario: 1 model vs N data sequences (vmap over data only)."""
    batch_size, timesteps, state_dim, obs_dim = 5, 10, 2, 1

    # single model instance
    model = KalmanFilter(
        initial_mean=jnp.ones(state_dim),
        initial_covariance=jnp.eye(state_dim),
        dynamics_weights=ConstantWeights(jnp.eye(state_dim)),
        dynamics_covariance=ConstantCovariance(jnp.eye(state_dim) * 0.1),
        emission_weights=ConstantWeights(jnp.zeros((obs_dim, state_dim))),
        emission_covariance=ConstantCovariance(jnp.eye(obs_dim))
    )

    emissions = jax.random.normal(jax.random.key(0), (batch_size, timesteps, obs_dim))

    # but vmap the 0th (batch) axis of emissions
    results = nnx.vmap(model, in_axes=0)(emissions)

    assert results.filtered_means.shape == (batch_size, timesteps, 2)
    assert results.marginal_log_likelihood.shape == (batch_size,)

def test_batch_filter_batch_data() -> None:
    """Scenario: N models vs N data sequences (vmap over both model and data)."""
    batch_size, timesteps, state_dim, obs_dim = 5, 10, 2, 1
    key = jax.random.key(2)
    k1, k2 = jax.random.split(key, 2)

    initial_mean_batched = jax.random.normal(k1, (batch_size, state_dim))

    # In nnx, vmap the constructor directly over the first argument (`initial_mean`).
    # This creates a stacked module where every attribute has a leading dimension of
    # batch_size
    batched_model = nnx.vmap(KalmanFilter, in_axes=(0, None, None, None, None, None))(
        initial_mean_batched,
        jnp.eye(state_dim),
        ConstantWeights(jnp.eye(state_dim)),
        ConstantCovariance(jnp.eye(state_dim) * 0.1),
        ConstantWeights(jnp.zeros((obs_dim, state_dim))),
        ConstantCovariance(jnp.eye(obs_dim))
    )

    emissions = jax.random.normal(k2, (batch_size, timesteps, obs_dim))

    results = batched_model(emissions)

    assert results.filtered_means.shape == (batch_size, timesteps, state_dim)
