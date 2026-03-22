import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from kalman_filter_jax.equinox.kalman_filter import KalmanFilter
from kalman_filter_jax.equinox.params import TrainableCovariance, TrainableWeights


def test_kalman_filter_recovers_noiseless_trajectory(linear_motion_data):
    params, emissions, true_states = linear_motion_data

    model = KalmanFilter(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_weights=lambda _t: params.dynamics_weights,
        dynamics_covariance=lambda _t: params.dynamics_covariance,
        emission_weights=lambda _t: params.emission_weights,
        emission_covariance=lambda _t: params.emission_covariance,
    )

    posterior = eqx.filter_jit(model)(emissions)

    assert jnp.allclose(posterior.filtered_means, true_states, atol=1e-4)


@pytest.mark.parametrize(
    "data_fixture",
    ["linear_motion_data", "noisy_linear_motion_data"],
)
def test_log_likelihood_is_finite(request, data_fixture):
    params, emissions, _ = request.getfixturevalue(data_fixture)

    model = KalmanFilter(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_weights=lambda _t: params.dynamics_weights,
        dynamics_covariance=lambda _t: params.dynamics_covariance,
        emission_weights=lambda _t: params.emission_weights,
        emission_covariance=lambda _t: params.emission_covariance,
    )

    posterior = model(emissions)

    assert jnp.isfinite(posterior.marginal_log_likelihood)


@pytest.mark.parametrize(
    "data_fixture",
    ["linear_motion_data", "noisy_linear_motion_data"],
)
def test_covariance_properties(request, data_fixture):
    params, emissions, _ = request.getfixturevalue(data_fixture)

    model = KalmanFilter(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_weights=lambda _t: params.dynamics_weights,
        dynamics_covariance=lambda _t: params.dynamics_covariance,
        emission_weights=lambda _t: params.emission_weights,
        emission_covariance=lambda _t: params.emission_covariance,
    )

    posterior = eqx.filter_jit(model)(emissions)
    covs = posterior.filtered_covariances

    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)

    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0)


def test_shared_model_batch_data() -> None:
    """
    Scenario: 1 model vs N data sequences (vmap over data only).
    Demonstrates broadcasting a single Equinox module over batched data.
    """
    batch_size, timesteps, state_dim, obs_dim = 5, 10, 2, 1

    # single model instance
    model = KalmanFilter(
        initial_mean=jnp.ones(state_dim),
        initial_covariance=jnp.eye(state_dim),
        dynamics_weights=lambda _t: jnp.eye(state_dim),
        dynamics_covariance=lambda _t: jnp.eye(state_dim) * 0.1,
        emission_weights=lambda _t: jnp.zeros((obs_dim, state_dim)),
        emission_covariance=lambda _t: jnp.eye(obs_dim),
    )

    emissions = jax.random.normal(
        jax.random.key(0), (batch_size, timesteps, obs_dim)
    )

    # jax.vmap(model) is a shortcut for vmap over the __call__ method
    results = jax.vmap(model)(emissions)

    assert results.filtered_means.shape == (batch_size, timesteps, 2)
    assert results.marginal_log_likelihood.shape == (batch_size,)


def test_batch_filter_batch_data():
    """
    Scenario: N models vs N data sequences (vmap over both).
    Uses eqx.Module callables so all pytree leaves are arrays,
    which is required for jax.vmap over the model.
    """
    batch_size, timesteps, state_dim, obs_dim = 5, 10, 2, 1
    key = jax.random.key(2)
    k1, k2 = jax.random.split(key, 2)

    def create_single_model(initial_mean):
        return KalmanFilter(
            initial_mean=initial_mean,
            initial_covariance=jnp.eye(state_dim),
            dynamics_weights=TrainableWeights(jnp.eye(state_dim)),
            dynamics_covariance=TrainableCovariance(
                jnp.eye(state_dim) * 0.1
            ),
            emission_weights=TrainableWeights(
                jnp.zeros((obs_dim, state_dim))
            ),
            emission_covariance=TrainableCovariance(jnp.eye(obs_dim)),
        )

    initial_mean_batched = jax.random.normal(k1, (batch_size, state_dim))
    batched_model = jax.vmap(create_single_model)(initial_mean_batched)

    emissions = jax.random.normal(k2, (batch_size, timesteps, obs_dim))

    batch_filter = jax.vmap(lambda m, d: m(d), in_axes=(0, 0))
    result = eqx.filter_jit(batch_filter)(batched_model, emissions)

    assert result.filtered_means.shape == (
        batch_size,
        timesteps,
        state_dim,
    )
