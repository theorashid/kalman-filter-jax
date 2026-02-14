import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kalman_filter_jax.equinox.kalman_filter import (
    AbstractCovariance,
    AbstractWeights,
    KalmanFilter,
)


class ConstantWeights(AbstractWeights):
    matrix: Float[Array, "out in"]
    def __call__(self, _t: Float[Array, ""]) -> Array:
        return self.matrix

class ConstantCovariance(AbstractCovariance):
    matrix: Float[Array, "out out"]
    def __call__(self, _t: Float[Array, ""]) -> Array:
        return self.matrix


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

    posterior = eqx.filter_jit(model)(emissions)

    assert jnp.allclose(
        posterior.filtered_means,
        true_states,
        atol=1e-4
    ), "The Kalman Filter failed to recover the deterministic trajectory."


def test_log_likelihood_is_finite(linear_motion_data) -> None:
    params, emissions, _ = linear_motion_data

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


def test_covariance_properties(linear_motion_data) -> None:
    """
    Checks that filtered covariances remain symmetric and positive-definite.
    """
    params, emissions, _ = linear_motion_data

    model = KalmanFilter(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_weights=ConstantWeights(params.dynamics_weights),
        dynamics_covariance=ConstantCovariance(params.dynamics_covariance),
        emission_weights=ConstantWeights(params.emission_weights),
        emission_covariance=ConstantCovariance(params.emission_covariance)
    )

    posterior = model(emissions)
    covs = posterior.filtered_covariances

    # Check symmetry: P == P.T
    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)

    # Check positive-definiteness (all eigenvalues > 0)
    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0), "Filtered covariances are not positive-definite."


def test_shared_model_batch_data() -> None:
    """
    Scenario: 1 model vs N data sequences.
    Demonstrates broadcasting a single Equinox module over batched data.
    """
    batch_size, timesteps, state_dim, obs_dim = 5, 10, 2, 1

    # single model instance
    model = KalmanFilter(
        initial_mean=jnp.ones(state_dim),
        initial_covariance=jnp.eye(state_dim),
        dynamics_weights=ConstantWeights(matrix=jnp.eye(state_dim)),
        dynamics_covariance=ConstantCovariance(matrix=jnp.eye(state_dim) * 0.1),
        emission_weights=ConstantWeights(matrix=jnp.zeros((obs_dim, state_dim))),
        emission_covariance=ConstantCovariance(matrix=jnp.eye(obs_dim))
    )

    emissions = jax.random.normal(
        jax.random.PRNGKey(0),
        (batch_size, timesteps, obs_dim),
    )

    # jax.vmap(model) is a shortcut for vmap over the __call__ method
    results = jax.vmap(model)(emissions)

    assert results.filtered_means.shape == (batch_size, timesteps, 2)
    assert results.marginal_log_likelihood.shape == (batch_size,)


def test_batch_filter_batch_data() -> None:
    """
    Scenario: N models vs N data sequences.
    Matches the test in basic: random data, batched params, checking shapes.
    """
    batch_size, timesteps, state_dim, obs_dim = 5, 10, 2, 1
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key, 2)

    # function to create many different models (perturbed initial_mean)
    def create_single_model(initial_mean):
        return KalmanFilter(
            initial_mean=initial_mean,
            initial_covariance=jnp.eye(state_dim),
            dynamics_weights=ConstantWeights(matrix=jnp.eye(state_dim)),
            dynamics_covariance=ConstantCovariance(matrix=jnp.eye(state_dim) * 0.1),
            emission_weights=ConstantWeights(matrix=jnp.zeros((obs_dim, state_dim))),
            emission_covariance=ConstantCovariance(matrix=jnp.eye(obs_dim))
        )

    # vmap the constructor to create a PyTree with batched model leaves
    initial_mean_batched = jax.random.normal(k1, (batch_size, state_dim))
    batched_model = jax.vmap(create_single_model)(initial_mean_batched)

    emissions = jax.random.normal(k2, (batch_size, timesteps, obs_dim))

    # vmap the call over _both_ the model and the data (N vs N)
    batch_filter = jax.vmap(lambda m, d: m(d), in_axes=(0, 0))
    result = eqx.filter_jit(batch_filter)(batched_model, emissions)

    assert result.filtered_means.shape == (batch_size, timesteps, state_dim)


