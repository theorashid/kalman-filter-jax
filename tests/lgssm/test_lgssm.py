import jax
import jax.numpy as jnp
import pytest

from kalman_filter_jax.lgssm.model import LinearGaussianSSM

jax.config.update("jax_enable_x64", val=True)

METHODS = ["kalman", "kalman_parallel"]


def _model_from_params(params):
    return LinearGaussianSSM(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        dynamics_weights=lambda _t: params.dynamics_weights,
        dynamics_covariance=lambda _t: params.dynamics_covariance,
        emission_weights=lambda _t: params.emission_weights,
        emission_covariance=lambda _t: params.emission_covariance,
    )


def test_construct_with_callables(linear_motion_data):
    params, _, _ = linear_motion_data
    model = _model_from_params(params)

    assert callable(model.dynamics_weights)
    F = model.dynamics_weights(jnp.array(0.0))
    assert jnp.array_equal(F, params.dynamics_weights)


@pytest.mark.parametrize("method", METHODS)
def test_recovers_noiseless_trajectory(linear_motion_data, method):
    params, emissions, true_states = linear_motion_data
    model = _model_from_params(params)
    posterior = model.infer(emissions, method=method)
    assert jnp.allclose(posterior.filtered_means, true_states, atol=1e-4)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "data_fixture",
    ["linear_motion_data", "noisy_linear_motion_data"],
)
def test_log_likelihood_is_finite(request, data_fixture, method):
    params, emissions, _ = request.getfixturevalue(data_fixture)
    model = _model_from_params(params)
    ll = model.infer(emissions, method=method).marginal_log_likelihood
    assert jnp.isfinite(ll)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "data_fixture",
    ["linear_motion_data", "noisy_linear_motion_data"],
)
def test_covariance_properties(request, data_fixture, method):
    params, emissions, _ = request.getfixturevalue(data_fixture)
    model = _model_from_params(params)
    posterior = model.infer(emissions, method=method)
    covs = posterior.filtered_covariances

    batch_transpose = jnp.transpose(covs, (0, 2, 1))
    assert jnp.allclose(covs, batch_transpose, atol=1e-8)

    eigenvals = jnp.linalg.eigvalsh(covs)
    assert jnp.all(eigenvals > 0)


def test_parallel_matches_sequential(noisy_linear_motion_data):
    params, emissions, _ = noisy_linear_motion_data
    model = _model_from_params(params)

    seq = model.infer(emissions, method="kalman")
    par = model.infer(emissions, method="kalman_parallel")

    assert jnp.allclose(seq.filtered_means, par.filtered_means, atol=1e-3)
    assert jnp.allclose(
        seq.marginal_log_likelihood, par.marginal_log_likelihood, atol=1e-2
    )


def test_invalid_method(linear_motion_data):
    params, emissions, _ = linear_motion_data
    model = _model_from_params(params)

    with pytest.raises((ValueError, Exception)):
        model.infer(emissions, method="ekf")  # type: ignore[arg-type]


def test_vmap_over_data():
    batch_size, timesteps, state_dim, obs_dim = 5, 10, 2, 1

    model = LinearGaussianSSM(
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

    results = jax.vmap(model.infer)(emissions)

    assert results.filtered_means.shape == (batch_size, timesteps, state_dim)
    assert results.marginal_log_likelihood.shape == (batch_size,)
