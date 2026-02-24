import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from flax import nnx
from gpjax.numpyro_extras import register_parameters
from jaxtyping import Array, Float
from numpyro.infer import SVI, Predictive
from numpyro.infer.autoguide import AutoDelta, Trace_ELBO
from numpyro.infer.util import initialize_model

from kalman_filter_jax.nnx.kalman_filter import KalmanFilter
from kalman_filter_jax.nnx.params import (
    ConstantCovariance,
    ConstantWeights,
    TrainableCovariance,
    TrainableWeights,
)

jax.config.update("jax_enable_x64", val=True)

def kalman_filter_model(
    initial_mean: Float[Array, " state"],
    initial_covariance: Float[Array, "state state"],
    static_emission_weights: Float[Array, "obs state"],
    static_emission_covariance: Float[Array, "obs obs"]
) -> KalmanFilter:
    state_dim = initial_covariance.shape[0]

    # prior representing identity dynamic z_t+1 ~ z_t, persistence of state
    weights_prior = dist.Normal(jnp.eye(state_dim), 0.5).to_event(2)
    cov_prior = dist.InverseWishart(
        concentration=state_dim + 1,
        scale_matrix=jnp.eye(state_dim),
    )

    return KalmanFilter(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        # trainable dynamics components
        dynamics_weights=TrainableWeights(
            matrix=jnp.eye(state_dim),
            prior=weights_prior,
        ),
        dynamics_covariance=TrainableCovariance(
            matrix=jnp.eye(state_dim),
            prior=cov_prior,
        ),
        # assume emission (sensor) model H and R are known/fixed
        emission_weights=ConstantWeights(static_emission_weights),
        emission_covariance=ConstantCovariance(static_emission_covariance)
    )


def numpyro_lgssm(kalman_filter: KalmanFilter, y: Float[Array, "time obs"]):
    # `register_parameters` handles all the `numpyro.sample` calls
    # and replaces the values in the model with sampled ones
    sampled_kalman_filter = register_parameters(kalman_filter)
    posterior = sampled_kalman_filter(y)

    numpyro.factor("log_likelihood", posterior.marginal_log_likelihood)

    numpyro.deterministic("filtered_means", posterior.filtered_means)
    numpyro.deterministic("filtered_covariances", posterior.filtered_covariances)


def test_numpyro_trace_structure(noisy_linear_motion_data):
    params_data, emissions, _ = noisy_linear_motion_data

    kalman_filter_template = kalman_filter_model(
        initial_mean=params_data.initial_mean,
        initial_covariance=params_data.initial_covariance,
        static_emission_weights=params_data.emission_weights,
        static_emission_covariance=params_data.emission_covariance
    )

    key = jax.random.key(2)

    with numpyro.handlers.trace() as trace, numpyro.handlers.seed(rng_seed=key):
        numpyro_lgssm(kalman_filter_template, emissions)

    sampled_sites = trace.keys()

    # verify that only Trainable parameters appear as sample sites
    # trainable dynamics are present
    assert any(
        "dynamics_weights.matrix" in s for s in sampled_sites
    ), "Dynamics weights missing from trace"
    assert any(
        "dynamics_covariance.matrix" in s for s in sampled_sites
    ), "Dynamics covariance missing from trace"

    # constant emissions are absent
    assert not any(
        "emission_weights" in s for s in sampled_sites
    ), "Static weights leaked into trace"
    assert not any(
        "emission_covariance" in s for s in sampled_sites
    ), "Static covariance leaked into trace"


def test_numpyro_log_prob(noisy_linear_motion_data):
    params_data, emissions, _ = noisy_linear_motion_data

    kalman_filter_template = kalman_filter_model(
        initial_mean=params_data.initial_mean,
        initial_covariance=params_data.initial_covariance,
        static_emission_weights=params_data.emission_weights,
        static_emission_covariance=params_data.emission_covariance
    )

    key = jax.random.key(1)
    k1, k2, k3 = jax.random.split(key, num=3)

    model_info = initialize_model(
        rng_key=k1,
        model=numpyro_lgssm,
        model_args=(kalman_filter_template, emissions)
    )
    init_params, potential_fn, postprocess_fn, _ = model_info

    # test the potential function (negative log-joint density)
    with numpyro.handlers.seed(rng_seed=k2):
        initial_lp = potential_fn(init_params)

    assert jnp.isfinite(initial_lp), f"Potential returned non-finite val: {initial_lp}"

    # check that post-processing returns the deterministic sites, so
    # the KF forward pass (`nnx.scan`) works with sampled values
    with numpyro.handlers.seed(rng_seed=k3):
        processed_values = postprocess_fn(init_params)

    assert "filtered_means" in processed_values
    assert jnp.all(jnp.isfinite(processed_values["filtered_means"]))


def test_numpyro_svi(noisy_linear_motion_data):
    params_data, emissions, _ = noisy_linear_motion_data
    state_dim = params_data.initial_covariance.shape[0]

    kalman_filter_template = kalman_filter_model(
        initial_mean=params_data.initial_mean,
        initial_covariance=params_data.initial_covariance,
        static_emission_weights=params_data.emission_weights,
        static_emission_covariance=params_data.emission_covariance
    )

    _, initial_state = nnx.split(kalman_filter_template)

    key = jax.random.key(1)
    k1, k2 = jax.random.split(key, num=2)

    # AutoDelta for MAP
    guide = AutoDelta(numpyro_lgssm)
    optimizer = optax.adam(5e-3)
    svi = SVI(numpyro_lgssm, guide, optimizer, loss=Trace_ELBO())

    svi_result = svi.run(k1, 150, kalman_filter_template, emissions)

    # test denoising: loss should decrease significantly
    assert svi_result.losses[-1] < svi_result.losses[0], "SVI loss did not improve"

    map_params = guide.median(svi_result.params)
    with numpyro.handlers.substitute(data=map_params):
        final_model = register_parameters(kalman_filter_template)

    # convert to State object to avoid Module attribute type error
    final_state = nnx.state(final_model)

    # check substitute worked as expected
    f_key = next(k for k in map_params if "dynamics_weights" in k)
    assert jnp.array_equal(map_params[f_key], final_state.dynamics_weights.matrix[...])

    # check constant components were not touched during optimisation
    assert not any(
        "emission" in k for k in map_params
    ), "Constant parameters in sample sites"
    assert jnp.array_equal(
        final_state.emission_weights.matrix[...],
        params_data.emission_weights
    ), "Constant emission weights were modified."

    # test trainable components moved from initial values
    Q_final = final_state.dynamics_covariance.matrix[...]
    assert not jnp.array_equal(
        final_state.dynamics_weights.matrix[...],
        initial_state.dynamics_weights.matrix[...]
    ), "Trainable dynamics weights did not move."
    assert not jnp.array_equal(
        Q_final,
        initial_state.dynamics_covariance.matrix[...]
    ), "Trainable dynamics covariance did not move."

    # double check we still have a PSD matrix
    assert Q_final.shape == (state_dim, state_dim)
    assert jnp.all(jnp.linalg.eigvalsh(Q_final) > 0)

    # check Predictive and trajectory
    predictive = Predictive(numpyro_lgssm, guide=guide, num_samples=1)
    preds = predictive(k2, kalman_filter_template, emissions)

    # shape: (num_samples, time, state_dim) vs (time, obs_dim)
    assert preds["filtered_means"][0].shape == (emissions.shape[0], state_dim)
