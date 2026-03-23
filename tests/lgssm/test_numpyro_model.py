# Outer loop: fully Bayesian via numpyro.
#
# The model constructs a LinearGaussianSSM from sampled values and
# passes the log-likelihood to numpyro.factor.

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from jaxtyping import Array, Float
from numpyro.infer import SVI, Predictive
from numpyro.infer.autoguide import AutoDelta, Trace_ELBO
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from kalman_filter_jax.lgssm.model import LinearGaussianSSM

jax.config.update("jax_enable_x64", val=True)


def numpyro_lgssm(
    y: Float[Array, "time obs"],
    initial_mean: Float[Array, " state"],
    initial_covariance: Float[Array, "state state"],
    emission_weights: Float[Array, "obs state"],
    emission_covariance: Float[Array, "obs obs"],
):
    state_dim = initial_covariance.shape[0]

    F = numpyro.sample(
        "dynamics_weights",
        dist.Normal(jnp.eye(state_dim), 0.5).to_event(2),
    )
    Q = numpyro.sample(
        "dynamics_covariance",
        dist.InverseWishart(
            concentration=state_dim + 1,
            scale_matrix=jnp.eye(state_dim),
        ),
    )

    model = LinearGaussianSSM(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_weights=lambda _t, f=F: f,
        dynamics_covariance=lambda _t, q=Q: q,
        emission_weights=lambda _t, h=emission_weights: h,
        emission_covariance=lambda _t, r=emission_covariance: r,
    )

    posterior = model.infer(y)

    numpyro.factor("log_likelihood", posterior.marginal_log_likelihood)
    numpyro.deterministic("filtered_means", posterior.filtered_means)
    numpyro.deterministic("filtered_covariances", posterior.filtered_covariances)


def _model_args(params_data, emissions):
    return (
        emissions,
        params_data.initial_mean,
        params_data.initial_covariance,
        params_data.emission_weights,
        params_data.emission_covariance,
    )


def _init_values(state_dim):
    # Perturb off the identity: cuthbert's Cholesky-parameterised scan
    # produces NaN gradients at exactly F = I (degenerate point for the
    # Cholesky factorisation in the predict step).
    return {
        "dynamics_weights": jnp.eye(state_dim) + 0.01,
        "dynamics_covariance": jnp.eye(state_dim) * 1.01,
    }


def test_numpyro_trace_structure(noisy_linear_motion_data):
    params_data, emissions, _ = noisy_linear_motion_data
    key = jax.random.key(2)

    with numpyro.handlers.trace() as trace, numpyro.handlers.seed(
        rng_seed=key
    ):
        numpyro_lgssm(*_model_args(params_data, emissions))

    sampled_sites = trace.keys()

    # trainable dynamics are present
    assert "dynamics_weights" in sampled_sites
    assert "dynamics_covariance" in sampled_sites

    # constant emissions are absent
    assert "emission_weights" not in sampled_sites
    assert "emission_covariance" not in sampled_sites


def test_numpyro_log_prob(noisy_linear_motion_data):
    params_data, emissions, _ = noisy_linear_motion_data
    state_dim = params_data.initial_covariance.shape[0]

    key = jax.random.key(1)
    k1, k2, k3 = jax.random.split(key, num=3)

    model_info = initialize_model(
        rng_key=k1,
        model=numpyro_lgssm,
        model_args=_model_args(params_data, emissions),
        init_strategy=init_to_value(values=_init_values(state_dim)),
    )
    init_params, potential_fn, postprocess_fn, _ = model_info

    with numpyro.handlers.seed(rng_seed=k2):
        initial_lp = potential_fn(init_params.z)

    assert jnp.isfinite(initial_lp)

    with numpyro.handlers.seed(rng_seed=k3):
        processed_values = postprocess_fn(init_params.z)

    assert "filtered_means" in processed_values
    assert jnp.all(jnp.isfinite(processed_values["filtered_means"]))


def test_numpyro_svi(noisy_linear_motion_data):
    params_data, emissions, _ = noisy_linear_motion_data
    state_dim = params_data.initial_covariance.shape[0]

    args = _model_args(params_data, emissions)

    key = jax.random.key(1)
    k1, k2 = jax.random.split(key, num=2)

    init_values = _init_values(state_dim)
    guide = AutoDelta(
        numpyro_lgssm,
        init_loc_fn=init_to_value(values=init_values),
    )
    optimizer = optax.adam(5e-3)
    svi = SVI(numpyro_lgssm, guide, optimizer, loss=Trace_ELBO())

    svi_result = svi.run(k1, 150, *args)

    # loss decreased
    assert svi_result.losses[-1] < svi_result.losses[0]

    map_params = guide.median(svi_result.params)

    # constant emissions not in sample sites
    assert not any("emission" in k for k in map_params)

    F_final = map_params["dynamics_weights"]
    Q_final = map_params["dynamics_covariance"]

    assert jnp.all(jnp.isfinite(F_final))
    assert jnp.all(jnp.isfinite(Q_final))

    # InverseWishart ensures PSD
    assert Q_final.shape == (state_dim, state_dim)
    assert jnp.all(jnp.linalg.eigvalsh(Q_final) > 0)

    # check Predictive works end-to-end
    predictive = Predictive(numpyro_lgssm, guide=guide, num_samples=1)
    preds = predictive(k2, *args)

    assert preds["filtered_means"][0].shape == (
        emissions.shape[0],
        state_dim,
    )
