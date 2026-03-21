"""Kalman filter implementation using cuthbert."""

import jax.numpy as jnp
from cuthbert import filter as cuthbert_filter
from cuthbert.gaussian import kalman
from cuthbertlib.types import ArrayTreeLike
from jax import Array
from jaxtyping import Float

from kalman_filter_jax.basic.kalman_filter import KalmanParams, PosteriorFilter


def kalman_filter(
    params: KalmanParams,
    emissions: Float[Array, "time obs"],
    *,
    parallel: bool = False,
) -> PosteriorFilter:
    chol_Q = jnp.linalg.cholesky(params.dynamics_covariance)
    chol_R = jnp.linalg.cholesky(params.emission_covariance)
    chol_P0 = jnp.linalg.cholesky(params.initial_covariance)
    # dynamics bias (no offset in our model)
    c = jnp.zeros_like(params.initial_mean)
    # emission bias (no offset in our model)
    d = jnp.zeros(emissions.shape[-1])

    # cuthbert iterates over `model_inputs` (here jnp.arange(n_time + 1)),
    # passing each element to these callbacks:
    #   init receives model_inputs[0], dynamics/observation receive [t].
    # We use the time index in get_observation_params to index emissions.
    # init/dynamics are time-invariant so the input is unused (_).

    def get_init_params(_model_inputs: ArrayTreeLike) -> tuple[Array, Array]:
        # m0, chol_P0
        return params.initial_mean, chol_P0

    def get_dynamics_params(_model_inputs: ArrayTreeLike) -> tuple[Array, Array, Array]:
        # F, c, chol_Q
        return params.dynamics_weights, c, chol_Q

    def get_observation_params(
        model_inputs: ArrayTreeLike,
    ) -> tuple[Array, Array, Array, Array]:
        # H, d, chol_R, y_t
        # model_inputs is 1-indexed, so subtract 1 to index emissions
        t = model_inputs - 1
        return params.emission_weights, d, chol_R, emissions[t]

    filter_obj = kalman.build_filter(
        get_init_params,  # type: ignore[arg-type]
        get_dynamics_params,  # type: ignore[arg-type]
        get_observation_params,  # type: ignore[arg-type]
    )

    n_time = emissions.shape[0]
    # [0, 1, ..., n_time]: index 0 is for init, 1..n_time for filter steps
    model_inputs = jnp.arange(n_time + 1)

    states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)

    # slice off the init state (index 0) to match kalman_filter.py output
    filtered_means = states.mean[1:]
    filtered_chol_covs = states.chol_cov[1:]
    # reconstruct full covariances from Cholesky for PosteriorFilter compat
    filtered_covariances = (
        filtered_chol_covs @ jnp.matrix_transpose(filtered_chol_covs)
    )
    # cumulative log normalizing constant, equivalent to sum of per-step lls
    marginal_log_likelihood = states.log_normalizing_constant[-1]

    return PosteriorFilter(
        marginal_log_likelihood=marginal_log_likelihood,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
    )
