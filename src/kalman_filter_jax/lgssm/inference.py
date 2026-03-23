# Inner loop inference methods for LinearGaussianSSM.
#
# All methods delegate to cuthbert. The model's callable fields are
# evaluated at each timestep to autogenerate the get_*_params
# callbacks. The user never touches build_filter directly.
#
#   | method              | cuthbert backend                  | exact? |
#   |---------------------|-----------------------------------|--------|
#   | kalman              | gaussian.kalman (sequential)      | yes    |
#   | kalman_parallel     | gaussian.kalman (associative)     | yes    |
#
# Future (NonlinearGaussianSSM etc):
#   | ekf                 | gaussian.taylor                   | approx |
#   | ukf                 | gaussian.moments                  | approx |
#   | particle            | smc.particle_filter               | yes*   |

import jax.numpy as jnp
from jaxtyping import Array, Float

from .model import LinearGaussianSSM, PosteriorFilter


def filter_kalman(
    model: LinearGaussianSSM,
    emissions: Float[Array, "time obs"],
    *,
    parallel: bool = False,
) -> PosteriorFilter:
    from cuthbert import filter as cuthbert_filter  # noqa: PLC0415
    from cuthbert.gaussian import kalman  # noqa: PLC0415

    dtype = model.initial_mean.dtype
    state_dim = model.initial_mean.shape[0]
    obs_dim = emissions.shape[-1]

    # cuthbert's native parametrisation is Cholesky factors. We decompose
    # the user's full covariance matrices here. If the model stored Cholesky
    # factors directly, these decompositions could be skipped.
    chol_P0 = jnp.linalg.cholesky(model.initial_covariance)

    c = jnp.zeros(state_dim)
    d = jnp.zeros(obs_dim)

    # cuthbert iterates over model_inputs = [0, 1, ..., n_time].
    # Index 0 -> get_init_params, indices 1..n_time -> dynamics/observation.
    # The callbacks evaluate the model's callables at each timestep.

    def get_init_params(
        _model_inputs: object,
    ) -> tuple[Array, Array]:
        return model.initial_mean, chol_P0

    def get_dynamics_params(
        model_inputs: object,
    ) -> tuple[Array, Array, Array]:
        t = jnp.array(model_inputs - 1, dtype=dtype)  # type: ignore[operator]
        F_t = model.dynamics_weights(t)
        Q_t = model.dynamics_covariance(t)
        chol_Q_t = jnp.linalg.cholesky(Q_t)
        return F_t, c, chol_Q_t

    def get_observation_params(
        model_inputs: object,
    ) -> tuple[Array, Array, Array, Array]:
        t = jnp.array(model_inputs - 1, dtype=dtype)  # type: ignore[operator]
        t_idx = jnp.array(model_inputs - 1, dtype=jnp.int32)  # type: ignore[operator]
        H_t = model.emission_weights(t)
        R_t = model.emission_covariance(t)
        chol_R_t = jnp.linalg.cholesky(R_t)
        return H_t, d, chol_R_t, emissions[t_idx]

    filter_obj = kalman.build_filter(
        get_init_params,  # type: ignore[arg-type]
        get_dynamics_params,  # type: ignore[arg-type]
        get_observation_params,  # type: ignore[arg-type]
    )

    n_time = emissions.shape[0]
    model_inputs = jnp.arange(n_time + 1)

    states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)

    # slice off the init state (index 0) to match our PosteriorFilter
    filtered_means = states.mean[1:]
    filtered_chol_covs = states.chol_cov[1:]
    filtered_covariances = (
        filtered_chol_covs @ jnp.matrix_transpose(filtered_chol_covs)
    )
    marginal_log_likelihood = states.log_normalizing_constant[-1]

    return PosteriorFilter(
        marginal_log_likelihood=marginal_log_likelihood,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
    )
