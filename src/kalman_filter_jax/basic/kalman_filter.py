"""
Basic implementation, inspired by filterjax and dynamax.

No optimisation as no constrain/unconstrain implemented.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jaxtyping import Float


class PosteriorFilter(NamedTuple):
    # log-likelihood is a scalar for one seq, but 1D array for a batch
    marginal_log_likelihood: Float[Array, "..."]
    # "... state" allows for (state,) AND (batch, state)
    filtered_means: Float[Array, "... time state"]
    filtered_covariances: Float[Array, "... time state state"]


class KalmanParams(NamedTuple):
    initial_mean: Float[Array, "... state"]
    initial_covariance: Float[Array, "... state state"]

    # Dynamics model: z_{t+1} = F_t z_t + q_t
    dynamics_weights: Float[Array, "... state state"]    # F
    dynamics_covariance: Float[Array, "... state state"] # Q

    # Observation model: y_t = H_t z_t + u_t + r_t
    emission_weights: Float[Array, "... obs state"]  # H
    emission_covariance: Float[Array, "... obs obs"] # R


def predict(
    state_mean: Float[Array, " state"],
    state_covariance: Float[Array, "state state"],
    dynamics_weights: Float[Array, "state state"],
    dynamics_covariance: Float[Array, "state state"]
) -> tuple[Float[Array, " state"], Float[Array, "state state"]]:
    """Time update step: predict the next state prior: p(z_t | y_{1:t-1})"""
    # m_{t|t-1} = F_t @ m_{t-1}
    prior_mean = dynamics_weights @ state_mean
    # P_{t|t-1} = F_t @ P_{t-1} @ F_t.T + Q_t
    prior_covariance = dynamics_weights @ state_covariance @ dynamics_weights.T + \
        dynamics_covariance
    return prior_mean, prior_covariance


def update(
    observation: Float[Array, " obs"],
    prior_mean: Float[Array, " state"],
    prior_covariance: Float[Array, "state state"],
    emission_weights: Float[Array, "obs state"],
    emission_covariance: Float[Array, "obs obs"]
) -> tuple[Float[Array, " state"], Float[Array, "state state"]]:
    """Measurement, update belief with observation: p(z_t | y_{1:t})"""
    # y_hat = H @ m_prior; residual = y_t - y_hat
    residual = observation - (emission_weights @ prior_mean)

    # S = H @ P_prior @ H.T + R
    innovation_covariance = emission_weights @ prior_covariance @ emission_weights.T + \
        emission_covariance

    # --- Possible to calculate log-likelihood here ---
    # This is equivalent to the separate log_likelihood function but more efficient
    # as it reuses 'residual' and 'innovation_covariance' already computed above.
    # The Residual approach uses the error vector:
    # MVN(mean=0, cov=S).log_prob(y - H @ z_prior)
    # This is a "zero-centered" version of the same probability calculation.
    #
    # ll = jsp.stats.multivariate_normal.logpdf(
    #   residual,
    #   mean=jnp.zeros_like(residual),
    #   cov=innovation_covariance,
    # )

    # Kalman Gain: K = P_prior @ H.T @ S^-1
    kalman_gain = jsp.linalg.solve(
        innovation_covariance,
        emission_weights @ prior_covariance,
        assume_a="pos",
    ).T

    # Posterior mean update belief: m_t = m_prior + K @ residual
    filtered_mean = prior_mean + kalman_gain @ residual

    # Posterior Covariance: P_t = P_prior - K @ S @ K.T
    filtered_covariance = prior_covariance - \
        kalman_gain @ innovation_covariance @ kalman_gain.T

    # --- Joseph Form (Alternative for numerical stability) ---
    # identity = jnp.eye(prior_mean.shape[0])
    # weight_residual = identity - kalman_gain @ emission_weights
    # filtered_covariance = (
    #     weight_residual @ prior_covariance @ weight_residual.T +
    #     kalman_gain @ emission_covariance @ kalman_gain.T
    # )

    return filtered_mean, filtered_covariance


def log_likelihood(
    observation: Float[Array, " obs"],
    prior_mean: Float[Array, " state"],
    prior_covariance: Float[Array, "state state"],
    emission_weights: Float[Array, "obs state"],
    emission_covariance: Float[Array, "obs obs"]
) -> Float[Array, ""]:
    """Compute log p(y_t | y_{1:t-1}) using the observation and predicted mean."""
    predicted_obs_mean = emission_weights @ prior_mean
    innovation_covariance = emission_weights @ prior_covariance @ emission_weights.T + \
        emission_covariance

    return jsp.stats.multivariate_normal.logpdf(
        observation,
        mean=predicted_obs_mean,
        cov=innovation_covariance
    )


def kalman_filter(
    params: KalmanParams,
    emissions: Float[Array, "time obs"]
) -> PosteriorFilter:
    def step(carry, y_t):
        m_prev, P_prev = carry

        # Predict: p(z_t | y_{1:t-1})
        m_prior, P_prior = predict(
            m_prev, P_prev, params.dynamics_weights, params.dynamics_covariance
        )

        # Log-Likelihood: p(y_t | y_{1:t-1})
        ll = log_likelihood(
            y_t, m_prior, P_prior,
            params.emission_weights, params.emission_covariance
        )

        # Update: p(z_t | y_{1:t})
        m_filt, P_filt = update(
            y_t, m_prior, P_prior,
            params.emission_weights, params.emission_covariance
        )

        return (m_filt, P_filt), (m_filt, P_filt, ll)

    initial_carry = (params.initial_mean, params.initial_covariance)
    # Passing `emissions` (shape (time, obs)) as the second arg to scan iterates through
    #  the time dimension (axis 0)
    _, (means, covs, lls) = jax.lax.scan(step, initial_carry, emissions)

    return PosteriorFilter(
        marginal_log_likelihood=jnp.sum(lls),
        filtered_means=means,
        filtered_covariances=covs,
    )
