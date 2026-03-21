from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy as jsp
from flax import nnx
from jaxtyping import Array, Float


class PosteriorFilter(NamedTuple):
    marginal_log_likelihood: Float[Array, "..."]
    filtered_means: Float[Array, "... time state"]
    filtered_covariances: Float[Array, "... time state state"]

class KalmanFilter(nnx.Module):
    def __init__(
        self,
        initial_mean: Float[Array, "... state"],
        initial_covariance: Float[Array, "... state state"],
        dynamics_weights: Callable[[Float[Array, ""]], Array],
        dynamics_covariance: Callable[[Float[Array, ""]], Array],
        emission_weights: Callable[[Float[Array, ""]], Array],
        emission_covariance: Callable[[Float[Array, ""]], Array],
    ):
        # static
        self.initial_mean = initial_mean
        self.initial_covariance = initial_covariance

        # Dynamics model: z_{t+1} = F_t z_t + q_t
        self.dynamics_weights = dynamics_weights       # F_t
        self.dynamics_covariance = dynamics_covariance # Q_t

        # Observation model: y_t = H_t z_t + u_t + r_t
        self.emission_weights = emission_weights        # H_t
        self.emission_covariance = emission_covariance  # R_t

    def predict(
        self,
        state_mean: Float[Array, "... state"],
        state_covariance: Float[Array, "... state state"],
        t: Float[Array, ""],
    ) -> tuple[Float[Array, "... state"], Float[Array, "... state state"]]:
        dynamics_weights_t = self.dynamics_weights(t)
        dynamics_covariance_t = self.dynamics_covariance(t)
        # m_{t|t-1} = F_t @ m_{t-1}
        prior_mean = jnp.einsum("...ij,...j->...i", dynamics_weights_t, state_mean)
        # P_{t|t-1} = F_t @ P_{t-1} @ F_t.T + Q_t
        prior_covariance = (
            jnp.einsum(
                "...ij,...jk,...lk->...il",
                dynamics_weights_t,
                state_covariance,
                dynamics_weights_t
            )
            + dynamics_covariance_t
        )
        return prior_mean, prior_covariance

    def update(
        self,
        observation: Float[Array, "... obs"],
        prior_mean: Float[Array, "... state"],
        prior_covariance: Float[Array, "... state state"],
        t: Float[Array, "..."],
    ) -> tuple[Float[Array, "... state"], Float[Array, "... state state"]]:
        """Time update step: predict the next state prior: p(z_t | y_{1:t-1})"""
        emission_weights_t = self.emission_weights(t)
        emission_covariance_t = self.emission_covariance(t)

        # y_hat = H_t @ m_prior; residual = y_t - y_hat
        residual = observation - jnp.einsum(
            "...ij,...j->...i",
            emission_weights_t,
            prior_mean
        )

        # S_t = H_t @ P_prior @ H.T + R
        innovation_covariance = (
            jnp.einsum("...ij,...jk,...lk->...il",
            emission_weights_t,
            prior_covariance,
            emission_weights_t
        )
            + emission_covariance_t
        )

        # Kalman Gain: K = P_prior @ H.T @ S^-1
        kalman_gain = jsp.linalg.solve(
            innovation_covariance,
            jnp.einsum("...ij,...jk->...ik", emission_weights_t, prior_covariance),
            assume_a="pos",
        ).mT

        # Posterior mean update belief: m_t = m_prior + K @ residual
        filtered_mean = prior_mean + jnp.einsum(
            "...ij,...j->...i",
            kalman_gain,
            residual
        )

        # Posterior Covariance: P_t = P_prior - K @ S @ K.T
        filtered_covariance = (
            prior_covariance
            - jnp.einsum(
                "...ij,...jk,...lk->...il",
                kalman_gain,
                innovation_covariance,
                kalman_gain
            )
        )

        return filtered_mean, filtered_covariance

    def log_likelihood(
        self,
        observation: Float[Array, "... obs"],
        prior_mean: Float[Array, "... state"],
        prior_covariance: Float[Array, "... state state"],
        t: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """Compute log p(y_t | y_{1:t-1}) using the observation and predicted mean."""
        emission_weights_t = self.emission_weights(t)
        emission_covariance_t = self.emission_covariance(t)

        predicted_obs_mean = jnp.einsum(
            "...ij,...j->...i",
            emission_weights_t,
            prior_mean
        )
        innovation_covariance = (
            jnp.einsum(
                "...ij,...jk,...lk->...il",
                emission_weights_t,
                prior_covariance,
                emission_weights_t
            )
            + emission_covariance_t
        )

        return jsp.stats.multivariate_normal.logpdf(
            observation,
            mean=predicted_obs_mean,
            cov=innovation_covariance
        )

    def __call__(
        self,
        emissions: Float[Array, "... time obs"]
    ) -> PosteriorFilter:
        # time axis index relative to the emissions rank
        # (T, O), time_dim = 0, (B, T, O), time_dim = 1
        # just -2 does not work
        time_dim = emissions.ndim - 2
        num_timesteps = emissions.shape[time_dim]

        # ts is always 1D, so its time axis is always 0
        ts = jnp.arange(num_timesteps, dtype=jnp.float32)

        # in_axes:
        #   - emissions: slice at time_dim (0 or 1)
        #   - ts: slice at 0
        # out_axes:
        #   - results: stack at time_dim
        @nnx.scan(in_axes=(nnx.Carry, (time_dim, 0)), out_axes=(nnx.Carry, time_dim))
        def step(carry, inputs):
            m_prev, P_prev = carry
            y_t, t = inputs

            # Predict: p(z_t | y_{1:t-1})
            m_prior, P_prior = self.predict(m_prev, P_prev, t)

            # Log-Likelihood: p(y_t | y_{1:t-1})
            ll = self.log_likelihood(y_t, m_prior, P_prior, t)

            # Update: p(z_t | y_{1:t})
            m_filt, P_filt = self.update(y_t, m_prior, P_prior, t)

            return (m_filt, P_filt), (m_filt, P_filt, ll)

        initial_carry = (self.initial_mean, self.initial_covariance)
        _, (means, covs, lls) = step(initial_carry, (emissions, ts))

        return PosteriorFilter(
            marginal_log_likelihood=jnp.sum(lls, axis=time_dim),
            filtered_means=means,
            filtered_covariances=covs,
        )
