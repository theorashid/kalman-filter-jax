import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float


class AbstractWeights(eqx.Module):
    @abc.abstractmethod
    def __call__(self, t: Float[Array, ""]) -> Float[Array, "out in"]:
        ...

class AbstractCovariance(eqx.Module):
    @abc.abstractmethod
    def __call__(self, t: Float[Array, ""]) -> Float[Array, "out out"]:
        ...


class PosteriorFilter(eqx.Module):
    marginal_log_likelihood: Float[Array, ""]
    filtered_means: Float[Array, "time state"]
    filtered_covariances: Float[Array, "time state state"]


class KalmanFilter(eqx.Module):
    initial_mean: Float[Array, " state"]
    initial_covariance: Float[Array, "state state"]

    # Dynamics model: z_{t+1} = F_t z_t + q_t
    # could probably do callable[Float[Array, ""], Float[Array, "out in"]]
    # but keeping things in Equinox is safer for training
    dynamics_weights: AbstractWeights       # F_t
    dynamics_covariance: AbstractCovariance # Q_t

    # Observation model: y_t = H_t z_t + u_t + r_t
    emission_weights: AbstractWeights        # H_t
    emission_covariance: AbstractCovariance  # R_t

    def predict(
        self,
        state_mean: Float[Array, " state"],
        state_covariance: Float[Array, "state state"],
        t: Float[Array, ""],
    ) -> tuple[Float[Array, " state"], Float[Array, "state state"]]:
        """Time update step: predict the next state prior: p(z_t | y_{1:t-1})"""
        dynamics_weights_t = self.dynamics_weights(t)
        dynamics_covariance_t = self.dynamics_covariance(t)
        # m_{t|t-1} = F_t @ m_{t-1}
        prior_mean = dynamics_weights_t @ state_mean
        # P_{t|t-1} = F_t @ P_{t-1} @ F_t.T + Q_t
        prior_covariance = (
            dynamics_weights_t @ state_covariance @ dynamics_weights_t.T
            + dynamics_covariance_t
        )
        return prior_mean, prior_covariance

    def update(
        self,
        observation: Float[Array, " obs"],
        prior_mean: Float[Array, " state"],
        prior_covariance: Float[Array, "state state"],
        t: Float[Array, ""],
    ) -> tuple[Float[Array, " state"], Float[Array, "state state"]]:
        """Measurement, update belief with observation: p(z_t | y_{1:t})"""
        emission_weights_t = self.emission_weights(t)
        emission_covariance_t = self.emission_covariance(t)

        # y_hat = H_t @ m_prior; residual = y_t - y_hat
        residual = observation - (emission_weights_t @ prior_mean)

        # S_t = H_t @ P_prior @ H.T + R
        innovation_covariance = (
            emission_weights_t @ prior_covariance @ emission_weights_t.T
            + emission_covariance_t
        )

        # Kalman Gain: K = P_prior @ H.T @ S^-1
        kalman_gain = jsp.linalg.solve(
            innovation_covariance,
            emission_weights_t @ prior_covariance,
            assume_a="pos",
        ).T

        # Posterior mean update belief: m_t = m_prior + K @ residual
        filtered_mean = prior_mean + kalman_gain @ residual

        # Posterior Covariance: P_t = P_prior - K @ S @ K.T
        filtered_covariance = (
            prior_covariance - kalman_gain @ innovation_covariance @ kalman_gain.T
        )

        return filtered_mean, filtered_covariance

    def log_likelihood(
        self,
        observation: Float[Array, " obs"],
        prior_mean: Float[Array, " state"],
        prior_covariance: Float[Array, "state state"],
        t: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Compute log p(y_t | y_{1:t-1}) using the observation and predicted mean."""
        emission_weights_t = self.emission_weights(t)
        emission_covariance_t = self.emission_covariance(t)

        predicted_obs_mean = emission_weights_t @ prior_mean
        innovation_covariance = (
            emission_weights_t @ prior_covariance @ emission_weights_t.T
            + emission_covariance_t
        )

        return jsp.stats.multivariate_normal.logpdf(
            observation,
            mean=predicted_obs_mean,
            cov=innovation_covariance
        )

    def __call__(
        self,
        emissions: Float[Array, "time obs"]
    ) -> PosteriorFilter:
        num_timesteps = emissions.shape[0]
        ts = jnp.arange(num_timesteps, dtype=jnp.float32)

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
        _, (means, covs, lls) = jax.lax.scan(step, initial_carry, (emissions, ts))

        return PosteriorFilter(
            marginal_log_likelihood=jnp.sum(lls),
            filtered_means=means,
            filtered_covariances=covs,
        )
