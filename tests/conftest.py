import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from kalman_filter_jax.basic.kalman_filter import KalmanParams


def simulate_motion(
    params: KalmanParams,
    n_time: int,
    key: Array
) -> tuple[Float[Array, "time state"], Float[Array, "time obs"]]:
    F: Float[Array, "state state"] = params.dynamics_weights
    H: Float[Array, "obs state"] = params.emission_weights
    Q: Float[Array, "state state"] = params.dynamics_covariance
    R: Float[Array, "obs obs"] = params.emission_covariance

    state_dim = F.shape[0]
    obs_dim = H.shape[0]

    def step(state: Float[Array, " state"], key: Array):
        k_q, k_r = jax.random.split(key)

        q_noise = jax.random.multivariate_normal(k_q, jnp.zeros(state_dim), Q)
        next_state = F @ state + q_noise

        r_noise = jax.random.multivariate_normal(k_r, jnp.zeros(obs_dim), R)
        obs = H @ next_state + r_noise

        return next_state, (next_state, obs)

    keys = jax.random.split(key, n_time)
    _, (states, emissions) = jax.lax.scan(step, params.initial_mean, keys)

    return states, emissions


@pytest.fixture(scope="session")
def linear_motion_data() -> tuple[
    KalmanParams,
    Float[Array, "time obs"],
    Float[Array, "time state"]
]:
    """
    Simulates a deterministic constant-velocity 1D motion.
    State: [position, velocity]
    """
    timesteps = 10
    dt = 1.0

    F = jnp.array([[1.0, dt], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])
    Q = jnp.eye(2) * 1e-10 # near-zero noise for "exact" recovery testing
    R = jnp.eye(1) * 1e-10

    initial_mean = jnp.array([0.0, 1.0]) # position at 0, velocity 1
    initial_covariance = jnp.eye(2) * 1e-10

    params = KalmanParams(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_weights=F,
        dynamics_covariance=Q,
        emission_weights=H,
        emission_covariance=R
    )
    states, obs = simulate_motion(params, timesteps, jax.random.key(0))
    return params, obs, states

@pytest.fixture(scope="session")
def noisy_linear_motion_data() -> tuple[
    KalmanParams,
    Float[Array, "time obs"],
    Float[Array, "time state"]
]:
    """
    Simulates a stochastic constant-velocity 1D motion.
    State: [position, velocity]
    """
    timesteps = 500  # longer is better for learning noise stats
    dt = 1.0

    F = jnp.array([[1.0, dt], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])
    Q = jnp.eye(2) * 0.05  # some noise
    R = jnp.eye(1) * 0.1

    initial_mean = jnp.array([0.0, 1.0])
    initial_covariance = jnp.eye(2)

    params = KalmanParams(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_weights=F,
        dynamics_covariance=Q,
        emission_weights=H,
        emission_covariance=R
    )
    states, obs = simulate_motion(params, timesteps, jax.random.key(2))
    return params, obs, states

