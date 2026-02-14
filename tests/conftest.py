import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from kalman_filter_jax.basic.kalman_filter import KalmanParams


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

    # Dynamics: pos_t = pos_{t-1} + vel_{t-1}, vel_t = vel_{t-1}
    F = jnp.array([[1.0, dt],
                   [0.0, 1.0]])
    # Observation: we only see position
    H = jnp.array([[1.0, 0.0]])

    # Noise: Set to near-zero for "exact" recovery testing
    Q = jnp.eye(2) * 1e-10
    R = jnp.eye(1) * 1e-10

    initial_mean = jnp.array([0.0, 1.0]) # Starts at 0, velocity 1
    initial_covariance = jnp.eye(2) * 1e-10

    params = KalmanParams(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_weights=F,
        dynamics_covariance=Q,
        emission_weights=H,
        emission_covariance=R
    )

    # Generate true states and observations
    true_states = [initial_mean]
    emissions = []

    curr_state = initial_mean
    for _t in range(timesteps):
        # Predict next state (deterministic)
        curr_state = F @ curr_state
        true_states.append(curr_state)
        # Generate observation (deterministic)
        emissions.append(H @ curr_state)

    return params, jnp.array(emissions), jnp.array(true_states[1:])
