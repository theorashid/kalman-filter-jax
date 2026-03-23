# Separates model specification from inference method.
# The user defines the SSM (initial distribution, dynamics, observations),
# then calls .infer(emissions, method=...) to choose the inner loop.

from collections.abc import Callable
from typing import Literal

import equinox as eqx
from jaxtyping import Array, Float

Method = Literal["kalman", "kalman_parallel"]

VALID_METHODS: set[Method] = {"kalman", "kalman_parallel"}


def _resolve_method(method: Method) -> Method:
    if method not in VALID_METHODS:
        msg = (
            f"Unknown method {method!r}. "
            f"Choose from {sorted(VALID_METHODS)}."
        )
        raise ValueError(msg)
    return method


class PosteriorFilter(eqx.Module):
    marginal_log_likelihood: Float[Array, ""]
    filtered_means: Float[Array, "time state"]
    filtered_covariances: Float[Array, "time state state"]


class LinearGaussianSSM(eqx.Module):
    # z_0 ~ N(initial_mean, initial_covariance)
    initial_mean: Float[Array, " state"]
    initial_covariance: Float[Array, "state state"]

    # z_{t+1} = F_t z_t + q_t,   q_t ~ N(0, Q_t)
    dynamics_weights: Callable[[Float[Array, ""]], Array]
    dynamics_covariance: Callable[[Float[Array, ""]], Array]

    # y_t = H_t z_t + r_t,       r_t ~ N(0, R_t)
    emission_weights: Callable[[Float[Array, ""]], Array]
    emission_covariance: Callable[[Float[Array, ""]], Array]

    def infer(
        self,
        emissions: Float[Array, "time obs"],
        method: Method = "kalman",
    ) -> PosteriorFilter:
        from .inference import filter_kalman  # noqa: PLC0415

        method = _resolve_method(method)

        if method == "kalman_parallel":
            return filter_kalman(self, emissions, parallel=True)
        return filter_kalman(self, emissions, parallel=False)
