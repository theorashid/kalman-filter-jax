import jax.numpy as jnp
import numpyro.distributions.transforms as npt
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    T,
    _check_is_square,
    _safe_assert,
)
from jax.experimental import checkify
from jaxtyping import ArrayLike

# forward: vector -> L -> PSD matrix (LL^T)
# inverse: PSD matrix -> L (Cholesky) -> vector
PSD_BIJECTOR = npt.ComposeTransform([
    npt.SoftplusLowerCholeskyTransform(),
    npt.CholeskyTransform().inv,
])

DEFAULT_BIJECTION["psd"] = PSD_BIJECTOR

class PSDMatrix(Parameter[T]):
    def __init__(self, value: T, tag: str = "psd", **kwargs):
        super().__init__(value=value, tag=tag, **kwargs)

        if (
            not isinstance(value, jnp.ndarray)
            or getattr(value, "aval", None) is not None
        ):
            _safe_assert(_check_is_square, self[...])
            _safe_assert(_check_is_symmetric, self[...])
            _safe_assert(_check_is_positive_definite, self[...])


@checkify.checkify
def _check_is_symmetric(value: ArrayLike) -> None:
    checkify.check(
        jnp.allclose(value, value.T, atol=1e-6),
        "value needs to be symmetric, got {value}",
        value=value,
    )

@checkify.checkify
def _check_is_positive_definite(value: ArrayLike) -> None:
    eigenvals = jnp.linalg.eigvalsh(value)
    checkify.check(
        jnp.all(eigenvals > 0),
        "value needs to be positive-definite, got eigenvalues {eigenvals}",
        eigenvals=eigenvals,
    )
