from flax import nnx
from gpjax.parameters import Real
from jaxtyping import Array, Float

from .gpjax_parameters_extras import PSDMatrix


class TrainableCovariance(nnx.Module):
    matrix: PSDMatrix

    def __init__(self, matrix: Float[Array, "dim dim"], **kwargs):
        self.matrix = PSDMatrix(matrix, **kwargs)

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "dim dim"]:
        return self.matrix[...]


class TrainableWeights(nnx.Module):
    matrix: Real

    def __init__(self, matrix: Float[Array, "out in"], **kwargs):
        self.matrix = Real(matrix, **kwargs)

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "out in"]:
        return self.matrix[...]
