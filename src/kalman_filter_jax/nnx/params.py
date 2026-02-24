from abc import abstractmethod

from flax import nnx
from gpjax.parameters import Real
from jaxtyping import Array, Float

from .gpjax_parameters_extras import PSDMatrix


class AbstractWeights(nnx.Module):
    @abstractmethod
    def __call__(self, t: Float[Array, ""]) -> Float[Array, "out in"]:
        ...

class AbstractCovariance(nnx.Module):
    @abstractmethod
    def __call__(self, t: Float[Array, ""]) -> Float[Array, "out out"]:
        ...

class ConstantWeights(AbstractWeights):
    def __init__(self, matrix: Float[Array, "out in"]):
        self.matrix = matrix
    def __call__(self, _t: Float[Array, ""]) -> Array:
        return self.matrix

class ConstantCovariance(AbstractCovariance):
    def __init__(self, matrix: Float[Array, "out out"]):
        self.matrix = matrix
    def __call__(self, _t: Float[Array, ""]) -> Array:
        return self.matrix


class TrainableCovariance(AbstractCovariance):
    matrix: PSDMatrix

    def __init__(self, matrix: Float[Array, "dim dim"], **kwargs):
        # kwargs (which includes 'prior') to Real -> Parameter
        self.matrix = PSDMatrix(matrix, **kwargs)

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "dim dim"]:
        # transform(inverse=False) handles the vector -> PSD mapping
        return self.matrix[...]


class TrainableWeights(AbstractWeights):
    matrix: Real

    def __init__(self, matrix: Float[Array, "out in"], **kwargs):
        self.matrix = Real(matrix, **kwargs)

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "out in"]:
        # transform(inverse=False) reshapes the flat vector back to (out, in)
        return self.matrix[...]
