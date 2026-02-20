from abc import abstractmethod

from flax import nnx
from jaxtyping import Array, Float


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
