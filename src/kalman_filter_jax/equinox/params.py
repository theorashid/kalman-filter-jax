import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions.transforms as npt
from jaxtyping import Array, Float


class AbstractWeights(eqx.Module):
    @abc.abstractmethod
    def __call__(self, t: Float[Array, ""]) -> Float[Array, "out in"]:
        ...

class AbstractCovariance(eqx.Module):
    @abc.abstractmethod
    def __call__(self, t: Float[Array, ""]) -> Float[Array, "out out"]:
        ...


class AbstractConstantMatrix(eqx.Module):
    matrix: eqx.AbstractVar[Array]


class ConstantWeights(AbstractWeights, AbstractConstantMatrix):
    matrix: Float[Array, "out in"]
    def __call__(self, _t: Float[Array, ""]) -> Array:
        return self.matrix

class ConstantCovariance(AbstractCovariance, AbstractConstantMatrix):
    matrix: Float[Array, "out out"]
    def __call__(self, _t: Float[Array, ""]) -> Array:
        return self.matrix


class PrecomputedWeights(AbstractWeights, AbstractConstantMatrix):
    matrix: Float[Array, "time out in"]
    def __call__(self, t: Float[Array, ""]) -> Array:
        # Cast t to int32 for JAX indexing within scan
        return self.matrix[t.astype(jnp.int32)]


class PrecomputedCovariance(AbstractCovariance, AbstractConstantMatrix):
    matrix: Float[Array, "time out out"]
    def __call__(self, t: Float[Array, ""]) -> Array:
        return self.matrix[t.astype(jnp.int32)]


class TrainableWeights(AbstractWeights):
    matrix: Float[Array, "out in"]

    def __init__(self, out_dim: int, in_dim: int, key: Array):
        # no need for bijections + constraints
        self.matrix = jax.random.normal(key, shape=(out_dim, in_dim)) * 0.1

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "out in"]:
        return self.matrix


class TrainableCovariance(AbstractCovariance):
    unconstrained_params: Float[Array, " dim_tril"] # vector has length n(n+1)/2
    bijection: npt.Transform = eqx.field(static=True)

    def __init__(self, dim: int, key: Array):
        self.bijection = npt.ComposeTransform([
            # fill a vector of length n(n+1)/2 to an n x n lower triangular matrix
            # and diagonal positivity
            npt.SoftplusLowerCholeskyTransform(),
            # L @ L.T to PSD covariance matrix
            npt.CholeskyTransform().inv,
        ])

        vector_len = dim * (dim + 1) // 2
        self.unconstrained_params = jax.random.normal(key, shape=(vector_len,)) * 0.1

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "dim dim"]:
        return self.bijection(self.unconstrained_params)


# Currently, we uses the weights and covariances as matrices, but if we were to update
# the logic for [nonlinear/generalised SSMs](https://probml.github.io/dynamax/notebooks/nonlinear_gaussian_ssm/ekf_mlp.html)
# we could have the Module output computed quantities of the right shape.
#
# As one example, rather than `self.dynamics_weights(t) @ state_mean`, we could have the
# `dynamic_weights(t, state_mean)` and apply some general function over the
# latent state.

# Firstly, a diagonal covariance matrix (think independent noise for each site).
# This could be done with a `jnp.diag()` and output a matrix, but this version will
# require elementwise multiplication in the underlying Kalman filter implementation.

class ConstantDiagonalCovariance(eqx.Module):
    diag: Float[Array, " dim"]

    def __init__(self, dim: int, key: Array):
        self.diag = jax.random.normal(key, shape=(dim,)) * 0.1

    def __call__(
        self,
        _t: Float[Array, ""],
    ) -> Float[Array, " dim"]:
        return jax.nn.softplus(self.diag)

# Secondly, we could have a neural network that applies on the latent state space. This
# could get really complicated with time-varying networks, but the one below keeps it
# very simple.
# This one outputs a covariance so can be used in the standard Kalman filter.

class TrainableNeuralCovariance(eqx.Module):
    mlp: eqx.nn.MLP
    bijection: npt.Transform = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        mlp_width_size: int,
        mlp_depth: int,
        key: Array,
    ):
        self.bijection = npt.ComposeTransform([
            npt.SoftplusLowerCholeskyTransform(),
            npt.CholeskyTransform().inv,
        ])

        dim_tril = dim * (dim + 1) // 2

        self.mlp = eqx.nn.MLP(
            in_size=dim,
            out_size=dim_tril,
            width_size=mlp_width_size,
            depth=mlp_depth,
            key=key,
        )


    def __call__(
        self,
        _t: Float[Array, ""],
        state: Float[Array, " dim"],
    ) -> Float[Array, "dim dim"]:
        tril = self.mlp(state)
        return self.bijection(tril)
