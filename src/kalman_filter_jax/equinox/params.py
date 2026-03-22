import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to

# numpyro's biject_to is a ConstraintRegistry
#
#   biject_to(constraints.positive_definite)
#     -> ComposeTransform([LowerCholeskyTransform(), CholeskyTransform().inv])
#
# i.e. vector of length n(n+1)/2  ->  lower-triangular L (exp on diagonal)
#                                  ->  L @ L.T (PSD matrix)
#
# from numpyro source:
#   @biject_to.register(constraints.positive_definite)
#   @biject_to.register(constraints.positive_semidefinite)
#   def _transform_to_positive_definite(constraint):
#       return ComposeTransform([LowerCholeskyTransform(), CholeskyTransform().inv])
#
# Each class below stores a constraint and resolves the bijection via
# biject_to(self._constraint) at init/call time.


class TrainableWeights(eqx.Module):
    matrix: Float[Array, "out in"]

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "out in"]:
        return self.matrix


class TrainableCovariance(eqx.Module):
    _unconstrained: Float[Array, " dim_tril"]
    _constraint: constraints.Constraint = eqx.field(static=True)

    def __init__(
        self,
        matrix: Float[Array, "dim dim"],
        constraint: constraints.Constraint = constraints.positive_definite,
    ):
        self._constraint = constraint
        self._unconstrained = biject_to(constraint).inv(matrix)

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "dim dim"]:
        return biject_to(self._constraint)(self._unconstrained)


# Currently, we use weights and covariances as matrices, but for
# nonlinear / generalised SSMs (see dynamax EKF-MLP example:
# https://probml.github.io/dynamax/notebooks/nonlinear_gaussian_ssm/ekf_mlp.html)
# the callable could output computed quantities of the right shape.
#
# For example, rather than `self.dynamics_weights(t) @ state_mean`, we could
# have `dynamics_weights(t, state_mean)` and apply a general function over
# the latent state.


# Diagonal covariance (independent noise per dimension).
# Stores unconstrained parameters and applies softplus for positivity.
# Returns a full diagonal matrix via jnp.diag().
#
# With lineax, the innovation solve could use
# lx.DiagonalLinearOperator(diag) for O(n) instead of O(n^3).
class DiagonalCovariance(eqx.Module):
    _unconstrained_diag: Float[Array, " dim"]

    def __init__(self, diag: Float[Array, " dim"]):
        self._unconstrained_diag = jax.nn.softplus(diag)

    def __call__(self, _t: Float[Array, ""]) -> Float[Array, "dim dim"]:
        return jnp.diag(jax.nn.softplus(self._unconstrained_diag))


# State-dependent covariance via an MLP.
# Maps the latent state to the lower-triangular parameterisation,
# then applies the same biject_to constraint as TrainableCovariance.
#
# Note: this has a different call signature -- it takes (t, state)
# rather than just (t) -- so the KalmanFilter would need to be
# adapted to pass the state through to the covariance callable.
class NeuralCovariance(eqx.Module):
    mlp: eqx.nn.MLP
    _constraint: constraints.Constraint = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        mlp_width_size: int,
        mlp_depth: int,
        key: Array,
        constraint: constraints.Constraint = constraints.positive_definite,
    ):
        self._constraint = constraint
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
        return biject_to(self._constraint)(tril)
