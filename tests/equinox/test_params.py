import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from kalman_filter_jax.equinox.params import (
    AbstractConstantMatrix,
    AbstractCovariance,
    AbstractWeights,
    ConstantCovariance,
    PrecomputedWeights,
    TrainableCovariance,
)


def get_trainable_filter_spec(model):
    """Mark arrays as True if they belong to a non-Constant module."""
    def _is_trainable_leaf(module):
        # if Constant class, the whole branch is False
        if isinstance(module, AbstractConstantMatrix):
            return jtu.tree_map(lambda _: False, module)
        # otherwise follow the standard eqx.is_array pattern for this branch
        return jtu.tree_map(eqx.is_array, module)

    return jtu.tree_map(
        _is_trainable_leaf,
        model,
        is_leaf=lambda x: isinstance(
            x,
            (AbstractWeights, AbstractCovariance, eqx.nn.MLP)
        )
    )


@pytest.mark.parametrize(
    ("cov_cls", "init_args", "is_constant"),
    [
        (ConstantCovariance, {"matrix": jnp.eye(3)}, True),
        (TrainableCovariance, {"dim": 3, "key": jax.random.key(2)}, False),
        # (TrainableNeuralCovariance, {"dim": 3, "mlp_depth": 2, ...}, False),
    ]
)
def test_covariance_properties_and_trainable(cov_cls, init_args, is_constant):
    cov_module = cov_cls(**init_args)
    dim = 3 # matches init_args

    # test covariance matrix properties
    cov_matrix = cov_module(jnp.array(0.0))
    assert cov_matrix.shape == (dim, dim), "Covariances not square"
    assert jnp.allclose(cov_matrix, cov_matrix.T, atol=1e-6), "Covariances asymmetric"
    assert jnp.all(jnp.linalg.eigvalsh(cov_matrix) > 0), "Covariances not PSD"

    # test trainable
    filter_spec = get_trainable_filter_spec(cov_module)
    dynamic, static = eqx.partition(cov_module, filter_spec)

    # convert to leaves to check if any arrays exist in the dynamic partition
    dynamic_leaves = jtu.tree_leaves(dynamic, is_leaf=eqx.is_array)

    if is_constant:
        # dynamic part should have no arrays
        assert len(dynamic_leaves) == 0
        assert dynamic.matrix is None
        assert static.matrix is not None
        assert isinstance(cov_module, AbstractConstantMatrix)
    else:
        # dynamic part should have the raw parameters
        assert len(dynamic_leaves) > 0
        assert dynamic.unconstrained_params is not None
        assert static.unconstrained_params is None
        assert static.bijection is not None
        assert not isinstance(cov_module, AbstractConstantMatrix)

@pytest.mark.parametrize(
    ("t_val", "expected_idx"),
    [
        (0.0, 0),
        (2.0, 2),
        (4.9, 4),
    ]
)
def test_precomputed_indexing_accuracy(t_val, expected_idx):
    data = jnp.arange(10, dtype=jnp.float32).reshape(5, 2, 1)

    cov_module = PrecomputedWeights(matrix=data)

    result = cov_module(jnp.array(t_val))

    # check the correct slice
    assert jnp.allclose(result, data[expected_idx])
    # check output is still a float
    assert jnp.issubdtype(result.dtype, jnp.floating)

