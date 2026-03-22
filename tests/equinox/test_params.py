import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from kalman_filter_jax.equinox.params import TrainableCovariance


def test_trainable_covariance_properties():
    dim = 3
    cov = TrainableCovariance(jnp.eye(dim))

    cov_matrix = cov(jnp.array(0.0))
    assert cov_matrix.shape == (dim, dim)
    assert jnp.allclose(cov_matrix, cov_matrix.T, atol=1e-6)
    assert jnp.all(jnp.linalg.eigvalsh(cov_matrix) > 0)

    # roundtrip: forward(inverse(I)) ~ I
    assert jnp.allclose(cov_matrix, jnp.eye(dim), atol=1e-4)

    # only the unconstrained vector is a trainable leaf
    dynamic_leaves = jtu.tree_leaves(
        eqx.partition(cov, eqx.is_array)[0]
    )
    assert len(dynamic_leaves) == 1
