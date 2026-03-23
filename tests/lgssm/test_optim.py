# Outer loop: MLE via optax.
#
# dynamics_weights uses TrainableWeights (unconstrained eqx.Module).
# dynamics_covariance uses TrainableCovariance (PSD bijection).
# Both are callables: (t) -> Array.

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from kalman_filter_jax.equinox.params import TrainableCovariance, TrainableWeights
from kalman_filter_jax.lgssm.model import LinearGaussianSSM


def test_lgssm_training_loop(noisy_linear_motion_data):
    params_data, emissions, _true_states = noisy_linear_motion_data
    state_dim = params_data.initial_mean.shape[0]
    key = jax.random.key(2)

    dynamics_weights_init = jax.random.normal(key, (state_dim, state_dim))

    model = LinearGaussianSSM(
        initial_mean=params_data.initial_mean,
        initial_covariance=params_data.initial_covariance,
        dynamics_weights=TrainableWeights(dynamics_weights_init),
        dynamics_covariance=TrainableCovariance(jnp.eye(state_dim)),
        emission_weights=lambda _t: params_data.emission_weights,
        emission_covariance=lambda _t: params_data.emission_covariance,
    )

    assert isinstance(model.dynamics_weights, TrainableWeights)
    init_weights = model.dynamics_weights.matrix.copy()

    # mark trainable components
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda m: (m.dynamics_weights, m.dynamics_covariance),
        filter_spec,
        replace=(
            jtu.tree_map(lambda _: True, model.dynamics_weights),
            jtu.tree_map(lambda _: True, model.dynamics_covariance),
        ),
    )

    opt = optax.adam(5e-3)
    trainable, _static = eqx.partition(model, filter_spec)
    opt_state = opt.init(trainable)

    @eqx.filter_value_and_grad
    def loss_fn(trainable, static, y):
        m = eqx.combine(trainable, static)
        return -m.infer(y).marginal_log_likelihood

    @eqx.filter_jit
    def make_step(m, state, y):
        t, s = eqx.partition(m, filter_spec)
        loss, grads = loss_fn(t, s, y)
        updates, state = opt.update(grads, state)
        t = eqx.apply_updates(t, updates)
        return loss, eqx.combine(t, s), state

    initial_nll, model, opt_state = make_step(model, opt_state, emissions)
    current_nll = initial_nll
    for _ in range(50):
        current_nll, model, opt_state = make_step(model, opt_state, emissions)

    # loss decreased
    assert current_nll < initial_nll

    # fixed emissions unchanged
    assert jnp.array_equal(
        model.emission_weights(jnp.array(0.0)),
        params_data.emission_weights,
    )

    # trainable weights moved
    assert isinstance(model.dynamics_weights, TrainableWeights)
    assert not jnp.array_equal(model.dynamics_weights.matrix, init_weights)

    # covariance still PSD
    Q_final = model.dynamics_covariance(jnp.array(0.0))
    assert jnp.all(jnp.linalg.eigvalsh(Q_final) > 0)
