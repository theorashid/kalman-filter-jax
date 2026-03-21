"""Following [gpjax.fit](https://github.com/thomaspinder/GPJax/blob/b620398bd4d45b317f2199410b570924d7fa7a3a/gpjax/fit.py#L132-L178)"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from gpjax.parameters import Parameter, transform

from kalman_filter_jax.nnx.gpjax_parameters_extras import DEFAULT_BIJECTION
from kalman_filter_jax.nnx.kalman_filter import KalmanFilter
from kalman_filter_jax.nnx.params import TrainableCovariance, TrainableWeights


def test_kalman_filter_training_loop(noisy_linear_motion_data):
    """
    Train a basic Linear Gaussian State Space Model.

    Verifies that the Kalman Filter improves its state estimation (denoising)
    on a stochastic trajectory by optimising its internal parameters.

    Checks that:
    1. Gradients are only computed for trainable parameters.
    2. The loss decreases after optimization steps.
    """
    params_data, emissions, _true_states = noisy_linear_motion_data
    state_dim = params_data.initial_mean.shape[0]
    key = jax.random.key(2)

    # make random dynamics_weights init
    dynamics_weights_init = jax.random.normal(key, (state_dim, state_dim))

    # make PSD dynamics_covariance init
    dynamics_covariance_init = jnp.eye(state_dim)

    # initialise with a random wrong dynamics matrix to give Adam something to do
    model = KalmanFilter(
        initial_mean=params_data.initial_mean,
        initial_covariance=params_data.initial_covariance,
        # trainable dynamics components
        dynamics_weights=TrainableWeights(dynamics_weights_init),
        dynamics_covariance=TrainableCovariance(dynamics_covariance_init),
        # assume emission (sensor) model H and R are known/fixed
        emission_weights=lambda _t: params_data.emission_weights,
        emission_covariance=lambda _t: params_data.emission_covariance,
    )

    # Model state filtering
    graphdef, params, *static_state = nnx.split(model, Parameter, ...)

    # Parameters bijection to unconstrained space
    initial_params = transform(
        params=params,
        params_bijection=DEFAULT_BIJECTION,
        inverse=True
    )

    # setup optimiser. opt.init should only see the trainable part of the tree
    opt = optax.adam(5e-3)
    opt_state = opt.init(initial_params)

    def loss_fn(params, y):
        # convert back to constrained space to calculate likelihood
        params = transform(
            params=params,
            params_bijection=DEFAULT_BIJECTION,
            inverse=False,
        )
        model = nnx.merge(graphdef, params, *static_state)
        posterior = model(y)
        return -posterior.marginal_log_likelihood


    @jax.jit
    def step(carry, _):
        p_unconstrained, o_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(p_unconstrained, emissions)
        updates, next_o_state = opt.update(grads, o_state, p_unconstrained)
        next_p_unconstrained = optax.apply_updates(p_unconstrained, updates)
        return (next_p_unconstrained, next_o_state), loss

    num_iters = 50
    (final_params_unconstrained, _), history = jax.lax.scan(
        step,
        (initial_params, opt_state),
        xs=None,
        length=num_iters
    )

    # back to constrained space
    final_params_constrained = transform(
        params=final_params_unconstrained,
        params_bijection=DEFAULT_BIJECTION,
        inverse=False
    )

    # reconstruct model
    final_model = nnx.merge(graphdef, final_params_constrained, *static_state)

    # tests:
    # test denoising: loss should decrease significantly
    assert history[-1] < history[0], "Optimisation failed, loss did not improve"

    # check constant components were not touched during optimisation
    assert jnp.array_equal(
        final_model.emission_weights(jnp.array(0.0)),
        params_data.emission_weights
    )

    # test trainable components moved from initial values
    changed = jax.tree_util.tree_map(
        lambda initial, final: jnp.any(jnp.not_equal(initial, final)),
        initial_params,
        final_params_unconstrained
    )
    assert jax.tree_util.tree_all(changed), "Some trainable parameters did not move."

    # double check we still have a PSD matrix
    Q_final = final_model.dynamics_covariance(jnp.array(0.0))
    assert jnp.all(jnp.linalg.eigvalsh(Q_final) > 0)
