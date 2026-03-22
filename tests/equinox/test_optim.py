import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from kalman_filter_jax.equinox.kalman_filter import KalmanFilter
from kalman_filter_jax.equinox.params import TrainableCovariance, TrainableWeights


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

    # initialise with a random wrong dynamics matrix to give Adam something to do
    dynamics_weights_init = jax.random.normal(key, (state_dim, state_dim))
    dynamics_covariance_init = jnp.eye(state_dim)

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

    assert isinstance(model.dynamics_weights, TrainableWeights)
    init_weights = model.dynamics_weights.matrix.copy()

    # Filter spec marks arrays in trainable classes as True, others as False.
    # Follows the diffrax pattern:
    # https://docs.kidger.site/diffrax/examples/kalman_filter/
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda m: (m.dynamics_weights, m.dynamics_covariance),
        filter_spec,
        replace=(
            jtu.tree_map(lambda _: True, model.dynamics_weights),
            jtu.tree_map(lambda _: True, model.dynamics_covariance),
        ),
    )

    # setup optimiser -- opt.init should only see the trainable part
    opt = optax.adam(5e-3)
    trainable, _static = eqx.partition(model, filter_spec)
    opt_state = opt.init(trainable)

    @eqx.filter_value_and_grad
    def loss_fn(trainable, static, y):
        m = eqx.combine(trainable, static)
        posterior = m(y)
        return -posterior.marginal_log_likelihood

    @eqx.filter_jit
    def make_step(m, state, y):
        t, s = eqx.partition(m, filter_spec)
        loss, grads = loss_fn(t, s, y)
        updates, state = opt.update(grads, state)
        t = eqx.apply_updates(t, updates)
        return loss, eqx.combine(t, s), state

    initial_nll, model, opt_state = make_step(
        model, opt_state, emissions
    )
    current_nll = initial_nll
    for _ in range(50):
        current_nll, model, opt_state = make_step(
            model, opt_state, emissions
        )

    # test denoising: loss should decrease significantly
    assert current_nll < initial_nll

    # constant emissions unchanged
    assert jnp.array_equal(
        model.emission_weights(jnp.array(0.0)),
        params_data.emission_weights,
    )

    # trainable weights moved
    assert not jnp.array_equal(model.dynamics_weights.matrix, init_weights)

    # covariance still PSD
    Q_final = model.dynamics_covariance(jnp.array(0.0))
    assert jnp.all(jnp.linalg.eigvalsh(Q_final) > 0)
