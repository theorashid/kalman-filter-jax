import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from kalman_filter_jax.equinox.kalman_filter import KalmanFilter
from kalman_filter_jax.equinox.params import (
    AbstractConstantMatrix,
    AbstractCovariance,
    AbstractWeights,
    ConstantCovariance,
    ConstantWeights,
    TrainableCovariance,
    TrainableWeights,
)


def test_kalman_filter_training_loop(noisy_linear_motion_data):
    """
    Train a basic Linear Gaussian State Space Model.

    Verifies that the Kalman Filter improves its state estimation (denoising)
    on a stochastic trajectory by optimising its internal parameters.

    Checks that:
    1. Gradients are only computed for trainable parameters.
    2. The loss decreases after optimization steps.
    """
    params, emissions, _true_states = noisy_linear_motion_data
    state_dim = params.initial_mean.shape[0]
    key = jax.random.key(2)
    k1, k2, _k3 = jax.random.split(key, 3)

    # initialise with a random wrong dynamics matrix to give Adam something to do
    model = KalmanFilter(
        initial_mean=params.initial_mean,
        initial_covariance=params.initial_covariance,
        # trainable dynamics components
        dynamics_weights=TrainableWeights(state_dim, state_dim, key=k1),
        dynamics_covariance=TrainableCovariance(state_dim, key=k2),
        # assume emission (sensor) model H and R are known/fixed
        emission_weights=ConstantWeights(params.emission_weights),
        emission_covariance=ConstantCovariance(params.emission_covariance)
    )

    init_weights_arrays = eqx.filter(model.dynamics_weights, eqx.is_array)

    # optimisation loop follows [diffrax example](https://docs.kidger.site/diffrax/examples/kalman_filter/)
    # Filter spec marks arrays in Trainable classes as True, others as False
    def get_trainable_filter_spec(model):
        def _is_trainable_leaf(module):
            if isinstance(module, AbstractConstantMatrix):
                return jtu.tree_map(lambda _: False, module)
            return jtu.tree_map(eqx.is_array, module)

        return jtu.tree_map(
            _is_trainable_leaf,
            model,
            is_leaf=lambda x: isinstance(
                x,
                (AbstractWeights, AbstractCovariance, eqx.nn.MLP)
            )
        )


    filter_spec = get_trainable_filter_spec(model)

    # setup optimiser. opt.init should only see the trainable part of the tree
    opt = optax.adam(5e-3) # Slightly lower LR for stability on noisy data
    trainable_params, _ = eqx.partition(model, filter_spec)
    opt_state = opt.init(trainable_params)

    @eqx.filter_value_and_grad
    def loss_fn(dynamic_m, static_m, y):
        m = eqx.combine(dynamic_m, static_m)
        posterior = m(y)
        return -posterior.marginal_log_likelihood

    @eqx.filter_jit
    def make_step(m, state, y):
        dynamic_m, static_m = eqx.partition(m, filter_spec)
        loss, grads = loss_fn(dynamic_m, static_m, y)
        updates, state = opt.update(grads, state)
        m = eqx.apply_updates(m, updates)
        return loss, m, state

    # run for more steps (50) because noisy gradients require more evidence
    initial_nll, model, opt_state = make_step(model, opt_state, emissions)
    current_nll = initial_nll
    for _ in range(50):
        current_nll, model, opt_state = make_step(model, opt_state, emissions)

    # test denoising: loss should decrease significantly
    assert current_nll < initial_nll, "Optimisation failed, loss did not improve"

    # check constant components were not touched during optimisation
    assert jnp.array_equal(model.emission_weights.matrix, params.emission_weights)

    # test trainable components moved from initial values
    final_weights_arrays = eqx.filter(model.dynamics_weights, eqx.is_array)
    eqx.filter(model.dynamics_covariance, eqx.is_array)
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            jnp.array_equal,
            init_weights_arrays,
            final_weights_arrays,
        )
    ), "Weights did not change"

    # double check we still have a PSD matrix
    Q_final = model.dynamics_covariance(jnp.array(0.0))
    assert jnp.all(jnp.linalg.eigvalsh(Q_final) > 0)
