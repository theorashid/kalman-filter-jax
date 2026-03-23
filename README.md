# kalman-filter-jax

Trying out different jax-based libraries by implementing a [Kalman filter](https://theorashid.github.io/notes/kalman-filtering-and-smoothing) to solve [linear Gaussian state space models](https://theorashid.github.io/notes/linear-gaussian-ssm).

While I was playing around with this, [cuthbert](https://github.com/state-space-models/cuthbert) was released, which is much more feature-complete (numerically stable, other filters, etc).
The Kalman filter in this repo is very simple and follows the textbook maths rather than using numerical stability tricks.
This repo serves more as an exercise in how to interface other jax packages (numpyro, nnx, Equinox) with a custom inference method like a Kalman filter.

## basic

This is a minimal implementation.
It is adapted from Gabriel Stechschulte's [filterjax](https://github.com/GStechschulte/filter-jax), which is readable and great.
I also took some inspiration from [dynamax](https://github.com/probml/dynamax), although I did not bother with the `constrain`/`unconstrain` logic as I find it cumbersome (too many `NamedTuple` and overloading their functions) and I think we could simplify this using Equinox/nnx.

The main test checks to see if the filtered means recover the underlying state of a noiseless trajectory.
I would like to add noisy test here.
It also shows how to use `jax.vmap` to batch the filters.

To keep things easy, I assumed all the parameters of the dynamics and observation model are time invariant.
In dynamax, these matrices can be time-varying.
The `ParamsLGSSM*` elements can be overloaded with shapes `Float[Array, "state_dim state_dim"] | Float[Array, "ntime state_dim state_dim"]` for the time-invariant and varying cases.
In the Kalman filter step, the [helper](https://github.com/probml/dynamax/blob/main/dynamax/linear_gaussian_ssm/inference.py#L151-L170) `_get_one_param` plucks the variable depending on whether the input is a time-varying matrix, static matrix or a function.

### basic (cuthbert)

A thin wrapper around [cuthbert](https://github.com/state-space-models/cuthbert)'s numerically stable Kalman filter, reusing the same `KalmanParams` and `PosteriorFilter` types from the basic implementation.
Internally it converts covariances to Cholesky factors and sets up cuthbert's callback-based API (`get_init_params`, `get_dynamics_params`, `get_observation_params`).
It supports parallel filtering via the `parallel` flag (using cuthbert's associative scan).
Batching still works via `jax.vmap` as in the basic case.

## Equinox

`KalmanFilter` is an `eqx.Module` whose dynamics and observation components are callables `(t) -> Array`.
For constant parameters, a lambda (`lambda _t: matrix`).
For trainable parameters, we use `eqx.Module` subclasses (`TrainableWeights`, `TrainableCovariance`) that store parameters internally and apply bijections in their `__call__`.

The Kalman gain solve uses [lineax](https://docs.kidger.site/lineax/) with `MatrixLinearOperator` and a `positive_semidefinite_tag` for Cholesky-based solves.
This is extensible: for structured covariance (e.g. diagonal noise), swapping to `lx.DiagonalLinearOperator` gives O(n) solves without materialising the full matrix.

### Optimisation

Optimisation follows the Equinox [pattern](https://docs.kidger.site/diffrax/examples/kalman_filter/) of partitioning the model into trainable and static parts with `eqx.partition` and `eqx.tree_at`.
`TrainableCovariance` initialises from a constrained PSD matrix, stores an unconstrained vector internally, and maps it back to PSD via numpyro bijectors (`SoftplusLowerCholeskyTransform` + `CholeskyTransform.inv`) in `__call__`.
The optimiser works directly on the unconstrained parameters.

I considered using [paramax](https://github.com/danielward27/paramax) (as [GPJax is doing](https://github.com/thomaspinder/GPJax/pull/614)) where `paramax.unwrap(model)` resolves all constrained parameters tree-wide, eliminating the need for partition.
But the explicit `eqx.tree_at` filter spec is more transparent about what's trainable here.

`tests/equinox/test_optim.py` shows the optimisation loop.

### Fully Bayesian (numpyro)

For numpyro, we write standard `numpyro.sample` statements and pass the sampled arrays directly to the `KalmanFilter` as lambdas (`lambda _t: F`).
Numpyro distributions handle constraints themselves (e.g. `InverseWishart` produces PSD matrices), so no bijections are needed on this path.

`tests/equinox/test_numpyro_model.py` shows how to write a model with priors on the dynamics parameters.

### Extensions

Sketched out in `equinox/params.py`:

- `DiagonalCovariance`: diagonal noise per dimension, with a lineax `DiagonalLinearOperator` opportunity for O(n) solves.
- `NeuralCovariance`: state-dependent covariance via an MLP. Has a `(t, state)` signature, so the KalmanFilter would need adapting to pass the latent state through.
- Generalise for nonlinear dynamics: the callable design already supports this `dynamics_weights(t, state)`. Could apply a general function over the latent state rather than a matrix multiply.

## nnx (via GPJax)

`KalmanFilter` is an `nnx.Module` whose dynamics and observation components are callables `(t) -> Array`, following [cuthbert](https://github.com/state-space-models/cuthbert)'s callback-based design.
For constant parameters, a lambda (`lambda _t: matrix`).
For trainable parameters, we use `nnx.Module` subclasses (`TrainableWeights`, `TrainableCovariance`) that wrap GPJax `Parameter` types with bijections for constrained optimisation.

The implementation of the Kalman filter itself was more difficult for the batch case.
Unlike Equinox, I could not just use `jax.vmap` over the batch dimension.
I needed `...` in all the type hints, and ended up using `.mT` and `einsum` everywhere.
Note, I'm now using `nnx.vmap`, `nnx.scan` and `nnx.jit`, just in case we have any stateful parameters anywhere.

### Optimisation

I followed the GPJax `Parameter` pattern for optimisation.
We can [partition](https://docs.jaxgaussianprocesses.com/_examples/backend/?h=backend#parameter-transforms) the model based on what we do and do not want to optimise.
Each `Parameter` has a bijection that we can `transform` [to and from the unconstrained space](https://theorashid.github.io/notes/jax-and-state#nnx) using `nnx.split` to partition.
Inference (optimisation, MAP) is done in the unconstrained space.
The loss function is defined in the constrained space.

I extended GPJax to add a `PSDMatrix(Parameter[T])` (with GPJax-style tests in `tests/nnx/test_gpjax_parameters_extras.py`).
Unlike Equinox, the `TrainableCovariance` matrix is initialised in the constrained space, rather than the flat unconstrained vector.

`tests/nnx/test_optim.py` shows how to perform optimisation using the [GPJax fit pattern](https://github.com/thomaspinder/GPJax/blob/b620398bd4d45b317f2199410b570924d7fa7a3a/gpjax/fit.py#L132-L178).

### Fully Bayesian (numpyro)

I originally used `gpjax.numpyro_extras.register_parameters` to replace values in the `nnx.Module` with sampled ones from numpyro distributions.
`register_parameters` walks the `nnx.Module` tree, finds all `Parameter` leaves with priors, and calls `numpyro.sample` for each.
But numpyro already handles constraints through its distributions (e.g. `InverseWishart` produces PSD matrices, `Normal` produces unconstrained reals), so the `Parameter` bijection machinery is redundant in this path.
We don't need `register_parameters` at all.

Instead, we write standard `numpyro.sample` statements and pass the sampled arrays directly to the `KalmanFilter` as lambdas.
This is cleaner, more idiomatic numpyro, and decouples the sampling path from the GPJax dependency.

`tests/nnx/test_numpyro_model.py` shows how to write a model with priors on the dynamics parameters of the Kalman filter, for fully-Bayesian inference.

## Equinox vs nnx

I prefer the Equinox version.

- The linear algebra reads like maths: `@` everywhere vs `jnp.einsum("...ij,...j->...i", ...)` in nnx, because nnx bakes the batch dimension into the filter itself.
- Batching is just `jax.vmap`. Equinox modules are regular pytrees. nnx needs `nnx.vmap` with explicit axis config and `...` dims threaded through the whole filter. Same story for `jax.lax.scan` vs `nnx.scan`.
- Bijections use numpyro's constraint registry: `biject_to(constraints.positive_definite)`. nnx needs a custom `PSDMatrix(Parameter)` subclass and a mutation to GPJax's global `DEFAULT_BIJECTION` dict.
- The optimisation loop is simpler: `eqx.partition`/`combine` with the bijection hidden inside `__call__`. nnx needs `nnx.split`/`merge` + explicit `transform(params, bijection, inverse=True/False)` on every loss evaluation.
- lineax gives structured solves (`MatrixLinearOperator` with PSD tag, swappable to `DiagonalLinearOperator` for O(n)). nnx uses `scipy.linalg.solve` with a full matrix.

## Linear Gaussian State Space Model

`LinearGaussianSSM` separates model specification from inference (see [cuthbert#218](https://github.com/state-space-models/cuthbert/issues/218) and [dynestyx](https://github.com/BasisResearch/dynestyx)).
Define the model, then call `.infer(emissions, method=...)` to choose the inner loop.
Inference is done by [cuthbert](https://github.com/state-space-models/cuthbert): `"kalman"` (sequential) or `"kalman_parallel"` (associative scan).

All parameter fields are callables `(t) -> Array`, matching cuthbert's callback contract.
For static parameters, wrap in a lambda.
For trainable parameters, use `TrainableWeights` or `TrainableCovariance`.

The inner loop (filtering) is handled by cuthbert.
The outer loop (parameter estimation) is the user's choice: MLE via optax with `eqx.partition`/`combine` (`tests/lgssm/test_optim.py`), or fully Bayesian via numpyro with `numpyro.sample` + `numpyro.factor` on the marginal log-likelihood (`tests/lgssm/test_numpyro_model.py`).
The model doesn't care which outer loop you use -- it just returns a posterior given emissions.

## remaining

To try:

- extending to do forecasting and missing values
- PyTensor (maybe? for the linalg graph speedups)

## resources

I like Kevin Murphy’s notation from _Probabilistic Machine Learning: Advanced Topics_ (2023), but Durbin and Koopman’s _Time Series Analysis by State Space Methods_ (2012) is the key resource.
I have some notes [on state space models](https://theorashid.github.io/notes/#ssm).

Related projects:

- [cuthbert](https://github.com/state-space-models/cuthbert)
- [dynestyx](https://github.com/BasisResearch/dynestyx)
- [filterjax](https://github.com/GStechschulte/filter-jax)
- [dynamax](https://github.com/probml/dynamax)
- [PyMC Statespace](https://github.com/pymc-devs/pymc-extras/tree/main/pymc_extras/statespace)
- [filterpy](https://github.com/rlabbe/filterpy)
- [GPJax](https://github.com/thomaspinder/GPJax)

## dev

```sh
uv run ruff check --fix
uv run ty check
uv run pytest
```
