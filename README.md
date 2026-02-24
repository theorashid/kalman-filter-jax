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

## Equinox

The parameters of the dynamics and observation model are now `eqx.Module` with `__call__` methods (and _only_ this for simplicity), meaning they can be time-varying functions.

This means we can also do optimisation on the state space model parameters following the Equinox [pattern](https://docs.kidger.site/diffrax/examples/kalman_filter/) of partitioning the model on what to train and what not to.
I created different `AbstractCovariance` and `AbstractWeights` modules (`equinox/params.py`), some designed to be static and some to be trainable by initialising in an unconstrained space.
The unconstrained covariance matrix is mapped to positive semidefinite using numpyro bijectors.

An example optimisation loop is shown in `tests/equinox/test_optim.py`.

I tried using [FlowJax](https://danielward27.github.io/flowjax/api/bijections.html) for the bijections (and maybe even priors on covariance matrices for a fully Bayesian approach) as it keeps everything in Equinox, but it does not have a the bijections we need here. I would like to use numpyro anyway.

Possible extensions:

- Fully Bayesian approach? Maybe this slots into a numpyro model as is, and we could set a prior over the `TrainableCovariance` unconstrained vector with `numpyro.contrib.module.random_eqx_module()`. However, this sets a prior on the unconstrained space, which is not ideal. I'm also not entirely sure how our marginal log likelihood (include with `numpyro.factor`) and the log prior terms play with each other here to construct our log posterior. It's probably more sensible to look at [what GPJax does](https://docs.jaxgaussianprocesses.com/_examples/numpyro_integration/).
- Let the `__call__` take in the latent state to have more complicated neural networks that depend on the previous state. This is sketched out at the bottom of `equinox/params.py` with `TrainableNeuralCovariance`
- Generalise the Kalman filter for nonlinear dynamics

## nnx (via GPJax)

The parameters of the dynamics and observation model are now `nnx.Module` with `__call__` methods.
The implementation of the Kalman filter itself was more difficult for the batch case.
Unlike Equinox, I could not just use `jax.vmap` over the batch dimension.
I needed `...` in all the type hints, and ended up using `.mT` and `einsum` everywhere.
Note, I'm now using `nnx.vmap`, `nnx.scan` and `nnx.jit`, just in case we have any stateful parameters anywhere.

I closely followed the GPJax `Parameter` pattern so I could eventually piggback on the [numpyro integration](https://docs.jaxgaussianprocesses.com/_examples/numpyro_integration/).
Like Equinox, we can [partition](https://docs.jaxgaussianprocesses.com/_examples/backend/?h=backend#parameter-transforms) the model based on what we do and do not want to optimise.
Each `Parameter` has a bijection that we can `transform` [to and from the unconstrained space](https://theorashid.github.io/notes/jax-and-state#nnx) using `nnx.split` to partition.
Inference (optimisation, MAP, NUTS sampling etc) is done in the unconstrained space.
The loss function is defined in the constrained space.

I extended GPJax to add a `PSDMatrix(Parameter[T])` (with GPJax-style tests in `tests/nnx/test_gpjax_parameters_extras.py`).
Unlike Equinox, the `TrainableCovariance` matrix is initialised in the constrained space, rather than the flat unconstrained vector.

`tests/nnx/test_optim.py` shows how to perform optimisation using the [GPJax fit pattern](https://github.com/thomaspinder/GPJax/blob/b620398bd4d45b317f2199410b570924d7fa7a3a/gpjax/fit.py#L132-L178).

Having followed the GPJax pattern throughout, I could now use the `gpjax.numpyro_extras.register_parameters` to replace values in the `nnx.Module` with sampled ones from numpyro distributions.
`tests/nnx/test_numpyro_model.py` shows how to write a model with priors on the dynamics/emissions parameters of the Kalman filter, for fully-Bayesian inference.

## remaining

To try:

- get Equinox working with numpyro
- extending to do forecasting and missing values
- PyTensor (maybe? for the linalg graph speedups)

## resources

I like Kevin Murphy’s notation from _Probabilistic Machine Learning: Advanced Topics_ (2023), but Durbin and Koopman’s _Time Series Analysis by State Space Methods_ (2012) is the key resource.
I have some notes [on state space models](https://theorashid.github.io/notes/#ssm).

Related projects:

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
