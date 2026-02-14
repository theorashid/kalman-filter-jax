# kalman-filter-jax

Trying out different jax-based libraries by implementing a [Kalman filter](https://theorashid.github.io/notes/kalman-filtering-and-smoothing) to solve [linear Gaussian state space models](https://theorashid.github.io/notes/linear-gaussian-ssm).

While I was playing around with this, [cuthbert](https://github.com/state-space-models/cuthbert) was released, which is much more feature-complete (numerically stable, other filters, etc).

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
This also makes it easier for optimisation.

Possible extensions:

- Let the `__call__` take in `state_mean` to have more complicated neural networks that depend on the previous state (although could encode this in the weights)
- The `__call__` emit a matrix and follow the standard linear Kalman filter, but this could be generalised for extended Kalman filter

## remaining

To try:

- nnx
- confirming these work with numpyro
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
```
