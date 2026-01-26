# kalman-filter-jax

Trying out different jax-based libraries by implementing a [Kalman filter](https://theorashid.github.io/notes/kalman-filtering-and-smoothing) to solve [linear Gaussian state space models](https://theorashid.github.io/notes/linear-gaussian-ssm).

## basic

This is a minimal implementation.
It is adapted from Gabriel Stechschulte's [filterjax](https://github.com/GStechschulte/filter-jax), which is readable and great.
I also took some inspiration from [dynamax](https://github.com/probml/dynamax), although I did not bother with the `constrain`/`unconstrain` logic as I find it cumbersome (too many `NamedTuple` and overloading their functions) and I think we could simplify this using Equinox/nnx.

The main test checks to see if the filtered means recover the underlying state of a noiseless trajectory.
I would like to add noisy test here.
It also shows how to use `jax.vmap` to batch the filters.

## remaining

To try:

- Equinox
- nnx
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
