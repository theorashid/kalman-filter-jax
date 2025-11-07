# kalman-filter-jax

Trying out different jax-based libraries by implementing a [Kalman filter](https://theorashid.github.io/notes/kalman-filtering-and-smoothing) to solve [linear Gaussian state space models](https://theorashid.github.io/notes/linear-gaussian-ssm).

This is a minimal implementation.
It is largely adapted from Gabriel Stechschulte's [filterjax](https://github.com/GStechschulte/filter-jax), which is readable and great.

I like Kevin Murphy’s notation from _Probabilistic Machine Learning: Advanced Topics_ (2023), but Durbin and Koopman’s _Time Series Analysis by State Space Methods_ (2012) is the key resource.
I have some notes [on state space models](https://theorashid.github.io/notes/#ssm).

To try:

- jax
- Equinox
- nnx
- PyTensor

Related projects:

- [filterjax](https://github.com/GStechschulte/filter-jax)
- [dynamax](https://github.com/probml/dynamax)
- [PyMC Statespace](https://github.com/pymc-devs/pymc-extras/tree/main/pymc_extras/statespace)
- [filterpy](https://github.com/rlabbe/filterpy)
- [GPJax](https://github.com/thomaspinder/GPJax)
