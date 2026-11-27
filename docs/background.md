# Background

Four papers are behind what is here, and behind what comes next. Cite them if you use it.

- [Tempest, Beylich & Craig (2026)](https://arxiv.org/abs/2604.20467), *Mechanistic
  Interpretability Tool for AI Weather Models*
  ([doi:10.1007/978-3-032-29915-4_10](https://doi.org/10.1007/978-3-032-29915-4_10);
  [code](https://github.com/ktempestuous/latent_space_visualiser_weather_models)). Its
  workflow — a region, the channels that respond there, cosine similarity, a PCA fitted
  in the region and mapped everywhere — is what [`latents`](regions.md)
  reimplements as a library, and what [the app](app.md) puts widgets on. Where exege
  departs from it, and why, is in [Design](design.md#differences-from-the-original-visualiser).
- [MacMillan & Ouellette (2025)](https://arxiv.org/abs/2512.24440), *Towards mechanistic
  understanding in a data-driven weather model: internal activations reveal interpretable
  physical features*
  ([code](https://github.com/theodoremacmillan/graphcast-interpretability)). Sparse
  autoencoders on GraphCast's node embeddings, and interventions on the features they
  find. The TopK autoencoder in [`nn`](nn.md) is theirs in form; held-out
  [evaluation](evaluation.md) and [steering](steering.md), against a toy system so far,
  follow them; their auxiliary loss for dead features is not here yet.
- [Gu & Last (2026)](https://arxiv.org/abs/2611.25119), *Auditing Learned Dynamics in
  Weather Models*. GraphCast's sparse features as experimental coordinates: a feature
  locked to the mesh that matters for skill, and features associated with convection and
  low-level vorticity that steer tropical-cyclone intensity. Several pieces here answer
  questions it raises, written from the paper and not from its code:
  [associations against a rotation null](associations.md), a [rotation of a running
  model's inputs](rotation.md) that names what is turned, steering an activation's
  [excess above a reference](steering.md#modes-and-references) in an
  [open or closed loop](steering.md#open-and-closed-loops), [matched and comparison
  arms](steering.md#comparison-and-matched-arms), [many cases](steering.md#cases),
  [an amount chosen leave-one-out](steering.md#dose-curves-and-choosing-an-amount),
  [mediation](mediation.md), and the [free-running against teacher-forced](teacher-forcing.md)
  comparison.
- [Cheon (2026)](https://arxiv.org/abs/2605.17493), *Beyond Linear Superposition:
  Discovering Climate Features in AI Weather Models with KAN-SAE*. A sparse autoencoder
  whose ReLU is replaced by a learnable B-spline per feature. **Not implemented here
  yet**: the `bspline` activation in `nn` predates our reading of it and is a different
  thing ([see there](nn.md#activation-functions)).

exege was started as `xaig` in
[E3SM-Project/aigroup](https://github.com/E3SM-Project/aigroup).
