# Background

Three papers are behind what is here, and behind what comes next. Cite them if you use it.

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
  find. The TopK autoencoder in [`nn`](nn.md) is theirs in form; their auxiliary
  loss for dead features and their steering are not here yet; held-out
  [evaluation](evaluation.md) is.
- [Cheon (2026)](https://arxiv.org/abs/2605.17493), *Beyond Linear Superposition:
  Discovering Climate Features in AI Weather Models with KAN-SAE*. A sparse autoencoder
  whose ReLU is replaced by a learnable B-spline per feature. **Not implemented here
  yet**: the `bspline` activation in `nn` predates our reading of it and is a different
  thing ([see there](nn.md#activation-functions)).

exege was started as `xaig` in
[E3SM-Project/aigroup](https://github.com/E3SM-Project/aigroup).
