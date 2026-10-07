"""exege -- tools for understanding and evaluating scientific machine-learning models.

Domains, and the presentation downstream of them:

- ``exege.latents``  what emulators hold inside: the latent space, on a grid
- ``exege.nn``       torch modules trained on those latents: a sparse autoencoder
- ``exege.figures``  figures of what ``latents`` computes, with no web framework
- ``exege.app``      a local web app over all of the above; nothing imports it

Everything framework-specific lives in ``exege.adapters``. See ``AGENTS.md``.
"""

from __future__ import annotations

__version__ = "0.6.0"

__all__ = ["__version__"]
