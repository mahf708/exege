"""nn -- torch modules trained on a model's latents: a sparse autoencoder.

- ``exege.nn.sae``    a sparse autoencoder / transcoder, and a learnable B-spline
                     activation: plain ``nn.Module``s over tensors
- ``exege.nn.train``  fitting one to a model's latents, and handing the result
                     to ``exege.latents`` as a ``Dictionary``

Needs the ``nn`` extra (torch). This package module stays importable without
it, so that ``exege --help`` can list the ``nn`` command on a base install.
"""

from __future__ import annotations

__all__: list[str] = []
