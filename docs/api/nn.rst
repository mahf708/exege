Sparse autoencoder API
======================

Needs the ``nn`` extra (torch).

Training
--------

.. autofunction:: exege.nn.train.fit_sae

.. autofunction:: exege.nn.train.pick_device

Modules
-------

Plain ``torch.nn.Module`` subclasses over tensors, with no knowledge of archives, grids or loops.

.. autoclass:: exege.nn.sae.SparseAutoencoder
   :members:

.. autoclass:: exege.nn.sae.BSplineActivation
   :members:

.. autofunction:: exege.nn.sae.topk_mask
