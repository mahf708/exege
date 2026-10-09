Adapters and errors
===================

Adapters
--------

The adapters shipped with exege. Each is found through the ``exege.adapters`` entry-point
group; see :doc:`Writing an adapter <../adapters>`.

.. autoclass:: exege.adapters.latent_archive.LatentArchive
   :members:

.. autofunction:: exege.adapters.latent_archive.write_archive

.. autoclass:: exege.adapters.bundle_dir.BundleDir
   :members:

.. autofunction:: exege.adapters.bundle_dir.write_bundle

.. autoclass:: exege.adapters.toy_dynamics.ToyDynamics
   :members:

Registry
--------

Adapters are found through the ``exege.adapters`` entry-point group; ``register`` adds
one for the current process, which tests and notebooks use.

.. autofunction:: exege.core.registry.available

.. autofunction:: exege.core.registry.register

.. autofunction:: exege.core.registry.create

Errors
------

Errors exege raises on purpose are ``ExegeError`` exceptions, and reach a terminal as one line.

.. autoexception:: exege.core.ExegeError

.. autoexception:: exege.core.RequestError

.. autoexception:: exege.core.AdapterError

.. autoexception:: exege.core.MissingExtraError
