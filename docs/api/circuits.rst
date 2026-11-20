Circuits API
============

Every name below is importable from ``exege.circuits`` itself; the heading says which
module defines it. They are loaded on first use.

Graph
-----

Defined in ``exege.circuits.graph``.

.. autoclass:: exege.circuits.Graph
   :members:

.. autofunction:: exege.circuits.residual_graph

Contract
--------

Defined in ``exege.circuits.contract``.

.. autoclass:: exege.circuits.Patchable
   :members:

.. autoclass:: exege.circuits.EdgeGradients
   :members:

.. autoclass:: exege.circuits.EdgePatch

.. autoclass:: exege.circuits.ComponentPatch

.. autofunction:: exege.circuits.open_patchable

Metrics
-------

Defined in ``exege.circuits.metrics``.

.. autoclass:: exege.circuits.LinearMetric

.. autofunction:: exege.circuits.field_mean

.. autofunction:: exege.circuits.recovery

Finding circuits
----------------

Defined in ``exege.circuits.discover``.

.. autofunction:: exege.circuits.attribute

.. autoclass:: exege.circuits.Attribution
   :members:

.. autofunction:: exege.circuits.prune

.. autoclass:: exege.circuits.Circuit
   :members:

.. autofunction:: exege.circuits.faithfulness_curve

.. autoclass:: exege.circuits.FaithfulnessCurve
   :members:

.. autofunction:: exege.circuits.patched

.. autofunction:: exege.circuits.endpoints

.. autofunction:: exege.circuits.top_k

.. autofunction:: exege.circuits.jaccard

.. autofunction:: exege.circuits.edge_frequency

.. autofunction:: exege.circuits.core
