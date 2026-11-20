Laws API
========

Every name below is importable from ``exege.laws`` itself; the heading says which module
defines it. They are loaded on first use.

Laws
----

Defined in ``exege.laws.law`` and ``exege.laws.waves``.

.. autoclass:: exege.laws.Law
   :members:

.. autofunction:: exege.laws.consistency

.. autoclass:: exege.laws.ZonalAdvection

.. autoclass:: exege.laws.Rossby
   :members: flow, beta

.. autoclass:: exege.laws.Envelope
   :members:

.. autofunction:: exege.laws.projection

.. autofunction:: exege.laws.phase_rate

LAWFUL
------

Defined in ``exege.laws.lawful``.

.. autofunction:: exege.laws.measure

.. autoclass:: exege.laws.LawResult
   :members:

.. autofunction:: exege.laws.deviations

.. autoclass:: exege.laws.Deviations
   :members:

.. autofunction:: exege.laws.search

.. autoclass:: exege.laws.Search

.. autoclass:: exege.laws.LawCircuit

.. autofunction:: exege.laws.components
