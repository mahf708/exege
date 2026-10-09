Latents API
===========

Every name below is importable from ``exege.latents`` itself; the heading says which
module defines it. They are loaded on first use, so ``import exege.latents`` alone does
not need numpy.

Sources
-------

Defined in ``exege.latents.source``.

.. autoclass:: exege.latents.LatentInfo
   :members:

.. autoclass:: exege.latents.LatentSource
   :members:

.. autoclass:: exege.latents.LayerInfo
   :members:

.. autoclass:: exege.latents.ReferenceFields
   :members:

.. autofunction:: exege.latents.check_comparable

.. autofunction:: exege.latents.open_source

.. autofunction:: exege.latents.parse_time

.. autofunction:: exege.latents.read_latents

One region at one time
----------------------

Defined in ``exege.latents.analysis``.

.. autodata:: exege.latents.AnyRegion
   :annotation:

.. autoclass:: exege.latents.Box
   :members:

.. autoclass:: exege.latents.ChannelRanking
   :members:

.. autoclass:: exege.latents.Region
   :members:

.. autoclass:: exege.latents.RegionAnalysis
   :members:

.. autofunction:: exege.latents.analyze_region

.. autofunction:: exege.latents.cosine_similarity

.. autofunction:: exege.latents.load_channels

.. autofunction:: exege.latents.rank_channels

Bases: PCA and dictionaries
---------------------------

Defined in ``exege.latents.basis``.

.. autoclass:: exege.latents.Decomposition
   :members:

.. autoclass:: exege.latents.Dictionary
   :members:

.. autoclass:: exege.latents.PCA
   :members:

.. autofunction:: exege.latents.basis_hash

.. autofunction:: exege.latents.basis_provenance

.. autofunction:: exege.latents.bspline_activation

.. autofunction:: exege.latents.result_provenance

.. autofunction:: exege.latents.fit_pca

.. autofunction:: exege.latents.load_basis

.. autofunction:: exege.latents.save_basis

.. autofunction:: exege.latents.spline_knots

.. autofunction:: exege.latents.top_loadings

Many times at once
------------------

Defined in ``exege.latents.samples``.

.. autoclass:: exege.latents.Moments
   :members:

.. autofunction:: exege.latents.accumulate_moments

.. autofunction:: exege.latents.iter_batches

.. autofunction:: exege.latents.pca_from_moments

Through time and between runs
-----------------------------

Defined in ``exege.latents.through``.

.. autoclass:: exege.latents.DifferenceGrowth
   :members:

.. autoclass:: exege.latents.FieldRanking
   :members:

.. autoclass:: exege.latents.FieldStoryline
   :members:

.. autoclass:: exege.latents.Hovmoller
   :members:

.. autoclass:: exege.latents.PairedDifference
   :members:

.. autoclass:: exege.latents.RegionSeries
   :members:

.. autofunction:: exege.latents.correlate_field

.. autofunction:: exege.latents.difference

.. autofunction:: exege.latents.difference_growth

.. autofunction:: exege.latents.field_storyline

.. autofunction:: exege.latents.hovmoller

.. autofunction:: exege.latents.rank_by_field

.. autofunction:: exege.latents.region_series

Census and profile
------------------

Defined in ``exege.latents.features``.

.. autoclass:: exege.latents.FeatureCensus
   :members:

.. autoclass:: exege.latents.FeatureProfile
   :members:

.. autofunction:: exege.latents.feature_census

.. autofunction:: exege.latents.feature_profile

Evaluation
----------

Defined in ``exege.latents.evaluate``.

.. autoclass:: exege.latents.Evaluation
   :members:

.. autoclass:: exege.latents.FidelityCurve
   :members:

.. autoclass:: exege.latents.Split
   :members:

.. autoclass:: exege.latents.Stability
   :members:

.. autofunction:: exege.latents.evaluate_basis

.. autofunction:: exege.latents.fidelity_curve

.. autofunction:: exege.latents.save_result

.. autofunction:: exege.latents.seed_stability

.. autofunction:: exege.latents.split_archives

.. autofunction:: exege.latents.split_groups

.. autofunction:: exege.latents.split_time_blocks

Steering
--------

Defined in ``exege.latents.steering``.

.. autoclass:: exege.latents.Effect
   :members:

.. autoclass:: exege.latents.Hook
   :members:

.. autoclass:: exege.latents.Intervenable
   :members:

.. autoclass:: exege.latents.Pairing
   :members:

.. autoclass:: exege.latents.Rollout
   :members:

.. autoclass:: exege.latents.Run
   :members:

.. autoclass:: exege.latents.Steer
   :members:

.. autoclass:: exege.latents.SteeringResult
   :members:

.. autofunction:: exege.latents.feature_direction

.. autofunction:: exege.latents.open_intervenable

.. autofunction:: exege.latents.run_steering

Experiment records
------------------

Defined in ``exege.latents.record``.

.. autoclass:: exege.latents.ExperimentRecord
   :members:

.. autofunction:: exege.latents.evaluation_record

.. autofunction:: exege.latents.load_record

.. autofunction:: exege.latents.record_from_dict

.. autofunction:: exege.latents.save_record

.. autofunction:: exege.latents.steering_record

Grids
-----

Defined in ``exege.latents.grid``.

.. autoclass:: exege.latents.grid.Grid
   :members:

.. autofunction:: exege.latents.grid.cell_area_weights

.. autofunction:: exege.latents.grid.great_circle_km

.. autofunction:: exege.latents.grid.small_circle

The toy emulator
----------------

Defined in ``exege.latents.toy``.

.. autofunction:: exege.latents.toy.write_toy

.. autofunction:: exege.latents.toy.toy_run

.. autoclass:: exege.latents.toy.ToyModel
   :members:

.. autoclass:: exege.latents.toy.ToyRun
   :members:

.. autofunction:: exege.latents.toy.toy_grid
