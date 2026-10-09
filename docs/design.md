# Design

## Why it is built this way

Three concerns are kept apart, because each has a different answer:

- **Framework coupling lives in adapters.** The group expects to move to systems
  profoundly unlike ACE/FME/Samudra, so everything that knows a real file layout, log
  format or scheduler lives in `exege.adapters`, behind a small protocol. Supporting a new
  system means [writing a new adapter](adapters.md), never editing the code that uses it.
  Adapters are found through the `exege.adapters` entry-point group and nothing else, so
  one can ship from a completely separate package.
- **Science lives in the subpackage that uses it**, with the dependencies it honestly
  needs: needing numpy does not make something an adapter. What it may not know is a file
  format or a user interface.
- **Weight lives behind extras.** `exege.core` depends on the standard library alone, and
  `import exege` never pulls in the scientific stack.

Every API returns objects and prints nothing; the CLI is one client of it, a notebook
another. These rules are enforced by `tests/test_purity.py`, not by convention.

Errors exege raises on purpose are `ExegeError`s and reach a terminal as one line; anything
else is a bug and keeps its traceback, as does everything under `exege --debug`.

## Differences from the original visualiser

`exege.latents` reimplements the workflow of the
[latent space visualiser](https://github.com/ktempestuous/latent_space_visualiser_weather_models).
Checked on the real SamudrACE-E3SMv3 atmosphere latents: region selection and the
uncentered ranking are identical to the app's, node for node, and the unweighted PCA
matches scikit-learn's to 1e-7. Three things differ because they should:

- **Means are area-weighted.** Rows of a lat-lon grid crowd the poles, so an unweighted
  "global mean" over-counts them. Weighting moves channel 321's global mean by 0.35 —
  40% of its standard deviation — and changes the centered top five.
- **The similarity reference is a stated policy.** The app compares against whichever
  region node comes first in the array, which for this region is its south-west corner
  at (7.5°S, 144.5°W), 13° from the center asked for. Here it is the node nearest the
  center, or the region's mean.
- **Invalid nodes are left out.** Over land an ocean model's activations mean nothing —
  and are not zero: at the last layer of the SamudrACE-E3SMv3 ocean they are larger than
  over the sea (RMS 0.60 against 0.50), so nothing in the latents gives them away:
  they are excluded from regions and means and read NaN in every map, with no divide
  warnings and no borrowed numbers.

PCA signs are also fixed (each component's largest loading is positive), so a map does not
flip color between two runs of the same analysis.
