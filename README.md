# mplsoccer-viz-tests

Each test in [tests/soccer/](tests/soccer/) is a self-contained example that
draws a figure and compares it against a baseline image with
[pytest-mpl](https://github.com/matplotlib/pytest-mpl). The image tests are
parametrized over every valid pitch type (plus a custom
`center_scale_dims` pitch) and most draw the horizontal and vertical pitch
side by side so differences between the two are easy to spot. Text is
stripped from the figures (`remove_text=True`) so the comparison does not
depend on font rendering, and the baselines are saved at low dpi.
Tests that only check numbers (bin statistic
counts, standardizer round-trips, NaN handling) are plain assertion tests.

## Setup

```bash
uv sync
```

## Running the tests

Compare against the committed baselines (do this to validate a new
mplsoccer version):

```bash
uv run pytest --mpl
```

On failure, write the diff images (baseline / actual / difference) to a
results directory for inspection:

```bash
uv run pytest --mpl --mpl-results-path=mpl-results
```

## Testing an unreleased mplsoccer

To test a local mplsoccer checkout (e.g. a development branch) instead
of the locked PyPI release, overlay it with `--with-editable`.
Assuming the two repositories are
cloned side by side (e.g. `git/mplsoccer` and `git/mplsoccer-viz-tests`,
where `git` is just an example parent directory), run from this
repository:

```bash
uv run --with-editable ../mplsoccer pytest --mpl
```

To regenerate baselines after an intentional change:

```bash
uv run --with-editable ../mplsoccer python generate_baselines.py heatmap
```

## Regenerating the baselines

After hand-validating that the current output is correct,
regenerate and commit the baselines:

```bash
uv run python generate_baselines.py            # all modules
uv run python generate_baselines.py heatmap    # just one module
```

Each test module keeps its baselines in its own directory
(`tests/soccer/baseline/<module>/`); the script runs pytest once per
module because pytest-mpl's `--mpl-generate-path` only writes to a single
flat directory. The generate run skips the image comparison, so run
`pytest --mpl` afterwards to confirm everything passes, and hand-inspect
the changed images (`git diff --stat tests/soccer/baseline`) before
committing.

## Layout

All the current examples are for soccer pitches and live in
`tests/soccer/`; other sports can be added as sibling directories.
