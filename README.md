# Key Signature Classifier

A learning-focused classifier for the 24 major/minor keys. It extracts three
chroma-derived components from a track—first pitch class, last pitch class, and
a track-wide 12-bin histogram—then compares a learned model with the classic
Krumhansl–Schmuckler profile-matching baseline.

The product requirements live in
[`specs/key-signature-classifier-spec.md`](specs/key-signature-classifier-spec.md).
The executable acceptance contract is documented in
[`HARNESS_README.md`](HARNESS_README.md), and resumable implementation state is
recorded in [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).

## Install

Python 3.11 and ffmpeg are required.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -m harness
```

## Build the feature cache

The command sparse-clones annotations only, streams previews one at a time,
deletes every temporary audio file, and resumes from per-track checkpoints:

```bash
python -m keyclf.features
```

Useful development flags include `--limit 24`, `--workers 2`, `--chroma-type
stft`, and `--rebuild`. The only canonical cache is
`data/features/giantsteps_features.parquet`.

## Train and evaluate

```bash
python -m keyclf.train
KEYCLF_REQUIRE_IMPL=1 pytest --junitxml reports/pytest-junit.xml
python -m eval.evaluate --junit reports/pytest-junit.xml
python -m eval.evaluate --criteria-only reports/metrics.json \
  --junit reports/pytest-junit.xml
```

Training uses the fixed split stored in the feature cache, performs model
selection on validation data, refits on train plus validation, and never uses
the test split for selection.

## Run the app

```bash
streamlit run keyclf/app.py
```

Upload an MP3, MP4, or WAV to see the detected key, model confidence, the top
three guesses, and the track's chroma distribution. The classifier selector
offers the locally trained models.

The Community Cloud entry point, `streamlit_app.py`, accepts a microphone
recording or MP3/MP4 and ranks three keys with the custom model or Essentia.
The custom model learns from overall chroma, harmonic chroma, and bass
chroma. Its training data consists of 662 CC0/CC BY Freesound loops and 480
original synthetic phrases. [Open the public app](https://music-key-analyzer.streamlit.app/).

| Held-out set | First choice | Top three |
|---|---:|---:|
| GiantSteps | 47/91 (51.6%) | 67/91 (73.6%) |
| FMAK | 40/75 (53.3%) | 56/75 (74.7%) |

See [deployment and local run instructions](DEPLOYMENT.md) and
[training data provenance](DATA_PROVENANCE.md#training-sources-for-the-open-key-model).

## Held-out chroma and Hooktheory comparison

| Training setup | GiantSteps first / top three | FMAK first / top three |
|---|---|---|
| GiantSteps, chroma plus first and last pitch summaries | 42/91 (46.2%) / 66/91 (72.5%) | — |
| GiantSteps, chroma only | 47/91 (51.6%) / 65/91 (71.4%) | 29/75 (38.7%) / 54/75 (72.0%) |
| GiantSteps + FMAK, chroma only | 42/91 (46.2%) / 65/91 (71.4%) | 42/75 (56.0%) / 54/75 (72.0%) |
| GiantSteps + FMAK + 500 Hooktheory renders, chroma only | 45/91 (49.5%) / 64/91 (70.3%) | 41/75 (54.7%) / 54/75 (72.0%) |

All four runs used the same 500-tree forest settings and transposition
augmentation. Hooktheory examples were weighted at 0.25, selected on real-audio
validation before testing. The held-out effect was mixed. Both test sets contain
only real recordings. Run `python -m keyclf.hooktheory` and then
`python -m keyclf.hook_experiment` with the existing GiantSteps and FMAK caches
to reproduce or resume the comparison. See the [status ledger](IMPLEMENTATION_STATUS.md)
for measured details and [data provenance](DATA_PROVENANCE.md) for limitations.

## Accuracy research

The opt-in research path preserves the v1 artifact and original test IDs. It
adds harmonic/bass pitch summaries, beat accents, pitch persistence, and
section-level distributions. Persistence is a duration proxy, and beat accents
do not identify bar downbeats or transcribe individual notes.

```bash
python -m keyclf.research_features --workers 2
python -m keyclf.experiments --features data/work/research_features.parquet \
  --out reports/research-v2
```

The experiment runner uses three mode-stratified development folds and applies
12 transpositions only inside each training fold. It ranks configurations by
exact accuracy, top-three accuracy, and log loss. Add `--finalize --projections`
only after fixing the experiment matrix to evaluate the selected model once
on the original test partition and export PCA, t-SNE, and exploratory K-means
coordinates. It records exact recall by key, macro recall, weighted recall,
and major/minor mode performance. The final test receipt prevents silently
evaluating a different selection in the same output directory.

Copy the resulting `research_model.joblib` into `models/` to enable the
research classifier in the upload app. The app keeps the original histogram
for display and selects the model's feature columns in the recorded order.

`scripts/prepare_technical_report.py` prepares the measured learning curve and
presentation data. `scripts/build_technical_deck.mjs` creates an editable
PowerPoint using the Codex presentation runtime. Generated caches, models,
point coordinates, and slides remain under ignored work/report directories.

The completed first research batch selected a 500-tree, transposition-augmented
histogram random forest. On the unchanged 91-track benchmark, exact accuracy
increased from 46.2% to 51.6%, top-three accuracy from 69.2% to 71.4%, and macro
recall over the 23 represented keys from 27.3% to 41.5%. Training accuracy was
98.8%, so substantial overfitting remains. Richer feature groups did not win
development selection, and the original 80%/95% acceptance targets are not met.
This previously examined benchmark is not a fresh blind test.

`scripts/verify_research_model.py` checks a second fixed-configuration refit
without reselection. `scripts/measure_research_latency.py` times the selected
model's actual upload extraction path without warm-up. The status ledger records
results and local artifact paths, including the 19-slide technical PowerPoint.

## Data policy

No raw audio is retained. Dataset and derived artifact rights are separate from
the MIT-licensed code; see [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

## FMAKv2 expansion pilot

The optional pilot adds artist-separated FMAK recordings to the existing
histogram random forest. It requires the original GiantSteps feature cache,
annotations and frozen `models/research_model.joblib`.

```bash
python -m keyclf.fmak plan
python -m keyclf.fmak extract
python -m keyclf.fmak_experiment
```

The plan fixes 500 recording IDs and splits before extraction. The downloader
uses HTTP byte ranges to read selected ZIP members, checks each member's CRC,
and deletes temporary audio. Extraction can be resumed without fetching completed
recordings again. A server that ignores ranges is rejected instead of downloading
the full archive.

The experiment selects the new data weight on validation only, then compares
the existing and expanded models on the original GiantSteps test and a new
artist-separated FMAK test. The upload app offers the expanded model as
`FMAK pilot (experimental)` while retaining the previous default. Local evidence
is in `reports/fmak-pilot/test_results.json`; caches and models stay out of Git.

The completed 500-track pilot improved exact accuracy on the fresh FMAK test
from 29/75 to 39/75 (38.7% to 52.0%), but reduced exact accuracy on the unchanged
GiantSteps test from 47/91 to 44/91 (51.6% to 48.4%). It is an optional model for
experimentation, not a replacement for the previous default.
