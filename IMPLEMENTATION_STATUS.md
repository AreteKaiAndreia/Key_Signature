# Implementation status

This is the durable resume ledger for the key-signature classifier.

## Sources of truth

1. Product behavior: [`specs/key-signature-classifier-spec.md`](specs/key-signature-classifier-spec.md)
2. Acceptance contract: [`HARNESS_README.md`](HARNESS_README.md) and [`.github/workflows/acceptance.yml`](.github/workflows/acceptance.yml)
3. Progress only: this file

## Completion state

- Local v1 accepted: **no**
- Public release accepted: **no**
- GitHub target: [`AreteKaiAndreia/Key_Signature`](https://github.com/AreteKaiAndreia/Key_Signature) (public)
- Data publication: **blocked pending documented redistribution basis**

## Milestones

| Milestone | State | Evidence |
|---|---|---|
| Canonical repository and packaging | complete | Spec moved; pyproject, README, MIT license, provenance added |
| Harness self-tests | complete | 81 passed, 1 optional skip on Python 3.11 |
| Feature interfaces and synthetic tests | complete | 31/31 required feature tests passed |
| Resumable GiantSteps pipeline | complete | 604/604 CQT tracks, 0 failures, 135 resumed, no audio retained |
| Deterministic model training | complete | 125 tests passed; two isolated train runs reproduced exactly |
| Interpretive analysis and notebook | implemented | LR and RF paths plus exploration notebook added |
| Streamlit application | complete | AppTest imports/runs; regular Streamlit selected after Stlite dependency review |
| Strict C1–C7 acceptance | blocked by measured accuracy | C1 46.15% and C2 69.23% fail; C3–C7 pass |
| Public GitHub source release | complete | Repository now under `AreteKaiAndreia/Key_Signature`; source-only boundary |
| Accuracy research requested 2026-10-30 | complete, targets not met | 30 configurations, three development folds, one selected-model fixed-test evaluation; exact accuracy improved to 51.65% |
| Technical PowerPoint requested 2026-10-30 | complete | 19 slides; editable charts/tables; all previews inspected and package/import/layout validation passed |
| FMAKv2 expansion pilot | complete, mixed accuracy effect | 500/500 extracted; fresh FMAK accuracy 38.67% → 52.00%; unchanged GiantSteps benchmark 51.65% → 48.35%; default unchanged |

## Current checkpoint

### Publishable custom model

- Target: at least 50% first choice and 70% top three on each unchanged
  GiantSteps and FMAK test partition before releasing a custom engine.
- The existing forest has no identified blanket publication ban. Its training
  corpus has mixed terms, and its frozen GiantSteps test result misses the
  new first-choice threshold (45/91). Train a new model from original synthetic
  material and permissively licensed recordings. Use GiantSteps and FMAK
  only for validation and final testing.
- Freesound Loop Dataset work is checkpointed under `data/work/open-key`.
  The official ZIP supports bounded byte ranges. The license and human
  annotation filters found 666 eligible members. The first pass retained
  662 unique loops (356 CC0 and 306 CC BY), with three rate-limited entries
  and one too short. It also generated features for 480 original phrases.
- Chroma-only logistic regression with Krumhansl–Schmuckler scores was chosen
  on validation. Held-out FMAK 47/75 first and 56/75 top three, GiantSteps
  41/91 first and 63/91 top three. This candidate failed the joint threshold.
  The frozen selection, artifact, and receipt are in `reports/open-key/`.
- Linear chroma, harmonic and bass model reached 45/91 first and 67/91 top
  three on GiantSteps and 39/75 first and 57/75 top three on FMAK. It missed
  the GiantSteps first-choice threshold by one track.
- The selected nonlinear model scores each of 24 keys with a shared
  feedforward network and three 12-bin pitch summaries rotated to the
  candidate tonic. It was fitted on 662 Freesound loops and 480 original
  phrases. Candidate selection used 513 GiantSteps development tracks and
  75 FMAK validation tracks. Neither benchmark contributed fitted examples.
  Configuration and JSON SHA-256 were frozen before held-out evaluation.
- Held-out GiantSteps: **47/91 first (51.6%), 67/91 top three (73.6%)**.
  Held-out FMAK: **40/75 first (53.3%), 56/75 top three (74.7%)**.
  All four targets passed. This was the third candidate family evaluated
  on the same benchmarks. Use a fresh holdout for future claims of
  performance on completely unseen recordings.
- Artifacts and receipts are under `reports/open-key-candidate/`. The
  published model is `keyclf/open_key_model.json`. Public app verification
  is the remaining step. For a local rerun of the frozen gate, run
  `.venv/bin/python -m keyclf.train_open_candidate --final`.

### Essentia-only public app

- The Community Cloud entry point uses Essentia when no forest is configured.
  The public app needs no project dataset or trained weights.
- The Essentia-only configuration passed app checks in Linux with synthetic
  MP3, MP4, and recorded-format WAV inputs. All three ranked C major first
  for the C-major chord. The eight existing audio and research tests passed.
- The owner completed GitHub authorization. The
  [public Essentia app](https://aretekaiandreia-key-signature.streamlit.app/)
  is live on Streamlit Community Cloud from `main`, entry point
  `streamlit_app.py`, Python 3.11, with empty secrets.
- Verified the public URL while signed out. A synthetic C-major MP3 ranked
  C major, E minor, and G major in that order. Evidence is in
  `reports/ui-checks/streamlit-public-result.jpg`. The original v1 accuracy
  and data publication gates above apply to the custom model.

### Chroma-only, Hooktheory, and Community Cloud

- Date: 2026-11-13 America/Los_Angeles. The matched ablation, annotation pilot,
  held-out comparison, and local upload redesign are complete. Public use of
  the custom weights is held pending the decision in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).
- Frozen pilot: 500 of 9,699 eligible training excerpts, 300 major and 200 minor.
  Manifest `66e9ae6d30a325829677ec59bcd61562553cf1e646d6b4501fbd4d1082b7ab09`.
  Replaying `python -m keyclf.hooktheory && python -m keyclf.hook_experiment`
  resumed all 500 rows and the frozen model receipt; no original recordings
  fetched, no rendered audio retained, and no second test fit or scoring.
- Unchanged recording splits: GiantSteps 422/91/91 and FMAK 350/75/75.
  Excluded known FMAK and Sheet Sage held-out matches by identity; the locally
  available GiantSteps data did not include artist/title identities. Held-out
  tracks and labels were untouched. The test set contains zero renders.
- Matched seeded 500-tree forest and twelve training transpositions for every
  comparison. The 36-versus-12 comparison uses the same 513 GiantSteps
  development recordings. Expanded refits use 938 real development recordings;
  the Hooktheory candidate appends 500 weighted training-only renders. This is
  a fresh matched experiment, separate from the earlier FMAK pilot below.

| Matched fit | GiantSteps first guess | GiantSteps top three | FMAK first guess | FMAK top three |
|---|---:|---:|---:|---:|
| Chroma + beginning/ending (GiantSteps) | 42/91 (46.2%) | 66/91 (72.5%) | — | — |
| Chroma only (GiantSteps) | 47/91 (51.6%) | 65/91 (71.4%) | 29/75 (38.7%) | 54/75 (72.0%) |
| Chroma + FMAK data | 42/91 (46.2%) | 65/91 (71.4%) | 42/75 (56.0%) | 54/75 (72.0%) |
| Chroma + FMAK + Hooktheory data | 45/91 (49.5%) | 64/91 (70.3%) | 41/75 (54.7%) | 54/75 (72.0%) |

- Validation selected `hook-0.25` among weights 0.25, 0.5, and 1.0; it raised
  mean first-guess validation accuracy from 50.0% to 59.6% over the
  GiantSteps-only histogram, and GiantSteps validation rose from 51/91 to 55/91.
  The held-out GiantSteps first-guess score was lower than the GiantSteps-only
  histogram. The local upload app retains its Research default; the Cloud
  artifact uses the choice frozen on validation.
- Local artifacts: `data/work/hooktheory/{manifest.json,features.parquet,checkpoints/}`,
  `reports/hooktheory/{selection.json,results.json}`,
  `models/cloud_model.joblib` (179 MiB, version
  `real-song-hist-ablation-hook-v1`, SHA-256
  `184002c949d07c934e1fdbbdd9ccdecbf5d120821dfc01d871173d477f4cbd1a`).
- Browser QA: initial upload, analysis spinner, and ranked results at 1280 px
  and 390 px, including MP3/MP4 and the three local classifier choices.
  Screenshots are under `reports/ui-checks/`. The local Cloud entry point ran
  both engines on the same MP3. Recording controls and upload fallback were
  checked without granting live microphone access. No public app was deployed.
- Full strict pytest: **150 passed, 1 optional skip** (`reports/pytest-current.xml`).
  Eight new tests cover direct histogram parity, label-blind renders,
  duplicate exclusion, checkpoint interruption, both ranking engines on WAV,
  MP3, and MP4, silence/corrupt/oversize handling, truncation, and rejected
  weights. Original criteria replay: C3–C7 pass; C1/C2 remain below the 80%/95%
  thresholds. Hosted real-data acceptance remains gated.
- Clean Debian Linux x86-64 image with Python 3.11, FFmpeg, and Essentia
  2.1b6.dev1389: application startup and both engines passed on synthetic WAV,
  MP3, and MP4 with the actual model mounted read-only. The Essentia winner
  and strength matched the native API. The initial forest deserialization
  took 12 seconds in the emulated container; actual Community Cloud cold
  start and memory limits still need verification after clearance.
- Source-only implementation pushed at `115f987`.
  [GitHub harness self-tests passed](https://github.com/AreteKaiAndreia/Key_Signature/actions/runs/43558275504);
  full real-data acceptance was skipped because no cleared cache URL is set.

### FMAKv2 expansion pilot 2026-11-10

- Complete: integrated 500 artist-separated, quality-screened FMAKv2 recordings with the existing histogram classifier.
- Official Zenodo FMAK archive returned HTTP 206 for a 32-byte range probe. Use bounded range reads and discard temporary audio; do not download the 39.3 GB release to the laptop.
- Preserve the original 604 rows, labels, and split. Freeze FMA artist groups and pilot IDs before audio extraction. Reserve a fresh FMA test split. Fit only on training data during selection, then refit on development data. Compare the frozen research model and the expanded candidate on identical test rows. No FMA features or models are published.
- Manifest frozen: 5,489 label rows, 5,271 v1/v2 agreements, 1,619 eligible after archive, duration, license, and checksum-index checks. The pilot is 350/75/75 with FMA artist groups disjoint. Manifest `4d0273311ec9fef498c674719055c5f9ad6748fa41f2a262c201d31bf3bf4ef4`.
- Selection plan: hold the 500-tree histogram forest and augmentation fixed. Compare added-recording weights 0.25, 0.5, and 1.0 on the mean of exact accuracy on the two validation sets; use mean top-three accuracy and lower mean log loss as tie-breaks. Fit the winner on both development sets. Score the frozen reference and winner on the unchanged GiantSteps test and new FMA test, and keep the added model opt-in.
- Extraction: 500/500, zero failures; 80 resumed after a deliberate SIGINT, 420 newly extracted with parallel range reads and two decode slots. Repeat run reused all 500 without a download. No audio or ZIP archives retained. Feature cache is 112 KiB; persistent pilot working directory is 4.4 MiB including metadata and checkpoints.
- Selection: validation chose FMAK weight 1.0 (GiantSteps 59.34%, FMAK 57.33%). Final fit used 513 original plus 425 FMAK development recordings, 938 total before rotations. Neither test set was scored until the selection was frozen.
- Held-out GiantSteps (91): exact 47/91 → 44/91 (51.65% → 48.35%); top-three 71.43% → 69.23%; macro recall 41.46% → 35.94%.
- Fresh held-out FMAK (75, 39 artists): exact 29/75 → 39/75 (38.67% → 52.00%); top-three 72.00% → 73.33%; macro recall 41.32% → 53.13%.
- Pooled pilot exact: 76/166 → 83/166 (45.78% → 50.00%). This particular mixture improved, but the expansion did not improve the original benchmark. Cross-source remixes and artist overlap remain unverified. Keep Research as the app default, with the expansion opt-in.
- Last successful checks: 142 passed, one optional skip (strict pytest, `reports/fmak-pilot/pytest.xml`); selection and test receipt replayed without fitting or evaluating again; actual Streamlit option verified. Expanded-path cold maximum 1.269 s on three synthetic 120-second MP3 calls, excluding model loading (`reports/fmak-pilot/latency.json`).
- Local artifacts: `data/work/fmak-pilot/{manifest.json,features.parquet,checkpoints/}`, `reports/fmak-pilot/{selection.json,test_results.json}`, and `models/fmak_model.joblib`. The latter is available as **FMAK pilot (experimental)**. The original cache and models were not replaced.
- Source and aggregate results published at `652074e` (integration `e34e12f`). [GitHub harness self-tests passed](https://github.com/AreteKaiAndreia/Key_Signature/actions/runs/42972205742). Full hosted data acceptance was skipped under the existing redistribution gate.

### Previous benchmark checkpoint

- Date: 2026-10-30 America/Los_Angeles
- Active operation: none; requested research batch, PowerPoint and source handoff complete
- Last successful action: exported and validated `reports/research-v2/presentation/Key_Signature_Technical_Review.pptx` (19 slides, 8 native charts, 8 native tables); visually inspected all slides, including final revised previews
- Runtime recovery: Desktop cache and virtualenv contain macOS `dataless` cloud placeholders. A new local runtime under `/private/tmp/keyclf-research.1I9aUa` avoids stalled imports. The earlier attribution solely to policy-service contention was not established.
- Research checkpoint: 604/604 new feature rows complete, zero failures, two checkpoint resumes, no audio retained. SHA256 `a11c602fab1edb3f7aeff716d518b6db657a2574b2aef2bb4284a2edf45a6f15`.
- Research selection complete: histogram-only random forest, 500 trees, leaf size 2, sqrt feature subset, balanced_subsample and 12 transpositions inside training folds. Development CV exact accuracy 55.36%.
- Research fixed test: exact 51.65% (47/91), top-three 71.43% (65/91), macro recall over represented keys 41.46%, weighted recall 51.65%, major exact recall 33.33% (5/15), minor exact recall 55.26% (42/76), log loss 1.8601. These remain below the original 80%/95% targets.
- Research training resubstitution: 98.83% exact and 99.36% macro recall on 513 original development tracks after fitting 6,156 rotated training rows. This is training fit, not a generalization estimate; the training/test gap still indicates overfitting.
- All original IDs/labels/splits match; the 36 original feature columns reproduced exactly (maximum difference 0.0). New harmonic/beat/persistence/temporal groups did not beat the augmented histogram on development selection.
- Verification: latest complete strict run passed **136 tests with one optional criterion skip** in 19.58 seconds. JUnit: `reports/research-v2/baseline_acceptance/pytest-junit.xml`. Original gate replay reproduced 46.15% exact and 69.23% top-three with C3-C7 passing.
- Development CV: original reference forest 46.98% exact; transposition-augmented histogram forest 55.36%. The selected configuration was fixed before research test inference. The 91-track benchmark was already examined during v1, so a fresh blind test is needed for a stronger generalization claim.
- Reproducibility: a second fixed-configuration refit retained byte-equivalent split IDs and matched all 604 probability vectors within 1.12e-16 (absolute tolerance 1e-12), without reselection or recomputing test metrics. Evidence: `reports/research-v2/reproducibility.json`.
- Research inference: selected histogram extraction/model path, three synthetic 120-second MP3 runs with no warm-up; maximum 1.145 seconds. Model loading excluded, matching the harness timing boundary. The complete 180-feature extraction path was not timed.
- Cluster diagnostics: development-fitted PCA explains 36.90% in two dimensions; exploratory 24-cluster K-means silhouette 0.128, adjusted Rand index 0.156. The classifier is supervised random forest, not K-means. Test points are transformed only after PCA fitting; t-SNE uses development tracks only.
- Additional-data metadata: 1,349 single-key candidates (429 major / 920 minor), no exact ID or MD5 overlaps, 137 ambiguous/unknown entries recorded separately. No additional audio collected.
- Source milestones: `c6e6d62` added the research pipeline; `d7fb278` fixed cold synthetic self-tests. The harness stand-in uses FFT/FFmpeg to avoid timing first-time numba compilation; real C3 still times the actual extractor with the unchanged strict limit and no warm-up.
- Upload verification: synthetic WAV, MP3, MP4 decoding and Streamlit top-three/relative-key rendering passed all four smoke tests. The histogram-only research app fast path has its own regression test.
- Final source verification: GitHub harness self-tests passed at `2764175b4e80eb02e9accbb503307dbe2c0824e1` (run [`41126521319`](https://github.com/AreteKaiAndreia/Key_Signature/actions/runs/41126521319)). Full hosted acceptance remains intentionally pending an authorized cache.
- Environment: Python 3.11 virtualenv installed; ffmpeg available
- Committed source milestones: `12e4b28` (v1 build) and `6291ad3` (acceptance diagnostics and handoff)
- Blockers:
  - local v1: honest held-out accuracy is below C1 and C2; the bounded failure policy is exhausted
  - public release acceptance: feature-cache redistribution rights are not documented, and the measured C1/C2 shortfall remains
- Generated local artifacts (all intentionally ignored):
  - `data/features/giantsteps_features.parquet` (604 CQT rows)
  - `data/work/` (atomic per-track checkpoints)
  - `models/model.joblib`
  - `reports/pytest-junit.xml`, `reports/metrics.json`, confusion matrices, profile artifacts, and diagnostics
  - `data/work/research_features.parquet`, `data/work/research-v2/` (604 research checkpoints)
  - `models/research_model.joblib` (available in the upload app separately from v1)
  - `reports/research-v2/` (candidate results, final test receipt, recall, projections, learning curve, latency, reproducibility, additional-data metadata)
  - `reports/research-v2/presentation/Key_Signature_Technical_Review.pptx` (SHA256 `6235a08349d02b6a78597b03d3313ebce335ef91ce4fa6ae7b43ddc0cd903c19`)
  - `reports/research-v2/.deck-build/validation.json`, rendered previews and private draft

## Original v1 measured results

- Fixed split, seed 42: 422 train / 91 validation / 91 held-out test; no track leakage.
- Selected on validation only: 500-tree random forest, `min_samples_leaf=1`,
  `max_features=0.5`, `class_weight=balanced_subsample`.
- CQT validation: exact 0.637363, top-three 0.714286, log loss 2.535430.
- Held-out test: exact 0.461538, top-three 0.692308, 91 tracks.
- Krumhansl-Schmuckler baseline on the same test split: exact 0.406593.
- Latest cold-path maximum latency: 0.868 seconds (three synthetic 120-second MP3 runs; historical first run was 1.505 seconds).
- Latest learned/profile correlation: major 0.843700, minor 0.701450 (fresh Python 3.11/scikit-learn environment; source definitions unchanged).
- Bounded STFT alternative: validation exact 0.582418 and top-three 0.714286;
  rejected without evaluating its test partition.
- Gate: C3, C4, C5, C6, and C7 pass; C1 and C2 fail. No threshold,
  label, row, or split was changed after observing test performance.

## Operation log

| Operation | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | harness self-test: 81 passed, 1 optional criterion skip |
| `KEYCLF_REQUIRE_IMPL=1 .venv/bin/python -m pytest -q --junitxml reports/pytest-junit.xml` | complete run: 125 passed, 1 optional skip; isolated training predictions and split IDs matched |
| `.venv/bin/python -m keyclf.features --workers 2` | official cache: 604/604 extracted, 0 failed, 135 compatible checkpoints resumed, no audio retained |
| `.venv/bin/python -m keyclf.train` | deterministic CQT selection and train+validation refit completed |
| `.venv/bin/python -m eval.evaluate --junit reports/pytest-junit.xml` | C1/C2 failed honestly; C3-C7 passed |
| `.venv/bin/python -m eval.evaluate --criteria-only reports/metrics.json --junit reports/pytest-junit.xml` | reproduced the same C1/C2 gate failure |
| `.venv/bin/python -m keyclf.diagnostics --alternate-training /tmp/keyclf-stft-training.json` | per-class report written; STFT rejected on validation |
| strict pytest rerun after profile vectorization | 117 passed and 1 skipped before interruption at 14m32s; a fresh collector then blocked in stdlib `binascii` import while macOS `syspolicyd` saturated the host. An evaluator replay likewise blocked reading NumPy during import. Manual training and targeted changed-code tests had already passed; these are recorded as host contention, not passing reruns. |
| `gh repo create benschwierking/Key_Signature --public --source=. --remote=origin ...` | public repository created after reauthenticating `benschwierking` |
| `git push --progress -u origin main` | 61 objects / 78.33 KiB pushed; remote `main` established and verified public |
| `python -m keyclf.research_features --workers 2` | 604/604 research tracks, 0 failures, 2 resumed; original 36 features match exactly; no audio retained |
| `python -m keyclf.experiments --features /private/tmp/keyclf-research.1I9aUa/research_features.parquet --reference /private/tmp/keyclf-research.1I9aUa/original_features.parquet --out /private/tmp/keyclf-research.1I9aUa/final_experiments --finalize --projections` | 30 configurations / 90 development fold fits; histogram RF selected; fixed test 47/91 exact, 65/91 top-three |
| `KEYCLF_REQUIRE_IMPL=1 pytest --junitxml reports/pytest-junit.xml` in the isolated acceptance sandbox | 136 passed, 1 optional skip, 0 failures/errors; latest JUnit copied to research report directory |
| `python scripts/verify_research_model.py --features ... --model ... --out reports/research-v2/reproducibility.json` | second refit passed; same split IDs; maximum probability difference 1.11e-16 |
| `python scripts/measure_research_latency.py --model ... --out .../research_latency.json` | actual selected-model path, no warm-up, maximum 1.145 seconds; temporary synthetic audio deleted |
| `scripts/audit_additional_data.py` | 1,349 metadata candidates, including 429 major; 137 ambiguous/unknown entries kept separate; no extra audio collected or used for training |
| `scripts/prepare_technical_report.py` and `scripts/build_technical_deck.mjs` | measured learning curve and 19-slide editable PowerPoint exported; first-party re-import, native chart validation, package and layout checks passed |
| `git push origin main` and `gh run watch 41126521319 --exit-status` | final source commit `2764175` pushed; GitHub harness self-test passed in 1m1s; full hosted acceptance skipped by the documented data-rights gate |

## Resume invariant

The FMA and Hooktheory pilots use frozen manifests and reuse compatible
per-track checkpoints. The original benchmark and model artifacts are intact.
Do not publish
`data/`, `models/`, or `reports/` unless `DATA_PROVENANCE.md` first records a
valid redistribution basis. The commit containing this ledger is the current
handoff commit (`git rev-parse HEAD`).

## Exact next action

```bash
cd "/Users/ben/Desktop/ML and Big Data/Key_Signature/keyclf-harness" && .venv/bin/python -m streamlit run streamlit_app.py
```
