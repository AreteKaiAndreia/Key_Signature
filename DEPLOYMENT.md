# Running and deploying the apps

## Public app on Community Cloud

[Public key classifier](https://musickeyanalyzer.streamlit.app/)

Deploy `AreteKaiAndreia/Key_Signature` from `main` with
`streamlit_app.py` as the entry point and **Python 3.11** in Advanced
settings. Leave Secrets empty. The app defaults to Neural Network
(`keyclf/open_key_model.json`). The engine selector also offers Linear SVM,
Random Forest, Gradient Boosting, Essentia, and Soft Voting. The selected
method shows how it works and its held-out accuracy, recall, and precision.
Each predicts three relative
key pairs from the audio recorded so far while the microphone runs. Uploading a file
or stopping the recorder starts full analysis, up to the first 120 seconds.

The owner signs in at [Streamlit Community Cloud](https://share.streamlit.io/)
and completes any GitHub connection prompt. See the
[Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [code license notices](NOTICE.md).

To run the same app locally with Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The pinned `imageio-ffmpeg` wheel supplies the upload decoder. Cloud installs
only `libatomic1` through `packages.txt`; it avoids the system FFmpeg package's
large recommended graphics/media dependency chain. All supported audio formats
remain available. Local dataset/training commands can still use system FFmpeg.
PyArrow is pinned to the version Cloud actually permits, avoiding its automatic
uninstall of 26.0.0 and second wheel download during provisioning.

## Startup performance gate

Every push and pull request runs `.github/workflows/startup-performance.yml`.
It benchmarks the previous source and current source on the same Linux runner,
with the deployed Python 3.11 dependency lock and Chrome. CI logs and its job
summary include before, after, signed delta, and limit for each measure.
The build fails when any current value reaches its limit:

| Measure | Required |
|---|---:|
| Median warm URL-to-widget load, 7 visits | < 3,000 ms |
| Worst cold URL-to-widget load, 3 fresh processes | < 30,000 ms |
| Longest individual startup operation | < 500 ms |

A second job checks seven fresh signed-out visits to the actual public Cloud
URL and fails if its median reaches 3 seconds or a browser startup Long Task
reaches 500 ms. It logs deltas against the previous retained hosted report when
available. Cloud redeployment is asynchronous, so this checks public availability
and latency while the local source test checks the exact commit under review.

The clock starts immediately before browser navigation. A cold request launches
a fresh Streamlit process. It stops after the default engine selector and upload
control are visible and two browser animation frames have painted. Warm visits
reuse that server with fresh browser contexts. Dependency installation and
Chrome launch occur before the timer. This process-start test does not measure
Community Cloud's container provisioning or dependency installation.

The opt-in server profiler records the maximum wall time of **each invocation**,
including nested work, imports, the app module, and `main`. Browser Long Tasks
are measured through widget paint. Upload decoding and inference are outside
this startup-only budget. Full function profiles, samples, and server logs are
retained in the CI artifact. The profiler is disabled in the public app.

All six methods, descriptions, diagrams, metrics, uploads, and cumulative live
recording remain enabled. Audio and model libraries load on the first analysis;
the initial page uses lightweight display modules. The original Geist and IBM
Plex Mono font files ship with their SIL OFL notices and are inlined once per
process to avoid Google Fonts round trips. WebSocket compression reduces the
initial page payload. Reproduce with:

```bash
python -m pip install -r requirements.txt playwright==1.65.2
python scripts/benchmark_startup.py --output reports/startup/local.json
python scripts/benchmark_startup.py --url https://musickeyanalyzer.streamlit.app/ \
  --output reports/startup/hosted.json
```

Chrome must be installed. Use a fresh output path for each local run. Hosted
measurements include the Community Cloud wrapper and actual upload widget.
They do not assert that a remote server is cold unless its restart or wake
was independently observed. The local and hosted timings are recorded separately.

## Optional research forest

From the repository root, after the local feature caches and the deployment
weights have been built:

```bash
python -m pip install -r requirements.txt
KEYCLF_LOCAL_MODEL=1 python -m streamlit run streamlit_app.py
```

Use Python 3.11 with FFmpeg available. `requirements.txt` pins the Linux build
and the compatible Essentia wheel. The source-only Linux image can be built
with `docker build --platform linux/amd64 -t keyclf-linux-check .`.
The local opt-in looks for `models/cloud_model.joblib`. With the forest
enabled, the app keeps results in the session when switching engines.

Once a hosting and
redistribution basis for the exact weights is recorded in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md):

1. Upload the cleared compressed artifact as a versioned GitHub Release asset
   in [AreteKaiAndreia/Key_Signature](https://github.com/AreteKaiAndreia/Key_Signature).
   Record its SHA-256 and the corresponding source commit.
2. In the app's secrets, set top-level `KEYCLF_MODEL_URL` to the GitHub Release
   asset URL and `KEYCLF_MODEL_SHA256` to the expected lower-case digest.
   No credentials, audio, labels, or feature caches belong in the repository or app.
3. Verify the release loads and both engines rank three key pairs in the hosted
   browser using a recording and synthetic MP3 and MP4 files, then share the URL.
   Supply the deployed build's corresponding source per [NOTICE.md](NOTICE.md).

## v1 choice: regular Streamlit

The app is served with regular, server-side Streamlit:

```bash
streamlit run keyclf/app.py
```

This preserves the exact tested Python feature path, including librosa chroma,
scikit-learn model loading, and the system ffmpeg fallback for MP4 decoding.

## Streamlit Lite feasibility check

Checked on 2026-10-29:

- Stlite runs Streamlit on the Pyodide WebAssembly runtime:
  https://stlite.net/
- Pyodide currently provides scikit-learn and soundfile, but its package list
  does not provide librosa or numba:
  https://pyodide.org/en/stable/usage/packages-in-pyodide.html
- The production feature extractor imports librosa, which depends on numba for
  the CQT path, and the upload contract also includes MP4/ffmpeg decoding.

Therefore the existing, acceptance-tested extractor cannot be installed as a
faithful browser-only Stlite app. A browser deployment would require a separate
WebAudio/WASM feature extractor and parity tests against `keyclf.features`,
which is outside v1. Regular Streamlit is the documented fallback authorized by
specification §5.
