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
Plex Mono font definitions ship with their SIL OFL notices, avoiding a Google
Fonts CSS request. Only needed subsets load from their pinned upstream WOFF2
URLs. Inlining every subset increased hosted load time and was rejected.
WebSocket compression reduces the
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

### Measured status on 2026-11-18

The app remains on Community Cloud, as requested. Performance acceptance is
**pending**, even though the same six methods, MP3/MP4 uploads, microphone,
cumulative predictions, diagrams, metrics, and phone/desktop views passed the
[public browser check](https://github.com/AreteKaiAndreia/Key_Signature/actions/runs/44390945747).

| Current measurement | Result | Budget |
|---|---:|---:|
| Public URL, warm server / fresh signed-out browser median | 4,020.6 ms | < 3,000 ms, **failed** |
| Linux CI warm browser median | 1,025.2 ms | < 3,000 ms |
| Linux CI worst cold process | 1,654.8 ms | < 30,000 ms |
| Linux CI longest individual app startup call | 289.1 ms | < 500 ms |

The [performance run](https://github.com/AreteKaiAndreia/Key_Signature/actions/runs/44390025874)
keeps the public failure visible and retains before/after values and raw profiles.
A supported embedded view with toolbar and padding retained measured 3,477.8 ms
in a separate seven-visit diagnostic, so it is not promoted as meeting the target.
The internal guest path was unreliable for fresh clients and is not the public
entry point. No view or threshold has been changed to conceal a failing result.

During an observed Cloud reboot, URL navigation took 28,067.3 ms to widget paint.
It began 9,352.5 ms after the reboot request; total reboot-to-paint was 37,419.8 ms.
This is one reboot observation, rather than a guarantee for hibernation wakeups.
Cold process CI measurements do not include the provider's provisioning delay.

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
