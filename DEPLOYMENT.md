# Running and deploying the apps

## Public app on Community Cloud

[Public key classifier](https://musickeyanalyzer.streamlit.app/)

Deploy `AreteKaiAndreia/Key_Signature` from `main` with
`streamlit_app.py` as the entry point and **Python 3.11** in Advanced
settings. Leave Secrets empty. The app defaults to the custom classifier
bundled as `keyclf/open_key_model.json`. The engine selector also offers
Linear SVM (`keyclf/linear_svm.json`) and Essentia. All rank three relative-key pairs
by each pair's higher key score. They accept a microphone recording or an
MP3/MP4 upload, with a 120-second analysis limit.

The owner signs in at [Streamlit Community Cloud](https://share.streamlit.io/)
and completes any GitHub connection prompt. See the
[Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [code license notices](NOTICE.md).

To run the same app locally with Python 3.11 and FFmpeg:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

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
