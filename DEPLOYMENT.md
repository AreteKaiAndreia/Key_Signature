# Running and deploying the apps

## Two-engine app for Community Cloud

From the repository root, after the local feature caches and the deployment
weights have been built:

```bash
python -m pip install -r requirements.txt
KEYCLF_LOCAL_MODEL=1 python -m streamlit run streamlit_app.py
```

Use Python 3.11 with FFmpeg available. `requirements.txt` pins the Linux build
and the compatible Essentia wheel. The source-only Linux image can be built
with `docker build --platform linux/amd64 -t keyclf-linux-check .`.
The local opt-in looks only for `models/cloud_model.joblib`; omit it for public
hosting. The app can accept a recorded WAV or an MP3/MP4 upload, analyze up to
the first 120 seconds, and keep the results in the current session when
switching between the forest and Essentia.

**Public deployment is blocked by weights clearance.** Once a hosting and
redistribution basis for the exact weights is recorded in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md):

1. Upload the cleared compressed artifact as a versioned GitHub Release asset
   in [AreteKaiAndreia/Key_Signature](https://github.com/AreteKaiAndreia/Key_Signature).
   Record its SHA-256 and the corresponding source commit.
2. In [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
   deploy `AreteKaiAndreia/Key_Signature`, branch `main`, file
   `streamlit_app.py`, Python 3.11. The owner completes any GitHub OAuth prompt.
   In the app's secrets, set top-level `KEYCLF_MODEL_URL` to the GitHub Release
   asset URL and `KEYCLF_MODEL_SHA256` to the expected lower-case digest.
   No credentials, audio, labels, or feature caches belong in the repository or app.
3. Verify the release loads and both engines rank three keys in the hosted
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
