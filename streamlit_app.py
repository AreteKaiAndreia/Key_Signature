# SPDX-License-Identifier: AGPL-3.0-only
"""Public-ready Streamlit entry point for recording or uploading a soundbite."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Lock

import pandas as pd
import requests
import streamlit as st

from keyclf.cloud_audio import decode_clip
from keyclf.cloud_engines import custom_scores, essentia_scores, local_model, rank_three, released_model
from keyclf.keys import PITCH_NAMES
from keyclf.studio import STUDIO_CSS, STUDIO_HEADER

ML = "Custom ML — chroma-only random forest"
KS = "Essentia — Krumhansl–Schmuckler"
LOCAL = Path(__file__).resolve().parent / "models" / "cloud_model.joblib"
SOURCE = "https://github.com/AreteKaiAndreia/Key_Signature"


@st.cache_resource
def inference_slot() -> Lock:
    """Avoid concurrent expensive analyses on the small hosted instance."""
    return Lock()


@st.cache_resource(show_spinner=False)
def get_model(url: str, sha256: str, allow_local: bool, revision: int):
    """Load one verified model per server or one explicitly enabled local model."""
    if url or sha256:
        return released_model(url, sha256)
    if allow_local:
        return local_model(LOCAL)
    raise FileNotFoundError("Custom ML is unavailable until the approved model is configured.")


def main() -> None:
    """Display two inference modes and an explicit analysis action."""
    st.set_page_config(page_title="Key Signature Classifier", layout="centered")
    st.html(STUDIO_CSS)
    st.html(STUDIO_HEADER)
    with st.container(key="track_input", border=True):
        source = st.radio("Audio source", ["Upload MP3/MP4", "Record a soundbite"], horizontal=True)
        if source == "Record a soundbite":
            st.caption("Allow microphone access when prompted. If it’s blocked, use the file upload option.")
            clip = st.audio_input("Record a soundbite", sample_rate=44100)
        else:
            clip = st.file_uploader("Upload a track", type=["mp3", "mp4"])
        st.caption("Audio is sent to this server for analysis. It is not saved or used for training.")
    engine = st.selectbox("Prediction engine", [ML, KS])
    content = clip.getvalue() if clip is not None else b""
    identity = hashlib.sha256(content).hexdigest() if content else None
    if st.session_state.get("clip_id") != identity:
        st.session_state["clip_id"] = identity
        st.session_state["predictions"] = {}
    predictions = st.session_state.setdefault("predictions", {})
    if st.button("Analyze", type="primary", disabled=not content):
        predictions.pop(engine, None)
        slot = inference_slot()
        if not slot.acquire(blocking=False):
            st.warning("The server is analyzing another clip. Try again in a moment.")
        else:
            try:
                model_label = None
                with st.spinner("Analyzing your soundbite…"):
                    if engine == ML:
                        allow_local = os.environ.get("KEYCLF_LOCAL_MODEL") == "1"
                        revision = LOCAL.stat().st_mtime_ns if allow_local and LOCAL.is_file() else 0
                        model, variant, model_version = get_model(
                            os.environ.get("KEYCLF_MODEL_URL", ""), os.environ.get("KEYCLF_MODEL_SHA256", ""),
                            allow_local, revision)
                        model_label = f"{variant} · {model_version}"
                    samples, clipped = decode_clip(content, "recording.wav" if source == "Record a soundbite" else clip.name)
                    profile, scores = custom_scores(model, samples) if engine == ML else essentia_scores(samples)
                    predictions[engine] = (rank_three(scores), profile.tolist(), clipped, model_label)
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                st.error(str(exc))
            except (OSError, ImportError, requests.RequestException):
                st.error("This engine could not complete analysis. Check the app configuration and try again.")
            finally:
                slot.release()
    if engine in predictions:
        ranking, profile, clipped, model_label = predictions[engine]
        if clipped:
            st.info("Only the first 120 seconds were analyzed.")
        if model_label:
            st.caption(f"Model: {model_label}")
        key, score = ranking[0]
        number = f"{score:.1%}" if engine == ML else f"{score:.3f}"
        label = "model score" if engine == ML else "match score"
        st.html(f"""<section class="studio-result" aria-label="Detected key">
          <p class="studio-kicker">Detected key</p><h2>{key}</h2>
          <p class="studio-confidence"><strong>{number}</strong> {label}</p>
        </section>""")
        st.html('<h3 class="studio-section">Top three keys</h3>')
        st.table(pd.DataFrame([
            {"Rank": i, "Key": key, "Model score" if engine == ML else "Match score":
             f"{score:.1%}" if engine == ML else f"{score:.3f}"}
            for i, (key, score) in enumerate(ranking, 1)
        ]).set_index("Rank"))
        st.caption("Match scores are correlations, not probabilities." if engine == KS
                   else "Forest scores rank the candidates; they are not calibrated confidence estimates.")
        st.html('<h3 class="studio-section">Pitch-class distribution</h3>')
        st.bar_chart(pd.DataFrame({"Pitch class": PITCH_NAMES, "Strength": profile}),
                     x="Pitch class", y="Strength", sort=False, x_label="Pitch class", height=145,
                     color="#b8db84")
    st.caption(f"[Corresponding source and build instructions]({SOURCE}/blob/main/DEPLOYMENT.md) · "
               "[Essentia license](https://essentia.upf.edu/licensing_information.html)")


if __name__ == "__main__":
    main()
