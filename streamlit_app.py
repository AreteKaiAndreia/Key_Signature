# SPDX-License-Identifier: AGPL-3.0-only
"""Streamlit key detection with the open-trained model and Essentia."""

from __future__ import annotations

import hashlib
from html import escape
import os
from pathlib import Path
from threading import Lock

import requests
import streamlit as st

from keyclf.cloud_audio import decode_clip
from keyclf.cloud_engines import custom_scores, essentia_scores, local_model, rank_three, released_model
from keyclf.keys import PITCH_NAMES
from keyclf.open_candidate import CandidateKeyModel
from keyclf.studio import STUDIO_HEADER, studio_css
from threadpoolctl import threadpool_limits

ML = "Research ML (random forest)"
OPEN_ML = "Custom ML (open-trained)"
KS = "Essentia (Krumhansl–Schmuckler)"
LOCAL = Path(__file__).resolve().parent / "models" / "cloud_model.joblib"
OPEN_ARTIFACT = Path(__file__).resolve().parent / "keyclf" / "open_key_model.json"
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


@st.cache_resource(show_spinner=False)
def get_open_model() -> CandidateKeyModel:
    return CandidateKeyModel.load(OPEN_ARTIFACT)


def render_readout(stage: str, result=None, message: str = "", *, essentia: bool = False) -> str:
    """Render the same readout through empty, loading, error, and full states."""
    header = '<section class="studio-result" aria-labelledby="readout-title"><h2 class="studio-panel-title" id="readout-title">Detected key</h2>'
    if stage == "loading":
        return header + '<div class="studio-wait" aria-hidden="true"></div><p class="studio-empty" role="status">Analyzing audio…</p></section>'
    if result is None:
        role = ' role="alert"' if stage == "error" else ""
        return f'{header}<p class="studio-empty"{role}>{escape(message)}</p></section>'

    ranking, profile, clipped, model_label = result
    key, score = ranking[0]
    score_format = ".3f" if essentia else ".1%"
    label = "match score" if essentia else "model score"
    rows = "".join(f'<tr><th scope="row">{escape(name)}</th><td class="studio-measure">{value:{score_format}}</td></tr>'
                   for name, value in ranking[1:])
    # Use the full 0–100% scale for each pitch class and expose exact readings.
    bars = "".join(
        f'<div class="studio-pitch" aria-label="{pitch}: {value:.1%}" title="{pitch}: {value:.1%}">'
        f'<div class="studio-track" aria-hidden="true"><div class="studio-bar" style="--pitch: {value:.2%}"></div></div>'
        f'<span>{pitch}</span></div>'
        for pitch, value in zip(PITCH_NAMES, profile, strict=True)
    )
    notices = ""
    if essentia:
        notices += '<p class="studio-confidence">Match scores measure correlation with each key profile.</p>'
    if clipped:
        notices += '<p class="studio-confidence">Only the first 120 seconds were analyzed.</p>'
    if model_label:
        notices += f'<p class="studio-confidence">{escape(model_label)}</p>'
    return f"""{header}
      <p class="studio-reading">{escape(key)}</p>
      <p class="studio-confidence"><strong class="studio-measure">{score:{score_format}}</strong> {label}</p>
      <table class="studio-alternates"><caption>Alternate keys</caption><tbody>{rows}</tbody></table>
      <figure class="studio-chart" aria-label="Pitch-class distribution">
        <figcaption>Chroma distribution</figcaption><div class="studio-pitches">{bars}</div>
      </figure>
      {notices}
    </section>"""


def main() -> None:
    """Offer both public engines and the research forest when configured."""
    st.set_page_config(page_title="Key Signature Classifier", layout="centered")
    st.html(studio_css())
    st.html(STUDIO_HEADER)
    allow_local = os.environ.get("KEYCLF_LOCAL_MODEL") == "1"
    model_url = os.environ.get("KEYCLF_MODEL_URL", "")
    model_sha = os.environ.get("KEYCLF_MODEL_SHA256", "")
    engines = [OPEN_ML, KS]
    if (allow_local and LOCAL.is_file()) or (model_url and model_sha):
        engines.append(ML)
    with st.container(key="console_workbench"):
        controls, output = st.columns([.86, 1.14], gap="large")
        with controls, st.container(key="console_input"):
            st.html('<h2 class="studio-panel-title">Audio</h2>')
            with st.container(key="track_input"):
                source = st.radio("Audio source", ["Upload MP3/MP4", "Record a soundbite"],
                                  horizontal=True, label_visibility="collapsed")
                if source == "Record a soundbite":
                    clip = st.audio_input("Record a soundbite", sample_rate=44100, label_visibility="collapsed")
                    st.caption("Allow microphone access when prompted. If it’s blocked, use the file upload option.")
                else:
                    clip = st.file_uploader("Upload a track", type=["mp3", "mp4"], label_visibility="collapsed")
                st.caption("Audio is sent to this server for analysis. It is not saved or used for training.")
            engine = st.selectbox("Prediction engine", engines)
            content = clip.getvalue() if clip is not None else b""
            analyze = st.button("Analyze", type="primary", disabled=not content)
        readout = output.empty()
    identity = hashlib.sha256(content).hexdigest() if content else None
    if st.session_state.get("clip_id") != identity:
        st.session_state["clip_id"] = identity
        st.session_state["predictions"] = {}
    predictions = st.session_state.setdefault("predictions", {})
    error = ""
    if analyze:
        predictions.pop(engine, None)
        slot = inference_slot()
        if not slot.acquire(blocking=False):
            error = "The server is analyzing another clip. Try again in a moment."
        else:
            try:
                readout.html(render_readout("loading"))
                model_label = None
                if engine == ML:
                    revision = LOCAL.stat().st_mtime_ns if allow_local and LOCAL.is_file() else 0
                    model, variant, version = get_model(model_url, model_sha, allow_local, revision)
                    model_label = f"{variant} ({version})"
                samples, clipped = decode_clip(content, "recording.wav" if source == "Record a soundbite" else clip.name)
                if engine == OPEN_ML:
                    with threadpool_limits(limits=1):
                        profile, scores = get_open_model().score_audio(samples)
                elif engine == KS:
                    profile, scores = essentia_scores(samples)
                else:
                    profile, scores = custom_scores(model, samples)
                predictions[engine] = (rank_three(scores), profile.tolist(), clipped, model_label)
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                error = str(exc)
            except (OSError, ImportError, requests.RequestException):
                error = "Analysis failed. Try another clip or switch engines."
            finally:
                slot.release()
    if error:
        readout.html(render_readout("error", message=error))
    elif engine in predictions:
        readout.html(render_readout("result", predictions[engine], essentia=engine == KS))
    else:
        message = "Select Analyze to find its key." if content else "Upload a track or record a soundbite to find its key."
        readout.html(render_readout("empty", message=message))
    st.html(f"""<footer class="studio-footer">
      <a href="{SOURCE}/blob/main/DEPLOYMENT.md">Corresponding source and build instructions</a>
      <a href="https://essentia.upf.edu/licensing_information.html">Essentia license</a>
    </footer>""")


if __name__ == "__main__":
    main()
