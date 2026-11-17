# SPDX-License-Identifier: AGPL-3.0-only
"""Streamlit key detection with open-trained models and Essentia."""

from __future__ import annotations

import hashlib
from html import escape
import os
from pathlib import Path
from threading import Lock

import requests
import streamlit as st

from keyclf.cloud_audio import SAMPLE_RATE, decode_clip
from keyclf.cloud_engines import custom_scores, essentia_scores, local_model, released_model
from keyclf.keys import PITCH_NAMES, key_name
from keyclf.method_details import (
    BOOSTING, FOREST, KS, LINEAR_SVM, METHODS, OPEN_ML, SOFT_VOTE, render_method_details,
)
from keyclf.microphone import MIN_CHROMA_SPREAD, NO_SIGNAL, clean_recording
from keyclf.open_candidate import CandidateKeyModel
from keyclf.open_ensemble import SoftVotingKeyModel
from keyclf.open_linear_svm import LinearKeySVM
from keyclf.open_tree_models import load_tree_models, tree_scores
from keyclf.relative_pairs import rank_relative_pairs
from keyclf.studio import STUDIO_HEADER, studio_css
from keyclf.streaming_recorder import Recording, record_soundbite
from threadpoolctl import threadpool_limits

ML = "Research ML (random forest)"
TREE_ENGINES = {FOREST: "forest", BOOSTING: "boosting"}
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


@st.cache_resource(show_spinner=False)
def get_linear_svm() -> LinearKeySVM:
    """Cache the published SVM coefficients for the hosted app."""
    return LinearKeySVM.load()


@st.cache_resource(show_spinner=False)
def get_tree_models() -> dict:
    """Cache the verified set of public tree models."""
    return load_tree_models()


@st.cache_resource(show_spinner=False)
def get_ensemble() -> SoftVotingKeyModel:
    """Load the voting calibration shipped with the public engines."""
    return SoftVotingKeyModel.load(OPEN_ARTIFACT.with_name("soft_vote_calibration.json"))


def score_clip(content, filename, engine, recording, research_config):
    """Apply the selected engine to either a preview or a complete clip."""
    try:
        samples, clipped = decode_clip(content, filename)
        if recording:
            samples = clean_recording(samples, SAMPLE_RATE)
        model_label = None
        with threadpool_limits(limits=1):
            if engine in (OPEN_ML, LINEAR_SVM):
                model = get_open_model() if engine == OPEN_ML else get_linear_svm()
                profile, scores = model.score_audio(samples)
            elif engine in TREE_ENGINES:
                profile, scores = tree_scores(get_tree_models()[TREE_ENGINES[engine]], samples)
            elif engine == KS:
                profile, scores = essentia_scores(samples)
            elif engine == SOFT_VOTE:
                trees = get_tree_models()
                scorers = {
                    "current": get_open_model().scores, "linear": get_linear_svm().scores,
                    "forest": trees["forest"].predict_proba, "boosting": trees["boosting"].predict_proba,
                }
                profile, scores = get_ensemble().score_audio(samples, feature_scorers=scorers)
            else:
                allow_local, model_url, model_sha = research_config
                revision = LOCAL.stat().st_mtime_ns if allow_local and LOCAL.is_file() else 0
                model, variant, version = get_model(model_url, model_sha, allow_local, revision)
                model_label = f"{variant} ({version})"
                profile, scores = custom_scores(model, samples)
        if recording and max(profile) - min(profile) < MIN_CHROMA_SPREAD:
            raise ValueError(NO_SIGNAL)
        return (rank_relative_pairs(scores)[:3], profile.tolist(), clipped, model_label), ""
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return None, str(exc)
    except (OSError, ImportError, requests.RequestException):
        return None, "Analysis failed. Try another clip or switch engines."


def render_readout(stage: str, result=None, message: str = "", *, engine: str = OPEN_ML,
                   ack: str = "", preview_duration: float = 0) -> str:
    """Render the same readout through empty, loading, error, and full states."""
    title = "Live key prediction" if stage in ("live", "listening") else "Detected key pair"
    classes = "studio-result studio-result-live" if stage == "live" else "studio-result"
    header = (f'<section class="{classes}" aria-labelledby="readout-title"'
              f' data-engine="{escape(engine)}" data-preview-ack="{escape(ack)}">'
              f'<h2 class="studio-panel-title" id="readout-title">{title}</h2>')
    if stage == "loading":
        return header + '<div class="studio-wait" aria-hidden="true"></div><p class="studio-empty" role="status">Analyzing audio…</p></section>'
    if result is None:
        role = ' role="alert"' if stage == "error" else ' role="status"' if stage == "listening" else ""
        return f'{header}<p class="studio-empty"{role}>{escape(message)}</p></section>'

    ranking, profile, clipped, model_label = result
    key, relative, score = ranking[0]
    if engine == KS:
        score_format, label = ".3f", "higher match score in pair"
    elif engine == LINEAR_SVM:
        score_format, label = ".2f", "higher SVM ranking score in pair"
    else:
        score_format, label = ".1%", "higher model score in pair"
    rows = "".join(
        f'<tr><th scope="row">{key_name(first)} / {key_name(second)}</th>'
        f'<td class="studio-measure">{value:{score_format}}</td></tr>'
        for first, second, value in ranking[1:]
    )
    # Use the full 0–100% scale for each pitch class and expose exact readings.
    bars = "".join(
        f'<div class="studio-pitch" aria-label="{pitch}: {value:.1%}" title="{pitch}: {value:.1%}">'
        f'<div class="studio-track" aria-hidden="true"><div class="studio-bar" style="--pitch: {value:.2%}"></div></div>'
        f'<span>{pitch}</span></div>'
        for pitch, value in zip(PITCH_NAMES, profile, strict=True)
    )
    notices = ""
    if stage == "live" and preview_duration:
        notices += (f'<p class="studio-confidence" data-preview-duration="{preview_duration:.3f}">'
                    f'Based on {preview_duration:.1f} seconds so far.</p>')
    if engine == KS:
        notices += '<p class="studio-confidence">Match scores measure correlation with each key profile.</p>'
    if clipped:
        notices += '<p class="studio-confidence">Only the first 120 seconds were analyzed.</p>'
    if model_label:
        notices += f'<p class="studio-confidence">{escape(model_label)}</p>'
    chart = "" if stage == "live" else f"""
      <figure class="studio-chart" aria-label="Pitch-class distribution">
        <figcaption>Chroma distribution</figcaption><div class="studio-pitches">{bars}</div>
      </figure>"""
    return f"""{header}
      <p class="studio-reading studio-pair"><span>{key_name(key)}</span> <span>{key_name(relative)}</span></p>
      <p class="studio-confidence"><strong class="studio-measure">{score:{score_format}}</strong> {label}</p>
      <table class="studio-alternates"><caption>Alternate key pairs</caption><tbody>{rows}</tbody></table>
      {chart}
      {notices}
    </section>"""


def main() -> None:
    """Offer the public engines and the research forest when configured."""
    st.set_page_config(page_title="Key Signature Classifier", layout="centered")
    st.html(studio_css())
    st.html(STUDIO_HEADER)
    allow_local = os.environ.get("KEYCLF_LOCAL_MODEL") == "1"
    model_url = os.environ.get("KEYCLF_MODEL_URL", "")
    model_sha = os.environ.get("KEYCLF_MODEL_SHA256", "")
    engines = list(METHODS)
    if (allow_local and LOCAL.is_file()) or (model_url and model_sha):
        engines.append(ML)
    with st.container(key="console_workbench"):
        controls, output = st.columns([.86, 1.14], gap="large")
        with controls, st.container(key="console_input"):
            st.html('<h2 class="studio-panel-title">Audio</h2>')
            with st.container(key="track_input"):
                source = st.radio("Audio source", ["Upload MP3/MP4", "Record a soundbite"],
                                  horizontal=True, label_visibility="collapsed")
                recording = source == "Record a soundbite"
                mic = Recording()
                if recording:
                    mic = record_soundbite()
                    content, filename = mic.content, "recording.wav"
                    st.caption("Record 5–15 seconds close to the music. Include a few notes or chords.")
                    st.caption("Live predictions use the recording so far. Audio is used only for analysis during this session.")
                else:
                    clip = st.file_uploader("Upload a track", type=["mp3", "mp4"], label_visibility="collapsed")
                    content, filename = (clip.getvalue(), clip.name) if clip is not None else (b"", "")
                    st.caption("Audio is sent to this server for analysis. It is not saved or used for training.")
            engine = st.selectbox("Prediction engine", engines, filter_mode=None)
        readout = output.empty()
    if engine in METHODS:
        st.html(render_method_details(engine))
    predictions = st.session_state.setdefault("automatic_key_predictions", {})
    identity = (source, hashlib.sha256(content).hexdigest()) if content else None
    if st.session_state.get("clip_id") != identity:
        st.session_state["clip_id"] = identity
        predictions.clear()
    research_config = allow_local, model_url, model_sha
    if mic.take_id != st.session_state.get("preview_take"):
        st.session_state.update(preview_take=mic.take_id, preview_ack="", live_key_result=None,
                                live_preview_duration=0)
    if mic.active:
        ack = st.session_state["preview_ack"]
        if mic.token and mic.token != ack:
            slot = inference_slot()
            if mic.preview and slot.acquire(blocking=False):
                try:
                    result, _ = score_clip(mic.preview, "recording.wav", engine, True, research_config)
                    st.session_state["live_key_result"] = (engine, result)
                    st.session_state["live_preview_duration"] = mic.preview_duration
                finally:
                    slot.release()
            ack = st.session_state["preview_ack"] = mic.token
        live_engine, live_result = st.session_state["live_key_result"] or (None, None)
        if live_result and live_engine == engine:
            readout.html(render_readout("live", live_result, engine=engine, ack=ack,
                                        preview_duration=st.session_state.get("live_preview_duration", 0)))
        else:
            readout.html(render_readout("listening", message="Listening for clear musical notes…", engine=engine, ack=ack))
    elif content:
        if engine not in predictions:
            readout.html(render_readout("loading", engine=engine))
            with inference_slot():
                predictions[engine] = score_clip(content, filename, engine, recording, research_config)
        result, error = predictions[engine]
        readout.html(render_readout("error" if error else "result", result, message=error, engine=engine))
    else:
        readout.html(render_readout("empty", message="Upload a track or record a soundbite to find its key.", engine=engine))
    st.html(f"""<footer class="studio-footer">
      <a href="{SOURCE}/blob/main/DEPLOYMENT.md">Corresponding source and build instructions</a>
      <a href="https://essentia.upf.edu/licensing_information.html">Essentia license</a>
    </footer>""")


if __name__ == "__main__":
    main()
