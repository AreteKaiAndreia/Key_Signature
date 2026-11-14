"""Streamlit upload demo for the trained key-signature classifier."""

from __future__ import annotations

import tempfile
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from keyclf.features import FeatureConfig, extract_features, feature_names
from keyclf.keys import KEY_NAMES, is_related
from keyclf.studio import STUDIO_CSS, STUDIO_HEADER


@st.cache_resource
def load_artifact(path: str = "models/model.joblib"):
    """Load and cache the exported estimator plus metadata."""
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        return artifact["model"], artifact
    return artifact, {}


def probability_24(estimator, vector: np.ndarray, metadata: dict) -> np.ndarray:
    """Map model probability columns to canonical key order."""
    names = metadata.get("feature_names")
    sample = (
        pd.DataFrame([vector], columns=names)[list(estimator.feature_names_in_)]
        if names and hasattr(estimator, "feature_names_in_")
        else vector.reshape(1, -1)
    )
    raw = np.asarray(estimator.predict_proba(sample), dtype=float)[0]
    result = np.zeros(24, dtype=float)
    class_names = metadata.get("classes")
    for column, value in enumerate(estimator.classes_):
        key = KEY_NAMES.index(class_names[int(value)]) if class_names and isinstance(value, (int, np.integer)) else int(value)
        result[key] += raw[column]
    return result


def prepare_input(path: str | Path, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return display features and the artifact's correctly ordered model input."""
    if metadata.get("research_version"):
        requested = metadata["feature_names"]
        base_names = feature_names()
        if set(requested) <= set(base_names):
            base = extract_features(path, FeatureConfig(**metadata.get("feature_config", {})))
            return base, base[[base_names.index(name) for name in requested]]
        from keyclf.research_features import BLOCKS, extract_research_features

        full, _ = extract_research_features(path)
        names = [f"{block}_{pc}" for block in BLOCKS for pc in range(12)]
        columns = [names.index(name) for name in requested]
        return full[:36], full[columns]
    vector = extract_features(path, FeatureConfig(**metadata.get("feature_config", {})))
    return vector, vector


def render_result(vector: np.ndarray, probability: np.ndarray) -> None:
    """Lead with the detected key and keep diagnostic evidence secondary."""
    order = np.argsort(-probability, kind="stable")[:3]
    best = int(order[0])
    st.html(f"""<section class="studio-result" aria-label="Detected key">
      <p class="studio-kicker">Detected key</p>
      <h2>{KEY_NAMES[best]}</h2>
      <p class="studio-confidence"><strong>{probability[best]:.1%}</strong> model confidence</p>
    </section>""")
    with st.container(key="result_details"):
        st.html('<h3 class="studio-section">Top three keys</h3>')
        for rank, key in enumerate(order, start=1):
            related = " — related key" if key != best and is_related(best, int(key)) else ""
            st.markdown(f"{rank}. **{KEY_NAMES[int(key)]}** — {probability[int(key)]:.1%}{related}")
        st.html('<h3 class="studio-section">Chroma distribution</h3>')
        chart_data = pd.DataFrame({"pitch class": [name.split()[0] for name in KEY_NAMES[:12]],
                                   "energy": vector[24:36]})
        chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#b8db84").encode(
            x=alt.X("pitch class:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("energy:Q", title=None, axis=alt.Axis(format="%", tickCount=3)),
            tooltip=[alt.Tooltip("pitch class:N"), alt.Tooltip("energy:Q", format=".1%")],
        ).properties(height=125)
        st.altair_chart(chart, width="stretch")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Key Signature Classifier", page_icon="🎼", layout="centered")
    st.html(STUDIO_CSS)
    st.html(STUDIO_HEADER)
    with st.container(key="track_input", border=True):
        st.html('<h2 class="studio-upload-title">Upload a track</h2>'
                '<p class="studio-upload-hint">Choose a file to detect its key automatically.</p>')
        upload = st.file_uploader("Upload a track", type=["mp3", "mp4", "wav"], label_visibility="collapsed")
    available = {label: Path(filename) for label, filename in (
        ("Original", "models/model.joblib"), ("Research", "models/research_model.joblib"),
        ("FMAK pilot (experimental)", "models/fmak_model.joblib"),
    ) if Path(filename).exists()}
    if not available:
        st.error("Train the model first with `python -m keyclf.train`.")
        return
    options = list(available)
    default = options.index("Research") if "Research" in available else 0
    choice = st.selectbox("Classifier", options, index=default) if len(options) > 1 else options[0]
    model_path = available[choice]
    if upload is None:
        return
    suffix = Path(upload.name).suffix.lower() or ".audio"
    try:
        with st.spinner("Analyzing your track…", show_time=True):
            estimator, metadata = load_artifact(str(model_path))
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                temporary.write(upload.getbuffer())
                temporary.flush()
                vector, model_input = prepare_input(temporary.name, metadata)
            probability = probability_24(estimator, model_input, metadata)
    except (OSError, RuntimeError, ValueError):
        st.error("This track couldn’t be analyzed. Try a valid MP3, MP4 with audio, or WAV file.")
        return
    render_result(vector, probability)


if __name__ == "__main__":
    main()
