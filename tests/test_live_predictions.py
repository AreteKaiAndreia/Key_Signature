"""Keep the last live key result across temporary inference failures."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from keyclf.cloud_audio import SAMPLE_RATE
from keyclf.microphone import NO_SIGNAL

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_live_result_survives_a_failed_update_and_resets_for_a_new_take():
    from streamlit.testing.v1 import AppTest

    from keyclf.streaming_recorder import Recording

    wave = BytesIO()
    sf.write(wave, np.full(3 * SAMPLE_RATE, 0.1), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    packet = wave.getvalue()
    major_c, major_g = np.zeros(24), np.zeros(24)
    major_c[0], major_g[7] = 1, 1
    profile = np.eye(12)[0]
    with (
        patch("keyclf.streaming_recorder.record_soundbite") as microphone,
        patch("keyclf.microphone.clean_recording", side_effect=lambda samples, _: samples),
        patch("keyclf.open_candidate.CandidateKeyModel.score_audio",
              side_effect=[(profile, major_c), ValueError(NO_SIGNAL), (profile, major_g)]) as predict,
    ):
        microphone.return_value = Recording(active=True, take_id="first-take")
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.radio[0].set_value("Record a soundbite").run()
        if app.checkbox:
            app.checkbox[0].uncheck().run()
        microphone.return_value = Recording(active=True, take_id="first-take", preview=packet, token="first/1")
        app.run()
        assert not app.exception
        result = next(item.value for item in app.get("html") if 'class="studio-reading studio-pair"' in item.value)
        assert "<span>C major</span> <span>A minor</span>" in result
        assert 'data-preview-duration="3.000"' in result

        microphone.return_value = Recording(active=True, take_id="first-take", preview=packet,
                                            token="first/2", preview_offset=3 * SAMPLE_RATE)
        app.run()
        assert not app.exception
        retained = next(item.value for item in app.get("html") if 'class="studio-reading studio-pair"' in item.value)
        assert "<span>C major</span> <span>A minor</span>" in retained
        assert 'data-preview-duration="3.000"' in retained
        assert 'data-preview-frames="132300"' in retained

        microphone.return_value = Recording(active=True, take_id="first-take", preview=packet,
                                            token="first/3", preview_offset=6 * SAMPLE_RATE)
        app.run()
        assert not app.exception
        updated = next(item.value for item in app.get("html") if 'class="studio-reading studio-pair"' in item.value)
        assert "<span>G major</span> <span>E minor</span>" in updated
        assert 'data-preview-duration="9.000"' in updated
        assert predict.call_count == 3

        microphone.return_value = Recording(active=True, take_id="second-take")
        app.run()
        assert not app.exception
        assert not any('class="studio-reading studio-pair"' in item.value for item in app.get("html"))
