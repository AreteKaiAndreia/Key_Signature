"""Exercise noise handling with synthetic recordings and the Streamlit app."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from keyclf.cloud_audio import SAMPLE_RATE
from keyclf.microphone import NO_SIGNAL, clean_recording

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.mark.parametrize("hum_hz", [0, 50, 60])
def test_recording_rejects_room_noise_without_music(hum_hz):
    time = np.arange(6 * SAMPLE_RATE) / SAMPLE_RATE
    samples = 0.04 * np.random.default_rng(331).normal(size=time.size)
    if hum_hz:
        samples += 0.2 * np.sin(2 * np.pi * hum_hz * time + 0.6)
        samples += 0.08 * np.sin(4 * np.pi * hum_hz * time + 0.4)
    with pytest.raises(ValueError, match="enough clear musical notes"):
        clean_recording(samples.astype(np.float32), SAMPLE_RATE)


@pytest.mark.parametrize("engine", ["Neural Network", "Linear SVM"])
def test_noisy_microphone_wav_reaches_the_prediction_readout(engine):
    from streamlit.testing.v1 import AppTest

    from keyclf.streaming_recorder import Recording

    recorder_rate = 44100
    time = np.arange(5 * recorder_rate) / recorder_rate
    noise = 0.2 * np.random.default_rng(75).normal(size=time.size)
    chord = sum(0.10 * np.sin(2 * np.pi * hz * time) for hz in (261.626, 329.628, 391.995))
    chord *= (time >= 0.7) & (time < 4.3)
    recording = BytesIO()
    sf.write(recording, np.clip(chord + noise, -1, 1), recorder_rate, format="WAV")
    with patch("keyclf.streaming_recorder.record_soundbite", return_value=Recording(content=recording.getvalue())) as microphone:
        app = AppTest.from_file(APP, default_timeout=45).run()
        assert app.selectbox[0].value == "Neural Network"
        app.selectbox[0].set_value(engine).run()
        app.radio[0].set_value("Record a soundbite").run()
        assert not app.button
        assert not app.exception
        assert any("<span>C major</span> <span>A minor</span>" in item.value for item in app.get("html"))

        noise_only = BytesIO()
        sf.write(noise_only, noise, recorder_rate, format="WAV")
        microphone.return_value = Recording(content=noise_only.getvalue())
        app.run()
        assert not app.exception
        assert any(NO_SIGNAL in item.value and 'role="alert"' in item.value for item in app.get("html"))
        assert not any('class="studio-reading studio-pair"' in item.value for item in app.get("html"))


def test_live_audio_append_recovers_overlapping_retries():
    from keyclf.streaming_recorder import LiveAudio

    def packet(values):
        buffer = BytesIO()
        sf.write(buffer, np.asarray(values, dtype=np.int16), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    live = LiveAudio()
    live.append(0, packet([100, 200]))
    live.append(0, packet([100, 200]))
    live.append(3, packet([999]))  # Keep the prefix until the gap arrives.
    live.append(1, packet([800, 999]))  # A retry must agree with the existing samples.
    assert live.frames == 2

    live.append(1, packet([200, 300, 400]))
    assert live.frames == 4
    waveform, rate = sf.read(BytesIO(live.wav()), dtype="int16")
    assert rate == SAMPLE_RATE
    np.testing.assert_array_equal(waveform, [100, 200, 300, 400])
