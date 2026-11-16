const audioTypes = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"];

function addPitchEnergy(spectrum, binWidth, energy) {
  const first = Math.ceil(65 / binWidth);
  const last = Math.min(spectrum.length - 2, Math.floor(5000 / binWidth));
  let peak = -Infinity;
  for (let i = first; i <= last; i += 1) {
    peak = Math.max(peak, spectrum[i]);
  }
  const cutoff = Math.max(-68, peak - 25);
  for (let i = first; i <= last; i += 1) {
    const db = spectrum[i];
    const left = spectrum[i - 1];
    const right = spectrum[i + 1];
    if (db < cutoff || db <= left || db <= right) continue;

    // Compare with energy outside the peak to suppress broadband noise.
    const nearbyLeft = spectrum[Math.max(0, i - 4)];
    const nearbyRight = spectrum[i + 4] ?? -Infinity;
    if (db - (nearbyLeft + nearbyRight) / 2 < 9) continue;

    const offset = Number.isFinite(left + right) ? 0.5 * (left - right) / (left - 2 * db + right) : 0;
    const midi = Math.round(69 + 12 * Math.log2((i + offset) * binWidth / 440));
    energy[((midi % 12) + 12) % 12] += 10 ** (db / 10);
  }
}

function encodeBlob(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",", 2)[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

export default function ({parentElement, data, setStateValue}) {
  const record = parentElement.querySelector("[data-record]");
  const clear = parentElement.querySelector("[data-clear]");
  const elapsed = parentElement.querySelector("[data-elapsed]");
  const playback = parentElement.querySelector("audio");
  const chart = parentElement.querySelector("figure");
  const message = parentElement.querySelector("[role=status]");
  const pitches = [...chart.querySelectorAll("[data-pitch]")];
  const bars = pitches.map(pitch => pitch.querySelector(".studio-bar"));
  const energy = new Float64Array(pitches.length);
  let stream, context, recorder, refresh, timeout, playbackURL;
  let disposed = false;

  function showMessage(text) {
    message.textContent = text;
    message.hidden = !text;
  }

  function drawBars() {
    const total = energy.reduce((sum, value) => sum + value, 0);
    bars.forEach((bar, index) => {
      const percent = total ? 100 * energy[index] / total : 0;
      bar.style.setProperty("--pitch", percent.toFixed(2) + "%");
      const description = pitches[index].dataset.pitch + ": " + percent.toFixed(1) + "%";
      pitches[index].title = description;
      pitches[index].setAttribute("aria-label", description);
    });
  }

  function releaseCapture() {
    clearInterval(refresh);
    clearTimeout(timeout);
    stream?.getTracks().forEach(track => track.stop());
    if (context && context.state !== "closed") context.close().catch(() => {});
    stream = context = null;
  }

  function removePlayback() {
    if (!playbackURL) return;
    playback.pause();
    playback.removeAttribute("src");
    playback.load();
    URL.revokeObjectURL(playbackURL);
    playbackURL = null;
    playback.hidden = clear.hidden = true;
    setStateValue("clip", null);
  }

  function stop() {
    if (recorder?.state !== "recording") return;
    record.disabled = true;
    recorder.stop();
    releaseCapture();
  }

  async function start() {
    record.disabled = clear.disabled = true;
    showMessage("");
    try {
      const mimeType = typeof MediaRecorder !== "undefined" && audioTypes.find(type => MediaRecorder.isTypeSupported(type));
      if (!navigator.mediaDevices?.getUserMedia || typeof AudioContext === "undefined" || !mimeType) {
        showMessage("Microphone recording is unavailable here. Upload an MP3 or MP4 instead.");
        return;
      }
      context = new AudioContext();
      const input = await navigator.mediaDevices.getUserMedia({audio: true});
      if (disposed) {
        input.getTracks().forEach(track => track.stop());
        return;
      }
      stream = input;
      await context.resume();
      if (disposed) return;
      const analyser = context.createAnalyser();
      analyser.fftSize = 8192;
      analyser.smoothingTimeConstant = 0.25;
      context.createMediaStreamSource(stream).connect(analyser);
      const frequencies = new Float32Array(analyser.frequencyBinCount);
      const binWidth = context.sampleRate / analyser.fftSize;
      const chunks = [];
      let failed = false;
      recorder = new MediaRecorder(stream, {mimeType, audioBitsPerSecond: 128000});
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = () => {
        failed = true;
        releaseCapture();
        if (!disposed) showMessage("The microphone stopped unexpectedly. Try recording again.");
      };
      recorder.onstop = async () => {
        releaseCapture();
        chart.hidden = true;
        record.textContent = "Record";
        try {
          if (disposed || failed) return;
          const blob = new Blob(chunks, {type: recorder.mimeType});
          if (!blob.size || blob.size > data.maxBytes) throw new Error("Invalid recording size");
          const audio = await encodeBlob(blob);
          if (disposed) return;
          playbackURL = URL.createObjectURL(blob);
          playback.src = playbackURL;
          playback.hidden = clear.hidden = false;
          setStateValue("clip", {mime: blob.type.split(";")[0], audio});
        } catch {
          if (!disposed) showMessage("The recording could not be read. Try a shorter soundbite.");
        } finally {
          record.disabled = clear.disabled = false;
        }
      };
      recorder.start();
      removePlayback();
      energy.fill(0);
      drawBars();
      chart.hidden = false;
      elapsed.textContent = "00:00";
      record.textContent = "Stop recording";
      const started = performance.now();
      refresh = setInterval(() => {
        analyser.getFloatFrequencyData(frequencies);
        addPitchEnergy(frequencies, binWidth, energy);
        drawBars();
        const seconds = Math.floor((performance.now() - started) / 1000);
        elapsed.textContent = [Math.floor(seconds / 60), seconds % 60].map(n => String(n).padStart(2, "0")).join(":");
      }, 100);
      timeout = setTimeout(stop, data.maxSeconds * 1000);
    } catch (error) {
      releaseCapture();
      if (!disposed) showMessage(error.name === "NotAllowedError"
        ? "Allow microphone access in your browser and try again."
        : "Could not start the microphone. Check your input device and try again.");
    } finally {
      if (!disposed) record.disabled = clear.disabled = false;
    }
  }

  record.onclick = () => recorder?.state === "recording" ? stop() : start();
  clear.onclick = () => {
    removePlayback();
    elapsed.textContent = "00:00";
    showMessage("");
  };
  return () => {
    disposed = true;
    record.onclick = clear.onclick = null;
    stop();
    releaseCapture();
    if (playbackURL) URL.revokeObjectURL(playbackURL);
  };
}
