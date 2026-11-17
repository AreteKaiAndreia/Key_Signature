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

async function recordingPcm(blob, sampleRate, maxFrames) {
  const decoder = new OfflineAudioContext(1, 1, sampleRate);
  const audio = await decoder.decodeAudioData(await blob.arrayBuffer());
  const frames = Math.min(audio.length, maxFrames);
  const samples = new Uint8Array(2 * frames);
  const pcm = new DataView(samples.buffer);
  const channels = Array.from({length: audio.numberOfChannels}, (_, index) => audio.getChannelData(index));
  for (let i = 0; i < frames; i += 1) {
    let sample = 0;
    for (const channel of channels) sample += channel[i];
    sample = Math.max(-1, Math.min(1, sample / channels.length));
    pcm.setInt16(2 * i, Math.round(sample * 32767), true);
  }
  return samples;
}

function recordingWav(parts, sampleRate, pcmBytes) {
  const header = new DataView(new ArrayBuffer(44));
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) header.setUint8(offset + i, text.charCodeAt(i));
  };
  writeText(0, "RIFF");
  header.setUint32(4, 36 + pcmBytes, true);
  writeText(8, "WAVEfmt ");
  header.setUint32(16, 16, true);
  header.setUint16(20, 1, true);  // PCM
  header.setUint16(22, 1, true);  // mono
  header.setUint32(24, sampleRate, true);
  header.setUint32(28, 2 * sampleRate, true);
  header.setUint16(32, 2, true);
  header.setUint16(34, 16, true);
  writeText(36, "data");
  header.setUint32(40, pcmBytes, true);
  return new Blob([header, ...parts], {type: "audio/wav"});
}

export default function ({parentElement, data, setStateValue, setTriggerValue}) {
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
  let segment, segmentTimer, takeId;
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
    clearTimeout(segmentTimer);
    if (segment?.state === "recording") {
      segment.onstop = null;
      segment.stop();
    }
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
  }

  function nextPreview(id, mimeType, capture) {
    const active = () => !disposed && takeId === id && recorder?.state === "recording" && !capture.failed;
    const unavailable = () => {
      if (active()) showMessage("Live predictions are unavailable. The finished recording will still be analyzed.");
      capture.failed = true;
    };
    if (!active()) return;
    try {
      const preview = new MediaRecorder(stream, {mimeType, audioBitsPerSecond: 64000});
      segment = preview;
      const chunks = [];
      preview.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      preview.onstop = () => {
        if (!active()) return;
        nextPreview(id, mimeType, capture);
        // Add every segment in capture order, even when the server is busy.
        capture.queue = capture.queue.then(async () => {
          if (!active()) return;
          const remaining = data.maxSeconds * data.sampleRate - capture.bytes / 2;
          if (!remaining) return;
          const blob = new Blob(chunks, {type: preview.mimeType});
          const samples = await recordingPcm(blob, data.sampleRate, remaining);
          if (!active() || !samples.length) return;
          capture.parts.push(samples);
          capture.bytes += samples.length;

          const acknowledged = document.querySelector("[data-preview-ack]")?.dataset.previewAck;
          if (capture.pending && acknowledged !== capture.pending && performance.now() - capture.sentAt < 30000) return;
          const token = id + "/" + (++capture.sequence);
          capture.pending = token;
          capture.sentAt = performance.now();
          const wav = recordingWav(capture.parts, data.sampleRate, capture.bytes);
          const audio = await encodeBlob(wav);
          if (active()) setTriggerValue("preview", {id, token, audio});
        }).catch(unavailable);
      };
      preview.start();
      segmentTimer = setTimeout(() => {
        if (preview.state === "recording") preview.stop();
      }, data.previewSeconds * 1000);
    } catch {
      unavailable();
    }
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
      if (!navigator.mediaDevices?.getUserMedia || typeof AudioContext === "undefined" ||
          typeof OfflineAudioContext === "undefined" || !mimeType) {
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
      const id = crypto.randomUUID();
      takeId = id;
      const capture = {parts: [], bytes: 0, queue: Promise.resolve(), sequence: 0};
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
          if (disposed) return;
          if (failed) {
            setStateValue("take", {id, active: false});
            return;
          }
          const blob = new Blob(chunks, {type: recorder.mimeType});
          if (!blob.size || blob.size > data.maxBytes) throw new Error("Invalid recording size");
          const samples = await recordingPcm(blob, data.sampleRate, data.maxSeconds * data.sampleRate);
          const wav = recordingWav([samples], data.sampleRate, samples.length);
          const audio = await encodeBlob(wav);
          if (disposed) return;
          playbackURL = URL.createObjectURL(wav);
          playback.src = playbackURL;
          playback.hidden = clear.hidden = false;
          setStateValue("take", {id, active: false, audio});
        } catch {
          if (!disposed) {
            setStateValue("take", {id, active: false});
            showMessage("The recording could not be read. Try a shorter soundbite.");
          }
        } finally {
          record.disabled = clear.disabled = false;
        }
      };
      recorder.start();
      removePlayback();
      setStateValue("take", {id, active: true});
      nextPreview(id, mimeType, capture);
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
    setStateValue("take", null);
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
