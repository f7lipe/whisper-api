const API_BASE = "";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusBadge    = document.getElementById("statusBadge");
const statusText     = document.getElementById("statusText");
const engineSelector = document.getElementById("engineSelector");
const modelSelect    = document.getElementById("modelSelect");
const engineBadge    = document.getElementById("engineBadge");
const recordBtn      = document.getElementById("recordBtn");
const recordLabel    = document.getElementById("recordLabel");
const timerEl        = document.getElementById("timer");
const timerDisplay   = document.getElementById("timerDisplay");
const visualizer     = document.getElementById("visualizer");
const visualizerIdle = document.getElementById("visualizerIdle");
const fileInput      = document.getElementById("fileInput");
const dropZone       = document.getElementById("dropZone");
const dropContent    = document.getElementById("dropContent");
const uploadBtn      = document.getElementById("uploadBtn");
const loading        = document.getElementById("loading");
const resultCard     = document.getElementById("resultCard");
const resultText     = document.getElementById("resultText");
const resultMeta     = document.getElementById("resultMeta");
const copyBtn        = document.getElementById("copyBtn");
const clearBtn       = document.getElementById("clearBtn");
const errorBanner    = document.getElementById("errorBanner");
const errorMsg       = document.getElementById("errorMsg");

// ── State ─────────────────────────────────────────────────────────────────────
let mediaRecorder  = null;
let audioChunks    = [];
let timerInterval  = null;
let secondsElapsed = 0;
let audioCtx       = null;
let analyser       = null;
let animFrameId    = null;
let selectedFile   = null;
let backendsData   = {};
let activeBackend  = "openai-whisper";

// ── Backend & model selector ──────────────────────────────────────────────────
async function loadBackends() {
  try {
    const res  = await fetch(`${API_BASE}/api/backends`);
    backendsData = await res.json();
    renderEngineSelector();
  } catch {
    // If the server is not reachable yet we'll retry with health check
  }
}

function renderEngineSelector() {
  engineSelector.innerHTML = "";

  Object.entries(backendsData).forEach(([key, info]) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id   = `engine-${key}`;
    btn.className = "engine-btn" + (key === activeBackend ? " active" : "");
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", key === activeBackend ? "true" : "false");
    btn.innerHTML = `
      <span class="engine-name">${info.label}</span>
      <span class="engine-desc">${info.description}</span>
      <span class="engine-license">${info.license} · ${info.author}</span>
    `;
    btn.addEventListener("click", () => selectBackend(key));
    engineSelector.appendChild(btn);
  });

  updateModelSelect();
}

function selectBackend(key) {
  activeBackend = key;

  engineSelector.querySelectorAll(".engine-btn").forEach((btn) => {
    const isActive = btn.id === `engine-${key}`;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-checked", isActive ? "true" : "false");
  });

  updateModelSelect();
}

function updateModelSelect() {
  const info = backendsData[activeBackend];
  if (!info) return;

  modelSelect.innerHTML = "";
  info.models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === info.default_model) opt.selected = true;
    modelSelect.appendChild(opt);
  });

  const colorMap = {
    "openai-whisper": "#4f6ef7",
    "faster-whisper": "#10b981",
  };
  const color = colorMap[activeBackend] || "#4f6ef7";
  engineBadge.textContent = info.label;
  engineBadge.style.setProperty("--badge-color", color);
}

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    // backends shape: { "openai-whisper": { ready: bool }, "faster-whisper": { ready: bool } }
    const readyList = Object.entries(data.backends || {})
      .filter(([, v]) => v?.ready)
      .map(([k]) => k);
    const anyReady = readyList.length > 0;

    if (anyReady) {
      statusBadge.classList.add("ready");
      statusBadge.classList.remove("error");
      statusText.textContent = `Prontos: ${readyList.join(", ")}`;
      if (!Object.keys(backendsData).length) await loadBackends();
    } else {
      statusBadge.classList.remove("ready", "error");
      statusText.textContent = "Nenhum modelo carregado ainda";
      if (!Object.keys(backendsData).length) await loadBackends();
    }
  } catch {
    statusBadge.classList.add("error");
    statusText.textContent = "Servidor offline";
    setTimeout(checkHealth, 5000);
  }
}

// ── Timer ─────────────────────────────────────────────────────────────────────
function startTimer() {
  secondsElapsed = 0;
  timerEl.classList.remove("hidden");
  timerInterval = setInterval(() => {
    secondsElapsed++;
    const m = String(Math.floor(secondsElapsed / 60)).padStart(2, "0");
    const s = String(secondsElapsed % 60).padStart(2, "0");
    timerDisplay.textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerEl.classList.add("hidden");
  timerDisplay.textContent = "00:00";
}

// ── Audio visualizer ──────────────────────────────────────────────────────────
function startVisualizer(stream) {
  audioCtx = new AudioContext();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  const source = audioCtx.createMediaStreamSource(stream);
  source.connect(analyser);

  const bufferLength = analyser.frequencyBinCount;
  const dataArray    = new Uint8Array(bufferLength);
  const ctx          = visualizer.getContext("2d");

  visualizerIdle.style.opacity = "0";

  function draw() {
    animFrameId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);

    const { width, height } = visualizer;
    ctx.clearRect(0, 0, width, height);

    const barW = (width / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barH = (dataArray[i] / 255) * height;
      const hue  = 200 + (i / bufferLength) * 100;
      ctx.fillStyle = `hsla(${hue}, 90%, 65%, 0.85)`;
      ctx.beginPath();
      ctx.roundRect(x, height - barH, barW - 1, barH, 2);
      ctx.fill();
      x += barW + 1;
    }
  }
  draw();
}

function stopVisualizer() {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  if (audioCtx)   { audioCtx.close(); audioCtx = null; }
  const ctx = visualizer.getContext("2d");
  ctx.clearRect(0, 0, visualizer.width, visualizer.height);
  visualizerIdle.style.opacity = "1";
}

// ── Record ────────────────────────────────────────────────────────────────────
recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }

  hideError();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];

    const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"].find(
      (t) => MediaRecorder.isTypeSupported(t)
    ) || "";

    mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      stopVisualizer();
      stopTimer();
      setRecordingUI(false);

      const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      const ext  = (mediaRecorder.mimeType || "audio/webm").includes("ogg") ? "ogg" : "webm";
      await transcribe(blob, `recording.${ext}`);
    };

    mediaRecorder.start(250);
    startVisualizer(stream);
    startTimer();
    setRecordingUI(true);
  } catch (err) {
    showError("Não foi possível acessar o microfone: " + err.message);
  }
});

function setRecordingUI(isRecording) {
  recordBtn.classList.toggle("recording", isRecording);
  recordBtn.setAttribute("aria-pressed", isRecording);
  recordLabel.textContent = isRecording ? "Parar Gravação" : "Iniciar Gravação";
  document.querySelector(".icon-mic").classList.toggle("hidden", isRecording);
  document.querySelector(".icon-stop").classList.toggle("hidden", !isRecording);
}

// ── File upload ───────────────────────────────────────────────────────────────
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) setSelectedFile(file);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) setSelectedFile(file);
});

uploadBtn.addEventListener("click", async () => {
  if (selectedFile) await transcribe(selectedFile, selectedFile.name);
});

function setSelectedFile(file) {
  selectedFile = file;
  const name = file.name.length > 40 ? file.name.slice(0, 37) + "…" : file.name;
  dropContent.innerHTML = `
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
    <p style="color:var(--text)">${name}</p>
    <p class="drop-hint">${(file.size / 1024 / 1024).toFixed(2)} MB — clique para trocar</p>
  `;
  uploadBtn.classList.remove("hidden");
}

// ── Transcribe ────────────────────────────────────────────────────────────────
async function transcribe(blobOrFile, filename) {
  hideError();
  hideResult();
  loading.classList.remove("hidden");

  const form = new FormData();
  form.append("file", blobOrFile, filename);

  const selectedModel   = modelSelect.value;
  const selectedBackend = activeBackend;
  const url = `${API_BASE}/api/transcribe?backend=${encodeURIComponent(selectedBackend)}`;

  try {
    const res = await fetch(url, { method: "POST", body: form });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Erro desconhecido");
    }

    const data = await res.json();
    showResult(data, selectedBackend, selectedModel);
  } catch (err) {
    showError(err.message);
  } finally {
    loading.classList.add("hidden");
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function showResult({ text, language, duration, segments }, backend, model) {
  resultText.textContent = text || "(sem texto detectado)";

  const backendLabel = backendsData[backend]?.label || backend;

  const metaParts = [
    `🌐 ${language?.toUpperCase() || "?"}`,
    `⏱ ${duration?.toFixed(2)}s`,
    `🤖 ${backendLabel} · ${model}`,
    segments ? `📄 ${segments.length} segmento${segments.length !== 1 ? "s" : ""}` : null,
  ].filter(Boolean);

  resultMeta.innerHTML = metaParts.map((p) => `<span>${p}</span>`).join("");
  resultCard.classList.remove("hidden");

  setTimeout(() => resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" }), 50);
}

function hideResult() {
  resultCard.classList.add("hidden");
  resultText.textContent = "";
  resultMeta.innerHTML = "";
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function hideError() {
  errorBanner.classList.add("hidden");
}

copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(resultText.textContent);
    const orig = copyBtn.innerHTML;
    copyBtn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copiado!`;
    setTimeout(() => { copyBtn.innerHTML = orig; }, 2000);
  } catch {
    showError("Não foi possível copiar para a área de transferência.");
  }
});

clearBtn.addEventListener("click", hideResult);

// ── Init ──────────────────────────────────────────────────────────────────────
loadBackends();
checkHealth();
