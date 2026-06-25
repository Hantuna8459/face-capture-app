"use client";

import { useEffect, useRef, useState } from "react";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const STEP_HOLD_MS = 650;

const RECORDING_FORMATS = [
  {
    id: "mp4",
    label: "MP4",
    extension: "mp4",
    mimeTypes: [
      "video/mp4;codecs=avc1.42E01E",
      "video/mp4;codecs=h264",
      "video/mp4",
    ],
  },
  {
    id: "webm",
    label: "WebM",
    extension: "webm",
    mimeTypes: [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ],
  },
];

function getSupportedFormats() {
  if (typeof MediaRecorder === "undefined") {
    return [];
  }

  return RECORDING_FORMATS.map((format) => ({
    ...format,
    mimeType: format.mimeTypes.find((type) => MediaRecorder.isTypeSupported(type)),
  })).filter((format) => format.mimeType);
}

function getExtensionFromMimeType(type) {
  return type?.includes("mp4") ? "mp4" : "webm";
}

function getSupportedMimeType() {
  return getSupportedFormats()[0]?.mimeType || "";
}

function formatElapsed(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

const FACE_STEPS = [
  { id: "center", label: "Look center" },
  { id: "left", label: "Turn left" },
  { id: "right", label: "Turn right" },
  { id: "up", label: "Look up" },
  { id: "down", label: "Look down" },
];

const DEFAULT_FACE_STATE = {
  found: false,
  ready: false,
  instruction: "Start camera to detect face",
  detail: "Keep your face inside the guide.",
  warning: "",
  stepId: "center",
  completed: [],
  metrics: null,
};

const SESSION_DONE_FACE_STATE = {
  ...DEFAULT_FACE_STATE,
  instruction: "Session complete",
  detail: "Camera closed. Press Retry to record again.",
};

function getBounds(landmarks) {
  return landmarks.reduce(
    (bounds, point) => ({
      minX: Math.min(bounds.minX, point.x),
      maxX: Math.max(bounds.maxX, point.x),
      minY: Math.min(bounds.minY, point.y),
      maxY: Math.max(bounds.maxY, point.y),
    }),
    { minX: 1, maxX: 0, minY: 1, maxY: 0 },
  );
}

function analyzeFace(landmarks, currentStepId, completedSteps) {
  if (!landmarks?.length) {
    return {
      ...DEFAULT_FACE_STATE,
      instruction: "No face detected",
      detail: "Move into view and face the camera.",
      stepId: currentStepId,
      completed: [...completedSteps],
    };
  }

  const bounds = getBounds(landmarks);
  const width = bounds.maxX - bounds.minX;
  const height = bounds.maxY - bounds.minY;
  const centerX = bounds.minX + width / 2;
  const centerY = bounds.minY + height / 2;
  const nose = landmarks[1] || landmarks[4] || landmarks[0];
  const chin = landmarks[152] || landmarks[175] || landmarks[0];
  const forehead = landmarks[10] || landmarks[151] || landmarks[0];
  const yaw = (nose.x - centerX) / Math.max(width, 0.001);
  const pitch = (nose.y - centerY) / Math.max(height, 0.001);
  const verticalSpan = Math.abs(chin.y - forehead.y) / Math.max(height, 0.001);

  const warnings = [];
  if (bounds.minX < 0.06) warnings.push("Move right");
  if (bounds.maxX > 0.94) warnings.push("Move left");
  if (bounds.minY < 0.08) warnings.push("Move down");
  if (bounds.maxY > 0.96) warnings.push("Move up");
  if (width < 0.18) warnings.push("Move closer");
  if (width > 0.62 || height > 0.82) warnings.push("Move back");

  const centered =
    centerX > 0.32 &&
    centerX < 0.68 &&
    centerY > 0.28 &&
    centerY < 0.72 &&
    width >= 0.18 &&
    width <= 0.62 &&
    bounds.minX >= 0.06 &&
    bounds.maxX <= 0.94 &&
    bounds.minY >= 0.08 &&
    bounds.maxY <= 0.96;

  const stepChecks = {
    center: centered && Math.abs(yaw) < 0.045 && Math.abs(pitch) < 0.055,
    left: centered && yaw > 0.16 && Math.abs(pitch) < 0.16,
    right: centered && yaw < -0.16 && Math.abs(pitch) < 0.16,
    up: centered && pitch < -0.14 && Math.abs(yaw) < 0.13 && verticalSpan > 0.62,
    down: centered && pitch > 0.14 && Math.abs(yaw) < 0.13,
  };

  let completed = [...completedSteps];
  const nextStep = FACE_STEPS.find((step) => !completed.includes(step.id)) || FACE_STEPS[FACE_STEPS.length - 1];
  const activeStepId = completed.includes(currentStepId) ? nextStep.id : currentStepId;
  const activeStep = FACE_STEPS.find((step) => step.id === activeStepId) || FACE_STEPS[0];
  const allDone = completed.length === FACE_STEPS.length;
  const warning = warnings[0] || "";
  const poseMatched = !warning && Boolean(stepChecks[activeStepId]);

  return {
    found: true,
    ready: centered && !warning,
    instruction: warning || (allDone ? "All motions captured" : activeStep.label),
    detail: allDone
      ? "Saving now. The full action sequence is complete."
      : centered
        ? "Hold each pose briefly until it checks off."
        : "Center your face before completing motion steps.",
    warning,
    stepId: activeStepId,
    completed,
    poseMatched,
    metrics: {
      bounds,
      centerX,
      centerY,
      yaw,
      pitch,
    },
  };
}

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const animationRef = useRef(null);
  const timerRef = useRef(null);
  const startedAtRef = useRef(0);
  const faceLandmarkerRef = useRef(null);
  const faceStateRef = useRef(DEFAULT_FACE_STATE);
  const faceStepRef = useRef("center");
  const completedStepsRef = useRef([]);
  const poseHoldRef = useRef({ stepId: "", startedAt: 0 });
  const lastFaceUiUpdateRef = useRef(0);
  const recordingRef = useRef(false);
  const savingRef = useRef(false);
  const completedSessionRef = useRef(false);
  const supportedFormatsRef = useRef([]);
  const formatIdRef = useRef("webm");

  const [cameraReady, setCameraReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showGuide, setShowGuide] = useState(true);
  const [filename, setFilename] = useState("face-motion-recording");
  const [elapsed, setElapsed] = useState("00:00");
  const [status, setStatus] = useState("Camera idle");
  const [savedFile, setSavedFile] = useState(null);
  const [supportedFormats, setSupportedFormats] = useState([]);
  const [formatId, setFormatId] = useState("webm");
  const [faceState, setFaceState] = useState(DEFAULT_FACE_STATE);
  const [sessionDone, setSessionDone] = useState(false);

  useEffect(() => {
    recordingRef.current = recording;
  }, [recording]);

  useEffect(() => {
    savingRef.current = saving;
  }, [saving]);

  useEffect(() => {
    supportedFormatsRef.current = supportedFormats;
  }, [supportedFormats]);

  useEffect(() => {
    formatIdRef.current = formatId;
  }, [formatId]);

  useEffect(() => {
    const formats = getSupportedFormats();
    setSupportedFormats(formats);
    setFormatId(formats.find((format) => format.id === "mp4")?.id || formats[0]?.id || "webm");
    const originalConsole = {
      info: console.info,
      warn: console.warn,
      error: console.error,
    };
    const shouldHideMediaPipeLog = (args) =>
      String(args[0] || "").includes("Created TensorFlow Lite XNNPACK delegate for CPU");

    console.info = (...args) => {
      if (!shouldHideMediaPipeLog(args)) originalConsole.info(...args);
    };
    console.warn = (...args) => {
      if (!shouldHideMediaPipeLog(args)) originalConsole.warn(...args);
    };
    console.error = (...args) => {
      if (!shouldHideMediaPipeLog(args)) originalConsole.error(...args);
    };

    return () => {
      console.info = originalConsole.info;
      console.warn = originalConsole.warn;
      console.error = originalConsole.error;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      cancelAnimationFrame(animationRef.current);
      clearInterval(timerRef.current);
      faceLandmarkerRef.current?.close();
      if (recorderRef.current?.state === "recording") {
        recorderRef.current.stop();
      }
    };
  }, []);

  useEffect(() => {
    if (!cameraReady) {
      return;
    }

    let disposed = false;

    async function loadFaceLandmarker() {
      if (faceLandmarkerRef.current) {
        return faceLandmarkerRef.current;
      }

      setStatus("Loading face detector");
      const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      const options = {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 2,
        minFaceDetectionConfidence: 0.55,
        minFacePresenceConfidence: 0.55,
        minTrackingConfidence: 0.55,
      };
      let landmarker;

      try {
        landmarker = await FaceLandmarker.createFromOptions(vision, options);
      } catch {
        landmarker = await FaceLandmarker.createFromOptions(vision, {
          ...options,
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "CPU",
          },
        });
      }

      faceLandmarkerRef.current = landmarker;
      setStatus("Face detector ready");
      return landmarker;
    }

    const draw = async () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas?.getContext("2d");

      if (!canvas || !ctx || !video) {
        animationRef.current = requestAnimationFrame(draw);
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.floor(rect.width * scale);
      canvas.height = Math.floor(rect.height * scale);
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      if (showGuide) {
        const safe = {
          x: rect.width * 0.24,
          y: rect.height * 0.14,
          width: rect.width * 0.52,
          height: rect.height * 0.78,
        };

        ctx.save();
        ctx.strokeStyle = faceStateRef.current.ready
          ? "rgba(117, 222, 195, 0.9)"
          : "rgba(255, 207, 90, 0.9)";
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 8]);
        ctx.strokeRect(safe.x, safe.y, safe.width, safe.height);
        ctx.restore();
      }

      try {
        const landmarker = await loadFaceLandmarker();
        if (!disposed && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
          const results = landmarker.detectForVideo(video, performance.now());
          const faces = results.faceLandmarks || [];
          const face = faces[0];
          const nextState =
            faces.length > 1
              ? {
                  ...DEFAULT_FACE_STATE,
                  found: true,
                  instruction: "Only one face please",
                  detail: "Move other faces out of the frame before recording.",
                  stepId: faceStepRef.current,
                  completed: [...completedStepsRef.current],
                }
              : analyzeFace(face, faceStepRef.current, completedStepsRef.current);

          const guidedState = applyPoseHold(nextState);

          faceStepRef.current = guidedState.stepId;
          completedStepsRef.current = guidedState.completed;
          faceStateRef.current = guidedState;
          syncAutoRecording(guidedState);

          if (face?.length && showGuide && guidedState.metrics) {
            const { bounds } = guidedState.metrics;
            const x = (1 - bounds.maxX) * rect.width;
            const y = bounds.minY * rect.height;
            const width = (bounds.maxX - bounds.minX) * rect.width;
            const height = (bounds.maxY - bounds.minY) * rect.height;

            ctx.save();
            ctx.strokeStyle = guidedState.ready ? "#75dec3" : "#ffcf5a";
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            ctx.strokeRect(x, y, width, height);
            ctx.fillStyle = guidedState.ready ? "#75dec3" : "#ffcf5a";
            for (const point of face.filter((_, index) => index % 18 === 0)) {
              ctx.beginPath();
              ctx.arc((1 - point.x) * rect.width, point.y * rect.height, 2, 0, Math.PI * 2);
              ctx.fill();
            }
            ctx.restore();
          }

          if (performance.now() - lastFaceUiUpdateRef.current > 120) {
            lastFaceUiUpdateRef.current = performance.now();
            setFaceState(guidedState);
          }
        }
      } catch (error) {
        const nextState = {
          ...DEFAULT_FACE_STATE,
          instruction: "Face detector unavailable",
          detail: error.message,
        };
        faceStateRef.current = nextState;
        setFaceState(nextState);
      }

      animationRef.current = requestAnimationFrame(draw);
    };

    cancelAnimationFrame(animationRef.current);
    draw();
  }, [cameraReady, showGuide]);

  async function startCamera() {
    try {
      setSessionDone(false);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setCameraReady(true);
      setSavedFile(null);
      setStatus("Camera ready");
    } catch (error) {
      setStatus(`Camera blocked: ${error.message}`);
    }
  }

  function resetGuideState(nextState = DEFAULT_FACE_STATE) {
    faceStepRef.current = "center";
    completedStepsRef.current = [];
    poseHoldRef.current = { stepId: "", startedAt: 0 };
    faceStateRef.current = nextState;
    setFaceState(nextState);
  }

  function closeCamera(nextState = DEFAULT_FACE_STATE) {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    cancelAnimationFrame(animationRef.current);
    clearInterval(timerRef.current);
    setCameraReady(false);
    setRecording(false);
    recordingRef.current = false;
    resetGuideState(nextState);
  }

  async function retrySession() {
    completedSessionRef.current = false;
    savingRef.current = false;
    setSaving(false);
    setElapsed("00:00");
    setStatus("Camera idle");
    setSessionDone(false);
    resetGuideState();
    await startCamera();
  }

  function applyPoseHold(nextState) {
    if (!nextState.found || !nextState.ready || !nextState.poseMatched) {
      poseHoldRef.current = { stepId: "", startedAt: 0 };
      return nextState;
    }

    const now = performance.now();
    const hold = poseHoldRef.current;

    if (hold.stepId !== nextState.stepId) {
      poseHoldRef.current = { stepId: nextState.stepId, startedAt: now };
      return {
        ...nextState,
        detail: "Hold steady for a moment.",
      };
    }

    if (now - hold.startedAt < STEP_HOLD_MS) {
      return {
        ...nextState,
        detail: "Hold steady for a moment.",
      };
    }

    if (nextState.completed.includes(nextState.stepId)) {
      return nextState;
    }

    const completed = [...nextState.completed, nextState.stepId];
    const nextStep = FACE_STEPS.find((step) => !completed.includes(step.id));
    const stepId = nextStep?.id || nextState.stepId;
    poseHoldRef.current = { stepId: "", startedAt: 0 };

    return {
      ...nextState,
      instruction: nextStep ? nextStep.label : "All motions captured",
      detail: nextStep
        ? "Hold each pose briefly until it checks off."
        : "Saving now. The full action sequence is complete.",
      stepId,
      completed,
      poseMatched: false,
    };
  }

  function syncAutoRecording(nextState) {
    const allStepsDone = nextState.completed.length === FACE_STEPS.length;

    if (allStepsDone && recordingRef.current) {
      completedSessionRef.current = true;
      stopRecording();
      return;
    }

    if (!nextState.ready && !recordingRef.current && !savingRef.current) {
      completedSessionRef.current = false;
    }

    if (
      nextState.ready &&
      !allStepsDone &&
      !recordingRef.current &&
      !savingRef.current &&
      !completedSessionRef.current
    ) {
      startRecording();
    }
  }

  function startRecording() {
    if (!streamRef.current || recordingRef.current || savingRef.current) {
      return;
    }

    chunksRef.current = [];
    completedStepsRef.current = [];
    faceStepRef.current = "center";
    poseHoldRef.current = { stepId: "", startedAt: 0 };
    completedSessionRef.current = false;
    setSavedFile(null);

    const selectedFormat = supportedFormatsRef.current.find(
      (format) => format.id === formatIdRef.current,
    );
    const mimeType = selectedFormat?.mimeType || getSupportedMimeType();
    const recorder = new MediaRecorder(
      streamRef.current,
      mimeType ? { mimeType } : undefined,
    );

    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    });

    recorder.addEventListener("stop", saveRecording);
    recorder.start();
    recorderRef.current = recorder;

    startedAtRef.current = Date.now();
    setElapsed("00:00");
    timerRef.current = setInterval(() => {
      setElapsed(formatElapsed(Date.now() - startedAtRef.current));
    }, 250);

    recordingRef.current = true;
    setRecording(true);
    setStatus("Auto recording");
  }

  function stopRecording() {
    if (!recordingRef.current) {
      return;
    }

    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }

    clearInterval(timerRef.current);
    recordingRef.current = false;
    savingRef.current = true;
    setRecording(false);
    setSaving(true);
    setStatus("Saving recording");
  }

  async function saveRecording() {
    const recorder = recorderRef.current;
    const type = recorder?.mimeType || "video/webm";
    const extension = getExtensionFromMimeType(type);
    const blob = new Blob(chunksRef.current, { type });
    const formData = new FormData();

    formData.append("file", blob, `${filename || "face-motion-recording"}.${extension}`);
    formData.append("filename", filename);
    formData.append("extension", extension);

    try {
      const response = await fetch("/api/recordings", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Could not save recording");
      }

      setSavedFile(result.file);
      setStatus("Recording saved");
    } catch (error) {
      setStatus(`Save failed: ${error.message}`);
    } finally {
      savingRef.current = false;
      completedSessionRef.current = true;
      setSaving(false);
      setSessionDone(true);
      closeCamera(SESSION_DONE_FACE_STATE);
    }
  }

  const statusClass = recording ? "recording" : cameraReady ? "ready" : "";

  return (
    <main className="app">
      <section className="stage" aria-label="Camera recorder">
        <video ref={videoRef} className="camera" autoPlay muted playsInline />
        <canvas ref={canvasRef} className="overlay" aria-hidden="true" />
        {(cameraReady || sessionDone) && (
          <div className={`screen-guide ${faceState.ready ? "ready" : faceState.found ? "warn" : ""}`}>
            <strong>{faceState.instruction}</strong>
            <span>{faceState.detail}</span>
          </div>
        )}
        {!cameraReady && !sessionDone && (
          <div className="empty-state">
            <h1>Face Motion Recorder</h1>
            <p>Use your laptop camera to record face movement into a local folder.</p>
          </div>
        )}
        {sessionDone && (
          <div className="empty-state done-state">
            <h1>Session Complete</h1>
            <p>The camera is closed and your recording has been saved.</p>
          </div>
        )}
      </section>

      <aside className="panel" aria-label="Recorder controls">
        <div className="status-row">
          <span className={`status-dot ${statusClass}`} />
          <span>{status}</span>
          <time dateTime={`PT${elapsed}S`}>{elapsed}</time>
        </div>

        <div className="controls">
          {sessionDone ? (
            <button type="button" className="retry-button" onClick={retrySession}>
              <span aria-hidden="true">RETRY</span>
              Try Again
            </button>
          ) : (
            <button type="button" onClick={startCamera} disabled={cameraReady || recording}>
              <span aria-hidden="true">CAM</span>
              Start Camera
            </button>
          )}
        </div>

        <div className={`face-card ${faceState.ready ? "ready" : faceState.found ? "warn" : ""}`}>
          <div>
            <strong>{faceState.instruction}</strong>
            <span>{faceState.detail}</span>
          </div>
          <div className="step-list" aria-label="Face motion checklist">
            {FACE_STEPS.map((step) => (
              <span
                key={step.id}
                className={[
                  "step-pill",
                  faceState.completed.includes(step.id) ? "done" : "",
                  faceState.stepId === step.id ? "active" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
              >
                {step.label}
              </span>
            ))}
          </div>
        </div>

        <div className="settings">
          <label className="check-row">
            <input
              type="checkbox"
              checked={showGuide}
              onChange={(event) => setShowGuide(event.target.checked)}
            />
            Show face guide
          </label>
          <label>
            Format
            <select
              value={formatId}
              onChange={(event) => setFormatId(event.target.value)}
              disabled={recording || saving}
            >
              {supportedFormats.length > 0 ? (
                supportedFormats.map((format) => (
                  <option key={format.id} value={format.id}>
                    {format.label}
                  </option>
                ))
              ) : (
                <option value="webm">Browser default</option>
              )}
            </select>
          </label>
          <label>
            Filename
            <input
              type="text"
              value={filename}
              onChange={(event) => setFilename(event.target.value)}
              disabled={recording || saving}
            />
          </label>
        </div>

        <div className="save-box" aria-live="polite">
          <strong>Save folder</strong>
          <span>recordings/</span>
          {savedFile && <span className="saved-file">{savedFile}</span>}
        </div>
      </aside>
    </main>
  );
}
