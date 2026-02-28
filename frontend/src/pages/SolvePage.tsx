import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { getAttempt, getIntervention, postEvents } from "../api";
import { AppShell } from "../components/AppShell";
import { CoachPanel } from "../components/CoachPanel";
import type { Attempt, ClientEvent, Intervention, StuckSignals } from "../types";

const INITIAL_SIGNALS: StuckSignals = {
  idle_ms: 0,
  erase_count_delta: 0,
  repeated_error_count: 0,
  stuck_score: 0
};

type SolveStage = "ready" | "lecture" | "solve";

function SoundToggleIcon({ muted }: { muted: boolean }) {
  if (muted) {
    return (
      <svg aria-hidden="true" className="sound-toggle-icon" fill="none" viewBox="0 0 24 24">
        <path
          d="M4 10h4l5-4v12l-5-4H4z"
          stroke="currentColor"
          strokeLinejoin="round"
          strokeWidth="1.8"
        />
        <path
          d="M16 9l5 5M21 9l-5 5"
          stroke="currentColor"
          strokeLinecap="round"
          strokeWidth="1.8"
        />
      </svg>
    );
  }
  return (
    <svg aria-hidden="true" className="sound-toggle-icon" fill="none" viewBox="0 0 24 24">
      <path
        d="M4 10h4l5-4v12l-5-4H4z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
      <path
        d="M16 9a4 4 0 0 1 0 6M18.5 7a7 7 0 0 1 0 10"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

export function SolvePage() {
  const { attemptId } = useParams();
  const navigate = useNavigate();

  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [stage, setStage] = useState<SolveStage>("ready");
  const [workText, setWorkText] = useState("");
  const [answer, setAnswer] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState<string | null>(null);
  const [lectureMuted, setLectureMuted] = useState(true);
  const [signals, setSignals] = useState<StuckSignals>(INITIAL_SIGNALS);
  const [intervention, setIntervention] = useState<Intervention | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const queueRef = useRef<ClientEvent[]>([]);
  const flushingRef = useRef(false);
  const lastActionRef = useRef(Date.now());
  const lectureVideoRef = useRef<HTMLVideoElement | null>(null);

  const enqueueEvent = useCallback((event: ClientEvent) => {
    const stamped = { ...event, ts: new Date().toISOString() };
    queueRef.current.push(stamped);
    if (event.type !== "idle_ping") {
      lastActionRef.current = Date.now();
    }
  }, []);

  const flushQueue = useCallback(async () => {
    if (!attemptId || flushingRef.current || queueRef.current.length === 0) {
      return;
    }

    flushingRef.current = true;
    const batch = [...queueRef.current];
    queueRef.current = [];

    try {
      const response = await postEvents(attemptId, batch);
      setSignals(response.stuck_signals);
      if (response.intervention) {
        setIntervention(response.intervention);
      }
      if (response.solved) {
        navigate(`/result/${attemptId}`);
      }
    } catch (err) {
      queueRef.current = [...batch, ...queueRef.current];
      setError(err instanceof Error ? err.message : "Failed to send events");
    } finally {
      flushingRef.current = false;
    }
  }, [attemptId, navigate]);

  useEffect(() => {
    if (!attemptId) {
      setError("Missing attempt id");
      setLoading(false);
      return;
    }
    getAttempt(attemptId)
      .then(setAttempt)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [attemptId]);

  useEffect(() => {
    if (!attemptId) {
      return;
    }

    const batchTimer = window.setInterval(() => {
      void flushQueue();
    }, 5000);

    const idleTimer = window.setInterval(() => {
      enqueueEvent({
        type: "idle_ping",
        payload: {
          idle_ms: Date.now() - lastActionRef.current
        }
      });
    }, 10000);

    const interventionTimer = window.setInterval(() => {
      getIntervention(attemptId)
        .then((latest) => {
          if (latest) {
            setIntervention(latest);
          }
        })
        .catch(() => {
          // Ignore polling errors and keep event flow running.
        });
    }, 7000);

    return () => {
      window.clearInterval(batchTimer);
      window.clearInterval(idleTimer);
      window.clearInterval(interventionTimer);
    };
  }, [attemptId, enqueueEvent, flushQueue]);

  useEffect(() => {
    return () => {
      if (uploadedPreviewUrl) {
        URL.revokeObjectURL(uploadedPreviewUrl);
      }
    };
  }, [uploadedPreviewUrl]);

  const onWorkChange = (value: string) => {
    setWorkText(value);
    enqueueEvent({
      type: "stroke_add",
      payload: {
        char_count: value.length
      }
    });
  };

  const clearWorkNotes = () => {
    if (!workText.length) {
      return;
    }
    setWorkText("");
    enqueueEvent({
      type: "stroke_erase",
      payload: {
        char_count: 0
      }
    });
  };

  const onUploadWork = (file: File | null) => {
    if (!file) {
      return;
    }
    setError(null);
    setUploadedFileName(file.name);
    if (uploadedPreviewUrl) {
      URL.revokeObjectURL(uploadedPreviewUrl);
    }
    if (file.type.startsWith("image/")) {
      setUploadedPreviewUrl(URL.createObjectURL(file));
    } else {
      setUploadedPreviewUrl(null);
    }
    enqueueEvent({
      type: "stroke_add",
      payload: {
        upload_name: file.name,
        upload_size: file.size
      }
    });
  };

  const toggleLectureMute = () => {
    const video = lectureVideoRef.current;
    if (!video) {
      return;
    }
    const nextMuted = !lectureMuted;
    video.muted = nextMuted;
    setLectureMuted(nextMuted);
    if (!nextMuted) {
      void video.play().catch(() => {
        video.muted = true;
        setLectureMuted(true);
      });
    }
  };

  const requestHint = async () => {
    enqueueEvent({
      type: "hint_request",
      payload: {
        source: "coach_button"
      }
    });
    await flushQueue();
  };

  const submitAnswer = async () => {
    if (!uploadedFileName) {
      setError("먼저 풀이 이미지나 파일을 업로드해 주세요.");
      return;
    }
    enqueueEvent({
      type: "answer_submit",
      payload: {
        answer
      }
    });
    await flushQueue();
  };

  if (loading) {
    return (
      <AppShell title="Practice Session" subtitle="Loading your attempt...">
        <section className="panel-card">
          <div className="loading-state">
            <div className="spinner" />
            <span>Loading attempt…</span>
          </div>
        </section>
      </AppShell>
    );
  }

  if (!attempt) {
    return (
      <AppShell title="Practice Session" subtitle="Unable to open this attempt">
        <section className="panel-card">
          <p className="error">{error ?? "Attempt not found"}</p>
        </section>
      </AppShell>
    );
  }

  return (
    <AppShell
      title={attempt.problem.title}
      subtitle={`Unit: ${attempt.problem.unit}`}
      immersive={stage !== "ready"}
      hideContentHeader={stage !== "ready"}
    >
      {stage === "ready" && (
        <section className="panel-card lecture-launch">
          <p className="overline">Lesson Setup</p>
          <h3>{attempt.problem.title}</h3>
          <p>
            강의를 먼저 시작하면 아바타가 핵심 개념을 설명하고, 종료 후에는 좌측 아바타 +
            우측 스케치 보드 화면으로 전환됩니다.
          </p>
          <div className="problem-statement">{attempt.problem.prompt}</div>
          <div className="entry-cta-row">
            <button className="btn-primary" onClick={() => setStage("lecture")} type="button">
              Start Lecture
            </button>
            <button className="btn-muted" onClick={() => setStage("solve")} type="button">
              Skip to Solving
            </button>
          </div>
        </section>
      )}

      {stage === "lecture" && (
        <section className="panel-card lecture-stage">
          <div className="lecture-video-wrap">
            <video
              autoPlay
              className="lecture-video"
              loop
              muted={lectureMuted}
              playsInline
              preload="metadata"
              ref={lectureVideoRef}
            >
              <source src="/intro.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
            <button
              aria-label={lectureMuted ? "Unmute lecture video" : "Mute lecture video"}
              className="video-sound-toggle"
              onClick={toggleLectureMute}
              type="button"
            >
              <SoundToggleIcon muted={lectureMuted} />
            </button>
          </div>

          <div className="lecture-copy">
            <p className="overline">AI Tutor Live Lecture</p>
            <h3>{attempt.problem.title}</h3>
            <p>{attempt.problem.prompt}</p>
            <div className="entry-cta-row">
              <button className="btn-primary" onClick={() => setStage("solve")} type="button">
                End Lecture and Open Sketch Board
              </button>
            </div>
          </div>
        </section>
      )}

      {stage === "solve" && (
        <div className="solve-lesson-grid">
          <section className="panel-card solve-lesson-split">
            <div className="solve-lesson-left">
              <div className="lecture-mini">
                <video
                  autoPlay
                  className="lecture-mini-video"
                  loop
                  muted
                  playsInline
                  preload="metadata"
                >
                  <source src="/intro.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              </div>
              <div className="problem-header">
                <p className="overline">Tutor Summary</p>
                <h3>{attempt.problem.title}</h3>
                <p className="muted">좌측은 강의 요약, 우측은 학생 풀이 업로드 영역입니다.</p>
              </div>
              <div className="problem-statement">{attempt.problem.prompt}</div>
              <div className="action-row">
                <button className="btn-muted" onClick={() => setStage("lecture")} type="button">
                  Replay Lecture
                </button>
                <button className="btn-teal" onClick={requestHint} type="button">
                  Ask Hint
                </button>
              </div>
            </div>

            <div className="solve-lesson-right">
              <div className="canvas-label">
                <span>Sketch Board Upload</span>
                <label className="upload-dropzone">
                  <input
                    accept="image/*,.pdf"
                    className="upload-input"
                    onChange={(event) => onUploadWork(event.target.files?.[0] ?? null)}
                    type="file"
                  />
                  <strong>{uploadedFileName ? "Uploaded" : "Upload your solved work"}</strong>
                  <small>
                    {uploadedFileName
                      ? uploadedFileName
                      : "Drag or select an image/PDF of your handwritten solution"}
                  </small>
                </label>
                {uploadedPreviewUrl && (
                  <img
                    alt="Uploaded solution preview"
                    className="upload-preview-image"
                    src={uploadedPreviewUrl}
                  />
                )}
              </div>

              <div className="canvas-label">
                <span>Work Notes (Optional)</span>
                <textarea
                  value={workText}
                  onChange={(event) => onWorkChange(event.target.value)}
                  placeholder="Write quick notes about your solving process…"
                  rows={7}
                />
              </div>

              <div className="action-row">
                <button className="btn-muted" onClick={clearWorkNotes} type="button">
                  Clear Notes
                </button>
              </div>

              <div className="answer-row">
                <span>Final Answer</span>
                <div className="answer-input-row">
                  <input
                    value={answer}
                    onChange={(event) => setAnswer(event.target.value)}
                    placeholder="Type your final answer…"
                  />
                  <button className="btn-primary" onClick={submitAnswer} type="button">
                    Submit
                  </button>
                </div>
              </div>
            </div>
          </section>

          {error && <p className="error">{error}</p>}

          <section className="panel-card">
            <CoachPanel signals={signals} intervention={intervention} onRequestHint={requestHint} />
          </section>

          <div className="action-row">
            <button className="btn-muted" onClick={() => navigate(`/result/${attempt.attempt_id}`)} type="button">
              View Summary
            </button>
          </div>
        </div>
      )}
    </AppShell>
  );
}
