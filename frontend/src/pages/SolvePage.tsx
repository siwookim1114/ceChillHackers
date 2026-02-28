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

export function SolvePage() {
  const { attemptId } = useParams();
  const navigate = useNavigate();

  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [workText, setWorkText] = useState("");
  const [answer, setAnswer] = useState("");
  const [signals, setSignals] = useState<StuckSignals>(INITIAL_SIGNALS);
  const [intervention, setIntervention] = useState<Intervention | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const queueRef = useRef<ClientEvent[]>([]);
  const flushingRef = useRef(false);
  const lastActionRef = useRef(Date.now());

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

  const onWorkChange = (value: string) => {
    setWorkText(value);
    enqueueEvent({
      type: "stroke_add",
      payload: {
        char_count: value.length
      }
    });
  };

  const eraseOne = () => {
    if (!workText.length) {
      return;
    }
    const next = workText.slice(0, -1);
    setWorkText(next);
    enqueueEvent({
      type: "stroke_erase",
      payload: {
        char_count: next.length
      }
    });
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
    <AppShell title={attempt.problem.title} subtitle={`Unit: ${attempt.problem.unit}`}>
      <section className="solve-layout panel-card">
        <div className="workspace">
          <div className="problem-header">
            <p className="overline">Practice Prompt</p>
            <h3>{attempt.problem.title}</h3>
          </div>

          <div className="problem-statement">{attempt.problem.prompt}</div>

          <div className="canvas-label">
            <span>Work Canvas</span>
            <textarea
              value={workText}
              onChange={(event) => onWorkChange(event.target.value)}
              placeholder="Write your steps here…"
              rows={10}
            />
          </div>

          <div className="action-row">
            <button className="btn-muted" onClick={eraseOne} type="button">
              ⌫ Erase Last
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

          {error && <p className="error">{error}</p>}

          <div className="action-row">
            <button className="btn-muted" onClick={() => navigate(`/result/${attempt.attempt_id}`)} type="button">
              View Summary
            </button>
          </div>
        </div>

        <CoachPanel signals={signals} intervention={intervention} onRequestHint={requestHint} />
      </section>
    </AppShell>
  );
}
