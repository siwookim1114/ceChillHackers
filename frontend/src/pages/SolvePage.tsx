import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { getAttempt, getIntervention, postEvents } from "../api";
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
          // Ignore polling errors; event batch flow still drives latest state.
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
      <main className="page">
        <section className="card">
          <p>Loading attempt...</p>
        </section>
      </main>
    );
  }

  if (!attempt) {
    return (
      <main className="page">
        <section className="card">
          <p className="error">{error ?? "Attempt not found"}</p>
        </section>
      </main>
    );
  }

  return (
    <main className="page solve-page">
      <section className="card solve-layout">
        <div className="workspace">
          <p className="overline">{attempt.problem.unit}</p>
          <h2>{attempt.problem.title}</h2>
          <p className="prompt">{attempt.problem.prompt}</p>

          <label>
            Work Canvas
            <textarea
              value={workText}
              onChange={(event) => onWorkChange(event.target.value)}
              placeholder="Write your steps here..."
              rows={10}
            />
          </label>

          <div className="action-row">
            <button className="btn-muted" onClick={eraseOne}>
              Erase Stroke
            </button>
            <button className="btn-muted" onClick={requestHint}>
              Ask Coach Hint
            </button>
          </div>

          <label>
            Final Answer
            <input
              value={answer}
              onChange={(event) => setAnswer(event.target.value)}
              placeholder="Type your final answer"
            />
          </label>

          <div className="action-row">
            <button className="btn-primary" onClick={submitAnswer}>
              Submit Answer
            </button>
            <button className="btn-muted" onClick={() => navigate(`/result/${attempt.attempt_id}`)}>
              View Summary
            </button>
          </div>

          {error && <p className="error">{error}</p>}
        </div>

        <CoachPanel signals={signals} intervention={intervention} />
      </section>
    </main>
  );
}
