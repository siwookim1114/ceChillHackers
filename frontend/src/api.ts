import type {
  Attempt,
  AttemptCreateResponse,
  ClientEvent,
  EventBatchResponse,
  Intervention,
  Problem,
  Summary
} from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {})
    },
    ...options
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  return (await response.json()) as T;
}

export function listProblems() {
  return request<Problem[]>("/api/problems");
}

export type CreateAttemptPayload = {
  guest_id: string;
  problem_id?: string;
  problem_text?: string;
  answer_key?: string;
  unit?: string;
};

export function createAttempt(payload: CreateAttemptPayload) {
  return request<AttemptCreateResponse>("/api/attempts", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function getAttempt(attemptId: string) {
  return request<Attempt>(`/api/attempts/${attemptId}`);
}

export function postEvents(attemptId: string, events: ClientEvent[]) {
  return request<EventBatchResponse>(`/api/attempts/${attemptId}/events`, {
    method: "POST",
    body: JSON.stringify({ events })
  });
}

export function getIntervention(attemptId: string) {
  return request<Intervention | null>(`/api/attempts/${attemptId}/intervention`);
}

export function getSummary(attemptId: string) {
  return request<Summary>(`/api/attempts/${attemptId}/summary`);
}
