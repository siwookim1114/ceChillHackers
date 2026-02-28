import { getAccessToken } from "./auth";
import type {
  Attempt,
  AttemptCreateResponse,
  AuthResponse,
  AuthUser,
  ClientEvent,
  EventBatchResponse,
  Intervention,
  LearningPace,
  LearningStyle,
  Problem,
  Summary,
  UserRole
} from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function parseErrorText(raw: string): string {
  if (!raw) {
    return "";
  }
  try {
    const parsed = JSON.parse(raw) as { detail?: string };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail;
    }
  } catch {
    // Keep raw text fallback when response is not JSON.
  }
  return raw;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getAccessToken();
  const headers = new Headers(options?.headers ?? {});
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_URL}${path}`, {
    headers,
    ...options
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(parseErrorText(text) || `HTTP ${response.status}`);
  }

  return (await response.json()) as T;
}

export type SignupPayload = {
  email: string;
  password: string;
  display_name: string;
  role: UserRole;
  learning_style: LearningStyle;
  learning_pace: LearningPace;
  target_goal?: string;
};

export type LoginPayload = {
  email: string;
  password: string;
};

export type CreateAttemptPayload = {
  guest_id: string;
  problem_id?: string;
  problem_text?: string;
  answer_key?: string;
  unit?: string;
};

export function signup(payload: SignupPayload) {
  return request<AuthResponse>("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function login(payload: LoginPayload) {
  return request<AuthResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function getMe() {
  return request<AuthUser>("/api/auth/me");
}

export function listProblems() {
  return request<Problem[]>("/api/problems");
}

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
