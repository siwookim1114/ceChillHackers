import { getAccessToken } from "./auth";
import type {
  Attempt,
  AttemptCreateResponse,
  AuthResponse,
  AuthUser,
  ClientEvent,
  CourseDetail,
  CourseFolder,
  DailyProgress,
  DailyProgressEventType,
  EventBatchResponse,
  Intervention,
  LectureFileInfo,
  LectureItem,
  LearningPace,
  LearningStyle,
  Problem,
  Summary,
  UserRole,
  VoiceSessionStartResponse,
  VoiceSessionTurnResponse,
} from "./types";

export const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

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
  const isFormDataBody = options?.body instanceof FormData;
  if (!isFormDataBody && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_URL}${path}`, {
    headers,
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(
      parseErrorText(text) || `HTTP ${response.status}`,
      response.status,
    );
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

export type DailyProgressEventPayload = {
  event_type: DailyProgressEventType;
  attempt_id?: string;
  topic?: string;
};

export type CreateCoursePayload = {
  title: string;
  syllabus?: string;
};

export type CreateLecturePayload = {
  title: string;
  description?: string;
  problem_prompt: string;
  answer_key: string;
  sort_order?: number;
};

export type VoiceSessionStartPayload = {
  attempt_id?: string;
  course_id?: string;
  lecture_id?: string;
};

export function signup(payload: SignupPayload) {
  return request<AuthResponse>("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function login(payload: LoginPayload) {
  return request<AuthResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getMe() {
  return request<AuthUser>("/api/auth/me");
}

export function getMeWithToken(accessToken: string) {
  return request<AuthUser>("/api/auth/me", {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
}

export function getGoogleAuthStartUrl(returnTo: string) {
  const params = new URLSearchParams({ return_to: returnTo });
  return `${API_URL}/api/auth/google/start?${params.toString()}`;
}

export function listCourses() {
  return request<CourseFolder[]>("/api/courses");
}

export function createCourse(payload: CreateCoursePayload) {
  return request<CourseFolder>("/api/courses", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getCourseDetail(courseId: string) {
  return request<CourseDetail>(`/api/courses/${courseId}`);
}

export function listCourseLectures(courseId: string) {
  return request<LectureItem[]>(`/api/courses/${courseId}/lectures`);
}

export function createLecture(courseId: string, payload: CreateLecturePayload) {
  return request<LectureItem>(`/api/courses/${courseId}/lectures`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listLectureFiles(courseId: string, lectureId: string) {
  return request<LectureFileInfo[]>(
    `/api/courses/${courseId}/lectures/${lectureId}/files`,
  );
}

export function uploadLectureFile(
  courseId: string,
  lectureId: string,
  file: File,
) {
  const formData = new FormData();
  formData.append("file", file);
  return request<LectureFileInfo>(
    `/api/courses/${courseId}/lectures/${lectureId}/files`,
    {
      method: "POST",
      body: formData,
    },
  );
}

export function listProblems() {
  return request<Problem[]>("/api/problems");
}

export function createAttempt(payload: CreateAttemptPayload) {
  return request<AttemptCreateResponse>("/api/attempts", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getAttempt(attemptId: string) {
  return request<Attempt>(`/api/attempts/${attemptId}`);
}

export function postEvents(attemptId: string, events: ClientEvent[]) {
  return request<EventBatchResponse>(`/api/attempts/${attemptId}/events`, {
    method: "POST",
    body: JSON.stringify({ events }),
  });
}

export function getIntervention(attemptId: string) {
  return request<Intervention | null>(
    `/api/attempts/${attemptId}/intervention`,
  );
}

export function getSummary(attemptId: string) {
  return request<Summary>(`/api/attempts/${attemptId}/summary`);
}

export function getDailyProgress() {
  return request<DailyProgress>("/api/progress/daily");
}

export function postDailyProgressEvent(payload: DailyProgressEventPayload) {
  return request<DailyProgress>("/api/progress/daily/events", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function startVoiceSession(payload: VoiceSessionStartPayload) {
  return request<VoiceSessionStartResponse>("/api/voice/session/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function postVoiceSessionTurn(sessionId: string, audioBlob: Blob) {
  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("audio", audioBlob, "student-voice.webm");
  return request<VoiceSessionTurnResponse>("/api/voice/session/turn", {
    method: "POST",
    body: formData,
  });
}
