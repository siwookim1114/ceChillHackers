export type Problem = {
  id: string;
  title: string;
  prompt: string;
  answer_key: string;
  unit: string;
};

export type Attempt = {
  attempt_id: string;
  started_at: string;
  solved_at: string | null;
  problem: Problem;
};

export type AttemptCreateResponse = {
  attempt_id: string;
  started_at: string;
  problem: Problem;
};

export type VoiceSessionStartResponse = {
  session_id: string;
  tutor_text: string;
  mediator_summary: string;
  audio_base64: string;
  audio_mime_type: string;
};

export type VoiceSessionTurnResponse = {
  session_id: string;
  transcript: string;
  tutor_text: string;
  mediator_summary: string;
  audio_base64: string;
  audio_mime_type: string;
};

export type EventType =
  | "stroke_add"
  | "stroke_erase"
  | "idle_ping"
  | "hint_request"
  | "answer_submit";

export type ClientEvent = {
  type: EventType;
  ts?: string;
  payload?: Record<string, unknown>;
};

export type StuckSignals = {
  idle_ms: number;
  erase_count_delta: number;
  repeated_error_count: number;
  stuck_score: number;
};

export type Intervention = {
  level: 1 | 2 | 3;
  reason: string;
  tutor_message: string;
  created_at: string;
};

export type EventBatchResponse = {
  accepted: number;
  stuck_signals: StuckSignals;
  intervention: Intervention | null;
  solved: boolean;
};

export type Summary = {
  attempt_id: string;
  metrics: {
    time_to_solve_sec: number | null;
    max_stuck: number;
    hint_max_level: number;
    erase_count: number;
  };
  timeline: Array<{
    at: string;
    type: string;
    label: string;
  }>;
};

export type UserRole = "student" | "teacher" | "parent";
export type LearningStyle = "explanation" | "question" | "problem_solving";
export type LearningPace = "fast" | "normal" | "slow";

export type AuthUser = {
  id: string;
  email: string;
  display_name: string;
  role: UserRole;
  learning_style: LearningStyle | null;
  learning_pace: LearningPace | null;
  target_goal: string | null;
};

export type AuthResponse = {
  access_token: string;
  token_type: "bearer";
  user: AuthUser;
};

export type DailyProgress = {
  date: string;
  solved_sessions: number;
  created_courses: number;
  coached_sessions: number;
  daily_target_sessions: number;
  current_course_topic: string | null;
};

export type DailyProgressEventType =
  | "session_solved"
  | "course_created"
  | "coached_session"
  | "set_current_topic";

export type CourseFolder = {
  id: string;
  title: string;
  syllabus: string | null;
  lecture_count: number;
  file_count: number;
  created_at: string;
};

export type LectureFileInfo = {
  id: string;
  file_name: string;
  content_type: string | null;
  size_bytes: number;
  storage_provider?: string;
  storage_key?: string | null;
  file_url?: string | null;
  created_at: string;
};

export type LectureItem = {
  id: string;
  title: string;
  description: string | null;
  problem_prompt: string;
  answer_key: string;
  sort_order: number;
  file_count: number;
  created_at: string;
  files: LectureFileInfo[];
};

export type CourseDetail = {
  id: string;
  title: string;
  syllabus: string | null;
  created_at: string;
  lectures: LectureItem[];
};
