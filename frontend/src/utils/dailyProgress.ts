type DailyProgressStore = {
  date: string;
  solved_attempt_ids: string[];
  course_attempt_ids: string[];
  coached_attempt_ids: string[];
};

export type DailyProgressSnapshot = {
  date: string;
  solvedSessions: number;
  createdCourses: number;
  coachedSessions: number;
  solvedAttemptIds: string[];
  createdCourseAttemptIds: string[];
  coachedAttemptIds: string[];
};

const STORAGE_KEY = "tc_daily_progress_v1";

function todayStamp(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function emptyStore(date = todayStamp()): DailyProgressStore {
  return {
    date,
    solved_attempt_ids: [],
    course_attempt_ids: [],
    coached_attempt_ids: []
  };
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((entry): entry is string => typeof entry === "string");
}

function readStore(): DailyProgressStore {
  const stamp = todayStamp();
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return emptyStore(stamp);
  }

  try {
    const parsed = JSON.parse(raw) as Partial<DailyProgressStore>;
    if (parsed.date !== stamp) {
      return emptyStore(stamp);
    }
    return {
      date: stamp,
      solved_attempt_ids: toStringArray(parsed.solved_attempt_ids),
      course_attempt_ids: toStringArray(parsed.course_attempt_ids),
      coached_attempt_ids: toStringArray(parsed.coached_attempt_ids)
    };
  } catch {
    return emptyStore(stamp);
  }
}

function writeStore(store: DailyProgressStore) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

function addUnique(items: string[], attemptId: string): string[] {
  if (!attemptId || items.includes(attemptId)) {
    return items;
  }
  return [...items, attemptId];
}

export function getDailyProgressSnapshot(): DailyProgressSnapshot {
  const store = readStore();
  return {
    date: store.date,
    solvedSessions: store.solved_attempt_ids.length,
    createdCourses: store.course_attempt_ids.length,
    coachedSessions: store.coached_attempt_ids.length,
    solvedAttemptIds: store.solved_attempt_ids,
    createdCourseAttemptIds: store.course_attempt_ids,
    coachedAttemptIds: store.coached_attempt_ids
  };
}

export function markCourseCreated(attemptId: string) {
  if (!attemptId) {
    return;
  }
  const store = readStore();
  store.course_attempt_ids = addUnique(store.course_attempt_ids, attemptId);
  writeStore(store);
}

export function markAttemptSummary(attemptId: string, solved: boolean, hintLevel: number) {
  if (!attemptId) {
    return;
  }
  const store = readStore();
  if (solved) {
    store.solved_attempt_ids = addUnique(store.solved_attempt_ids, attemptId);
  }
  if (hintLevel > 0) {
    store.coached_attempt_ids = addUnique(store.coached_attempt_ids, attemptId);
  }
  writeStore(store);
}
