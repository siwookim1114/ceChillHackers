import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ApiError,
  createAttempt,
  getCourseDetail,
  getMe,
  listCourses,
  postDailyProgressEvent,
} from "../api";
import {
  clearAuthSession,
  getAccessToken,
  getAuthUser,
  saveAuthSession,
} from "../auth";
import { AppShell } from "../components/AppShell";
import type {
  AuthUser,
  CourseDetail,
  CourseFolder,
  LectureItem,
} from "../types";

export function PracticePage() {
  const navigate = useNavigate();
  const [user, setUser] = useState<AuthUser | null>(() => getAuthUser());
  const [authLoading, setAuthLoading] = useState(true);

  const [courses, setCourses] = useState<CourseFolder[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<string | null>(null);
  const [selectedCourse, setSelectedCourse] = useState<CourseDetail | null>(
    null,
  );

  const [loadingCourses, setLoadingCourses] = useState(true);
  const [loadingLectures, setLoadingLectures] = useState(false);
  const [launchingLectureId, setLaunchingLectureId] = useState<string | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const token = getAccessToken();
    if (!token) {
      navigate("/login", { replace: true });
      return;
    }

    getMe()
      .then((me) => {
        saveAuthSession(token, me);
        setUser(me);
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearAuthSession();
          navigate("/login", { replace: true });
        }
      })
      .finally(() => setAuthLoading(false));
  }, [navigate]);

  const loadCourses = async () => {
    setLoadingCourses(true);
    try {
      const nextCourses = await listCourses();
      setCourses(nextCourses);
      setSelectedCourseId((previous) => {
        if (previous && nextCourses.some((course) => course.id === previous)) {
          return previous;
        }
        return nextCourses[0]?.id ?? null;
      });
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load course folders",
      );
    } finally {
      setLoadingCourses(false);
    }
  };

  useEffect(() => {
    if (authLoading) {
      return;
    }
    loadCourses();
  }, [authLoading]);

  useEffect(() => {
    if (!selectedCourseId) {
      setSelectedCourse(null);
      return;
    }

    setLoadingLectures(true);
    getCourseDetail(selectedCourseId)
      .then(setSelectedCourse)
      .catch((err: Error) => {
        setError(err.message || "Failed to load lectures");
        setSelectedCourse(null);
      })
      .finally(() => setLoadingLectures(false));
  }, [selectedCourseId]);

  const getActorId = () => {
    if (user) {
      return `user_${user.id}`;
    }
    const existing = localStorage.getItem("guest_id");
    if (existing) {
      return existing;
    }
    const next = `guest_${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem("guest_id", next);
    return next;
  };

  const startLecturePractice = async (lecture: LectureItem) => {
    if (!selectedCourse) {
      return;
    }

    setLaunchingLectureId(lecture.id);
    setError(null);

    try {
      const attempt = await createAttempt({
        guest_id: getActorId(),
        problem_text: lecture.problem_prompt,
        answer_key: lecture.answer_key,
        unit: selectedCourse.title,
      });

      if (getAccessToken() && user) {
        postDailyProgressEvent({
          event_type: "set_current_topic",
          topic: selectedCourse.title,
        }).catch(() => {
          // Keep solve flow uninterrupted when progress sync fails.
        });
      }

      navigate(`/solve/${attempt.attempt_id}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start lecture practice",
      );
    } finally {
      setLaunchingLectureId(null);
    }
  };

  if (authLoading) {
    return (
      <AppShell title="Practice Studio" subtitle="Checking your session...">
        <section
          className="panel-card session-skeleton"
          aria-label="Loading practice workspace"
        >
          <div className="skeleton-line skeleton-line-short" />
          <div className="skeleton-line skeleton-line-medium" />
          <div className="skeleton-pill-row">
            <span className="skeleton-pill" />
            <span className="skeleton-pill" />
          </div>
          <div className="skeleton-grid-4">
            <span className="skeleton-block" />
            <span className="skeleton-block" />
            <span className="skeleton-block" />
            <span className="skeleton-block" />
          </div>
        </section>
      </AppShell>
    );
  }

  return (
    <AppShell
      title="Practice Studio"
      subtitle="Open a course folder and start practice from its lectures."
    >
      <div className="practice-page-wrap">
        <section className="practice-head-banner reveal reveal-1">
          <div>
            <p className="overline">Course Folders</p>
            <h3>Practice by lecture, inside each course</h3>
            <p>
              Pick a folder on the left, then launch any lecture as a guided
              solving session.
            </p>
          </div>
          <button
            className="btn-primary"
            onClick={() => navigate("/create-course")}
            type="button"
          >
            Create New Course
          </button>
        </section>

        {error && <p className="error">{error}</p>}

        <section className="practice-course-layout reveal reveal-2">
          <aside className="panel-card course-folder-panel">
            <div className="home-header">
              <h3>Folders</h3>
              <span className="user-pill">
                <strong>{courses.length}</strong>
                <span>courses</span>
              </span>
            </div>

            {loadingCourses && (
              <div className="catalog-skeleton" aria-label="Loading folders">
                <div className="skeleton-line skeleton-line-medium" />
                <div className="skeleton-grid-3">
                  <span className="skeleton-block" />
                  <span className="skeleton-block" />
                  <span className="skeleton-block" />
                </div>
              </div>
            )}

            <div className="course-folder-list">
              {courses.map((course) => (
                <button
                  className={`course-folder-item${selectedCourseId === course.id ? " active" : ""}`}
                  key={course.id}
                  onClick={() => setSelectedCourseId(course.id)}
                  type="button"
                >
                  <strong>{course.title}</strong>
                  <small>
                    {course.lecture_count} lectures • {course.file_count} files
                  </small>
                </button>
              ))}
              {!loadingCourses && courses.length === 0 && (
                <div className="empty-course-state compact">
                  <p>No course folders yet.</p>
                  <button
                    className="btn-muted"
                    onClick={() => navigate("/create-course")}
                    type="button"
                  >
                    Create your first course
                  </button>
                </div>
              )}
            </div>
          </aside>

          <section className="panel-card lecture-list-panel">
            {!selectedCourseId && (
              <div className="empty-course-state">
                <h3>No folder selected</h3>
                <p>Select a course folder to view lectures.</p>
              </div>
            )}

            {selectedCourseId && (
              <>
                <div className="workspace-head">
                  <h3>{selectedCourse?.title ?? "Loading..."}</h3>
                  <p>{selectedCourse?.syllabus || "No syllabus yet."}</p>
                </div>

                {loadingLectures && (
                  <p className="muted">Loading lectures...</p>
                )}
                {!loadingLectures &&
                  selectedCourse &&
                  selectedCourse.lectures.length === 0 && (
                    <div className="empty-course-state compact">
                      <p>No lectures in this folder yet.</p>
                      <button
                        className="btn-muted"
                        onClick={() => navigate("/create-course")}
                        type="button"
                      >
                        Add lectures in Create New Course
                      </button>
                    </div>
                  )}

                <div className="lecture-practice-list">
                  {selectedCourse?.lectures.map((lecture) => (
                    <article className="lecture-practice-card" key={lecture.id}>
                      <div className="lecture-admin-head">
                        <div>
                          <h4>{lecture.title}</h4>
                          <p>{lecture.description || "No description"}</p>
                        </div>
                        <span className="unit-tag">
                          {lecture.file_count} files
                        </span>
                      </div>

                      <p className="problem-prompt">{lecture.problem_prompt}</p>

                      <button
                        className="btn-primary"
                        disabled={launchingLectureId === lecture.id}
                        onClick={() => startLecturePractice(lecture)}
                        type="button"
                      >
                        {launchingLectureId === lecture.id
                          ? "Starting..."
                          : "Start This Lecture"}
                      </button>
                    </article>
                  ))}
                </div>
              </>
            )}
          </section>
        </section>
      </div>
    </AppShell>
  );
}
