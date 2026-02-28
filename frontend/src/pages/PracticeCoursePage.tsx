import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  ApiError,
  createAttempt,
  getCourseDetail,
  getMe,
  postDailyProgressEvent,
} from "../api";
import {
  clearAuthSession,
  getAccessToken,
  getAuthUser,
  saveAuthSession,
} from "../auth";
import { AppShell } from "../components/AppShell";
import type { AuthUser, CourseDetail, LectureItem } from "../types";

export function PracticeCoursePage() {
  const navigate = useNavigate();
  const { courseId } = useParams<{ courseId: string }>();
  const [user, setUser] = useState<AuthUser | null>(() => getAuthUser());
  const [authLoading, setAuthLoading] = useState(true);
  const [loadingLectures, setLoadingLectures] = useState(false);
  const [course, setCourse] = useState<CourseDetail | null>(null);
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

  useEffect(() => {
    if (!courseId || authLoading) {
      return;
    }

    setLoadingLectures(true);
    setError(null);

    getCourseDetail(courseId)
      .then(setCourse)
      .catch((err: Error) => {
        setCourse(null);
        setError(err.message || "Failed to load lectures");
      })
      .finally(() => setLoadingLectures(false));
  }, [authLoading, courseId]);

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
    if (!course) {
      return;
    }

    setLaunchingLectureId(lecture.id);
    setError(null);

    try {
      const attempt = await createAttempt({
        guest_id: getActorId(),
        problem_text: lecture.problem_prompt,
        answer_key: lecture.answer_key,
        unit: course.title,
      });

      if (getAccessToken() && user) {
        postDailyProgressEvent({
          event_type: "set_current_topic",
          topic: course.title,
        }).catch(() => {
          // Keep solve flow uninterrupted when progress sync fails.
        });
      }

      const params = new URLSearchParams({
        courseId: course.id,
        lectureId: lecture.id,
      });
      navigate(`/solve/${attempt.attempt_id}?${params.toString()}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start lecture practice",
      );
    } finally {
      setLaunchingLectureId(null);
    }
  };

  const lectureCount = course?.lectures.length ?? 0;
  const fileCount =
    course?.lectures.reduce((sum, lecture) => sum + lecture.file_count, 0) ?? 0;

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
      subtitle="Select a lecture and start a guided practice attempt."
    >
      <div className="practice-page-wrap practice-course-layout-wrap">
        <section className="practice-course-hero reveal reveal-1">
          <div className="practice-course-hero-head">
            <button
              className="btn-muted"
              onClick={() => navigate("/practice")}
              type="button"
            >
              Back to Folders
            </button>
            <button
              className="btn-primary"
              onClick={() => navigate("/create-course")}
              type="button"
            >
              Create New Course
            </button>
          </div>

          <div className="practice-course-hero-main">
            <div>
              <p className="overline">Selected Folder</p>
              <h3>{course?.title ?? "Loading..."}</h3>
              <p>{course?.syllabus || "No syllabus yet."}</p>
            </div>
            <div className="practice-course-hero-stats">
              <article className="practice-course-stat">
                <small>Lectures</small>
                <strong>{lectureCount}</strong>
              </article>
              <article className="practice-course-stat">
                <small>Files</small>
                <strong>{fileCount}</strong>
              </article>
            </div>
          </div>
        </section>

        {error && <p className="error practice-alert">{error}</p>}

        <section className="panel-card lecture-list-panel practice-lecture-panel practice-course-board reveal reveal-2">
          <div className="practice-course-board-head">
            <h3>Lectures in this folder</h3>
            <span className="create-kpi-chip">{lectureCount} total</span>
          </div>

          {loadingLectures && <p className="muted">Loading lectures...</p>}

          {!loadingLectures && course && course.lectures.length === 0 && (
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

          <div className="lecture-practice-list practice-course-lecture-list">
            {course?.lectures.map((lecture, index) => (
              <article className="lecture-practice-card" key={lecture.id}>
                <div className="lecture-admin-head">
                  <div>
                    <h4>
                      <span className="practice-lecture-index">
                        L{index + 1}
                      </span>{" "}
                      {lecture.title}
                    </h4>
                    <p>{lecture.description || "No description"}</p>
                  </div>
                  <span className="unit-tag">{lecture.file_count} files</span>
                </div>

                <p className="problem-prompt">{lecture.problem_prompt}</p>

                <div className="lecture-practice-foot">
                  <small className="muted">
                    Lesson {index + 1} • Guided coaching enabled
                  </small>
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
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </AppShell>
  );
}
