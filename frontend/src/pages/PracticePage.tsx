import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, getMe, listCourses } from "../api";
import { clearAuthSession, getAccessToken, saveAuthSession } from "../auth";
import { AppShell } from "../components/AppShell";
import type { CourseFolder } from "../types";

function FolderTileIcon() {
  return (
    <svg
      aria-hidden="true"
      className="practice-folder-icon"
      viewBox="0 0 88 64"
    >
      <path
        className="folder-back"
        d="M7 22a8 8 0 0 1 8-8h18l7 7h33a8 8 0 0 1 8 8v21a8 8 0 0 1-8 8H15a8 8 0 0 1-8-8V22Z"
      />
      <path
        className="folder-front"
        d="M7 30h74v20a8 8 0 0 1-8 8H15a8 8 0 0 1-8-8V30Z"
      />
      <path className="folder-shine" d="M16 35h22a3 3 0 0 1 0 6H16a3 3 0 0 1 0-6Z" />
    </svg>
  );
}

export function PracticePage() {
  const navigate = useNavigate();
  const [authLoading, setAuthLoading] = useState(true);
  const [courses, setCourses] = useState<CourseFolder[]>([]);
  const [loadingCourses, setLoadingCourses] = useState(true);
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

  const totalLectureCount = courses.reduce(
    (sum, course) => sum + course.lecture_count,
    0,
  );
  const totalFileCount = courses.reduce((sum, course) => sum + course.file_count, 0);

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
        <section className="practice-studio-hero reveal reveal-1">
          <div className="practice-hero-copy">
            <p className="overline">Course Folders</p>
            <h3>Launch focused sessions from your own curriculum</h3>
            <p>
              Pick a folder, choose a lecture, and jump straight into guided
              solving. Every lecture starts a clean attempt flow.
            </p>
            <div className="practice-hero-actions">
              <button
                className="btn-primary"
                onClick={() => navigate("/create-course")}
                type="button"
              >
                Build New Course
              </button>
              <button
                className="btn-muted"
                onClick={() => void loadCourses()}
                type="button"
              >
                Refresh Folders
              </button>
            </div>
          </div>

          <div className="practice-hero-metrics">
            <article className="practice-metric-card">
              <small>Folders</small>
              <strong>{courses.length}</strong>
            </article>
            <article className="practice-metric-card">
              <small>Lectures</small>
              <strong>{totalLectureCount}</strong>
            </article>
            <article className="practice-metric-card">
              <small>Files</small>
              <strong>{totalFileCount}</strong>
            </article>
          </div>
        </section>

        {error && <p className="error practice-alert">{error}</p>}

        <section className="panel-card practice-folder-panel practice-folder-gallery reveal reveal-2">
          <div className="practice-section-head">
            <div>
              <p className="overline">Library</p>
              <h3>Course Folders</h3>
            </div>
            <span className="create-kpi-chip">
              {loadingCourses ? "Syncing..." : `${courses.length} total`}
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
                className="course-folder-item practice-folder-item"
                key={course.id}
                onClick={() => navigate(`/practice/course/${course.id}`)}
                title={course.title}
                type="button"
              >
                <span className="practice-folder-icon-wrap">
                  <FolderTileIcon />
                </span>
                <strong className="practice-folder-title">{course.title}</strong>
                <small className="practice-folder-meta">
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
        </section>
      </div>
    </AppShell>
  );
}
