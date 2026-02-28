import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ApiError,
  createCourse,
  createLecture,
  getCourseDetail,
  getMe,
  listCourses,
  uploadLectureFile,
} from "../api";
import { clearAuthSession, getAccessToken, saveAuthSession } from "../auth";
import { AppShell } from "../components/AppShell";
import type { CourseDetail, CourseFolder } from "../types";

export function CreateCoursePage() {
  const navigate = useNavigate();
  const [authLoading, setAuthLoading] = useState(true);
  const [loadingCourses, setLoadingCourses] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const [courses, setCourses] = useState<CourseFolder[]>([]);
  const [selectedCourseId, setSelectedCourseId] = useState<string | null>(null);
  const [selectedCourse, setSelectedCourse] = useState<CourseDetail | null>(
    null,
  );

  const [courseTitle, setCourseTitle] = useState("");
  const [courseSyllabus, setCourseSyllabus] = useState("");
  const [creatingCourse, setCreatingCourse] = useState(false);

  const [lectureTitle, setLectureTitle] = useState("");
  const [lectureDescription, setLectureDescription] = useState("");
  const [lecturePrompt, setLecturePrompt] = useState("");
  const [lectureAnswer, setLectureAnswer] = useState("");
  const [creatingLecture, setCreatingLecture] = useState(false);
  const [uploadingLectureId, setUploadingLectureId] = useState<string | null>(
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
      })
      .catch((err) => {
        // Only log out on 401. Server errors (5xx) should not clear a valid session.
        if (!(err instanceof ApiError && err.status === 401)) {
          return;
        }
        clearAuthSession();
        navigate("/login", { replace: true });
      })
      .finally(() => setAuthLoading(false));
  }, [navigate]);

  const loadCourses = async (preferredCourseId?: string) => {
    setLoadingCourses(true);
    try {
      const nextCourses = await listCourses();
      setCourses(nextCourses);

      if (nextCourses.length === 0) {
        setSelectedCourseId(null);
        setSelectedCourse(null);
        return;
      }

      setSelectedCourseId((previous) => {
        if (
          preferredCourseId &&
          nextCourses.some((course) => course.id === preferredCourseId)
        ) {
          return preferredCourseId;
        }
        if (previous && nextCourses.some((course) => course.id === previous)) {
          return previous;
        }
        return nextCourses[0].id;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load courses");
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

  const loadCourseDetail = async (courseId: string) => {
    setLoadingDetail(true);
    try {
      const detail = await getCourseDetail(courseId);
      setSelectedCourse(detail);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load course details",
      );
      setSelectedCourse(null);
    } finally {
      setLoadingDetail(false);
    }
  };

  useEffect(() => {
    if (!selectedCourseId) {
      setSelectedCourse(null);
      return;
    }
    loadCourseDetail(selectedCourseId);
  }, [selectedCourseId]);

  const handleCreateCourse = async () => {
    const title = courseTitle.trim();
    if (!title) {
      setError("Course title is required.");
      return;
    }

    setCreatingCourse(true);
    setError(null);
    try {
      const created = await createCourse({
        title,
        syllabus: courseSyllabus.trim() || undefined,
      });
      setCourseTitle("");
      setCourseSyllabus("");
      await loadCourses(created.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create course");
    } finally {
      setCreatingCourse(false);
    }
  };

  const handleCreateLecture = async () => {
    if (!selectedCourseId) {
      setError("Create or select a course first.");
      return;
    }
    if (!lectureTitle.trim()) {
      setError("Lecture title is required.");
      return;
    }
    if (!lecturePrompt.trim()) {
      setError("Lecture problem prompt is required.");
      return;
    }
    if (!lectureAnswer.trim()) {
      setError("Lecture answer key is required.");
      return;
    }

    setCreatingLecture(true);
    setError(null);
    try {
      await createLecture(selectedCourseId, {
        title: lectureTitle.trim(),
        description: lectureDescription.trim() || undefined,
        problem_prompt: lecturePrompt.trim(),
        answer_key: lectureAnswer.trim(),
      });
      setLectureTitle("");
      setLectureDescription("");
      setLecturePrompt("");
      setLectureAnswer("");

      await Promise.all([
        loadCourses(selectedCourseId),
        loadCourseDetail(selectedCourseId),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create lecture");
    } finally {
      setCreatingLecture(false);
    }
  };

  const handleUploadLectureFile = async (
    lectureId: string,
    file: File | null,
  ) => {
    if (!selectedCourseId || !file) {
      return;
    }
    setUploadingLectureId(lectureId);
    setError(null);
    try {
      await uploadLectureFile(selectedCourseId, lectureId, file);
      await Promise.all([
        loadCourses(selectedCourseId),
        loadCourseDetail(selectedCourseId),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload file");
    } finally {
      setUploadingLectureId(null);
    }
  };

  const totalLectureCount = courses.reduce(
    (sum, course) => sum + course.lecture_count,
    0,
  );
  const totalFileCount = courses.reduce((sum, course) => sum + course.file_count, 0);
  const selectedLectureCount = selectedCourse?.lectures.length ?? 0;
  const selectedFileCount =
    selectedCourse?.lectures.reduce((sum, lecture) => sum + lecture.file_count, 0) ??
    0;

  if (authLoading) {
    return (
      <AppShell title="Create New Course" subtitle="Checking your session...">
        <section
          className="panel-card session-skeleton"
          aria-label="Loading create course workspace"
        >
          <div className="skeleton-line skeleton-line-short" />
          <div className="skeleton-line skeleton-line-medium" />
          <div className="skeleton-pill-row">
            <span className="skeleton-pill" />
            <span className="skeleton-pill" />
          </div>
        </section>
      </AppShell>
    );
  }

  return (
    <AppShell
      title="Create New Course"
      subtitle="Create a course folder with syllabus, then add lectures and upload lecture files."
    >
      <div className="create-course-layout">
        <section className="create-course-hero reveal reveal-1">
          <div className="create-course-hero-copy">
            <p className="overline">Course Builder</p>
            <h3>Design your own practice curriculum</h3>
            <p>
              Create folders, add lecture prompts, and attach files in one
              workspace. Each lecture becomes a guided session in Practice
              Studio.
            </p>
            <div className="create-course-hero-actions">
              <button
                className="btn-primary"
                onClick={() => navigate("/practice")}
                type="button"
              >
                Open Practice Studio
              </button>
              <button
                className="btn-muted"
                onClick={() => void loadCourses(selectedCourseId ?? undefined)}
                type="button"
              >
                Refresh Folders
              </button>
            </div>
          </div>

          <div className="create-course-hero-metrics">
            <article className="create-course-metric">
              <small>Course Folders</small>
              <strong>{courses.length}</strong>
            </article>
            <article className="create-course-metric">
              <small>Total Lectures</small>
              <strong>{totalLectureCount}</strong>
            </article>
            <article className="create-course-metric">
              <small>Total Files</small>
              <strong>{totalFileCount}</strong>
            </article>
          </div>
        </section>

        {error && <p className="error create-course-alert">{error}</p>}

        <section className="create-course-grid reveal reveal-2">
          <aside className="panel-card course-builder-panel">
            <div className="create-card-head">
              <div className="create-card-title-row">
                <h3>New Course Folder</h3>
                <span className="create-kpi-chip">
                  {loadingCourses ? "Syncing..." : `${courses.length} total`}
                </span>
              </div>
              <p>
                Start with the parent folder. Lectures and files are grouped
                under this course.
              </p>
            </div>

            <label className="create-input-label">
              Course Name
              <input
                onChange={(event) => setCourseTitle(event.target.value)}
                placeholder="Example: Algebra Foundations"
                value={courseTitle}
              />
            </label>

            <label className="create-input-label">
              Syllabus
              <textarea
                onChange={(event) => setCourseSyllabus(event.target.value)}
                placeholder="Week 1: factoring, Week 2: quadratics, Week 3: applications"
                rows={4}
                value={courseSyllabus}
              />
            </label>

            <button
              className="btn-primary"
              disabled={creatingCourse}
              onClick={handleCreateCourse}
              type="button"
            >
              {creatingCourse ? "Creating..." : "Create Course Folder"}
            </button>

            <div className="course-folder-list-wrap">
              <div className="builder-header compact create-folder-head">
                <h4>Course Folders</h4>
                <p>
                  {loadingCourses ? "Loading..." : `${courses.length} total`}
                </p>
              </div>

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
                      {course.lecture_count} lectures • {course.file_count}{" "}
                      files
                    </small>
                  </button>
                ))}
                {!loadingCourses && courses.length === 0 && (
                  <div className="empty-course-state compact">
                    <p>No folders yet. Create your first course folder.</p>
                  </div>
                )}
              </div>
            </div>
          </aside>

          <section className="panel-card lecture-workspace-panel">
            {!selectedCourseId && (
              <div className="empty-course-state">
                <h3>Select or create a course</h3>
                <p>
                  Once you choose a folder, you can add lecture prompts and
                  attach notes/files for each lecture.
                </p>
              </div>
            )}

            {selectedCourseId && (
              <>
                <div className="workspace-head">
                  <div>
                    <h3>{selectedCourse?.title ?? "Loading course..."}</h3>
                    <p>{selectedCourse?.syllabus || "No syllabus yet."}</p>
                  </div>
                  <div className="create-workspace-kpis">
                    <span className="create-kpi-chip">
                      {selectedLectureCount} lectures
                    </span>
                    <span className="create-kpi-chip">
                      {selectedFileCount} files
                    </span>
                  </div>
                </div>

                <div className="lecture-form-grid">
                  <label className="create-input-label">
                    Lecture Title
                    <input
                      onChange={(event) => setLectureTitle(event.target.value)}
                      placeholder="Example: Solving by factoring"
                      value={lectureTitle}
                    />
                  </label>

                  <label className="create-input-label">
                    Description (Optional)
                    <input
                      onChange={(event) =>
                        setLectureDescription(event.target.value)
                      }
                      placeholder="Short summary for this lecture"
                      value={lectureDescription}
                    />
                  </label>

                  <label className="create-input-label">
                    Practice Prompt
                    <textarea
                      onChange={(event) => setLecturePrompt(event.target.value)}
                      placeholder="Example: Solve x^2 - 5x + 6 = 0"
                      rows={3}
                      value={lecturePrompt}
                    />
                  </label>

                  <label className="create-input-label">
                    Answer Key
                    <input
                      onChange={(event) => setLectureAnswer(event.target.value)}
                      placeholder="Example: 2,3"
                      value={lectureAnswer}
                    />
                  </label>
                </div>

                <button
                  className="btn-teal"
                  disabled={creatingLecture || loadingDetail}
                  onClick={handleCreateLecture}
                  type="button"
                >
                  {creatingLecture ? "Adding Lecture..." : "Add Lecture"}
                </button>

                <div className="lecture-card-list">
                  {loadingDetail && (
                    <p className="muted">Loading lectures...</p>
                  )}
                  {!loadingDetail && selectedCourse?.lectures.length === 0 && (
                    <p className="muted">
                      No lectures yet. Add your first lecture above.
                    </p>
                  )}

                  {selectedCourse?.lectures.map((lecture) => (
                    <article className="lecture-admin-card" key={lecture.id}>
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

                      <div className="lecture-upload-row">
                        <label
                          className="btn-muted file-upload-btn create-upload-btn"
                          htmlFor={`file-${lecture.id}`}
                        >
                          {uploadingLectureId === lecture.id
                            ? "Uploading..."
                            : "Upload Lecture File"}
                        </label>
                        <small className="muted">
                          PDF, docs, and notes supported
                        </small>
                        <input
                          className="upload-input"
                          id={`file-${lecture.id}`}
                          onChange={(event) => {
                            const nextFile = event.target.files?.[0] ?? null;
                            void handleUploadLectureFile(lecture.id, nextFile);
                            event.currentTarget.value = "";
                          }}
                          type="file"
                        />
                      </div>

                      {lecture.files.length > 0 && (
                        <ul className="lecture-file-list">
                          {lecture.files.map((file) => (
                            <li key={file.id}>
                              <span>{file.file_name}</span>
                              <small>
                                {Math.max(
                                  1,
                                  Math.round(file.size_bytes / 1024),
                                )}{" "}
                                KB
                              </small>
                            </li>
                          ))}
                        </ul>
                      )}
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
