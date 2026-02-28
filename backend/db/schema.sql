-- AI Tutoring Platform - Auth & Profile Schema
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- Enum Types
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
    CREATE TYPE user_role AS ENUM ('student', 'teacher', 'parent');
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'learning_style') THEN
    CREATE TYPE learning_style AS ENUM ('explanation', 'question', 'problem_solving');
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'learning_pace') THEN
    CREATE TYPE learning_pace AS ENUM ('fast', 'normal', 'slow');
  END IF;
END $$;

-- 1. Users - core auth
CREATE TABLE IF NOT EXISTS users (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email         TEXT        NOT NULL UNIQUE,
  password_hash TEXT        NOT NULL,
  role          user_role   NOT NULL,
  display_name  TEXT        NOT NULL DEFAULT '',
  is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- 2. student_profiles - learning prefs captured at onboarding
CREATE TABLE IF NOT EXISTS student_profiles (
  user_id        UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  learning_style learning_style NOT NULL DEFAULT 'explanation',
  learning_pace  learning_pace  NOT NULL DEFAULT 'normal',
  target_goal    TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_student_profiles_user_id ON student_profiles(user_id);