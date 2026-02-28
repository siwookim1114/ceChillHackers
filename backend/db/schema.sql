-- AI Tutoring Platform - Auth & Profile Schema
CREATE EXTENSION IF NOT EXISTS "pgcrypto";      --gen_random_uuid()

-- Enum Types
CREATE TYPE user_role AS ENUM ("student", "teacher", "parent");
CREATE TYPE learning_style AS ENUM ("explanation", "question", "problem_solving");
CREATE TYPE learning_pace AS ENUM ("fast", "normal", "slow");

-- 1. Users - core auth
CREATE TABLE users (
id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
email          TEXT        NOT NULL UNIQUE,           -- login identifier
password_hash  TEXT        NOT NULL,                  -- bcrypt / argon2 output
role           user_role   NOT NULL,                  -- student | teacher | parent
display_name   TEXT        NOT NULL DEFAULT '',       -- shown in UI
is_active      BOOLEAN     NOT NULL DEFAULT TRUE,     -- soft-disable account
created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. student_profiles - learning prefs captured at onboarding