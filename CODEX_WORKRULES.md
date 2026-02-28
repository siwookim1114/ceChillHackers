# CODEX_WORKRULES.md — Hackathon Lean Discipline

## A) Non-negotiables (P0 Guardrails)
- Never store raw handwriting images on the server (disk/DB/log). Process in-memory then discard.
- Mastery Graph is local-first (IndexedDB/LocalStorage). Server gets only anonymous aggregates + optional final integrity hash.
- STRICT mode: never send originals (raw notes/images) to external services.
- Avoid logging raw user content. If unavoidable, redact by default.

If a shortcut violates any of the above, DO NOT implement it.

---

## B) Work Loop (Lean): Plan → Implement → Smoke Test → Note
### B1) Plan Mode (mandatory, but lightweight)
Before coding, write a short PLAN in your response:
- Goal
- Files to touch
- Risks/edge cases (1–3 bullets)
- Smoke tests to run

DEVLOG update is required only at **milestones**, not every tiny patch.

### B2) Implement in small diffs
- Prefer small, reversible patches.
- Don’t mix refactors with feature changes.
- If API/schema is uncertain, ship a mock JSON stub first, then replace.

### B3) Smoke Test (mandatory every change)
Run the smallest set that proves functionality:
- 1) Chat request works (text or voice transcript path)
- 2) Scan pipeline returns valid JSON schema
- 3) Mastery updates and persists locally
- 4) Plan/PoW gating behaves (locked → submission → unlock)
- 5) No server-side image persistence (quick check)

Record what you tested in your response under “SMOKE TESTS”.

### B4) Notes (light documentation)
- Update `PITFALLS.md` only for:
  - repeats, time-sinks, schema breaks, privacy/security issues
- Update `SECURITY_NOTES.md` only when data flow/storage changes.
- Update `DEVLOG.md` only at milestones (roughly every 2–3 hours or per major feature).

---

## C) Schema Discipline (must)
- UI-consumed model outputs must be strict JSON with a fixed schema.
- Validate:
  - backend: Pydantic
  - frontend: Zod (or equivalent)
- If schema changes: update validators + one fixture example.

---

## D) Optional / If time remains (P1)
- EXIF stripping on client
- Rate limiting uploads
- JWT sessions / httpOnly cookie hardening
- Extended automated tests

---

## E) Definition of Done (Hackathon)
A feature is “done” when:
- Smoke tests pass
- Demo flow doesn’t break
- Guardrails still hold (no image storage, strict mode behavior)
- Any major pitfall recorded if encountered

