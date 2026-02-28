"""Deterministic scan parser for handwritten math submissions.

This parser is local/in-memory and does not persist raw images.
It extracts a structured `ScanParseInput` plus diagnostics scores.
"""

from __future__ import annotations

import base64
import binascii
import re
from io import BytesIO
from typing import Any

from db.models import (
    ScanParseInput,
    ScanParserDiagnostics,
    ScanParserRequest,
    ScanParserResponse,
    StudentStep,
)

_PROBLEM_PREFIX_RE = re.compile(
    r"^\s*(solve|find|compute|evaluate|determine|differentiate|integrate|prove|show)\b",
    re.IGNORECASE,
)
_NUMBERED_STEP_RE = re.compile(r"^\s*(?:step\s*)?(\d+)[\)\].:\-]\s*(.+)$", re.IGNORECASE)
_UNITS_RE = re.compile(
    r"\b(?:m/s\^2|m/s|cm|mm|km|m|kg|g|mg|s|min|h|n|pa|j|w|v|a|ohm|%|°c|k)\b",
    re.IGNORECASE,
)
_EQUATION_CHAR_RE = re.compile(r"[=+\-*/^()]")


def _clean_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def _is_problem_line(line: str) -> bool:
    if "?" in line:
        return True
    if _PROBLEM_PREFIX_RE.search(line):
        return True
    return False


def _extract_problem_statement(lines: list[str], hint: str | None) -> str | None:
    for line in lines[:3]:
        if _is_problem_line(line):
            return line
    if hint and hint.strip():
        return hint.strip()
    return lines[0] if lines else None


def _extract_final_answer(lines: list[str], answer_hint: str | None) -> str | None:
    markers = ("final answer", "answer:", "ans:", "therefore", "thus")
    for line in reversed(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in markers):
            if ":" in line:
                return line.split(":", 1)[1].strip() or line.strip()
            return line.strip()
    for line in reversed(lines):
        if "=" in line:
            return line.strip()
    if answer_hint and answer_hint.strip():
        return answer_hint.strip()
    return None


def _extract_units(text: str, units_hint: str | None) -> str | None:
    matches = [m.group(0) for m in _UNITS_RE.finditer(text or "")]
    if matches:
        # Keep deterministic, deduplicated order.
        seen: set[str] = set()
        ordered = []
        for unit in matches:
            normalized = unit.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(unit)
        return ", ".join(ordered)
    if units_hint and units_hint.strip():
        return units_hint.strip()
    return None


def _line_math_fragment(line: str) -> str | None:
    if not _EQUATION_CHAR_RE.search(line):
        return None
    fragment = re.sub(r"\s+", " ", line).strip()
    return fragment or None


def _extract_steps(
    lines: list[str],
    problem_statement: str | None,
    final_answer: str | None,
) -> list[StudentStep]:
    filtered = []
    for line in lines:
        if problem_statement and line == problem_statement:
            continue
        lowered = line.lower()
        if final_answer and (line == final_answer or final_answer in line):
            continue
        if "final answer" in lowered or lowered.startswith("ans:") or lowered.startswith("answer:"):
            continue
        filtered.append(line)

    steps: list[StudentStep] = []
    next_idx = 1
    for line in filtered:
        if len(line) < 2:
            continue
        numbered = _NUMBERED_STEP_RE.match(line)
        if numbered:
            idx = int(numbered.group(1))
            content = numbered.group(2).strip()
        else:
            idx = next_idx
            content = line
        next_idx = max(next_idx + 1, idx + 1)
        steps.append(
            StudentStep(
                step_index=idx,
                content=content,
                extracted_math=_line_math_fragment(content),
                units=_extract_units(content, None),
            )
        )

    # Ensure stable ordering + unique indices.
    sorted_steps = sorted(steps, key=lambda item: item.step_index)
    deduped: list[StudentStep] = []
    used: set[int] = set()
    auto_idx = 1
    for step in sorted_steps:
        idx = step.step_index
        if idx in used:
            while auto_idx in used:
                auto_idx += 1
            idx = auto_idx
        used.add(idx)
        deduped.append(
            StudentStep(
                step_index=idx,
                content=step.content,
                extracted_math=step.extracted_math,
                units=step.units,
            )
        )
    return deduped


def _compute_focus_effort(
    lines: list[str],
    steps: list[StudentStep],
    final_answer: str | None,
) -> tuple[int, int]:
    equation_like_lines = sum(1 for line in lines if _EQUATION_CHAR_RE.search(line))
    ambiguous_lines = sum(
        1
        for line in lines
        if any(token in line.lower() for token in ("idk", "not sure", "maybe", "guess"))
    )
    steps_count = len(steps)
    text_len = sum(len(line) for line in lines)

    focus = 30 + min(35, steps_count * 7) + min(20, equation_like_lines * 4)
    if final_answer:
        focus += 10
    focus -= min(20, ambiguous_lines * 6)
    focus = max(0, min(100, focus))

    effort = 15 + min(45, steps_count * 8) + min(20, equation_like_lines * 3)
    effort += min(20, int(text_len / 30))
    effort = max(0, min(100, effort))
    return focus, effort


def _decode_image_bytes(image_bytes_b64: str) -> bytes:
    try:
        decoded = base64.b64decode(image_bytes_b64.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_bytes_b64 is invalid base64") from exc
    if not decoded:
        raise ValueError("image_bytes_b64 decoded to empty bytes")
    if len(decoded) > 5 * 1024 * 1024:
        raise ValueError("image payload must be <= 5MB")
    return decoded


def _ocr_text_from_image(image_bytes: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        warnings.append(
            "Local OCR engine unavailable; provide ocr_text from frontend OCR."
        )
        return "", warnings

    try:
        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        text = text.strip()
        if not text:
            warnings.append("OCR produced empty text.")
        return text, warnings
    except Exception:
        warnings.append("OCR failed to parse image bytes.")
        return "", warnings


def _compute_confidence(
    *,
    problem_statement: str | None,
    steps: list[StudentStep],
    final_answer: str | None,
    image_input_present: bool,
    ocr_text_present: bool,
    used_hint_only: bool,
) -> float:
    score = 0.0
    if image_input_present:
        score += 0.08
    if ocr_text_present:
        score += 0.3
    if problem_statement:
        score += 0.2
    if steps:
        score += min(0.3, 0.12 + 0.06 * len(steps))
    if final_answer:
        score += 0.2
    if used_hint_only:
        score = min(score, 0.35)
    return round(max(0.0, min(1.0, score)), 3)


def parse_scan_submission(request: ScanParserRequest) -> ScanParserResponse:
    image_bytes = _decode_image_bytes(request.image_bytes_b64)
    image_input_present = bool(image_bytes)
    warnings: list[str] = []

    ocr_text = (request.ocr_text or "").strip()
    if not ocr_text:
        extracted_text, ocr_warnings = _ocr_text_from_image(image_bytes)
        warnings.extend(ocr_warnings)
        ocr_text = extracted_text

    lines = _clean_lines(ocr_text)
    ocr_text_present = bool(ocr_text)

    problem_statement = _extract_problem_statement(lines, request.problem_statement_hint)
    final_answer = _extract_final_answer(lines, request.answer_hint)
    steps = _extract_steps(lines, problem_statement, final_answer)

    if not any([problem_statement, final_answer, steps]):
        raise ValueError(
            "Could not extract problem content from image. Provide ocr_text or hint fields."
        )

    units_source = "\n".join(lines + [final_answer or ""])
    units = _extract_units(units_source, request.units_hint)

    used_hint_only = not ocr_text_present and any(
        [
            bool(request.problem_statement_hint and request.problem_statement_hint.strip()),
            bool(request.answer_hint and request.answer_hint.strip()),
            bool(request.units_hint and request.units_hint.strip()),
        ]
    )
    if used_hint_only:
        warnings.append("Parsed mostly from hints due to missing OCR text.")

    confidence = _compute_confidence(
        problem_statement=problem_statement,
        steps=steps,
        final_answer=final_answer,
        image_input_present=image_input_present,
        ocr_text_present=ocr_text_present,
        used_hint_only=used_hint_only,
    )

    if not steps:
        warnings.append("No explicit step-by-step lines detected.")
    if not final_answer:
        warnings.append("Final answer could not be confidently identified.")

    focus_score, effort_score = _compute_focus_effort(lines, steps, final_answer)

    # Ensure strict contract from existing TA schema.
    scan_parse = ScanParseInput(
        problem_statement=problem_statement,
        steps=steps,
        final_answer=final_answer,
        units=units,
        raw_parser_confidence=confidence,
    )
    diagnostics = ScanParserDiagnostics(
        focus_score=focus_score,
        effort_score=effort_score,
        text_lines=len(lines),
        equation_like_lines=sum(
            1 for line in lines if _EQUATION_CHAR_RE.search(line)
        ),
        image_input_present=image_input_present,
        warnings=warnings,
    )
    return ScanParserResponse(scan_parse=scan_parse, diagnostics=diagnostics)
