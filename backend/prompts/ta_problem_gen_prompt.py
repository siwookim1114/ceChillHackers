"""TA Problem Generation prompt templates.

Used by the problem generation pipeline to create actual, specific
practice problems grounded in course material retrieved via RAG.
"""

TA_PROBLEM_GEN_SYSTEM_PROMPT = """\
You are a TA problem generator for an AI tutoring platform. Your job is to \
create specific, well-structured practice problems that test the student's \
understanding of the given topic.

## Core Rules
1. Generate REAL, specific problems with concrete numbers, variables, or \
scenarios -- NOT generic descriptions like "Solve one easy exercise".
2. Each problem must have a clear, unambiguous statement a student can solve.
3. Adapt difficulty to the specified level.
4. Ground problems in the provided course material when available.
5. Include step-by-step solution outlines (key steps, not full solutions).
6. Include progressive hints from least to most revealing.
7. Identify common mistakes students make on each problem type.

## Difficulty Guidelines
- very_easy: Direct application of a single concept, small numbers.
- easy: One-step problems requiring basic understanding.
- medium: Multi-step problems combining 2-3 concepts.
- hard: Complex problems requiring insight or less obvious approaches.
- challenge: Problems requiring creative problem-solving or proofs.

## Response Format (STRICT JSON)
Return ONLY a JSON object with this structure:
{
  "problems": [
    {
      "statement": "The specific problem statement with concrete numbers/variables",
      "topic": "the specific sub-topic tested",
      "solution_outline": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
      "hint_ladder": ["Least revealing hint", "Medium hint", "Most revealing hint"],
      "common_mistakes": [
        {"label": "Name", "reason": "Why students make it", "fix": "How to avoid"}
      ]
    }
  ]
}

Do NOT include markdown code fences, extra keys, or commentary outside the JSON.
"""


def build_ta_problem_gen_user_prompt(
    user_message: str,
    topic: str,
    level: str,
    learning_style: str,
    pace: str,
    num_problems: int,
    difficulty_plan: list[str],
    rag_context: str,
    rag_found: bool,
    recent_error_tags: list[str] | None = None,
) -> str:
    """Assemble the user prompt for LLM-based problem generation."""
    parts: list[str] = []

    # RAG context section
    if rag_found and rag_context:
        parts.append(
            "## Course Material (from Knowledge Base)\n"
            f"{rag_context}\n\n"
            "Use this material to create problems that align with the course content. "
            "If the material contains practice problems or examples, adapt them."
        )
    else:
        parts.append(
            "## Course Material\n"
            "No specific course material was found in the knowledge base.\n"
            "Generate problems based on standard curriculum for this topic and level."
        )

    # Student information
    parts.append(
        "## Student Information\n"
        f"- Topic: {topic}\n"
        f"- Level: {level}\n"
        f"- Learning style: {learning_style}\n"
        f"- Pace: {pace}\n"
        f"- Student request: {user_message}"
    )

    # Difficulty plan
    difficulty_str = ", ".join(difficulty_plan)
    parts.append(
        f"## Generation Requirements\n"
        f"Generate exactly {num_problems} problem(s).\n"
        f"Difficulty sequence: {difficulty_str}"
    )

    # Error tags context
    if recent_error_tags:
        tags_str = ", ".join(recent_error_tags)
        parts.append(
            f"## Recent Weak Areas\n"
            f"The student recently struggled with: {tags_str}\n"
            "Design problems that help strengthen these specific areas."
        )

    # Output reminder
    parts.append(
        "## Output\n"
        "Return ONLY a strict JSON object with a 'problems' array.\n"
        f"Generate exactly {num_problems} problems with difficulties: {difficulty_str}.\n"
        "Each problem MUST have: statement, topic, solution_outline, hint_ladder, common_mistakes.\n"
        "The statement MUST be a SPECIFIC, solvable problem with actual numbers/expressions."
    )

    return "\n\n".join(parts)
