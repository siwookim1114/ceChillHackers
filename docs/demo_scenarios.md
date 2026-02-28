# Demo Scenarios (RevisionDojo MVP)

## 1) Quadratic Basics

- Problem: `x^2 - 5x + 6 = 0`
- Stuck acting point:
  - Pause for 20s before factoring
  - Erase repeatedly while trying signs
  - Submit wrong answer once (`x=2` only)
- Hint samples:
  - Level 1: "What two numbers multiply to 6 and add to -5?"
  - Level 2:
    - "Use factoring form `(x-a)(x-b)=0`."
    - "Find `a,b` from product/sum constraints."
    - "Then apply zero-product rule."
  - Level 3: "Mini-problem: Factor `x^2-5x+6` first, ignore solving. First step: list factor pairs of 6."

## 2) Derivative Basics

- Problem: `d/dx (3x^2 + 2x - 1)`
- Stuck acting point:
  - Stop writing after expansion attempt
  - Trigger two erase actions
  - Ask coach hint manually
- Hint samples:
  - Level 1: "Which differentiation rule applies term-by-term here?"
  - Level 2:
    - "Differentiate each term separately."
    - "`d/dx(3x^2)=6x`, `d/dx(2x)=2`, `d/dx(-1)=0`."
    - "Combine results."
  - Level 3: "Mini-problem: only compute `d/dx(x^2)` first. Then scale by 3."

## 3) Linear Equation

- Problem: `2x + 7 = 19`
- Stuck acting point:
  - Keep rewriting same line without isolating `x`
  - Submit wrong answer (`x=5`) and retry
- Hint samples:
  - Level 1: "What single operation removes `+7` from the left side?"
  - Level 2:
    - "Subtract 7 from both sides."
    - "Then divide both sides by 2."
    - "Check by substitution."
  - Level 3: "Mini-problem: Solve `x + 7 = 19` first, then adapt to `2x + 7 = 19`."
