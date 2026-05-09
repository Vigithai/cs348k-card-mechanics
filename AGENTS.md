# AGENTS.md

## Source Of Truth

Use `cs_348_k_balatro_mvp_spec.md` as the source of truth for game rules, scope, and behavior.

## Project Purpose

- Simplified Balatro-like MVP environment for CS348K.
- Focus on baseline bots, reproducible simulation, and strategy-trace analysis.

## Coding Preferences

- Keep environment logic separate from agent logic.
- Prefer clarity over cleverness.
- Keep functions small and readable.
- Use explicit naming.
- Add docstrings to public classes and functions.
- Add or update deterministic unit tests for nontrivial changes.

## Architecture Constraints

- Legal action enumeration stays in the environment.
- Agents only choose from provided legal actions.
- Do not add full Balatro mechanics unless explicitly requested.
- Keep behavior aligned with `cs_348_k_balatro_mvp_spec.md`.

## Explanation Preference

- For major changes, summarize what changed, why, and which files or functions were touched.
