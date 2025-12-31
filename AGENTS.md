# Repository Guidelines

## Project Structure & Module Organization
- `services/`, `libs/`, `utils/`, and `helper/` contain the core Python modules and helpers.
- `db_migration/` holds SQL and database migration assets (not Python package code).
- `docs/` stores project documentation and reference data.
- `examples/` includes runnable sample scripts (e.g., `examples/examples.py`).
- `tests/` is reserved for pytest tests (currently empty).

## Build, Test, and Development Commands
- `make install`: install dependencies via `uv sync`.
- `make test`: run the pytest suite with `uv run pytest`.
- `make format`: apply formatting and import sorting with Ruff.
- `make lint`: run Ruff checks in diff mode (no writes).
- Ad hoc examples: `uv run python examples/examples.py`.

## Coding Style & Naming Conventions
- Python is the primary language; use 4‑space indentation.
- Formatting and linting are enforced by Ruff (`ruff format` and `ruff check`).
- Prefer clear, descriptive snake_case for functions and variables, and PascalCase for classes.
- Keep modules small and focused; place shared utilities in `utils/` or `libs/`.

## Testing Guidelines
- Test runner: pytest (see `make test`).
- Place tests under `tests/` and name files `test_*.py`.
- No coverage thresholds are configured; add tests for new behavior and regressions.

## Commit & Pull Request Guidelines
- Commit messages in history are short, descriptive, and lowercase (e.g., “fixed the issue…”).
- Aim for one logical change per commit and avoid noisy formatting-only commits.
- PRs should include a concise description, steps to verify, and links to relevant issues.
- Add screenshots or logs when changes affect outputs or data flows.

## Configuration & Environment
- Local configuration uses `.env` (already in repo); avoid committing secrets.
- `setup.sh` provisions dependencies and creates `chroma_db/` for vector storage.
