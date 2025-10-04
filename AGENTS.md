# Repository Guidelines

## Project Structure & Module Organization
Core agents live in `functions/` (`actions/`, `filters/`, `pipes/`) and integrate directly with Open WebUI runtime events. Reusable automation lives in `tools/`, with Python entrypoints (for example `tools/writer_assistant/writer_assistant.py`) meant to be invoked by the UI. Shared documentation sits in `docs/`, while `.tests/` holds pytest suites that exercise the exported modules. Keep upstream mirrors inside `external/` and author prompt templates in `system_prompts/`.

Open WebUI’s upstream documentation is mirrored under `docs/openwebui/` for convenience when building local extensions.

> NOTE: `external/` is a read-only mirror of upstream Open WebUI—treat it as reference only.

## Build, Test, and Development Commands
Set up dependencies with `python -m pip install -e .[dev]`. Run linting locally via `nox -s lint` (equivalent to `ruff check functions tools .tests .scripts`). Execute the full test suite with `nox -s tests` or `pytest -vv --cov=functions --cov=tools`. Use `pre-commit run --all-files` before pushing when hooks are available.

## Coding Style & Naming Conventions
The project targets Python 3.11; prefer 4-space indentation, type hints, and explicit returns. `ruff` enforces style (line length 100, `select = ["ALL"]`, `ignore = ["D203", "D212"]`), so keep docstrings compact and import orders deterministic. Name modules with lowercase underscores (`writer_assistant.py`) and tests with `test_*.py`. For filters and pipes, mirror the behavior in the filename (e.g., `pipes/respond_with_summary.py`).

## Testing Guidelines
Place new tests under `.tests/` alongside fixtures in `conftest.py`. Follow pytest conventions (`test_function_behavior` naming) and keep async scenarios under `pytest.mark.asyncio` when needed. Preserve the existing coverage flags by running `pytest --cov-report=term-missing` and ensure added agents hit their success and failure paths. Update shared fixtures rather than duplicating Open WebUI client mocks.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects; the history mixes `chore: weekly OpenWebUI snapshot` and `updates`, but prefer Conventional Commit prefixes (`feat:`, `fix:`, `chore:`) for clarity. Each pull request should include: purpose summary, linked issue (if any), verification notes (`nox -s tests` output), and screenshots or prompt transcripts when altering UI-facing flows. Keep diffs focused, and mention follow-up work explicitly so downstream contributors can plan their agent releases.
