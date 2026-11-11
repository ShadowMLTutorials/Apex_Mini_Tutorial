# Repository Guidelines

## Compatibility

- Support Python 3.9 or newer.
- Keep core tests and smoke tests runnable on CPU without CUDA.
- Preserve existing PyTorch Lightning checkpoint compatibility unless a task explicitly requires otherwise.
- Preserve existing CLI behavior and defaults when adding options.

## Testing

- Add focused automated tests for behavioral changes.
- Tests must not require network access, downloaded training data, a model checkpoint, a GPU, or a Stockfish installation.
- Prefer Python's standard-library `unittest` unless an existing test framework is introduced for a clear reason.
- Keep smoke tests bounded so they finish quickly on CPU.

## Project Scope

- Keep changes localized to the requested feature.
- Do not modify `utils/indexUCI.py` or the model architecture unless the task specifically requires it.
- Avoid new runtime dependencies when the standard library or existing dependencies are sufficient.

## Repository Hygiene

- Do not add generated PGNs, JSONL datasets, checkpoints, logs, caches, or virtual environments.
- Do not commit or push unless explicitly requested.
- Report the exact validation commands run and their actual outcomes.
- Do not describe checks as passing if they were not run successfully.
