# Project Structure

This repository follows a common GitHub Python layout:

- `src/`: application source code packages
- `tests/`: test code
- `config/`: runtime and environment configuration files
- `docs/`: project documentation
- `scripts/`: runnable helper scripts
- `models/`: model and mesh assets
- `data/`: sample/input data

Notes:

- The root `main.py` keeps backward compatibility and adds `src/` to `PYTHONPATH` at runtime.
- New modules should be added under `src/`.
