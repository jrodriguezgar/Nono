# Contributing to Nono

Thank you for considering contributing to **Nono** — No Overhead, Neural Operations.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Nono.git
   cd Nono
   ```
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature
   ```
4. **Install dependencies** with [uv](https://docs.astral.sh/uv/):
   ```bash
   pip install uv
   uv sync
   ```

## Code Style

- Follow **PEP 8** conventions
- Use **type hints** on all functions and methods
- Use **Google-style docstrings** for public functions
- See [`.github/instructions/python-development.instructions.md`](.github/instructions/python-development.instructions.md) for full coding standards

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Purpose |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `chore:` | Maintenance, CI, tooling |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `test:` | Adding or updating tests |

Example: `feat: add support for Anthropic provider`

## Testing

Run the test suite before submitting a PR:

```bash
uv run pytest -v --tb=short
```

## Linting

```bash
uv run ruff check .
```

## Submitting a Pull Request

1. Ensure all tests pass and the linter reports no errors
2. Update `CHANGELOG.md` under `[Unreleased]` with your changes
3. Push your branch and open a PR against `main`
4. Fill in the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
5. Link any related issues (`Closes #123`)

## Reporting Bugs vs Requesting Features

- **Bug reports**: Use the [Bug Report](https://github.com/jrodriguezgar/Nono/issues/new?template=bug_report.yml) template
- **Feature requests**: Use the [Feature Request](https://github.com/jrodriguezgar/Nono/issues/new?template=feature_request.yml) template

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)
