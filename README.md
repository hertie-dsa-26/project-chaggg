# Chicago Crime Analysis Project

[![CI](https://github.com/hertie-dsa-26/project-chaggg/actions/workflows/ci.yml/badge.svg)](https://github.com/hertie-dsa-26/project-chaggg/actions/workflows/ci.yml)

This repository is for a group project for the Data Structures and Algorithms Course taught at the Hertie School in Berlin.

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — modules, data flow, design tradeoffs (rubric: Architecture & Design).
- **[Contributing](docs/CONTRIBUTING.md)** — branches, PRs, local checks, CI (rubric: Development Process).
- **[Rubric mapping](docs/RUBRIC_MAPPING.md)** — rubric maddeleri ↔ repodaki kanıtlar (review / savunma için).

## Testing

```bash
uv sync
CHAGGG_SKIP_DATA_LOAD=1 uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Pull requests run the same tests in GitHub Actions (see `.github/workflows/ci.yml`).

## Project Overview

Working in a team of 7 students, we are designing and developing a comprehensive **web application using Flask** that performs end-to-end analysis of a dataset.

## Dataset

We are using the City of Chicago's crime dataset, which provides comprehensive information on reported incidents from 2001 to present. We are using all data up to 31 December 2025.

[Crimes - 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data)

## Dataset Setup

The datasets are not stored in this repository. To download them, follow these steps in your terminal:
1. git clone <repo>
2. uv sync
3. uv run scripts/main.py

## Development Workflow

To maintain code quality and ensure collaboration, please follow this workflow:

1. **Create a branch** – Create a new branch either:
   - Locally in your IDE, or
   - On GitHub (then switch to it in your IDE)
2. **Develop locally** – Write and test code in your personal notebooks until you're confident it works
3. **Integrate code** – Copy your working code into the relevant script
4. **Commit changes** – Commit your changes to this branch with a descriptive commit message that everyone understand
6. **Open a pull request** – Create a pull request with a brief description of your changes
7. **Request review** – Tag a team member for code review
8. **Merge** – After approval, the pull request can be merged into `main`

### Important Guidelines

⚠️ **DO NOT commit directly to `main`**

✅ **ALL CODE must be reviewed by a second person before merging**


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D69TCBIW)
