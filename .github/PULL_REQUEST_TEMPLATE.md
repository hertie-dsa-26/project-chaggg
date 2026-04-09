## Summary

<!-- What does this PR do and why? -->

## How to test

<!-- Commands, manual steps, or “N/A” for docs-only -->

```bash
CHAGGG_SKIP_DATA_LOAD=1 uv run python -m unittest discover -s tests -p "test_*.py" -q
```

## Checklist

- [ ] Tests pass locally (and CI is green).
- [ ] At least one reviewer assigned (no self-merge without team rules).
- [ ] No secrets or personal tokens committed.
- [ ] User-facing changes documented or linked (`docs/` or README) if needed.

## Rubric / quality (optional)

- [ ] Architecture: changes fit existing module boundaries (`docs/ARCHITECTURE.md`).
- [ ] Edge cases or limitations noted where relevant.
