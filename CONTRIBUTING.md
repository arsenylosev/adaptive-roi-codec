# Contributing

## Commit messages (Conventional Commits)

This repository uses [Conventional Commits](https://www.conventionalcommits.org/) for all git history.

### Format

```
<type>[optional scope]: <short description>

[optional body — explain why, not what the diff already shows]
```

- **Subject line:** imperative mood, lowercase type, no trailing period, ≤ 72 characters.
- **Body:** optional; wrap at ~72 characters; separate from subject with a blank line.

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature or user-visible behavior |
| `fix` | Bug fix |
| `docs` | Documentation only (README, ADRs, guides) |
| `test` | Tests only (no production code change) |
| `refactor` | Code change that is not a fix or feature |
| `chore` | Tooling, dependencies, CI, config |
| `perf` | Performance improvement |

### Scopes (optional)

Use when the change is clearly bounded, e.g. `feat(train):`, `docs(cloud):`, `fix(loader):`.

### Examples

```
feat: add Kvasir video frame loader for 336×336 training

Stream labelled MP4s from S3 mount with frame_stride and train/val splits.
```

```
docs(cloud): document yc storage s3 upload workflow

Clarify ru-central1 requirement to avoid AWS SignatureDoesNotMatch errors.
```

```
fix(quantizer): correct batch dimension in spatial step
```

### Rules

- One logical change per commit (atomic commits).
- Do not mix refactors with features in the same commit.
- Never commit secrets (`.env`, keys, tokens).
- Run `uv run pytest` before pushing training or loader changes.

### Pull requests

PR titles should follow the same `type: description` format as the primary commit.
