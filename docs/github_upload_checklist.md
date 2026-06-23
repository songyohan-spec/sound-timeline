# GitHub Upload Checklist

This project generates large audio, model, and report artifacts. Do not commit
those files to GitHub unless you intentionally create a separate release bundle.

## Commit

Recommended files to commit:

- `.gitignore`
- `README.md`
- `environment.yml`
- `configs/`
- `docs/`
- `scripts/`
- `src/`

Ignored local artifacts:

- `data/`
- `models/`
- `outputs/`
- `.venv/`
- generated audio files (`*.wav`, `*.mp3`, `*.flac`, etc.)

## Local Commands

Run these from the project root in a terminal where Git is installed:

```powershell
git status
git add .gitignore README.md environment.yml configs docs scripts src
git status
git commit -m "Add stem-aware synth profiling pipeline"
```

Then connect a GitHub remote:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

## Important Caveat

If large files were already tracked before `.gitignore` was added, `.gitignore`
will not remove them from Git history. In that case, remove them from the index
without deleting local files:

```powershell
git rm -r --cached data models outputs
git commit -m "Stop tracking generated artifacts"
```
