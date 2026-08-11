# GitHub Publication Checklist

1. Confirm `.env`, SQLite databases, logs, traces and benchmark result artifacts are ignored.
2. Choose and add a license before public release.
3. Run `python -m unittest discover -s tests -v`.
4. Run `durable-agent benchmark` and confirm the quality gate passes.
5. Run one deterministic foreground and one background smoke test.
6. Review README claims and keep regression metrics distinct from held-out metrics.
7. Initialize and publish:

```powershell
git init
git add .
git status
git commit -m "Initial durable Agent runtime"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

Inspect `git status` before committing so local secrets and runtime artifacts are never uploaded.
