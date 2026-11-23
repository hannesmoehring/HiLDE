# Anonymous submission checklist

Goal: no reasonable reviewer stumbles onto the authors or institution by accident. This is
not foolproof, and the encoded URL in `index.html` is obfuscation, not secrecy.

## Live app

- [ ] Pick a neutral hostname that is not on any personal domain (not `3m0.de`) and has no
      name, initials or institution in it. Point the Cloudflare tunnel at it.
- [ ] Run the app live. Do not turn on cache-only mode.
- [ ] Do not look at Cloudflare analytics or access logs for this hostname while review is
      running. Visitor IPs could identify reviewers.

## GitHub Pages wrapper

- [ ] Create a neutral GitHub account for the Pages repo: no real name, avatar, bio or linked
      email. A repo under `hannesmoehring.github.io` would give the name away.
- [ ] Encode the final URL and write it into the page:
      `python submission/encode_url.py https://your-neutral-host --write`
      Check it with `python submission/encode_url.py --decode <printed string>`.
- [ ] Copy only `submission/index.html` into the new repo as `index.html` and turn on Pages.
      Commit as the neutral account (set `git config user.name` and `user.email` in that repo).
- [ ] Open the Pages URL in a private window and check that the app works inside the frame:
      loading, navigation, clipboard and fullscreen. Make sure no response sets
      `X-Frame-Options` or a CSP `frame-ancestors` that blocks it.

## Code upload (anonymous.4open.science)

- [ ] Run `python submission/check_terms.py`. It must report 0 hits.
- [ ] Upload the `submission-main` branch and paste the whole of
      `submission/anonymization-terms.txt` as the term list. The term file and this
      checklist name the terms on purpose; pasting the full list masks them too.
- [ ] Look through the anonymized preview: README, docs, file tree and figures,
      including PDF metadata. Term replacement does not reach binary files.
- [ ] Make sure the paper links the Pages URL and the anonymous repo, not the real ones.
- [ ] Push only the branch, never the tags: `final-thesis` and `v0.1.0-thesis` exist, and
      git history still has the real author, email and the removed thesis PDF.

## Term tracking

When something identifying turns up, add it to `anonymization-terms.txt`, remove it from
the code, add a row here, and rerun `check_terms.py`.

| Term / item | Where it was | What was done |
|---|---|---|
| Hannes, Moehring, Möhring, hannesmoehring | `pyproject.toml` authors, `.gitignore` comment, `scripts/download_dataset.sh` TODO, thesis PDF | Authors set to "Anonymous", comments removed, PDF deleted |
| `hannes.moehring@tum.de`, `acc@hannesmoehring.de` | `pyproject.toml`; git commit metadata | Removed from files; history is not uploaded |
| 3m0 / `hilde.3m0.de` | README badge and live-demo link | Badge removed, link replaced by `https://REPLACE-ME.example` |
| YouTube intro video | README badge, top-bar "Intro video" link in `frontend/src/App.tsx` | Both removed, with the related CSS |
| thesis, Bachelor | `thesis/thesis.pdf`, README, `pyproject.toml` description, `src_research/` docstrings, `scripts/`, `src/README.md` | PDF deleted; wording changed to "paper"; questionnaire PDF moved to `docs/` |
| §6.x / "Table 6.2" chapter numbers | `src_research/benchmark_workflow.py`, its output CSV name, README | Replaced with neutral names (`summary_table_cells.csv`) |
| Jacob Miller, Stephen Kobourov (also "Kuborov") | Only in the deleted thesis PDF | Nothing left in the tree |
| TUM, TU Munich, Munich and German spellings | `pyproject.toml` email, thesis PDF | Nothing left in the tree |
| Chair / group name | Not found in the tree | ____ add the exact name to the term file |
| Local paths `/Users/...` | `outputs/profiler_report*.html`, `/api/mode` `cache_dir`, server error messages | Reports deleted, `cache_dir` returns null, errors scrubbed by `backend/redact.py` |
| Personal hosting (desktop server, Portainer, NPM, home Cloudflare tunnel) | README, `docker-compose.yml`, `Dockerfile` comments | Reworded generically |
| Former project name SHD | README, compose service/image name, `frontend/package.json` | README sentence removed, renamed to `hilde`; `pyproject.toml` still uses `shd` (renaming needs a re-lock; low risk) |
| `hm_*` initials | `.gitignore`, `.dockerignore`, the `"hm"` predicate alias in `src/analysis/predicate_generator.py` | Kept: initials only, and the alias is code (left alone to avoid breaking predicates) |
| Reviewer IPs | uvicorn access log | `--no-access-log` in `Dockerfile`, `access_log=False` in `host.py` |
