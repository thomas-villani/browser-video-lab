# CLAUDE.md

Browser-only video understanding runtime; the library is **videodoc** (analyze once into
`<video>.videodoc.json`, query many times). Start with **HANDOFF.md** (current state, next
step, traps), then NOTES.md (measurements, checklist) and DESIGN.md (design, roadmap).

- Test: `npm test` · types: `npm run typecheck` · CLI: `npm run videodoc -- check|analyze|info|layers`
- Python only via `uv run ...`.
- Node runs `.ts` directly: Node-loaded code (`src/doc`, `src/track`, `cli`, `scripts/*.ts`,
  `tests`) imports with `.ts` extensions and uses only erasable TS syntax.
- Tools print JSON on stdout, progress and summaries on stderr.
- In Git Bash, prefix commands that pass `/media/...` URL paths with `MSYS_NO_PATHCONV=1`.
- Benchmark on AC power with the "Best performance" power overlay only.
- Record new measurements in NOTES.md; keep DESIGN.md's roadmap current, and docs/ (user
  documentation: CLI, MCP, format, library API, redaction) in step with what the code does.
