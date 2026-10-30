# Contributing

## Setup

```bash
npm install          # also copies ONNX Runtime's wasm files into public/ort
npm run models       # the default models into public/models (checked by SHA-256; YuNet needs uv)
npm run videodoc -- check
```

See [getting started](docs/getting-started.md) for what each model does and where it comes from.

## Before a pull request

```bash
npm run lint         # Biome: lint, formatting (TS/JS) and import order; `npm run format` fixes
npm run typecheck
npm test             # pure TS, seconds
npm run test:browser # headless Edge with WebGPU on the fixtures, ~4 min (`npm run fixtures` first)
```

CI runs everything but the browser tests on each pull request: those need a GPU, the models
and footage that is not in git, so run them locally when a change touches the browser side
(`src/runtime`, `src/app`, `src/analyze.ts`, the redaction pipeline). An intended change in a
measured number is accepted with `UPDATE_BASELINE=1 npm run test:browser`; say why in the pull
request.

Biome formats TypeScript and JavaScript only; CSS and Markdown keep their hand-written layout.
`git config blame.ignoreRevsFile .git-blame-ignore-revs` makes `git blame` skip the commit that
first formatted the code.

## Commit messages and releases

Commits on `main` follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat: ...` (a new capability), `fix: ...`, and `docs:`, `test:`, `refactor:`, `perf:`,
`build:`, `ci:`, `chore:` for the rest; `feat!:` or a `BREAKING CHANGE:` footer for a change
that breaks the doc format, the CLI or the library API. With squash merges, the pull request's
title is the commit message.

[release-please](https://github.com/googleapis/release-please) reads them: it keeps a release
pull request open with the next version and the CHANGELOG. Merging it tags `vX.Y.Z`, creates
the GitHub release and publishes the app to GitHub Pages. Before 1.0, `feat` bumps the minor
version and a breaking change does too; `fix` bumps the patch. Only `feat`, `fix` and `perf` (and
breaking changes) make a release.

## The hosted app

`.github/workflows/pages.yml` builds the app alone (`npm run build:pages`: `--mode pages`, into
`dist-pages/`) under the repository's Pages path, with the runtimes and the default models
(fetched by `npm run models`, cached between runs). GitHub Pages cannot send the
cross-origin isolation headers that threaded WebAssembly needs, so the site registers
[coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker), which adds them; the page
reloads once on the first visit. Run the workflow by hand (Actions → Pages → Run workflow) to
publish without a release.
