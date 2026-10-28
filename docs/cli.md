# CLI

```bash
npm run videodoc -- <command> [options]     # from anywhere: node /path/to/browser-video-lab/cli/videodoc.ts
```

Every command prints **one JSON object on stdout** (except `markdown` without `--out`, which
prints Markdown) and writes progress and a human summary to **stderr**, so it pipes into `jq`
or an agent as easily as it reads in a terminal. Exit codes: `0` ok, `1` failure (the message
says how to fix it), `2` usage error.

Commands that compute (`analyze`, `markdown` with missing layers, `redact`, `check`) start a
headless Edge (or `--browser chrome`; `--headed` to watch it). Commands that only read docs
(`search`, `info`, `layers`, and `markdown` when everything is analyzed) start no browser and
return in well under a second.

## analyze

```bash
videodoc analyze <video|folder> [--layers a,b] [--set layer.key=value]... [--out doc.json] [--force] [--dry-run]
```

Computes layers into `<video>.videodoc.json`. Given a folder, analyzes every video in it and its
subfolders (`.mp4 .m4v .mov .mkv .webm`) one after another.

- `--layers`: default `speech,transcript,scenes,keyframes`; available `speech, transcript,
  scenes, keyframes, faces`. Dependencies are added (`keyframes` brings `scenes` and `speech`).
- `--set`: override a setting, repeatable: `--set faces.every=4 --set transcript.language=de`.
  `videodoc layers` lists every setting with its default; unknown keys and wrong types are
  rejected with the list of valid ones.
- Layers already in the doc with the same settings (and computed from the same inputs) are
  reused; with nothing to compute no browser starts. `--force` recomputes the requested layers.
- `--dry-run` prints the plan (each layer `cached` or `compute`, and the decode passes) and exits.

stdout: `{ doc, source, layers: { <name>: { status, items, ms, device, model?, configHash, ... } }, wallMs, adapter? }`;
for a folder, `{ folder, videos: [...] }` with one such result per video (a video that fails
gets `{ video, error }` and the rest continue). Layer figures include speech seconds and percent, words,
keyframes by reason, face samples and face-frames.

## markdown

```bash
videodoc markdown <video> [--out file.md] [--images] [--image-size px] [--from t] [--to t] [--layers ...] [--set ...] [--force]
```

The video as Markdown: a header with the facts and provenance, one section per shot (with face
counts when the doc has faces), timestamped paragraphs of speech, `*(no speech a–b)*` for
silent gaps, and keyframe images where they fall. Analyzes `transcript` and `keyframes` (and
their dependencies, plus any `--layers`) first if they are missing.

- `--images`: link keyframe JPEGs, extracting missing ones into `<video>.keyframes/` (one seek
  per image, roughly 0.1 s each). `--image-size` sets the long side (default 768).
- `--out`: write the Markdown to a file (image links relative to it) and print a JSON summary
  instead; without it, Markdown goes to stdout with absolute image paths.
- `--from`, `--to`: only that stretch, in seconds, `mm:ss` or `h:mm:ss`; shots and paragraphs
  overlapping it are included. Use it to read around a search hit.

## search

```bash
videodoc search <query> <video|doc|folder>... [--top n]
```

Keyword search (BM25 with light stemming) over the transcripts of saved docs; folders are
searched recursively. Passages are 20-second windows of transcript; overlapping hits in one
video collapse to the best one; passages with the words in order rank higher.

```json
{"query": "turn signal",
 "hits": [{"video": "/videos/bodycam.mp4", "time": "00:42–01:01", "t0": 42.34, "t1": 61.38,
           "at": 42.34, "score": 9.39,
           "snippet": "…You didn't use your turd **signal** right away. Also, when you **turn** on…",
           "ref": "transcript/12", "keyframe": {"ref": "keyframes/6", "t": 40.6}}],
 "notAnalyzed": [], "searched": 7, "ms": 22}
```

`at` is where the first matched word is said (seek there); `notAnalyzed` lists videos without a
transcript, so a caller can analyze them and search again.

## redact

```bash
videodoc redact <video> [--out file.mp4] [--style blur|pixelate|fill] [--margin f] [--hold s] [--keep ids] [--no-plan] [--set ...] [--force]
```

Writes a copy with every face covered (H.264 MP4, audio copied) and a redaction log next to it
(`<out>.redactions.json`). Finds faces first if the doc has none (about real time on an
integrated GPU, slower in crowds).

- `--out`: default `<name>.redacted.mp4` next to the video; must be `.mp4` and not the input.
  The file is written as `<out>.part` and renamed when complete.
- `--style`: `blur` (default), `pixelate`, or `fill` (solid boxes; nothing left to undo).
- `--margin`: grow each face box by this fraction on every side (default 0.15).
- `--hold`: keep covering this many seconds before and after each face track, within its shot
  (default 0.3), for faces turning away or entering.
- `--keep 3,7`: face track ids to leave visible (ids are in the log, the doc and the app).
- A review saved by the app (`<video>.redact.json`) is used when present; options given here
  override it, `--no-plan` ignores it.

stdout includes the output and log paths, the settings used, totals (tracks, kept, added by
hand, frames with regions, regions, most per frame), and a caveat: detection misses some faces,
so watch the output before release. See [redaction](redaction.md).

## info

```bash
videodoc info <video|doc>
```

Summarizes a saved doc without a browser: source facts, and per layer its item count, time,
device, model, settings hash and figures, plus the layers still `missing`.

## layers

```bash
videodoc layers
```

Every layer: what it needs, which pass computes it, its description, and every setting with its
default and meaning (JSON). The CLI help and the MCP tool schemas are generated from the same
definition (`PRODUCERS` in `src/doc/plan.ts`).

## check

```bash
videodoc check [--browser chrome]
```

What this machine can run: browser version, WebGPU adapter, cross-origin isolation (threaded
wasm), which video and audio codecs decode, and which model files are present. `ok: false` and
exit code 1 when something a default layer needs is missing, with what to do.

## ui

```bash
videodoc ui [--port 4317] [--no-open]
```

Serves the app and opens it in your default browser (the next free port if 4317 is taken).
Runs until interrupted. See [the app](app.md).

## Examples

```bash
# Index a collection, then find things in it
videodoc analyze ~/meetings/
videodoc search "reading intervention" ~/meetings/ --top 5 | jq '.hits[] | {video, time, snippet}'

# Read around the best hit
videodoc markdown ~/meetings/2026-09-board.mp4 --from 41:00 --to 47:00

# Faces too, with denser refinement for a crowd, then counts per shot in the Markdown
videodoc analyze street.mp4 --layers faces --set faces.every=1
videodoc markdown street.mp4 --layers faces

# A redacted copy with solid boxes, leaving the speaker (track 12) visible
videodoc redact hearing.mp4 --style fill --keep 12 --out hearing.public.mp4

# What would be computed?
videodoc analyze talk.mp4 --layers faces --dry-run
```
