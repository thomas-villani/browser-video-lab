# CLI

```bash
npm run videodoc -- <command> [options]     # from anywhere: node /path/to/browser-video-lab/cli/videodoc.ts
```

Every command prints **one JSON object on stdout** (except `markdown` without `--out`, which
prints Markdown) and writes progress and a human summary to **stderr**, so it pipes into `jq`
or an agent as easily as it reads in a terminal. Exit codes: `0` ok, `1` failure (the message
says how to fix it), `2` usage error, `130` cancelled.

**Ctrl+C cancels** within a few seconds: the compute page is discarded, nothing half-done is
saved to the doc, and a partly written redaction is deleted. A second Ctrl+C quits at once.

Commands that compute (`analyze`, `markdown` with missing layers, `redact`, `check`) start a
headless Edge (or `--browser chrome`; `--headed` to watch it). Commands that only read docs
(`search`, `speakers`, `info`, `layers`, and `markdown` when everything is analyzed) start no
browser and return in well under a second. Speaker names (`<video>.speakers.json`, from
`speakers`) are used by `markdown`, `search`, `info` and `redact`.

## analyze

```bash
videodoc analyze <video|folder> [--layers a,b] [--set layer.key=value]... [--pipeline preset.json] [--out doc.json] [--force] [--dry-run]
```

Computes layers into `<video>.videodoc.json`. Given a folder, analyzes every video in it and its
subfolders (`.mp4 .m4v .mov .mkv .webm`) one after another.

- `--layers`: default `speech,transcript,scenes,keyframes`; available `speech, transcript,
  speakers, scenes, keyframes, faces, plates` (faces and plates share one decode; speakers is
  who speaks when, for interviews and podcasts). Dependencies are added (`keyframes` brings `scenes` and `speech`).
- `--set`: override a setting, repeatable: `--set faces.every=4 --set transcript.language=de`.
  `videodoc layers` lists every setting with its default; unknown keys and wrong types are
  rejected with the list of valid ones.
- `--pipeline preset.json`: a [pipeline preset](pipelines.md): its layers (when `--layers` is not
  given), its settings (`--set` wins) and its track layers on your own detectors (e.g. `vehicles`).
  Also for `markdown`, `redact` and `layers`.
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
videodoc search <query> <video|doc|folder>... [--top n] [--speaker name]
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

`at` is where to seek to hear the first matched word (half a second before it is said);
`speaker` ("Dana", or "Speaker 2" when unnamed) says who says it, for videos with a speakers layer;
`notAnalyzed` lists videos without a transcript, so a caller can analyze them and search again.

`--speaker Dana` searches only what Dana says (a name, or `2` / "Speaker 2"); videos without such a
speaker are listed in `noSpeaker`.

## speakers

```bash
videodoc speakers <video> [--name id=name]... [--merge a=b]... [--unmerge id]... [--move span=to]... [--unmove i]... [--voices folder]... [--reset]
```

Lists the speakers of an analyzed video (analyze with `--layers speakers` first) and edits them:
names, merges and moves, saved as `<video>.speakers.json` ([format](format.md#speaker-names-videospeakersjson)).
No browser.

- `--name 2=Dana`: name speaker 2 (`2=` removes the name). A name another speaker has is an
  error: they are one person, so merge them.
- `--merge 3=Dana`: speaker 3 is Dana (one person found as two); Dana keeps the name. Applied
  after `--name`, so `--name 1=Dana --merge 3=Dana` works in one call.
- `--unmerge 3`: separate it again.
- `--move 1:02-1:05=Dana`: all speech in that span is Dana's; `--move "Speaker 2@1:02-1:05=Dana"`
  moves only Speaker 2's. A name that is no speaker's adds one (a voice the analysis missed).
- `--unmove 0`: undo a move, by its index in `moves`.
- `--voices ~/podcast`: also suggest names from voices named under this folder (recursive); the
  video's own folder is always searched.
- `--reset`: delete the speakers file first.

Speakers are referred to by name, `Speaker 2` or `2`. stdout:

```json
{"video": "/videos/ep12.mp4", "doc": "...", "edits": "/videos/ep12.mp4.speakers.json", "changed": true,
 "speakers": [{"id": 1, "name": "Dana", "named": true, "seconds": 812.4, "share": 61.2, "turns": 140, "first": "00:03", "merged": [4]},
              {"id": 2, "name": "Speaker 2", "named": false, "seconds": 498.1, "share": 37.5, "turns": 131, "first": "00:41",
               "suggestion": {"name": "Sam", "distance": 0.08, "from": "/videos/ep11.mp4"}}],
 "moves": [{"index": 0, "t0": 61.2, "t1": 63.5, "from": "Speaker 2", "to": "Dana"}]}
```

A `suggestion` is a voice named in another video that sounds the same (cosine distance under
0.4; one person's voice in separate recordings measured at most 0.39, two people's mostly far
above: NOTES.md). It is never applied: check it, then `--name 2=Sam`. Edits follow the voices
when the speakers are found again; ones whose voice is gone are dropped with a warning.

## redact

```bash
videodoc redact <video> [--out file.mp4] [--style blur|pixelate|fill] [--margin f] [--hold s] [--keep ids] [--keep-plates ids] [--keep-tracks layer=ids]... [--cover faces,plates|none]
                [--words list] [--words-file path] [--profanity] [--speakers list] [--sound bleep|mute] [--lead s] [--tail s] [--keep-words keys] [--no-plan] [--pipeline preset.json] [--set ...] [--force]
```

Writes a copy with every face and license plate covered (H.264 MP4) and the listed words
bleeped, and a redaction log next to it (`<out>.redactions.json`). Finds faces and plates first
if the doc lacks them (one decode; on an integrated GPU faces take about real time, slower in
crowds, and plates up to about real time again in busy traffic), and the transcript with word
times when words are listed (about 3x faster than real time). Without words, the audio is
copied.

- `--out`: default `<name>.redacted.mp4` next to the video; must be `.mp4` and not the input.
  The file is written as `<out>.part` and renamed when complete.
- `--style`: `blur` (default), `pixelate`, or `fill` (solid boxes; nothing left to undo).
- `--margin`: grow each face box by this fraction on every side (default 0.15).
- `--hold`: keep covering this many seconds before and after each face track, within its shot
  (default 0.3), for faces turning away or entering.
- `--keep 3,7`: face track ids to leave visible (ids are in the log, the doc and the app).
- `--keep-plates 2`: plate track ids to leave visible (a sign taken for a plate). Ids are per
  layer: face 2 and plate 2 are different tracks.
- `--cover faces`: cover faces only (default `faces,plates`); `--cover none` copies the video
  and only bleeps words. A [pipeline's](pipelines.md) track layers are covered when listed:
  `--cover faces,plates,vehicles` (with `--pipeline`, or after analyzing with it: a track layer
  already in the doc is used as saved).
- `--keep-tracks vehicles=3,7`: a pipeline track layer's ids to leave visible, repeatable.
- `--words "Ackover,Daniel Reyes"`: words and phrases to bleep. Plain words also match plurals,
  possessives, -ed and -ing; `*` is a wildcard (`fuck*`); `"quoted"` matches only that word.
  Spell names as the transcript does (`videodoc markdown` shows it).
- `--words-file names.txt`: one word or phrase per line (`#` comments), added to `--words`.
- `--profanity`: also bleep the built-in strong-language list.
- `--speakers "Dana,3"`: bleep everything these speakers say (names from `speakers`, or ids):
  each of their turns, 0.25 s either side. Finds the speakers first if the doc lacks them.
- `--sound mute`: silence instead of a 1 kHz tone.
- `--lead 0.4 --tail 0.2` (defaults): seconds covered before and after each word (Whisper's word
  times run late).
- `--keep-words 41.3`: matches to leave audible, by word key from the log.
- A review saved by the app (`<video>.redact.json`) is used when present; options given here
  override it, `--no-plan` ignores it.

stdout includes the output and log paths, the settings used, totals (tracks, kept, added by
hand, plate tracks, frames with regions, regions, most per frame), `audio` totals when words were
bleeped (matches, kept, added, seconds), and a caveat: detection and transcription miss some
faces, plates and words, so watch and listen to the output before release. stderr lists the
first matches with their times. See [redaction](redaction.md).

## info

```bash
videodoc info <video|doc>
```

Summarizes a saved doc without a browser: source facts, and per layer its item count, time,
device, model, settings hash and figures, plus the layers still `missing`.

## layers

```bash
videodoc layers [--pipeline preset.json]
```

Every layer: what it needs, which pass computes it, its description, and every setting with its
default and meaning (JSON); with a pipeline, its track layers too. The CLI help and the MCP tool schemas are generated from the same
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

# Bleep two names and strong language, leaving the picture as it is
videodoc redact interview.mp4 --cover none --words "Ackover,Daniel Reyes" --profanity

# Who speaks: name them once, then read and search by name
videodoc analyze ep12.mp4 --layers speakers
videodoc speakers ep12.mp4 --name 1=Dana --name 2=Sam --merge 3=Dana
videodoc search "budget" ~/podcast/ --speaker Sam
videodoc redact ep12.mp4 --cover none --speakers Sam      # a guest who asked not to be heard

# What would be computed?
videodoc analyze talk.mp4 --layers faces --dry-run
```
