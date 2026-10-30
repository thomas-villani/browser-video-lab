# videodoc

[![CI](https://github.com/thomas-villani/browser-video-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/thomas-villani/browser-video-lab/actions/workflows/ci.yml)

**Read, search and redact your videos without uploading them.** videodoc analyzes a video once,
on your own machine's GPU inside a web browser, into a small document of what happens in it:
when people speak and what they say, where the shots change, which frames are worth looking
at, where the faces are. Everything after that (reading the video as text, finding where
something was said across hundreds of hours, blurring every face before you share a clip)
reads from that document and takes milliseconds to seconds.

It is built for people and for AI agents alike: a desktop-style app in the browser, a CLI
that prints JSON, and an MCP server so an agent like Claude can "watch" a video by reading it.

> Status: early and working. The format is `videodoc/0`; the code runs from this repository
> (not yet published to npm). Measurements below are from a laptop with an integrated Intel
> Iris Xe GPU.

**Try the app:** [thomas-villani.github.io/browser-video-lab](https://thomas-villani.github.io/browser-video-lab/)
in Chrome or Edge. It downloads its models (about 250 MB, once) from the same site; your videos
never leave your machine.

## Why

Video is the most opaque thing most of us keep. A folder of meeting recordings, interviews,
lectures or bodycam footage can't be searched, skimmed or quoted without someone watching it,
and the usual way around that (uploading it to a cloud service) is exactly what you can't do
with footage that is sensitive, confidential, evidentiary, or just someone else's privacy.

Three observations shape videodoc:

1. **The transcript carries most of the meaning.** In earlier work on semantic search over
   evidence footage, the transcript alone reached about 80% of full video-understanding
   performance. So videodoc is transcript-first: speech detection and Whisper run first, and
   visual analysis is spent where nobody is talking.
2. **Browsers can now do the heavy lifting.** WebCodecs decodes video in hardware, WebGPU runs
   neural networks, and ONNX Runtime Web ties them together. A laptop transcribes at 3–6x
   real time and finds faces at around real time, with nothing to install but a browser and
   nothing leaving the machine.
3. **Analyze once, ask many times.** Analysis is the expensive part, so it is saved next to the
   video as `<video>.videodoc.json`: versioned, with every setting and model recorded, and reused
   whenever the same question comes up again. Reading, searching and redaction plans need no GPU.

## What you can do with it

**Find where something was said, across a whole collection.** Point it at a folder of school
board meetings, depositions, user interviews or lectures and search: "reading intervention",
"budget cuts", a person's name. You get ranked passages with timestamps and a picture of the
moment; click one and the video plays from there. Searching 27,000 words of transcript takes
about 25 ms.

**Let an agent read a video.** Agents can't watch video, but they read well. `videodoc markdown`
(or the MCP `markdown` tool) renders a video as Markdown: one section per shot, timestamped
paragraphs of speech, silent stretches marked, keyframe images where they help. An agent can
summarize a 65-minute meeting, answer questions about it, or quote it with times, and follow up
with `search` and a `from`/`to` window around a hit.

**Blur faces and plates before you share a video.** For public records requests, journalism,
research footage or a neighborhood watch clip: `videodoc redact` writes a copy with every
detected face and license plate blurred, pixelated or blacked out, sound kept, and a log of
everything it covered. In the app, **Review faces and plates** shows exactly what will be
covered over the playing video; click a box to leave it visible (the official at the podium),
drag a box over anything the detector missed. Detection misses some faces (8.1% of hand-labeled
faces on a crowded street) and some plates (angled, blurred or distant), so a person reviews the
result; the tools say so. Spoken names, numbers and strong language can be bleeped too: list
them (`--words`, `--profanity`, or in the review) and every place they are said is covered,
from the transcript's word times.

**Skim a long recording.** The reading view puts the transcript beside the player with a frame
for each shot and each silent gap, so a two-hour recording becomes a page you can scroll.

On the roadmap: captions for keyframes from a
small vision model, audio-only files (podcasts, lectures), and transcript-driven editing
(finding the "um"s and long pauses in a podcast or YouTube video and cutting them).

## What a video looks like as a document

```markdown
# council-2026-10-14.mp4

1:12:40 · 1920×1080 · 30 fps · 9 shots · 94% speech · 11,204 words
<!-- videodoc/0 · speech (silero_vad v5) · transcript (onnx-community/whisper-base_timestamped) · scenes · keyframes -->

## Shot 3 · 0:14:05–0:31:50

![0:14:05](council-2026-10-14.mp4.keyframes/025365.jpg)

**[0:14:05]** Item four is the sidewalk repair contract. Staff recommends approval…

**[0:14:41]** I have a question about the timeline. When would the work on Elm Street start?

*(no speech 0:22:10–0:23:02)*

![0:22:36](council-2026-10-14.mp4.keyframes/040680.jpg)
```

The doc itself is JSON with one item per line, so it greps and diffs well. See
[the format](docs/format.md).

## Quick start

You need Node 24+, Chrome or Edge with WebGPU (any recent version on Windows, macOS or
ChromeOS; Linux may need a flag), and about 200 MB of models.

```bash
npm install
npm run models                   # the default models into public/models (docs/getting-started.md; YuNet needs uv)
npm run videodoc -- check        # browser, WebGPU adapter, decoders, models: what is missing and how to fix it
npm run ui                       # the app: open a folder of videos, analyze, read, search, redact
```

From the command line:

```bash
npm run videodoc -- analyze videos/                          # transcribe and index every video in a folder
npm run videodoc -- search "sidewalk contract" videos/       # where was it said? (no GPU, instant)
npm run videodoc -- markdown videos/council.mp4 --from 14:00 --to 20:00   # read that stretch
npm run videodoc -- redact videos/bodycam.mp4                # faces and plates blurred -> bodycam.redacted.mp4
npm run videodoc -- redact videos/bodycam.mp4 --profanity --words "Fairfield"   # ... and words bleeped
```

For an agent (Claude Code shown; any MCP client works):

```bash
claude mcp add videodoc -- node /path/to/browser-video-lab/cli/mcp.ts
```

Then ask it things like "what did the board decide about the sidewalk contract in the videos
in ~/meetings?"

## How it works

```
video file ──► WebCodecs decode ──► WebGPU ───────────────► layers ──► <video>.videodoc.json
               (in a browser:        Silero VAD   speech               │
                the app, or a        Whisper      transcript           ├─► markdown   read it
                headless Edge or     scene score  scenes               ├─► search     find it
                Chrome driven by     YuNet        faces                ├─► redact     blur it
                the CLI)             YOLOv9-t     plates               └─► your code
                                     (derived)    keyframes
```

- Audio is decoded once for speech and transcript; video is decoded once for every frame
  consumer (shots, faces and plates share a pass). Keyframe images seek to the chosen frames only.
- A planner works out which layers a request needs and which are already in the doc with the
  same settings, so repeating a request costs nothing and changing one setting recomputes only
  what depends on it.
- The CLI and MCP server plan in Node and only start a headless browser when something has to
  be computed; the video is served to it from disk over localhost.

Typical times on an integrated GPU: default analysis (speech, transcript, shots, keyframes)
3–4.5x faster than real time; speakers 8–10x; faces about real time (slower in crowds); redaction 40–80 fps;
search ~25 ms; Markdown of an analyzed video ~0.5 s. Details are in [NOTES.md](NOTES.md).

## Documentation

- [Getting started](docs/getting-started.md): install, models, the first video
- [The app](docs/app.md): folders, analysis, reading, search, face and plate review
- [CLI](docs/cli.md): every command and option, output, exit codes
- [MCP server](docs/mcp.md): tools for agents, setup, behavior
- [The VideoDoc format](docs/format.md): layers, settings, caching, files next to a video
- [Library API](docs/library.md): using videodoc from TypeScript, in Node and in the browser
- [Redaction](docs/redaction.md): how regions are chosen, review, the plan and the log, limits
- [The lab](docs/lab.md): benchmarks, evaluation scripts and labeling tools behind the numbers

Design and history: [DESIGN.md](DESIGN.md) (design and roadmap), [NOTES.md](NOTES.md) (every
measurement and what it taught), [HANDOFF.md](HANDOFF.md) (current state for contributors),
[CONTRIBUTING.md](CONTRIBUTING.md) (checks, commit messages, releases),
[CHANGELOG.md](CHANGELOG.md) (from the first release on).

## Limits worth knowing

- Chromium-based browsers only for analysis (WebGPU + WebCodecs); reading and searching work
  anywhere.
- Whisper makes mistakes ("turd signal" for "turn signal"); search absorbs some of them, fuzzy
  matching is planned. English is the default language (`--set transcript.language=...`).
- Speech detection finds no speech in sung vocals, so music has an empty transcript.
- Face detection misses some faces, especially small, turned or occluded ones; face tracks are
  for covering and counting, not yet for telling who is who. Plate detection misses angled,
  blurred and distant plates.
- No captions of what is on screen yet; the Markdown view shows the keyframes instead.

## License

[MIT](LICENSE). Models and test media are not in this repository; models keep their own licenses
(see [Getting started](docs/getting-started.md)), including the copies the hosted app serves.
