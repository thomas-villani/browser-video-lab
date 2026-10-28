# MCP server

`cli/mcp.ts` serves the CLI's commands as [Model Context Protocol](https://modelcontextprotocol.io)
tools over stdio, so an agent can read, search and redact videos on the machine it runs on.
It is the same implementation as the CLI (`cli/lib.ts`); results are the same JSON.

## Setup

Claude Code:

```bash
claude mcp add videodoc -- node /path/to/browser-video-lab/cli/mcp.ts
```

Any other MCP client: run `node /path/to/browser-video-lab/cli/mcp.ts` as a stdio server.
Options: `--browser chrome` (default Edge), `--headed` (watch the compute page). The models must
be in place first ([getting started](getting-started.md)); the `check` tool says what is missing.

## Tools

| Tool | What it does | Main arguments |
|---|---|---|
| `markdown` | The video as Markdown (shots, timed speech, silences, keyframes); analyzes missing layers first | `video`, `from`, `to`, `images`, `embedImages`, `imageSize`, `layers`, `out` |
| `search` | Where something is said across videos: ranked passages with times, snippet, nearest keyframe | `query`, `paths` (files or folders), `top` |
| `analyze` | Compute layers into the doc (a folder: every video) and summarize them | `video`, `layers`, `set`, `force`, `dryRun` |
| `redact` | A copy with faces and license plates covered, audio copied, plus the redaction log | `video`, `out`, `style`, `margin`, `hold`, `keep`, `keepPlates`, `cover`, `plan` |
| `info` | Summarize a saved doc (no compute) | `video` |
| `layers` | Layers, dependencies, every setting with its default | — |
| `check` | Browser, WebGPU, decoders, model files | — |

Argument schemas are generated from the layer definitions, so `layers` enums and `set` help
always match what the code accepts. Paths are absolute or relative to the server's working
directory (`~` is not expanded). `from`/`to` take seconds, `mm:ss` or `h:mm:ss`.

The server's instructions tell the agent the usual flow:

1. To understand a video: `markdown` (with `images: true` or `embedImages: 4` to see frames).
2. To find something across videos: `search` over a folder, then `markdown` with `from`/`to`
   around the hit's `at`.
3. To share a video without faces or plates: `redact`, then tell the user to watch it (the
   result carries a caveat that detection misses some faces and plates).

`embedImages: n` returns up to n keyframes as image content, spread evenly over the video (or the
`from`/`to` window), for agents that can see images; `images: true` only links them by path.

## Behavior

- **Warm browser.** The headless browser and its compute page start on the first call that needs
  them and stay open, so later calls skip startup. They close after `VIDEODOC_IDLE_MS`
  milliseconds idle (default 10 minutes). Calls that only read docs (`search`, `info`, `layers`,
  `markdown` of an analyzed video) never start it.
- **One at a time.** Calls that compute are queued (there is one GPU).
- **Progress.** Long calls send MCP progress notifications (stage, done, total) when the client
  asks for them; analysis of an hour of video takes many minutes.
- **Cancellation.** Cancelling a call closes the compute page, which stops the work; the next
  call starts a fresh one. A partly written redaction is deleted.
- **Errors** come back as tool errors (`isError: true`) whose text says how to fix the problem:
  an unknown setting lists the valid ones, a missing model names the file and the command to
  fetch it.

## Example session

> **User:** In ~/meetings, when did the board talk about the bus contract, and what did they
> decide?
>
> **Agent:** calls `search { query: "bus contract", paths: ["/home/me/meetings"] }`: the top hit
> is `2026-09-board.mp4` at 41:12; calls `markdown { video: "/home/me/meetings/2026-09-board.mp4",
> from: "40:00", to: "48:00" }`, reads the discussion and the vote, and answers with the times.

If `search` lists videos under `notAnalyzed`, the agent can `analyze` the folder first (one-time
cost: roughly a quarter to a third of the videos' length on an integrated GPU).
