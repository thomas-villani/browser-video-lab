# Redaction

videodoc covers faces and license plates in a copy of a video (blurred, pixelated or filled
with solid boxes) and bleeps or silences spoken words you list (names, places, numbers,
strong language), with a log of everything it covered. The regions come from the doc's `faces`
and `plates` layers and the words from its transcript, so the expensive part (detection,
transcription) runs once; changing the style, the word list or what stays visible only
re-encodes.

> **Detection and transcription miss things.** Against hand labels on a crowded street, 8.1%
> of faces were missed, mostly small ones in the background; turned, occluded and blurred
> faces are missed more often. Plates are missed when angled, motion-blurred or far away.
> Speech recognition misspells words (names most of all) and mishears some, and a misspelled
> name is not bleeped. Nothing here covers other identifying details: screens, name tags,
> tattoos, voices. **A person must watch and listen to the output before it is released.**
> Every surface says so: the CLI summary, the MCP result, the app, the log.

## Three ways to run it

- **App:** **Review faces and plates** to check what will be covered and adjust it, list words
  to bleep, then **Blur and bleep…** ([the app](app.md)).
- **CLI:** `videodoc redact video.mp4 [--style ...] [--keep 3,7] [--keep-plates 2] [--words "Okafor,Daniel Reyes"] [--profanity]` ([CLI](cli.md#redact)).
- **Agents:** the MCP `redact` tool ([MCP](mcp.md)).

All three use the same code and the same review file, so a review made in the app is honored
when an agent or a script makes the copy later.

## What gets covered

`cover` picks the layers: faces and plates by default; `--cover faces` (or `cover: ["faces"]`)
leaves plates alone, and `--cover none` (`cover: []`) copies the video as it is, for a copy with
only words bleeped. Faces and plates missing from the doc are detected first, in one decode.

For every frame, the regions are the covered tracks' boxes on that frame, adjusted:

1. **Margin.** Each box grows by `margin` (default 0.15, i.e. 15% of its size) on every side:
   detector boxes stop at the hairline and the ears.
2. **Hold.** Each track is covered for `hold` seconds (default 0.3) before it starts and after it
   ends, using its first and last box. Faces turning away, entering or leaving the frame are
   where the detector loses them for a few frames; plates flicker in and out the same way. A
   hold never crosses a shot cut (it would leave a stray blob in the next shot).
3. **No double cover.** Tracks fragment under motion, so a held box that overlaps a live box of
   the same kind by more than 30% is the same face (or plate) re-acquired, and is dropped.
4. **Keep.** Face tracks listed in `keep` and plate tracks in `keepPlates` are left visible (the
   official at the podium, the person who consented; a shop sign the detector took for a plate).
   Ids are per layer: face 3 and plate 3 are different tracks.
5. **Added.** Regions drawn by hand in a review are covered like any other (see below).

The shape is a rounded rectangle (corner radius 30% of the shorter side) that still covers the
detector's whole box at a margin of 10% or more.

**Styles.** `blur` scales with the region (radius about a fifth of its size, so a close-up is as
unreadable as a face in the crowd); `pixelate` uses blocks about a sixth of the region; `fill`
paints solid dark boxes and is the only style that leaves nothing to reverse. For anything
high-stakes, prefer `fill`.

**Output.** H.264 MP4 at 1.25x the source's average bitrate (clamped to 1–50 Mb/s), so files
stay near the original size. Frames without regions pass through decode and re-encode
unchanged in content. Audio is copied untouched when MP4 can hold it and no words are bleeped;
otherwise it is decoded, rewritten over the covered words and encoded again (AAC).

## Spoken words

`words` lists words and phrases to bleep wherever they are said; `profanity` adds a built-in
list of strong language and slurs (for platforms that restrict them; mild words such as damn
and hell are left to your own list). They are matched against the transcript's word times, so
the doc needs a transcript made with `transcript.words` (the default since 2026-10-28; an older
transcript is made again first).

How an entry matches:

| Entry | Matches |
|---|---|
| `Okafor` | Okafor, Okafor's, okafors (case, accents and punctuation are ignored) |
| `screw` | screw, screws, screwed, screwing, screwer (plurals, possessives, -ed, -ing, -er) |
| `Daniel Reyes` | the two words in a row |
| `555-213-4987` | the digits as Whisper writes them, together or apart |
| `fuck*`, `*shit*` | any letters in place of `*` (no inflections added) |
| `"spic"` | only that word (quotes turn inflections off: not "spiced") |

A match inside a longer one counts once ("son of a bitch" covers its "bitch").

**Spell names as the transcript does.** Whisper writes rare names as it hears them: in a test,
"Okafor" became "Ackover" and "Lindqvist" "Lindquist". Read the transcript (or search it) and
list the spellings it uses; the app suggests the transcript's recurring names to pick from.

**Timing.** Each match is covered from `lead` seconds before its first word to `tail` after its
last (defaults 0.4 and 0.2). Whisper's word times are approximate and run late: against
synthetic speech with exact word times, starts were 0.06–0.48 s late (median 0.17). With the
defaults, 110 of 117 words were covered whole; with lead 0.5 and tail 0.3, 116. A longer lead
clips more of the word before. The edges fade over 8 ms so they do not click.

**Sound.** `bleep` (default) replaces the stretch with a 1 kHz tone at -10 dBFS; `mute` with
silence. Both crossfade at the edges.

**Keep.** `keepWords` leaves matches audible by the key of their first word (`<transcript
item>.<word>`, as in the log). If the transcript is made again, the keys change, so a review's
`keepWords` is dropped with a warning (and those words are bleeped again).

## Plates

The `plates` layer uses a small YOLOv9 plate detector (`yolo-v9-t-640-license-plates` from
[open-image-models](https://github.com/ankandrew/open-image-models), 7.8 MB, trained on plates
from many countries). Plates are tracked like faces, with two differences for small, fast
objects: a track is linked by where its motion predicts it (a passing car's plate moves most of
its own width per frame, so consecutive boxes barely overlap), and it survives a few frames
without a detection (scores flicker), the gap interpolated. Settings: `videodoc layers`.

The detector sees the frame at 640 px, so plates under about 40 px wide in a 1080p frame are
often missed; review the output for distant plates.

## Review

The app's **Review faces and plates** draws the regions over the playing video (plates in cyan),
previewed live with the same drawing code as the output, and lets a person correct them:

- **Click a box** to leave it visible (green dashed outline); click again to cover it.
- **Drag a box** over anything the detector missed. It is covered from that frame for 2 s (to
  the end of the shot if sooner). If it moves, step forward, drag the box again where it is: the
  box follows, interpolated in between. **Starts here** / **Ends here** trim it; **Delete**
  removes it.
- The **list** groups faces and plates by shot, longest first, with thumbnails; click one to
  jump to it.
- Frame-accurate stepping: ← → (Shift: 10 frames), `[` `]` for shots, Space to play.
- **Spoken words** (above the shots): the words to bleep, one per line, with the transcript's
  recurring names to add in a click, a **Strong language too** box and bleep or silence. Every
  match is listed with its time: play it from a moment before (the preview mutes the player
  over it and plays the tone), or **Leave audible**. A transcript made without word times
  offers to be made again.

### The review file: `<video>.redact.json`

```json
{
 "schema": "videodoc-redact/0",
 "source": {"name": "hearing.mp4", "fingerprint": "v0:84426490:c114a982…"},
 "faces": "a1039772e5a1c7b3",
 "plates": "6f0e41c2d9b8a173",
 "transcript": "34d4be4619086f8f",
 "options": {"style": "blur", "margin": 0.15, "hold": 0.3, "keep": [12], "keepPlates": [4], "cover": ["faces", "plates"],
  "words": ["Ackover", "Daniel Reyes"], "profanity": true, "sound": "bleep", "lead": 0.4, "tail": 0.2, "keepWords": ["41.3"]},
 "addedAudio": [[612.5, 613.4]],
 "added": [
  {"id":-1,"f0":300,"f1":360,"samples":[[300,100,80,60,70],[330,160,82,60,70],[360,160,82,60,70]]}
 ]
}
```

- `source.fingerprint` ties it to the video; a review for another file is refused.
- `faces` and `plates` are the layers' `configHash`es. Track ids are only meaningful for that
  layer: if faces (or plates) are detected again with other settings, the ids change, so `keep`
  (or `keepPlates`) is dropped with a warning (and those tracks are covered again) while
  hand-drawn regions stay.
- `added` are tracks with negative ids, in source pixels.
- `transcript` is the transcript's `configHash`, which `keepWords` keys refer to; `addedAudio`
  are stretches to bleep given by hand, `[t0, t1]` seconds (words the transcript missed). The app
  does not draw these yet; edit the file to add them.

The app saves it next to the video (folder mode) or keeps it in the tab (dropped files; drop a
`.redact.json` with its video to reuse one). `videodoc redact` and MCP `redact` use it when
present; options given on the command line override it; `--no-plan` / `plan: false` ignore it.

## The redaction log: `<out>.redactions.json`

```json
{
 "schema": "videodoc-redactions/0",
 "source": {"name": "hearing.mp4", "fingerprint": "v0:…", "frames": 5326, "fps": 25, "width": 1920, "height": 1080},
 "output": {"name": "hearing.redacted.mp4", "frames": 5326, "ms": 65360},
 "detectors": {
  "faces": {"model": "yunet_face_dyn (2023mar)", "configHash": "a1039772e5a1c7b3", "device": "webgpu", "createdAt": "…"},
  "plates": {"model": "yolo-v9-t-640-license-plates (open-image-models)", "configHash": "6f0e41c2d9b8a173", "device": "webgpu", "createdAt": "…"}
 },
 "options": {"style": "blur", "margin": 0.15, "hold": 0.3, "keep": [12], "keepPlates": [4], "cover": ["faces", "plates"]},
 "totals": {"tracks": 412, "redacted": 410, "kept": 2, "added": 1, "plates": 42, "framesWithRegions": 4969, "regions": 9433, "maxPerFrame": 16},
 "caveat": "Regions come from automatic detection, which misses some faces (…) and some plates (…). Review the output before release.",
 "tracks": [
  {"id":1,"layer":"faces","t0":0.84,"t1":2.12,"frames":32,"kept":false},
  {"id":4,"layer":"plates","t0":3.2,"t1":4.6,"frames":35,"kept":true},
  {"id":-1,"layer":"faces","t0":12,"t1":14.44,"frames":61,"kept":false,"added":true}
 ],
 "audio": {"sound": "bleep", "totals": {"matches": 9, "kept": 1, "added": 1, "seconds": 8.3},
  "spans": [
  {"t0":2.68,"t1":3.66,"key":"0.4","text":"Ackover","term":"Ackover","kept":false},
  {"t0":15.57,"t1":16.36,"key":"5.4","text":"shit,","term":"profanity","kept":false},
  {"t0":612.5,"t1":613.4,"kept":false,"added":true}
 ]}
}
```

It records what was done and how, for whoever has to account for the release: the detectors and
their settings, the options, totals (`plates` counts the plate tracks among `tracks`), and every
track with its layer, times and whether it was kept or added by hand. One track per line, so it
stays readable for a crowd's tens of thousands. With words bleeped, `detectors.transcript` names
the transcript, and `audio` lists every match (times include lead and tail; `term` is the list
entry, or `profanity`) and every stretch added by hand; `seconds` is the audio rewritten,
overlaps counted once.

## Speed

On an integrated GPU (Intel Iris Xe), with faces already detected:

| Video | Faces | Redaction |
|---|---|---|
| Music video, 3:33, close-ups and cuts | 369 tracks | 78–81 fps |
| Street walk, 6:29, crowd (up to 48 faces per frame) | 25,278 tracks | 42 fps |
| Bodycam, 17:35, night scenes, from scratch | 1,238 tracks | faces ~330 s + redaction 414 s: 1.4x the video's length |
| Dashcam, 1:00, busy street, from scratch | 720 face + 92 plate tracks | faces 36 s, + plates 53 s (89 s together) + redaction 39 s (46 fps) |

The cost of drawing is grouped by blur strength rather than per region, so crowds stay fast. In
the app's review, playback with the live preview holds full frame rate on the crowd.

## Not covered yet

Text on screens and documents, voices (bleeping a word leaves the speaker's voice), names the
transcript misspells or misses, and identification of the same person across shots. Audio-only
files (no video track) cannot be redacted yet.
