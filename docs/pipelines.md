# Pipelines and your own models

A **pipeline preset** is one JSON file that says what to compute for a kind of footage: which
layers, their settings, and **track layers on your own detectors** (vehicles, logos, screens,
weapons, anything an ONNX object detector finds). Track layers share the video decode with faces
and plates, are tracked the same way, and can be covered by redaction and the review like faces
and plates.

```bash
videodoc analyze dashcam.mp4 --pipeline examples/pipelines/dashcam.json
videodoc redact dashcam.mp4 --pipeline examples/pipelines/dashcam.json --cover faces,plates,vehicles
videodoc layers --pipeline examples/pipelines/dashcam.json      # its layers and settings
```

The MCP tools `analyze`, `markdown`, `redact` and `layers` take the same file as `pipeline`;
the library takes the parsed object (`analyze(file, layers, { pipeline })`).

## The file

```json
{
  "schema": "videodoc-pipeline/0",
  "name": "dashcam",
  "description": "Dashcam footage: transcript, faces, plates and vehicles.",
  "layers": ["transcript", "keyframes", "faces", "plates", "vehicles"],
  "settings": { "transcript.words": true, "faces.every": 4 },
  "models": {
    "logos": { "url": "./logos.onnx", "decoder": "yolo", "labels": ["logo"] }
  },
  "tracks": {
    "vehicles": { "model": "rfdetr", "labels": "car,truck,bus,motorcycle", "minScore": 0.5 },
    "logos": { "model": "logos", "description": "Brand logos" }
  }
}
```

| Field | |
|---|---|
| `layers` | What to compute when no `--layers` is given; dependencies are added. Built-in layers and the pipeline's track layers. |
| `settings` | `"layer.key": value`, typed like the setting's default; `--set` overrides them. Unknown keys and wrong types are rejected with the valid ones. |
| `models` | Detector models by name, for track layers (and for `faces.model` / `plates.model`). Built-in names: `yunet`, `yolo-plate`, `rfdetr`, `rfdetr16`. |
| `tracks` | Track layers by name (lowercase letters, digits and `_`; not a built-in name): a `model`, an optional `description`, and any track settings below. |

The examples in `examples/pipelines/` run as they are. The built-in `rfdetr` model (91 COCO
classes) needs `public/models/rfdetr_nano.onnx` from huggingface.co/onnx-community/rfdetr_nano-ONNX
(Apache-2.0).

## Models

A model is an ONNX file plus a **decoder preset** that knows how to read its outputs. Fields
other than `url` and `decoder` default to what the decoder's exporters produce.

| Decoder | Outputs | Default input |
|---|---|---|
| `yolo` | Ultralytics YOLOv8-v11 exported without NMS: `[1, 4 + classes, anchors]` (cx, cy, w, h in input pixels, then a score per class). Needs `labels` (they set the class count). NMS at `nms` (0.45). | `images`, 640x640, RGB 0-1, letterboxed with gray 114 |
| `yolo-end2end` | NMS inside the model: rows of `[batch, x1, y1, x2, y2, class, score]` (open-image-models) or `[x1, y1, x2, y2, score, class]` (Ultralytics `nms=True`, YOLOv10), told apart by the output shape. | as `yolo` |
| `detr` | DETR, RT-DETR, RF-DETR: `logits` `[1, queries, classes]` (sigmoid) and `pred_boxes` `[1, queries, 4]` (cx, cy, w, h relative to the input). | `pixel_values`, 384x384, ImageNet mean/std, stretched |
| `yunet` | OpenCV YuNet 2023mar (`cls_*`, `obj_*`, `bbox_*` at strides 8, 16, 32). | `input`, 640x640 (any multiple of 32), BGR 0-255, letterboxed |

| Field | Default | |
|---|---|---|
| `url` | (required) | `/models/...` (served from `public/`), an `https://` URL (the server must allow cross-origin requests and range requests), or a local path, relative to the preset (the CLI serves it to the browser). |
| `decoder` | (required) | one of the above |
| `labels` | `class_<i>` | class index to label, in training order (COCO models: the 80 COCO names for YOLO, the 91 COCO ids for DETR; `rfdetr` has them) |
| `name` | the model's key | shown as the layer's model in the doc and the summaries |
| `score` | per decoder | default score threshold (track layers set their own with `minScore`) |
| `nms` | per decoder | overlap at which NMS drops the weaker box, per label |
| `inputName`, `width`, `height`, `sizeMultiple` | per decoder | input tensor name and size; `sizeMultiple` for models that accept any multiple (then `faces.longSide`-style inputs work) |
| `scale`, `bgr`, `mean`, `std`, `letterbox`, `pad` | per decoder | pixel values are `[0,1] × scale`, then `(v - mean) / std`; `letterbox` keeps the aspect ratio, padding with `pad` |

A model with outputs none of these fit can still be used from the library:
`registerModel(name, spec)` (`src/models.ts`) takes a spec with its own `decode` function.

## Track layers

A track layer runs its model over the shared decode with adaptive sampling (denser where things
move, never across a shot cut) and links detections into tracks like plates: by where their
motion predicts them, bridging a few missed frames. Its items are `Track`s
([format](format.md#layers)) and its producer has `kind: "track"`.

| Setting | Default | |
|---|---|---|
| `model` | (required) | detector model: built-in or from the pipeline's `models` |
| `labels` | `""` | keep only these labels, comma-separated (`car,truck,bus`); empty keeps everything the model finds |
| `maxStride` | 8 | coarsest spacing between detected frames; must divide the other track layers' (faces 16, plates 8) |
| `minIou` | 0.3 | overlap for two detections to count as the same object |
| `minScore` | 0.4 | detection score threshold |
| `lowScore` | 0.2 | an object seen at `minScore` at one end of a sampling interval still matches this weak a detection at the other |
| `maxShift` | 1 | link a detection to a track whose motion puts it at most this many object sizes away |
| `maxGap` | 6 | frames a track may go undetected and continue (interpolated); 0 = none |

Tracks hold boxes, not labels: a layer with `labels: "car,truck"` tracks cars and trucks together.
Make one layer per label when the difference matters.

**Speed.** Every track layer adds its own model run on the frames its sampling picks. YuNet and
the plate model run at tens of frames per second on an integrated GPU; RF-DETR nano (`rfdetr`,
108 MB, transformer) at about 3, so a `vehicles` layer on it takes several times the length of
a busy video (NOTES.md). A small YOLO is the faster choice for objects.

**Licenses.** videodoc ships no model files and is MIT; the models you point a pipeline at
carry their own licenses (the Ultralytics YOLO weights are AGPL-3.0, RF-DETR and YOLOX are
Apache-2.0, D-FINE is Apache-2.0). Which ones fit is your call; videodoc's defaults stick to
permissively licensed models.

## Caching

A track layer's identity (its `configHash`) includes its model: the decoder, labels and input
settings, and the model **file** (its fingerprint: size plus a hash of the first and last MiB).
Replacing the file recomputes the layers it made; moving it does not. For a remote model whose
server does not answer range requests, the URL stands in for the file.

## Redaction

`--cover` (CLI), `cover` (MCP, library) lists the track layers to cover: `faces,plates,vehicles`.
Faces and plates are the default; a pipeline's layers are covered when listed. Track ids to leave
visible are per layer: `--keep-tracks vehicles=3,7` (repeatable), `keepTracks: { vehicles: [3, 7] }`
in MCP and the review plan. The app's review shows every covered layer in its own color, and the
log lists each track with its layer. See [redaction](redaction.md).
