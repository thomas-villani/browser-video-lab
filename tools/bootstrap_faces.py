"""Step 2 of the face-labeling kit: propose faces with a stronger, independent detector.

RetinaFace-ResNet50 (batch-face, MIT) at native resolution plus a horizontally flipped pass,
merged by NMS. Far heavier than YuNet at 640 px, but it runs offline on ~300 frames only.

    uv run --python 3.12 --with batch-face --with torch --with opencv-python-headless --with "numpy<2.4" \
        python -I tools/bootstrap_faces.py public/label

Reads <dir>/frames.json; writes <dir>/retina.json {"<clip>/<frame>": [[x, y, w, h, score], ...]}.
stdout: JSON summary.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from batch_face import RetinaFace


def nms(boxes: np.ndarray, iou: float) -> list[int]:
    """Indices of boxes (x1, y1, x2, y2, score) kept by greedy NMS."""
    order = np.argsort(-boxes[:, 4])
    keep: list[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        o = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][o < iou]
    return keep


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "public/label")
    frames = json.loads((root / "frames.json").read_text())
    torch.set_num_threads(max(1, torch.get_num_threads()))
    det = RetinaFace(gpu_id=-1, network="resnet50", return_dict=True)
    out: dict[str, list[list[float]]] = {}
    t0 = time.time()
    for i, f in enumerate(frames):
        img = cv2.imread(str(root / f["clip"] / f"{f['frame']}.jpg"))
        h, w = img.shape[:2]
        found = []
        for flip in (False, True):
            src = img[:, ::-1].copy() if flip else img
            for d in det(src, threshold=0.3, cv=True, resize=1):
                x1, y1, x2, y2 = (float(v) for v in d["box"])
                if flip:
                    x1, x2 = w - x2, w - x1
                found.append([x1, y1, x2, y2, float(d["score"])])
        boxes = np.array(found, dtype=np.float32).reshape(-1, 5)
        kept = boxes[nms(boxes, 0.4)] if len(boxes) else boxes
        out[f"{f['clip']}/{f['frame']}"] = [
            [round(float(b[0]), 1), round(float(b[1]), 1), round(float(b[2] - b[0]), 1), round(float(b[3] - b[1]), 1), round(float(b[4]), 3)]
            for b in kept
        ]
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(frames)} frames, {(time.time() - t0) / (i + 1):.2f} s/frame", file=sys.stderr)
    (root / "retina.json").write_text(json.dumps(out))
    print(json.dumps({"frames": len(out), "faces": sum(len(v) for v in out.values()), "seconds": round(time.time() - t0, 1), "out": str(root / "retina.json")}))


if __name__ == "__main__":
    main()
