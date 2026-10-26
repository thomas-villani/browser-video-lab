"""Make YuNet's input/output spatial dims symbolic so it can run at any multiple of 32.

    uv run --with onnx --with onnxruntime python tools/dynamic_yunet.py public/models/yunet_face.onnx public/models/yunet_face_dyn.onnx
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
inp = m.graph.input[0].type.tensor_type.shape.dim
inp[2].dim_param, inp[3].dim_param = "H", "W"
for o in m.graph.output:
    d = o.type.tensor_type.shape.dim
    d[1].dim_param = f"N_{o.name.split('_')[-1]}"
del m.graph.value_info[:]  # stale fixed-size intermediate shapes
onnx.save(m, dst)

s = ort.InferenceSession(dst)
for h, w in [(640, 640), (320, 320), (160, 160), (384, 640)]:
    outs = s.run(None, {"input": np.random.rand(1, 3, h, w).astype(np.float32) * 255})
    print(h, w, [o.shape for o in outs[:3]])
