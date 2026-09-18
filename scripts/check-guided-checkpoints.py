"""Verify saved M6 policy reloads and inference without teacher commands."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from bb8_rl import guided, guided_dreamer
from bb8_rl.training import file_hash

parser = argparse.ArgumentParser()
parser.add_argument("--sac", type=Path, required=True)
parser.add_argument("--dreamer", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)


def forbidden_teacher(*args, **kwargs):
    raise AssertionError("Teacher action called during learned inference")


guided.teacher_action = forbidden_teacher
guided_dreamer.teacher_action = forbidden_teacher
vectors = np.array(
    [
        [1, 0.2, 0, 0, 0],
        [0.8, 0.1, 0.5, 0.1, 0],
        [0.3, 0.05, 0.5, 0, 0],
        [0.04, 0.01, 0.1, 0, 0],
        [0.2, -0.5, 0, 0, 1],
        [0.1, -0.2, 0.1, -0.3, 1],
        [0.03, -0.02, 0.02, -0.02, 1],
        [0.01, 0.01, 0, 0, 1],
    ],
    dtype=np.float32,
)
sac_traces = []
for _ in range(2):
    sac = guided.GuidedSAC.load(args.sac, device="cpu")
    sac_traces.append(np.stack([sac.predict(v, deterministic=True)[0] for v in vectors]))
np.testing.assert_array_equal(*sac_traces)

dreamer = guided_dreamer.GuidedDreamerPolicy(args.dreamer)
with args.dreamer.open("rb") as file:
    saved = pickle.load(file)
restored = dreamer.agent.save()
assert restored["params"].keys() == saved["params"].keys()
for key in saved["params"]:
    np.testing.assert_array_equal(saved["params"][key], restored["params"][key])
dreamer_traces = []
for _ in range(2):
    dreamer.agent.load(saved)
    dreamer.reset(888)
    dreamer_traces.append(np.stack([dreamer.action(v) for v in vectors]))
np.testing.assert_array_equal(*dreamer_traces)
for trace in (*sac_traces, *dreamer_traces):
    assert np.isfinite(trace).all() and np.max(abs(trace)) <= 1
report = {
    "status": "passed",
    "teacher_action_disabled": True,
    "sac_checkpoint_sha256": file_hash(args.sac),
    "dreamer_checkpoint_sha256": file_hash(args.dreamer),
    "dreamer_restored_parameter_tensors": len(saved["params"]),
    "seeded_trace_length": len(vectors),
    "sac_trace": sac_traces[0].tolist(),
    "dreamer_trace": dreamer_traces[0].tolist(),
    "scope": "Exact reload/inference within the selected local runtime",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
