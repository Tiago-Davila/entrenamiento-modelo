#!/usr/bin/env python3
"""Genera fixtures deterministas del contrato secuencial Python/Kotlin."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from config import PROJECT_DIR
from sequence_contract import COORDS_PER_LANDMARK, INPUT_DIM, MAX_FRAMES, build_input_tensor, sample_indices


FIXTURE_PATH = PROJECT_DIR / "reports" / "phase2" / "sequence_contract_fixture.json"


def f32le_base64(values: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(values, dtype="<f4")
    return base64.b64encode(little_endian.tobytes()).decode("ascii")


def make_sequence(frames: int, missing_right_every: int | None = None) -> np.ndarray:
    values = np.zeros((frames, 42, COORDS_PER_LANDMARK), dtype=np.float32)
    for frame in range(frames):
        for landmark in range(42):
            values[frame, landmark, 0] = 0.10 + frame * 0.003 + landmark * 0.001
            values[frame, landmark, 1] = -0.20 + frame * 0.002 - landmark * 0.0007
            values[frame, landmark, 2] = 0.01 * landmark - 0.002 * frame
        if missing_right_every and frame % missing_right_every == 0:
            values[frame, 21:, :] = 0.0
    return values


def make_record(name: str, raw: np.ndarray) -> dict[str, object]:
    return {
        "name": name,
        "rawShape": list(raw.shape),
        "rawF32LeBase64": f32le_base64(raw),
        "sampleIndices": sample_indices(raw.shape[0]).tolist(),
        "paddedRows": max(MAX_FRAMES - raw.shape[0], 0),
        "expectedInputShape": [MAX_FRAMES, INPUT_DIM],
        "expectedInputF32LeBase64": f32le_base64(build_input_tensor(raw)),
    }


def main() -> int:
    cases = [
        make_record("short_3_frames_with_missing_right_hand", make_sequence(3, 2)),
        make_record("exact_75_frames", make_sequence(75, 11)),
        make_record("long_79_frames_integer_sampling", make_sequence(79, 13)),
    ]
    payload = {
        "contractVersion": "lsa-t-seq2seq-input-v1",
        "frames": MAX_FRAMES,
        "coords": INPUT_DIM,
        "landmarkOrder": "left hand 21 then right hand 21; x,y,z interleaved",
        "missingHand": "63 zeros before normalization",
        "normalization": {
            "center": "per-frame mean of all 42 landmarks over x/y, including zeros",
            "scale": "global maximum absolute x/y after centering",
            "z": "unchanged",
        },
        "temporal": {
            "long": "floor(i * (T - 1) / (N - 1)) with integer arithmetic",
            "short": "zero-pad at end to 75 frames",
        },
        "cases": cases,
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[fixture] {FIXTURE_PATH} ({len(cases)} casos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
