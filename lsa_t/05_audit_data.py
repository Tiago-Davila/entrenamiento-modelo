#!/usr/bin/env python3
"""Audita el dataset procesado antes de entrenar o exportar un modelo.

No elimina ni modifica muestras. Produce un JSON con las mediciones completas
y un resumen Markdown para revisar las exclusiones en una fase posterior.

Uso:
    python 05_audit_data.py
    python 05_audit_data.py --report-dir reports/phase1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from config import DATA_DIR


# Conteos publicados por la anotacion de glosl-lsat. Se usan para detectar
# faltantes sin asumir que la carpeta esta completa solo porque contiene NPZ.
EXPECTED_COUNTS = {"train": 5413, "val": 1354, "test": 1692}
EXPECTED_PREFIXES = {"train": "train", "val": "val", "test": "test"}
EXPECTED_SHAPE = (42, 3)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p01": None, "p05": None, "p50": None,
                "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def scalar_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    return str(value).strip()


def numeric_ids(files: list[Path], prefix: str) -> list[int]:
    ids: list[int] = []
    marker = f"{prefix}_"
    for path in files:
        if path.stem.startswith(marker):
            try:
                ids.append(int(path.stem[len(marker):]))
            except ValueError:
                continue
    return sorted(set(ids))


def audit_split(split: str) -> dict[str, Any]:
    split_dir = DATA_DIR / "processed" / split
    files = sorted(split_dir.glob("*.npz")) if split_dir.exists() else []
    expected = EXPECTED_COUNTS[split]
    ids = numeric_ids(files, EXPECTED_PREFIXES[split])
    missing = sorted(set(range(expected)) - set(ids))

    zero_ratios: list[float] = []
    valid_frame_ratios: list[float] = []
    lengths: list[int] = []
    label_words: list[int] = []
    left_valid_ratios: list[float] = []
    right_valid_ratios: list[float] = []
    invalid_files: list[dict[str, str]] = []
    quality_candidates = {
        "allZero": [],
        "lowCoverage": [],
        "highZero": [],
        "longLabel": [],
    }
    labels = Counter()

    for path in files:
        try:
            with np.load(path, allow_pickle=True) as sample:
                if "keypoints" not in sample or "label" not in sample:
                    raise ValueError("faltan las claves keypoints o label")
                keypoints = np.asarray(sample["keypoints"])
                label = scalar_text(sample["label"])

            if keypoints.ndim != 3 or tuple(keypoints.shape[1:]) != EXPECTED_SHAPE:
                raise ValueError(
                    f"forma {tuple(keypoints.shape)}; se esperaba [T, 42, 3]",
                )
            if keypoints.shape[0] < 5:
                raise ValueError(f"solo tiene {keypoints.shape[0]} cuadros")
            if not np.issubdtype(keypoints.dtype, np.floating):
                raise ValueError(f"dtype {keypoints.dtype}; se esperaba float")
            if not np.isfinite(keypoints).all():
                raise ValueError("contiene NaN o infinito")
            if not label:
                raise ValueError("etiqueta vacia")

            finite_keypoints = keypoints.astype(np.float32, copy=False)
            frame_has_points = np.any(finite_keypoints != 0, axis=(1, 2))
            left_has_points = np.any(finite_keypoints[:, :21, :] != 0, axis=(1, 2))
            right_has_points = np.any(finite_keypoints[:, 21:, :] != 0, axis=(1, 2))
            zero_ratios.append(float(np.mean(finite_keypoints == 0)))
            valid_frame_ratios.append(float(np.mean(frame_has_points)))
            left_valid_ratios.append(float(np.mean(left_has_points)))
            right_valid_ratios.append(float(np.mean(right_has_points)))
            lengths.append(int(keypoints.shape[0]))
            words = len(label.split())
            label_words.append(words)
            labels[label] += 1

            zero_ratio = float(zero_ratios[-1])
            valid_frame_ratio = float(valid_frame_ratios[-1])
            candidate = {
                "file": str(path),
                "frames": int(keypoints.shape[0]),
                "zeroRatio": zero_ratio,
                "validFrameRatio": valid_frame_ratio,
                "labelWords": words,
                "labelPreview": label[:160],
            }
            if zero_ratio >= 1.0:
                quality_candidates["allZero"].append(candidate)
            if valid_frame_ratio < 0.10:
                quality_candidates["lowCoverage"].append(candidate)
            if zero_ratio >= 0.90:
                quality_candidates["highZero"].append(candidate)
            if words > 62:
                quality_candidates["longLabel"].append(candidate)
        except Exception as exc:  # el informe debe continuar con el resto
            invalid_files.append({"file": str(path), "reason": str(exc)})

    return {
        "directory": str(split_dir),
        "expectedCount": expected,
        "fileCount": len(files),
        "numericIds": len(ids),
        "missingIds": [f"{EXPECTED_PREFIXES[split]}_{i:05d}" for i in missing],
        "invalidCount": len(invalid_files),
        "invalidFiles": invalid_files,
        "duplicateLabels": sum(1 for count in labels.values() if count > 1),
        "frames": quantiles([float(v) for v in lengths]),
        "zeroRatio": quantiles(zero_ratios),
        "validFrameRatio": quantiles(valid_frame_ratios),
        "leftValidFrameRatio": quantiles(left_valid_ratios),
        "rightValidFrameRatio": quantiles(right_valid_ratios),
        "labelWords": quantiles([float(v) for v in label_words]),
        "zeroRatioAtLeast90Percent": sum(v >= 0.90 for v in zero_ratios),
        "zeroRatioAtLeast99Percent": sum(v >= 0.99 for v in zero_ratios),
        "zeroRatioExactly100Percent": sum(v >= 1.0 for v in zero_ratios),
        "validFrameRatioBelow10Percent": sum(v < 0.10 for v in valid_frame_ratios),
        "validFrameRatioExactlyZero": sum(v == 0.0 for v in valid_frame_ratios),
        "labelsOver62Words": sum(v > 62 for v in label_words),
        "qualityCandidates": quality_candidates,
    }


def audit_vocab() -> dict[str, Any]:
    vocab_path = DATA_DIR / "vocab.json"
    result: dict[str, Any] = {"file": str(vocab_path), "exists": vocab_path.exists()}
    if not vocab_path.exists():
        return result
    try:
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        result.update({
            "count": len(vocab) if isinstance(vocab, list) else None,
            "isList": isinstance(vocab, list),
            "specialTokens": vocab[:4] if isinstance(vocab, list) else None,
            "validSpecialTokens": vocab[:4] == ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
            if isinstance(vocab, list) else False,
        })
    except Exception as exc:
        result["error"] = str(exc)
    return result


def build_report(audit: dict[str, Any]) -> str:
    lines = [
        "# Auditoría de datos LSA-T — Fase 1",
        "",
        f"- Fuente: `{DATA_DIR}`",
        "- Este informe es de solo lectura: no se eliminaron ni modificaron muestras.",
        "",
        "## Resumen",
        "",
        f"- Vocabulario: {audit['vocabulary'].get('count', 'no disponible')} tokens.",
        "- Muestras esperadas: 8.459.",
        f"- Muestras encontradas: {audit['totalFiles']}.",
        f"- Muestras inválidas: {audit['totalInvalid']}.",
        f"- Faltantes: {audit['totalMissing']}.",
        "",
        "## Por split",
        "",
        "| Split | Esperadas | Encontradas | Inválidas | Faltantes | 100% ceros | <10% cuadros válidos | >62 palabras |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        item = audit["splits"][split]
        lines.append(
            f"| {split} | {item['expectedCount']} | {item['fileCount']} | "
            f"{item['invalidCount']} | {len(item['missingIds'])} | "
            f"{item['zeroRatioExactly100Percent']} | "
            f"{item['validFrameRatioBelow10Percent']} | "
            f"{item['labelsOver62Words']} |",
        )
    lines.extend([
        "",
        "## Hallazgos que requieren decisión",
        "",
        "- Las muestras con baja cobertura no se excluyen automáticamente en esta fase.",
        "- Los IDs faltantes deben explicarse antes de regenerar el dataset.",
        "- La longitud máxima del decoder y la política para etiquetas largas deben "
        "definirse antes de reentrenar.",
        "- Los umbrales de calidad deben validarse con ejemplos reales, no elegirse "
        "solamente por conveniencia técnica.",
        "",
        "## IDs faltantes",
        "",
    ])
    for split in ("train", "val", "test"):
        missing = audit["splits"][split]["missingIds"]
        lines.append(f"- `{split}`: {', '.join(missing) if missing else 'ninguno'}")
    lines.extend(["", "## Muestras para revisión manual", ""])
    lines.append("El auditor no elimina ninguna muestra. Estas son candidatas para decidir la limpieza:")
    lines.append("")
    lines.append("| Split | 100% ceros | Cobertura <10% | ≥90% ceros | >62 palabras |")
    lines.append("|---|---:|---:|---:|---:|")
    for split in ("train", "val", "test"):
        candidates = audit["splits"][split]["qualityCandidates"]
        lines.append(
            f"| {split} | {len(candidates['allZero'])} | {len(candidates['lowCoverage'])} | "
            f"{len(candidates['highZero'])} | {len(candidates['longLabel'])} |",
        )
    lines.extend([
        "",
        "Los archivos completos y sus métricas están en `quality_candidates.json`.",
        "",
        "## Próximo paso",
        "",
        "Revisar este informe y aprobar los criterios de limpieza antes de ejecutar un nuevo entrenamiento.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase1"))
    args = parser.parse_args()

    splits = {split: audit_split(split) for split in ("train", "val", "test")}
    audit = {
        "dataDirectory": str(DATA_DIR),
        "expectedShape": list(EXPECTED_SHAPE),
        "vocabulary": audit_vocab(),
        "splits": splits,
        "totalFiles": sum(item["fileCount"] for item in splits.values()),
        "totalInvalid": sum(item["invalidCount"] for item in splits.values()),
        "totalMissing": sum(len(item["missingIds"]) for item in splits.values()),
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "data_audit.json"
    markdown_path = args.report_dir / "data_audit.md"
    candidates_path = args.report_dir / "quality_candidates.json"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(build_report(audit), encoding="utf-8")
    candidates_path.write_text(
        json.dumps({
            "dataDirectory": str(DATA_DIR),
            "thresholds": {
                "allZero": "zeroRatio >= 1.0",
                "lowCoverage": "validFrameRatio < 0.10",
                "highZero": "zeroRatio >= 0.90",
                "longLabel": "labelWords > 62",
            },
            "splits": {
                split: item["qualityCandidates"]
                for split, item in splits.items()
            },
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[audit] datos: {DATA_DIR}")
    for split, item in splits.items():
        print(
            f"[audit] {split}: {item['fileCount']}/{item['expectedCount']} archivos, "
            f"{item['invalidCount']} inválidos, {len(item['missingIds'])} faltantes",
        )
    print(f"[audit] JSON: {json_path}")
    print(f"[audit] Markdown: {markdown_path}")
    print(f"[audit] Candidatos: {candidates_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
