#!/usr/bin/env python3
"""Congela la línea base antes de comenzar la implementación.

El script es de solo lectura sobre datos y modelos. Solo escribe el manifiesto
en ``reports/phase0`` para que una futura exportación pueda compararse con la
línea base seleccionada.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import BASE_DIR, CHECKPOINTS_DIR, DATA_DIR, EXPORTS_DIR, PROJECT_DIR


ARTIFACT_VERSION = "lsa-t-seq2seq-v1-experimental"
REPORT_DIR = PROJECT_DIR / "reports" / "phase0"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256(path),
    }
    if path.is_file():
        stat = path.stat()
        record.update({
            "bytes": stat.st_size,
            "modifiedUtc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc,
            ).isoformat(),
        })
    return record


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args, cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    names = (
        "numpy", "torch", "tensorflow", "onnx", "onnxruntime", "onnx2tf",
        "huggingface_hub", "pandas", "pose-format", "sacrebleu", "jiwer",
        "rouge-score", "tqdm",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_baseline() -> dict[str, Any]:
    tracked_files = [
        DATA_DIR / "vocab.json",
        CHECKPOINTS_DIR / "phase_b_best.pt",
        CHECKPOINTS_DIR / "phase_b_last.pt",
        CHECKPOINTS_DIR / "phase_b_history.json",
        PROJECT_DIR / "checkpoints_legacy" / "phase_b_best.pt",
        PROJECT_DIR / "checkpoints_legacy" / "phase_c_best.pt",
        PROJECT_DIR / "exports_legacy" / "encoder_int8.tflite",
        PROJECT_DIR / "exports_legacy" / "decoder_int8.tflite",
    ]
    return {
        "artifactVersion": ARTIFACT_VERSION,
        "capturedAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(PROJECT_DIR),
        "codeDirectory": str(BASE_DIR),
        "dataDirectory": str(DATA_DIR),
        "checkpointsDirectory": str(CHECKPOINTS_DIR),
        "exportsDirectory": str(EXPORTS_DIR),
        "git": {
            "branch": command("git", "branch", "--show-current"),
            "head": command("git", "rev-parse", "HEAD"),
            "statusPorcelain": command("git", "status", "--short"),
        },
        "canonicalCandidate": {
            "path": str(CHECKPOINTS_DIR / "phase_b_best.pt"),
            "role": "checkpoint candidato oficial para la próxima exportación",
            "reason": (
                "está junto al dataset raíz, coincide con la configuración vigente "
                "y es el checkpoint actual dentro de lsa_t/checkpoints"
            ),
        },
        "notSelectedAlternatives": [
            {
                "path": str(PROJECT_DIR / "checkpoints_legacy" / "phase_b_best.pt"),
                "reason": "copia histórica; se conserva para comparación",
            },
            {
                "path": str(PROJECT_DIR / "checkpoints_legacy" / "phase_c_best.pt"),
                "reason": "checkpoint de otra fase; no corresponde al candidato phase_b",
            },
        ],
        "files": [file_record(path) for path in tracked_files],
        "environment": {
            "python": f"Python {sys.version.split()[0]}",
            "pythonExecutable": sys.executable,
            "packagesFromActiveVenv": package_versions(),
        },
    }


def markdown(baseline: dict[str, Any]) -> str:
    canonical = baseline["canonicalCandidate"]
    files = baseline["files"]
    lines = [
        "# Línea base de implementación — Fase 0",
        "",
        f"- Versión de artefacto: `{baseline['artifactVersion']}`",
        f"- Capturada (UTC): `{baseline['capturedAtUtc']}`",
        f"- Repositorio: `{baseline['repository']}`",
        f"- Rama: `{baseline['git']['branch']}`",
        f"- Commit base: `{baseline['git']['head']}`",
        "",
        "## Candidato oficial",
        "",
        f"`{canonical['path']}`",
        "",
        canonical["reason"] + ".",
        "",
        "Las copias alternativas no se eliminan ni se reemplazan; quedan registradas para comparación.",
        "",
        "## Artefactos y hashes",
        "",
        "| Archivo | Existe | Bytes | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for record in files:
        lines.append(
            f"| `{record['path']}` | {'sí' if record['exists'] else 'no'} | "
            f"{record.get('bytes', '—')} | `{record.get('sha256') or '—'}` |",
        )
    lines.extend([
        "",
        "## Estado del repositorio",
        "",
        "El estado no se limpia automáticamente. Los archivos existentes y los cambios del usuario se conservan.",
        "",
        "## Entorno capturado",
        "",
        f"- Python: `{baseline['environment']['python'] or 'no disponible'}`",
    ])
    for name, version in baseline["environment"]["packagesFromActiveVenv"].items():
        lines.append(f"- {name}: `{version or 'no instalado en el entorno activo'}`")
    lines.extend([
        "",
        "## Decisiones de Fase 0",
        "",
        "- No se modificaron datos, checkpoints ni exportaciones.",
        "- No se seleccionó todavía ningún modelo para Android.",
        "- La exportación futura debe partir del candidato oficial y generar una nueva versión de artefacto.",
        "- La carpeta `exports_legacy` queda fuera de la línea base candidata hasta que pase la validación LiteRT.",
        "",
        "## Próximo paso",
        "",
        "Continuar con la auditoría de Fase 1 y decidir los criterios de limpieza antes de reentrenar.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    baseline = build_baseline()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "baseline.md").write_text(markdown(baseline), encoding="utf-8")
    print(f"[baseline] candidato: {baseline['canonicalCandidate']['path']}")
    print(f"[baseline] rama: {baseline['git']['branch']}")
    print(f"[baseline] JSON: {REPORT_DIR / 'baseline.json'}")
    print(f"[baseline] Markdown: {REPORT_DIR / 'baseline.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
