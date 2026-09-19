#!/usr/bin/env python3
"""Retira de forma recuperable muestras vacías o etiquetas demasiado largas.

El script consume ``reports/phase1/quality_candidates.json`` y mueve solo la
unión de estas dos categorías a ``data/quarantine/phase1``. Nunca elimina un
archivo. Por seguridad, primero ejecutar sin argumentos para revisar el plan y
usar ``--apply`` para moverlo.

Uso:
    python lsa/06_clean_data.py
    python lsa/06_clean_data.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DATA_DIR, PROJECT_DIR


REPORT_DIR = PROJECT_DIR / "reports" / "phase1"
CANDIDATES_PATH = REPORT_DIR / "quality_candidates.json"
QUARANTINE_DIR = DATA_DIR / "quarantine" / "phase1"
PROCESSED_DIR = DATA_DIR / "processed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidates() -> list[dict[str, Any]]:
    if not CANDIDATES_PATH.is_file():
        raise FileNotFoundError(
            f"No existe {CANDIDATES_PATH}. Ejecutar primero 05_audit_data.py",
        )
    payload = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    by_file: dict[str, dict[str, Any]] = {}
    for split, groups in payload.get("splits", {}).items():
        for reason_name in ("allZero", "longLabel"):
            for item in groups.get(reason_name, []):
                source = str(Path(item["file"]).resolve())
                record = by_file.setdefault(source, {
                    "source": source,
                    "split": split,
                    "reasons": [],
                    "frames": item.get("frames"),
                    "zeroRatio": item.get("zeroRatio"),
                    "validFrameRatio": item.get("validFrameRatio"),
                    "labelWords": item.get("labelWords"),
                    "labelPreview": item.get("labelPreview", ""),
                })
                if reason_name not in record["reasons"]:
                    record["reasons"].append(reason_name)
    return sorted(by_file.values(), key=lambda item: item["source"])


def validate_records(records: list[dict[str, Any]]) -> None:
    processed_root = PROCESSED_DIR.resolve()
    for record in records:
        source = Path(record["source"])
        if not source.is_file():
            raise FileNotFoundError(f"Candidato inexistente: {source}")
        try:
            source.relative_to(processed_root)
        except ValueError as exc:
            raise ValueError(f"Candidato fuera de data/processed: {source}") from exc

        relative = source.relative_to(processed_root)
        destination = (QUARANTINE_DIR / "processed" / relative).resolve()
        if destination.exists():
            raise FileExistsError(
                f"La cuarentena ya contiene {destination}; no se sobrescribe.",
            )
        record["destination"] = str(destination)


def report(records: list[dict[str, Any]]) -> str:
    counts = Counter(record["split"] for record in records)
    lines = [
        "# Limpieza de datos LSA-T — Fase 1",
        "",
        "Las muestras no se eliminaron: se movieron a una cuarentena recuperable.",
        "",
        "## Reglas aplicadas",
        "",
        "- `allZero`: `zeroRatio >= 1.0`.",
        "- `longLabel`: `labelWords > 62`.",
        "",
        "## Resultado",
        "",
        f"- Total movido: {len(records)}.",
        f"- train: {counts['train']}.",
        f"- val: {counts['val']}.",
        f"- test: {counts['test']}.",
        f"- Destino: `{QUARANTINE_DIR}`.",
        "",
        "Las muestras con cobertura baja pero no completamente vacías no fueron movidas.",
        "",
        "## Restauración",
        "",
        "Usar `cleaning_manifest.json` para identificar cada destino y restaurar manualmente una muestra si una revisión posterior lo requiere.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="mueve las muestras a cuarentena; sin esto solo muestra el plan",
    )
    args = parser.parse_args()

    records = load_candidates()
    validate_records(records)
    counts = Counter(record["split"] for record in records)
    print(f"[clean] candidatas: {len(records)}")
    print(f"[clean] train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"[clean] cuarentena: {QUARANTINE_DIR}")

    if not args.apply:
        print("[clean] simulación: usar --apply para moverlas")
        return 0

    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = Path(record["source"])
        destination = Path(record["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        record["sha256Before"] = sha256(source)
        record["bytes"] = source.stat().st_size
        shutil.move(str(source), str(destination))
        record["sha256After"] = sha256(destination)
        if record["sha256Before"] != record["sha256After"]:
            raise RuntimeError(f"hash cambió al mover {source}")

    manifest = {
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "dataDirectory": str(DATA_DIR),
        "processedDirectory": str(PROCESSED_DIR),
        "quarantineDirectory": str(QUARANTINE_DIR),
        "rules": {
            "allZero": "zeroRatio >= 1.0",
            "longLabel": "labelWords > 62",
        },
        "totalMoved": len(records),
        "bySplit": dict(counts),
        "records": records,
    }
    manifest_path = REPORT_DIR / "cleaning_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = REPORT_DIR / "cleaning_report.md"
    report_path.write_text(report(records), encoding="utf-8")
    print(f"[clean] manifiesto: {manifest_path}")
    print(f"[clean] reporte: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
