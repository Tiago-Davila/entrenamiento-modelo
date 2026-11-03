#!/usr/bin/env python3
"""
exportar_reporte.py — Junta los resultados de un entrenamiento LSA-94 en
archivos CSV chicos, listos para graficar (página de métricas).

Lee:
  --ab       runs/ab_augment.json          (comparación de variantes)
  --cv       runs/cv_*_s42.json            (uno o más: reps y loso)
  --export   exports/                      (metricas.json + lsa-manifest.json)

Escribe en --out:
  historia_final.csv   época, loss, accuracy del modelo final
  historia_cv.csv      cv, fold, época, loss, accuracy, val_loss, val_accuracy
  cv_grupos.csv        cv, grupo, accuracy, top5, n
  cv_folds.csv         cv, fold, accuracy, mejor_val, mejor_epoca, minutos
  ab_resumen.csv       variante, cv, grupo, media, desvio, semillas
  recall_clase.csv     cv, clase, glosa, grupo, recall, n
  confusiones.csv      cv, grupo, real, predicho, veces
  modelo.csv           una fila con los datos del modelo exportado

Uso:
    ..\\venv\\Scripts\\python.exe exportar_reporte.py --ab runs/ab_augment.json \\
        --cv runs/cv_reps_*.json runs/cv_loso_*.json --export exports --out runs/reporte
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from eva_contract import load_catalog, load_groups


def escribir(path: Path, columnas: list[str], filas: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    print(f"  {path.name}: {len(filas)} filas")


def grupo_de(clase: int, grupos: dict) -> str:
    """clase 1..N -> nombre del grupo."""
    return next(g for g, (a, b) in grupos.items() if a <= clase <= b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", type=Path, default=None)
    ap.add_argument("--cv", type=Path, nargs="*", default=[])
    ap.add_argument("--export", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _, glosas = load_catalog()
    grupos = load_groups()
    args.out.mkdir(parents=True, exist_ok=True)

    # --- Modelo final ---
    if args.export:
        m = json.loads((args.export / "metricas.json").read_text(encoding="utf-8"))
        man = json.loads((args.export / "lsa-manifest.json").read_text(encoding="utf-8"))
        h = m.get("historia", {})
        escribir(args.out / "historia_final.csv", ["epoca", "loss", "accuracy"], [
            {"epoca": i + 1, "loss": l, "accuracy": a}
            for i, (l, a) in enumerate(zip(h.get("loss", []), h.get("accuracy", [])))])
        hp = m.get("hiperparametros", {})
        escribir(args.out / "modelo.csv",
                 ["version", "clases", "catalogo", "contrato", "epocas", "arquitectura",
                  "features", "balance", "tamano_kb", "train_accuracy_final",
                  "equivalencia_tflite", "sha256"], [{
                     "version": man["modelVersion"], "clases": man["numClasses"],
                     "catalogo": man["catalogVersion"], "contrato": man["contractVersion"],
                     "epocas": m.get("epocas"), "arquitectura": hp.get("arch"),
                     "features": hp.get("features"), "balance": hp.get("balance"),
                     "tamano_kb": m.get("tamanoKB"),
                     "train_accuracy_final": m.get("trainAccuracyFinal"),
                     "equivalencia_tflite": m.get("equivalenciaMaxDiff"),
                     "sha256": man["modelSha256"]}])

    # --- Validación cruzada ---
    hist, grp, folds, recall, conf = [], [], [], [], []
    for ruta in args.cv:
        d = json.loads(ruta.read_text(encoding="utf-8"))
        cv = d["cv"]
        for nombre, g in d["porGrupo"].items():
            grp.append({"cv": cv, "grupo": nombre, **{k: g[k] for k in ("accuracy", "top5", "n")}})
        for f in d["folds"]:
            folds.append({"cv": cv, "fold": f["fold"], "accuracy": f["accuracy"],
                          "mejor_val": f["mejorVal"], "mejor_epoca": f["mejorEpoca"],
                          "minutos": f["minutos"]})
            h = f.get("historia", {})
            for i in range(len(h.get("loss", []))):
                hist.append({"cv": cv, "fold": f["fold"], "epoca": i + 1,
                             **{k: h[k][i] for k in ("loss", "accuracy", "val_loss", "val_accuracy")}})
        cm = np.array(d["matrizConfusionTotal"])
        for c in range(len(cm)):
            n = int(cm[c].sum())
            if n:
                recall.append({"cv": cv, "clase": c + 1, "glosa": glosas[c],
                               "grupo": grupo_de(c + 1, grupos),
                               "recall": round(float(cm[c, c] / n), 4), "n": n})
        pares = [(int(cm[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm))
                 if i != j and cm[i, j] > 0]
        for veces, i, j in sorted(pares, reverse=True)[:25]:
            conf.append({"cv": cv, "grupo": grupo_de(i + 1, grupos),
                         "real": glosas[i], "predicho": glosas[j], "veces": veces})
    if args.cv:
        escribir(args.out / "historia_cv.csv",
                 ["cv", "fold", "epoca", "loss", "accuracy", "val_loss", "val_accuracy"], hist)
        escribir(args.out / "cv_grupos.csv", ["cv", "grupo", "accuracy", "top5", "n"], grp)
        escribir(args.out / "cv_folds.csv",
                 ["cv", "fold", "accuracy", "mejor_val", "mejor_epoca", "minutos"], folds)
        escribir(args.out / "recall_clase.csv",
                 ["cv", "clase", "glosa", "grupo", "recall", "n"], recall)
        escribir(args.out / "confusiones.csv", ["cv", "grupo", "real", "predicho", "veces"], conf)

    # --- A/B ---
    if args.ab:
        d = json.loads(args.ab.read_text(encoding="utf-8"))
        filas = []
        for variante, tipos in d.get("resumen", {}).items():
            for cv, gs in tipos.items():
                for grupo, v in gs.items():
                    filas.append({"variante": variante, "cv": cv, "grupo": grupo,
                                  "media": v["media"], "desvio": v["desvio"],
                                  "semillas": d["config"]["seeds"]})
        escribir(args.out / "ab_resumen.csv",
                 ["variante", "cv", "grupo", "media", "desvio", "semillas"], filas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
