#!/usr/bin/env python3
"""
combinar_datasets.py — Une datasets ya procesados (contrato v3) en uno solo.

Eva LSA-94 se arma con:
  - ../data/processed_v3   las 3200 muestras de LSA64 (conversión exacta v2->v3,
                           no hace falta volver a correr MediaPipe);
  - data/processed_s012    los videos del sujeto 012 (preprocess_tasks.py).

Controles antes de escribir:
  - mismo contrato (contractVersion 3, 40 cuadros, 168 coordenadas);
  - mismo productor: si dos fuentes declaran taskSha256 distintos se aborta;
    si alguna no lo declara (processed_v3 es anterior), se avisa;
  - ninguna (seña, persona, repetición) repetida entre fuentes;
  - las etiquetas son exactamente 1..N del catálogo.

Uso:
    cd lsa-94
    ..\\venv\\Scripts\\python.exe combinar_datasets.py \\
        --fuentes ..\\data\\processed_v3 .\\data\\processed_s012 --out .\\data\\processed_94
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from eva_contract import CATALOG_PATH, CONTRACT_VERSION, COORDS, FRAMES, load_catalog, load_groups


def cargar_fuente(d: Path) -> dict:
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{d}: falta meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("contractVersion") != CONTRACT_VERSION:
        raise SystemExit(f"{d}: contrato v{meta.get('contractVersion')}, se espera v{CONTRACT_VERSION}")
    f = {k: np.load(d / f"{k}.npy") for k in ("X", "y", "subjects", "reps", "sizes")}
    if f["X"].shape[1:] != (FRAMES, COORDS):
        raise SystemExit(f"{d}: forma {f['X'].shape}, se espera (n, {FRAMES}, {COORDS})")
    n = len(f["X"])
    if any(len(v) != n for v in f.values()):
        raise SystemExit(f"{d}: los arrays no tienen el mismo largo")
    ruta_nombres = d / "nombres.npy"
    f["nombres"] = (np.load(ruta_nombres) if ruta_nombres.exists() else
                    np.array([f"{a:03d}_{b:03d}_{c:03d}"
                              for a, b, c in zip(f["y"], f["subjects"], f["reps"])]))
    f["meta"] = meta
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuentes", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    args = ap.parse_args()

    version, glosas = load_catalog(args.catalog)
    grupos = load_groups(args.catalog)
    fuentes = [cargar_fuente(d) for d in args.fuentes]

    shas = {str(d): f["meta"].get("taskSha256") for d, f in zip(args.fuentes, fuentes)}
    declarados = {s for s in shas.values() if s}
    if len(declarados) > 1:
        raise SystemExit(f"Las fuentes usaron distintos holistic_landmarker.task: {shas}")
    sin_sha = [d for d, s in shas.items() if not s]
    if sin_sha:
        print(f"  aviso: sin taskSha256 en {sin_sha}; se asume el mismo .task "
              f"(verificá que sea lsa64/assets/holistic_landmarker.task).")

    nombres = np.concatenate([f["nombres"] for f in fuentes])
    unicos, conteo = np.unique(nombres, return_counts=True)
    repetidos = unicos[conteo > 1]
    if len(repetidos):
        raise SystemExit(f"{len(repetidos)} muestras repetidas entre fuentes, p.ej. {repetidos[:5].tolist()}")

    X = np.concatenate([f["X"] for f in fuentes]).astype(np.float32)
    y = np.concatenate([f["y"] for f in fuentes]).astype(np.int32)
    sujetos = np.concatenate([f["subjects"] for f in fuentes]).astype(np.int32)
    reps = np.concatenate([f["reps"] for f in fuentes]).astype(np.int32)
    sizes = np.concatenate([f["sizes"] for f in fuentes]).astype(np.int32)

    clases = sorted(set(y.tolist()))
    if clases != list(range(1, len(glosas) + 1)):
        faltan = sorted(set(range(1, len(glosas) + 1)) - set(clases))
        sobran = sorted(set(clases) - set(range(1, len(glosas) + 1)))
        raise SystemExit(f"Las etiquetas no son 1..{len(glosas)}: faltan {faltan}, sobran {sobran}")

    # Orden canónico (seña, persona, repetición): determinista sin importar
    # el orden de las fuentes.
    orden = np.lexsort((reps, sujetos, y))
    X, y, sujetos, reps, sizes, nombres = (a[orden] for a in (X, y, sujetos, reps, sizes, nombres))

    args.out.mkdir(parents=True, exist_ok=True)
    for k, v in (("X", X), ("y", y), ("subjects", sujetos), ("reps", reps),
                 ("sizes", sizes), ("nombres", nombres)):
        np.save(args.out / f"{k}.npy", v)

    por_clase = {int(c): int((y == c).sum()) for c in clases}
    meta = {
        "contractVersion": CONTRACT_VERSION,
        "frames": FRAMES,
        "coordenadas": COORDS,
        "catalogo": {"version": version, "archivo": args.catalog.name, "clases": len(glosas)},
        "taskSha256": next(iter(declarados), None),
        "fuentes": [{"ruta": str(d.resolve()), "muestras": int(len(f["X"])),
                     "taskSha256": f["meta"].get("taskSha256")}
                    for d, f in zip(args.fuentes, fuentes)],
        "muestras": int(len(X)),
        "sujetos": {int(s): int((sujetos == s).sum()) for s in sorted(set(sujetos.tolist()))},
        "muestrasPorClase": por_clase,
        "resoluciones": sorted({f"{w}x{h}" for w, h in sizes}),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    print(f"X: {X.shape}  clases: {len(clases)}  sujetos: {meta['sujetos']}")
    for nombre, (a, b) in grupos.items():
        cuentas = sorted({por_clase[c] for c in range(a, b + 1)})
        suj = sorted({int(s) for s in sujetos[(y >= a) & (y <= b)]})
        print(f"  {nombre:11s} clases {a:3d}-{b:3d}: muestras por clase {cuentas}, sujetos {suj}")
    print(f"Guardado en {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
