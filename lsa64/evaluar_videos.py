#!/usr/bin/env python3
"""
evaluar_videos.py — Mide un modelo exportado sobre videos propios (por ejemplo,
grabados con el teléfono en horizontal). Es la métrica de uso real: el test de
LSA64 es laboratorio contra laboratorio.

Los videos siguen el formato de LSA64: SSS_PPP_RRR.mp4 (seña 001..064,
persona, repetición). Cada video debe contener UNA seña, recortada como la
recortaría el segmentador de la app (desde que arrancan las manos hasta que
vuelven al reposo).

Se pueden pasar varios directorios de export para comparar modelos sobre los
mismos videos. Cada uno se evalúa con el contrato que declara su
lsa-manifest.json (v2 o v3), así se puede medir el modelo viejo contra el
nuevo:

    python evaluar_videos.py --videos ./mis_videos --modelos ./exports_v2 ./exports \
        --model-task ./assets/holistic_landmarker.task

Los keypoints extraídos se guardan en --cache para no volver a correr
MediaPipe en cada evaluación.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from eva_contract import load_catalog


def extraer(videos: list[Path], model_task: Path, contrato: int, workers: int):
    import preprocess_tasks as pt
    from tqdm import tqdm

    X, y, sujetos, nombres, errores = [], [], [], [], []
    with Pool(processes=workers, initializer=pt._init_worker,
              initargs=(str(model_task), 0.5, contrato)) as pool:
        for r in tqdm(pool.imap(pt._procesar, [str(v) for v in videos], chunksize=1),
                      total=len(videos)):
            if r[0] != "ok":
                errores.append((r[1], r[2]))
                continue
            _, seq, sena, suj, rep, *_ = r
            X.append(seq); y.append(sena); sujetos.append(suj)
            nombres.append(f"{sena:03d}_{suj:03d}_{rep:03d}")
    return (np.array(X, dtype=np.float32), np.array(y), np.array(sujetos),
            nombres, errores)


def cargar_o_extraer(videos, model_task, contrato, cache: Path, workers):
    ruta = cache / f"keypoints_v{contrato}.npz"
    nombres_videos = sorted(v.stem for v in videos)
    if ruta.exists():
        d = np.load(ruta, allow_pickle=False)
        if sorted(d["origen"].tolist()) == nombres_videos:
            return d["X"], d["y"], d["sujetos"], d["nombres"].tolist(), []
    X, y, s, n, err = extraer(videos, model_task, contrato, workers)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(ruta, X=X, y=y, sujetos=s, nombres=np.array(n),
             origen=np.array(nombres_videos))
    return X, y, s, n, err


def predecir(tflite: Path, X: np.ndarray) -> np.ndarray:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_path=str(tflite))
    it.allocate_tensors()
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for x in X:
        it.set_tensor(ent["index"], x[None].astype(np.float32))
        it.invoke()
        out.append(it.get_tensor(sal["index"])[0].copy())
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--modelos", nargs="+", required=True, type=Path,
                    help="directorios con modelo_lsa.tflite + lsa-manifest.json")
    ap.add_argument("--model-task", type=Path,
                    default=Path(__file__).with_name("assets") / "holistic_landmarker.task")
    ap.add_argument("--cache", type=Path, default=None)
    # Cada worker crea un HolisticLandmarker (~500 MB). Con muchos, y más si
    # hay un entrenamiento corriendo, MediaPipe falla en AllocateTensors.
    ap.add_argument("--workers", type=int, default=min(2, cpu_count()))
    ap.add_argument("--out", type=Path, default=None, help="JSON con el reporte")
    args = ap.parse_args()

    videos = sorted(args.videos.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No hay .mp4 en {args.videos}")
    cache = args.cache or (args.videos / ".keypoints")
    _, catalog = load_catalog()

    reporte = {"videos": len(videos), "modelos": {}}
    for d in args.modelos:
        manifest = json.loads((d / "lsa-manifest.json").read_text(encoding="utf-8"))
        contrato = int(manifest.get("contractVersion", 2))
        X, y, sujetos, nombres, errores = cargar_o_extraer(
            videos, args.model_task, contrato, cache, args.workers)
        if errores:
            print(f"  {len(errores)} videos sin procesar (v{contrato}):")
            for n, e in errores[:10]:
                print(f"    {n}: {e}")
        y0 = y.astype(int) - 1
        logits = predecir(d / "modelo_lsa.tflite", X)
        e = np.exp(logits - logits.max(1, keepdims=True))
        prob = e / e.sum(1, keepdims=True)
        pred = prob.argmax(1)
        top5 = np.argsort(-prob, axis=1)[:, :5]
        acc = float((pred == y0).mean())
        acc5 = float(np.mean([y0[i] in top5[i] for i in range(len(y0))]))
        conf_ok = prob.max(1)[pred == y0]
        conf_mal = prob.max(1)[pred != y0]

        print(f"\n=== {manifest.get('modelVersion', d.name)} (contrato v{contrato}) — {len(y0)} videos ===")
        print(f"top-1 {acc:.3f}   top-5 {acc5:.3f}   clases distintas predichas: {len(set(pred.tolist()))}")
        if len(conf_ok):
            print(f"confianza media: aciertos {conf_ok.mean():.2f}", end="")
        if len(conf_mal):
            print(f" | errores {conf_mal.mean():.2f}", end="")
        print()
        for s in sorted(set(sujetos.tolist())):
            m = sujetos == s
            print(f"  persona {s:3d}: top-1 {(pred[m] == y0[m]).mean():.3f} ({m.sum()} videos)")
        errores_lista = [
            {"video": nombres[i], "real": catalog[y0[i]], "predicho": catalog[pred[i]],
             "confianza": round(float(prob[i].max()), 3)}
            for i in np.flatnonzero(pred != y0)]
        for err in errores_lista[:15]:
            print(f"  {err['video']}: {err['real']} -> {err['predicho']} ({err['confianza']:.2f})")

        reporte["modelos"][str(d)] = {
            "modelVersion": manifest.get("modelVersion"),
            "contractVersion": contrato,
            "top1": round(acc, 4), "top5": round(acc5, 4),
            "clasesPredichas": len(set(pred.tolist())),
            "errores": errores_lista,
        }

    if args.out:
        args.out.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReporte en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
