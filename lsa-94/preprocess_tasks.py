#!/usr/bin/env python3
"""
preprocess_tasks.py — videos de señas -> arrays .npy de keypoints Eva v3
usando MediaPipe **Tasks API** (HolisticLandmarker).

Es la versión de lsa64/preprocess_tasks.py para Eva LSA-94. Diferencias:

  - Solo contrato v3 (la rama --contract 2 quedó en lsa64/).
  - Acepta los videos del celular: .mp4, .mov/.MOV, .m4v, .avi, .mkv, .webm.
  - Nombres ESTRICTOS SSS_PPP_RRR (seña_persona_repetición, tres dígitos,
    guion bajo). Se validan todos antes de procesar: un nombre mal escrito
    aborta la corrida listando los inválidos, en lugar de etiquetar mal.
  - Trazabilidad: nombres.npy (nombre canónico por muestra), sha256 del .task
    en meta.json, fps y extensiones observados.
  - Control de calidad por video (qa.json): con 4 repeticiones por clase, un
    video donde MediaPipe pierde la mano es el 25% de esa clase.

CONTRATO EVA v3 — 168 COORDENADAS (definición completa en eva_contract.py):
    [  0: 63)  mano izquierda  21 landmarks x (x,y,z)
    [ 63:126)  mano derecha    21 landmarks x (x,y,z)
    [126:168)  pose 11..24     14 landmarks x (x,y,z)
                 (sin cara: pose 0..10; sin piernas: pose 25..32)
    pixeles (x*W, y*H, z*W), centro en el punto medio de los hombros y
    division por la mediana del ancho de hombros. z NO se centra.
    no detectado -> ceros.

Muestreo temporal con ARITMETICA ENTERA EXACTA (eva_contract.sample_indices),
reproducible bit a bit entre Python y Kotlin.

Modelo requerido: assets/holistic_landmarker.task (el mismo que la app).

Uso:
    cd lsa-94
    ..\\venv\\Scripts\\python.exe preprocess_tasks.py --videos .\\data\\raw\\sujeto012 \\
           --out .\\data\\processed_s012 --model .\\assets\\holistic_landmarker.task
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from eva_contract import (
    CONTRACT_VERSION,
    COORDS,
    FRAMES,
    LEFT_HAND_OFFSET,
    POSE_OFFSET,
    POSE_SOURCE_INDICES,
    RIGHT_HAND_OFFSET,
    VIDEO_EXTENSIONS,
    normalize_sequence_v3,
    parse_video_name,
    sample_indices,
)

N_COORDS = COORDS

# Un video cuya mano activa falta en más de esta fracción de cuadros se
# reporta en qa.json para revisarlo a mano.
UMBRAL_QA = 0.30

_CFG = None                    # (model_path, min_conf) por proceso worker


def vector_de_resultado(res) -> np.ndarray:
    """HolisticLandmarkerResult -> vector crudo de 168 coordenadas.

    Tasks devuelve left/right_hand_landmarks explicitos: no hay que resolver
    la mano por handedness.
    """
    v = np.zeros(N_COORDS, dtype=np.float32)

    for lms, off in ((getattr(res, "left_hand_landmarks", None), LEFT_HAND_OFFSET),
                     (getattr(res, "right_hand_landmarks", None), RIGHT_HAND_OFFSET)):
        if lms:
            for i, lm in enumerate(lms[:21]):
                v[off + i * 3] = lm.x
                v[off + i * 3 + 1] = lm.y
                v[off + i * 3 + 2] = lm.z

    pose = getattr(res, "pose_landmarks", None)
    if pose:
        for local_index, source_index in enumerate(POSE_SOURCE_INDICES):
            if source_index >= len(pose):
                break
            lm = pose[source_index]
            v[POSE_OFFSET + local_index * 3] = lm.x
            v[POSE_OFFSET + local_index * 3 + 1] = lm.y
            v[POSE_OFFSET + local_index * 3 + 2] = lm.z

    return v


def calidad(crudo: np.ndarray) -> dict:
    """Fracción de cuadros (todos, antes de muestrear) sin cada mano.

    "Mano activa" es la que más aparece: la dominante en señas de una mano.
    """
    izq = np.any(crudo[:, LEFT_HAND_OFFSET:RIGHT_HAND_OFFSET] != 0.0, axis=1)
    der = np.any(crudo[:, RIGHT_HAND_OFFSET:POSE_OFFSET] != 0.0, axis=1)
    sin_izq, sin_der = float(1.0 - izq.mean()), float(1.0 - der.mean())
    return {
        "sinManos": round(float(1.0 - (izq | der).mean()), 4),
        "sinIzq": round(sin_izq, 4),
        "sinDer": round(sin_der, 4),
        "sinManoActiva": round(min(sin_izq, sin_der), 4),
    }


def listar_videos(vid_dir: Path) -> tuple[list[Path], list[str], list[str]]:
    """Videos válidos, nombres inválidos y claves duplicadas.

    Duplicado: la misma (seña, persona, repetición) con distinta extensión.
    """
    videos, invalidos = [], []
    for v in sorted(vid_dir.iterdir()):
        if not v.is_file() or v.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            parse_video_name(v.stem)
        except ValueError:
            invalidos.append(v.name)
            continue
        videos.append(v)
    conteo = Counter(v.stem for v in videos)
    duplicados = sorted(k for k, n in conteo.items() if n > 1)
    return videos, invalidos, duplicados


def sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------

def _init_worker(model_path: str, min_conf: float):
    """Guarda la configuracion. NO crea el landmarker.

    El landmarker se crea por video en _procesar(). Ver el comentario ahi:
    reusarlo entre videos rompe el contrato de timestamps del modo VIDEO.
    """
    global _CFG

    # Silenciar el ruido de inicializacion de MediaPipe.
    #
    # Como se crea un landmarker por video, cada uno emite ~11 lineas (EGL,
    # GL version y siete inference_feedback_manager). Salen por stderr, que es
    # donde tqdm dibuja la barra, asi que la pisan constantemente. Son mensajes
    # de absl desde C++, que no consultan GLOG_minloglevel: la unica forma
    # confiable de callarlos es redirigir el descriptor 2 del proceso worker.
    #
    # Esto NO oculta fallas de video: los errores viajan por el valor de
    # retorno de _procesar y se imprimen al final.
    #
    # Para depurar y recuperar los logs nativos: EVA_MEDIAPIPE_LOGS=1
    if os.environ.get("EVA_MEDIAPIPE_LOGS") != "1":
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)          # fd 2 ya es una copia; este sobra

    _CFG = (model_path, min_conf)


def _crear_landmarker():
    """Landmarker nuevo, con estado limpio, para un unico video."""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    model_path, min_conf = _CFG
    return vision.HolisticLandmarker.create_from_options(
        vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_conf,
            min_pose_landmarks_confidence=min_conf,
            min_hand_landmarks_confidence=min_conf,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
    )


def _procesar(path_str: str):
    """("ok", secuencia, seña, sujeto, rep, frames_totales, ratio_ceros,
    (W, H), qa) o ("error", nombre, mensaje)."""
    import mediapipe as mp

    path = Path(path_str)
    sena, suj, rep = parse_video_name(path.stem)

    # Un landmarker NUEVO por video, y se cierra al terminar.
    #
    # En modo VIDEO los timestamps tienen que crecer monotonamente durante
    # toda la vida del landmarker, y cada video arranca en ts=0. Ademas el
    # modo VIDEO arrastra el ROI de los cuadros previos: una instancia limpia
    # por video equivale a una sesion nueva en Android, que es el contrato.
    landmarker = _crear_landmarker()

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ancho = alto = 0
    frames: list[np.ndarray] = []
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # W/H del cuadro que realmente ve MediaPipe (ya rotado por
            # OpenCV si el video trae rotación en metadatos).
            alto, ancho = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode exige timestamps en ms estrictamente crecientes
            ts = int(n * 1000.0 / fps)
            res = landmarker.detect_for_video(imagen, ts)
            frames.append(vector_de_resultado(res))
            n += 1
    except Exception as e:
        return ("error", path.name, str(e)[:120])
    finally:
        cap.release()
        landmarker.close()

    if not frames:
        return ("error", path.name, "sin frames (¿códec no soportado?)")

    crudo = np.array(frames, dtype=np.float32)
    try:
        seq = normalize_sequence_v3(crudo[sample_indices(len(crudo))], ancho, alto)
    except ValueError as e:
        return ("error", path.name, f"v3: {e}")

    qa = calidad(crudo)
    qa["fps"] = round(float(fps), 2)
    return ("ok", seq.astype(np.float32), sena, suj, rep, len(crudo),
            float(np.mean(seq == 0)), (ancho, alto), qa)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="ruta a holistic_landmarker.task")
    ap.add_argument("--signs", nargs="*", type=int, default=None)
    ap.add_argument("--subjects", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--min-conf", type=float, default=0.5)
    args = ap.parse_args()

    modelo = Path(args.model)
    if not modelo.exists():
        raise SystemExit(
            f"No existe {modelo}.\nUsá el mismo holistic_landmarker.task que la app "
            f"(lsa64/assets/ o app/src/main/assets/vision/).")

    vid_dir, out_dir = Path(args.videos), Path(args.out)
    videos, invalidos, duplicados = listar_videos(vid_dir)
    if invalidos or duplicados:
        for nombre in invalidos:
            print(f"  nombre inválido: {nombre}")
        for clave in duplicados:
            print(f"  duplicado (misma seña/persona/rep): {clave}")
        raise SystemExit(f"{len(invalidos)} nombres inválidos y {len(duplicados)} duplicados. "
                         f"Renombrá a SSS_PPP_RRR antes de procesar.")

    if args.signs:
        videos = [v for v in videos if parse_video_name(v.stem)[0] in set(args.signs)]
    if args.subjects:
        videos = [v for v in videos if parse_video_name(v.stem)[1] in set(args.subjects)]
    if not videos:
        raise SystemExit("No se encontraron videos. Revisá --videos, --signs y --subjects.")

    out_dir.mkdir(parents=True, exist_ok=True)
    task_sha = sha256_archivo(modelo)
    workers = args.workers or max(1, cpu_count() - 1)
    print(f"Procesando {len(videos)} videos | workers={workers}")
    print(f"Productor: MediaPipe Tasks HolisticLandmarker ({modelo.name}, sha256 {task_sha[:12]}…)")
    print(f"Contrato v{CONTRACT_VERSION}, {FRAMES} cuadros, muestreo entero exacto")

    resultados, errores = [], []
    with Pool(processes=workers, initializer=_init_worker,
              initargs=(str(modelo), args.min_conf)) as pool:
        # chunksize=1: los resultados vuelven de a un video, asi la barra avanza
        # de forma continua y el final se balancea mejor.
        for r in tqdm(pool.imap_unordered(_procesar, [str(v) for v in videos],
                                          chunksize=1), total=len(videos)):
            if r[0] != "ok":
                errores.append((r[1], r[2]))
            else:
                resultados.append(r)

    if not resultados:
        raise SystemExit("No se proceso ningun video correctamente.")

    # imap_unordered devuelve en cualquier orden: se ordena por clave para que
    # dos corridas sobre los mismos videos den archivos idénticos.
    resultados.sort(key=lambda r: (r[2], r[3], r[4]))
    X = np.stack([r[1] for r in resultados]).astype(np.float32)
    y = np.array([r[2] for r in resultados])
    sujetos = np.array([r[3] for r in resultados])
    reps = np.array([r[4] for r in resultados])
    largos = [r[5] for r in resultados]
    ceros = [r[6] for r in resultados]
    sizes = [r[7] for r in resultados]
    nombres = np.array([f"{r[2]:03d}_{r[3]:03d}_{r[4]:03d}" for r in resultados])
    qa = {str(n): r[8] for n, r in zip(nombres, resultados)}

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "subjects.npy", sujetos)
    np.save(out_dir / "reps.npy", reps)
    np.save(out_dir / "sizes.npy", np.array(sizes, dtype=np.int32))
    np.save(out_dir / "nombres.npy", nombres)

    sospechosos = sorted(((v["sinManoActiva"], k) for k, v in qa.items()
                          if v["sinManoActiva"] > UMBRAL_QA or v["sinManos"] > UMBRAL_QA),
                         reverse=True)
    (out_dir / "qa.json").write_text(json.dumps({
        "umbral": UMBRAL_QA,
        "sospechosos": [k for _, k in sospechosos],
        "porVideo": qa,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "productor": "mediapipe.tasks.vision.HolisticLandmarker",
        "modeloTask": modelo.name,
        "taskSha256": task_sha,
        "muestreo": "entero_exacto: T>=40 (i*(T-1))//39; T<40 (i*T)//40",
        "frames": FRAMES,
        "contractVersion": CONTRACT_VERSION,
        "coordenadas": N_COORDS,
        "bloques": {"mano_izq": [0, 63], "mano_der": [63, 126], "pose_11_24": [126, 168]},
        "poseDescartada": "0..10 (cara) y 25..32 (piernas)",
        "normalizacion": "v3: pixeles (x*W, y*H, z*W), centro en hombros, / mediana del ancho de hombros",
        "resoluciones": sorted({f"{w}x{h}" for w, h in sizes}),
        "extensiones": sorted({v.suffix for v in videos}),
        "fps": sorted({v["fps"] for v in qa.values()}),
        "minConfianza": args.min_conf,
        "videos": int(len(X)),
        "errores": len(errores),
        "largoVideoMin": int(min(largos)),
        "largoVideoMax": int(max(largos)),
        "largoVideoMedio": round(float(np.mean(largos)), 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    print(f"\nListo.")
    print(f"  X: {X.shape}   y: {y.shape}   clases: {len(set(y.tolist()))}")
    print(f"  sujetos: {sorted(set(sujetos.tolist()))}   resoluciones: {meta['resoluciones']}")
    print(f"  largo de video: min {min(largos)}  max {max(largos)}  "
          f"medio {np.mean(largos):.1f} frames   fps: {meta['fps']}")
    print(f"  ceros: medio {np.mean(ceros):.1%}  max {max(ceros):.1%}")

    # --- controles de sanidad ---
    print("\n--- Control ---")
    por_bloque = {
        "mano_izq": float(np.mean(X[..., 0:63] == 0)),
        "mano_der": float(np.mean(X[..., 63:126] == 0)),
        "pose": float(np.mean(X[..., 126:168] == 0)),
    }
    for k, v in por_bloque.items():
        print(f"  ceros {k:9s}: {v:5.1%}")
    print(f"  rango global: [{X.min():+.3f}, {X.max():+.3f}]  std {X.std():.4f}")

    if por_bloque["pose"] > 0.5:
        print("  *** ALERTA: pose mayormente en cero -> el centrado no es valido ***")
    cortos = sum(1 for l in largos if l < FRAMES)
    if cortos:
        print(f"  aviso: {cortos} videos con menos de {FRAMES} frames (se estiraron repitiendo cuadros)")
    if sospechosos:
        print(f"\n  {len(sospechosos)} videos con la mano activa ausente en más del "
              f"{UMBRAL_QA:.0%} de los cuadros (revisar, ver qa.json):")
        for frac, nombre in sospechosos[:15]:
            print(f"    {nombre}: {frac:.0%}")
    if errores:
        print(f"\n  {len(errores)} errores:")
        for nombre, msg in errores[:10]:
            print(f"    {nombre}: {msg}")

    print(f"\n  guardado en {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
