#!/usr/bin/env python3
"""
preprocess_tasks.py — LSA64 -> arrays .npy de keypoints
usando MediaPipe **Tasks API** (HolisticLandmarker).

Reemplaza a preprocess.py, que usaba mp.solutions.holistic (legacy, eliminado).

Dos cambios respecto de la version anterior. Los dos INVALIDAN el dataset
viejo, hay que regenerar:

1. Productor de keypoints: Tasks HolisticLandmarker en vez del Holistic legacy.
   Es la MISMA task que corre en Android, asi que entrenamiento e inferencia
   comparten implementacion.

2. Muestreo temporal con ARITMETICA ENTERA EXACTA:
       idx[i] = (i * (T-1)) // (N-1)
   np.linspace(...).astype(int) NO es equivalente: usa punto flotante y
   difiere en ~3% de los largos de video (ej. T=46, i=13 -> linspace da 14,
   el entero da 15, porque 13*45/39 = 14.999999999999998 en float).
   La version entera es reproducible bit a bit entre Python, Kotlin y JS.

CONTRATO DE 201 COORDENADAS (no negociable):
    [  0: 63)  mano izquierda  21 landmarks x (x,y,z)
    [ 63:126)  mano derecha    21 landmarks x (x,y,z)
    [126:201)  pose 0..24      25 landmarks x (x,y,z)   (sin piernas)
    centrado: se resta el punto medio de los hombros (pose 11 y 12) a x e y.
    z NO se centra.
    no detectado -> ceros.

Modelo requerido: holistic_landmarker.task
    Descargalo de la documentacion oficial de MediaPipe (seccion Holistic
    landmark detection > Models) y pasalo con --model.

Uso:
    pip install "mediapipe>=1.0" opencv-python numpy tqdm
    python preprocess_tasks.py --videos ./lsa64 --out ./data_tasks \
           --model ./holistic_landmarker.task
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# --- Contrato ---
FRAMES_FIJOS = 40
N_COORDS = 201
POSE_N = 25                    # landmarks 0..24 (descartamos piernas 25..32)
OFF_MANO_IZQ = 0
OFF_MANO_DER = 63
OFF_POSE = 126
IDX_HOMBRO_IZQ = OFF_POSE + 11 * 3
IDX_HOMBRO_DER = OFF_POSE + 12 * 3

_CFG = None                    # (model_path, min_conf) por proceso worker


# --------------------------------------------------------------------------
# Contrato: muestreo temporal
# --------------------------------------------------------------------------

def indices_muestreo(total: int, n: int = FRAMES_FIJOS) -> list[int]:
    """Indices de los n frames a conservar, en aritmetica entera exacta.

    Reproducible bit a bit en cualquier lenguaje con division entera.
    Equivalente en Kotlin:  idx[i] = (i * (T - 1)) / (N - 1)   // Int division
    """
    if total <= 0:
        return []
    if n == 1:
        return [0]
    if total >= n:
        return [(i * (total - 1)) // (n - 1) for i in range(n)]
    # secuencia mas corta que n: se repite el ultimo frame (padding)
    return list(range(total)) + [total - 1] * (n - total)


def centrar(seq: np.ndarray) -> np.ndarray:
    """Resta el punto medio de los hombros a x e y de todas las coordenadas."""
    out = seq.copy()
    for f in range(out.shape[0]):
        cx = (out[f, IDX_HOMBRO_IZQ] + out[f, IDX_HOMBRO_DER]) / 2.0
        cy = (out[f, IDX_HOMBRO_IZQ + 1] + out[f, IDX_HOMBRO_DER + 1]) / 2.0
        out[f, 0::3] -= cx
        out[f, 1::3] -= cy
    return out


def vector_de_resultado(res) -> np.ndarray:
    """HolisticLandmarkerResult -> vector de 201 coordenadas.

    Tasks devuelve left/right_hand_landmarks explicitos: no hay que resolver
    la mano por handedness (que en la POC web era fuente de dudas).
    """
    v = np.zeros(N_COORDS, dtype=np.float32)

    for lms, off in ((getattr(res, "left_hand_landmarks", None), OFF_MANO_IZQ),
                     (getattr(res, "right_hand_landmarks", None), OFF_MANO_DER)):
        if lms:
            for i, lm in enumerate(lms[:21]):
                v[off + i * 3] = lm.x
                v[off + i * 3 + 1] = lm.y
                v[off + i * 3 + 2] = lm.z

    pose = getattr(res, "pose_landmarks", None)
    if pose:
        for i, lm in enumerate(pose[:POSE_N]):
            v[OFF_POSE + i * 3] = lm.x
            v[OFF_POSE + i * 3 + 1] = lm.y
            v[OFF_POSE + i * 3 + 2] = lm.z

    return v


# --------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------

def _init_worker(model_path: str, min_conf: float):
    """Guarda la configuracion. NO crea el landmarker.

    El landmarker se crea por video en _procesar(). Ver el comentario ahi:
    reusarlo entre videos rompe el contrato de timestamps del modo VIDEO.
    """
    global _CFG
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
    """Devuelve (secuencia, seña, sujeto, repeticion, frames_totales, ratio_ceros)."""
    import mediapipe as mp

    path = Path(path_str)

    # Un landmarker NUEVO por video, y se cierra al terminar.
    #
    # No se puede reusar entre videos: en modo VIDEO los timestamps tienen que
    # crecer monotonamente durante toda la vida del landmarker. Como cada video
    # arranca en ts=0, el segundo video que tocara la misma instancia falla con
    # "Input timestamp must be monotonically increasing", asi que solo se
    # salvaba el primer video de cada worker (~7 de 3200 con los defaults).
    #
    # Compensar con un offset acumulado evitaria el error pero no alcanza: el
    # modo VIDEO arrastra el ROI de los frames previos, y ese estado del video
    # anterior contaminaria los primeros frames del siguiente. Instancia limpia
    # por video equivale a una sesion nueva en Android, que es el contrato.
    landmarker = _crear_landmarker()

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
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
        return ("vacio", path.name, "sin frames")

    crudo = np.array(frames, dtype=np.float32)
    idx = indices_muestreo(len(crudo))
    seq = centrar(crudo[idx])

    partes = path.stem.split("_")
    return ("ok", seq.astype(np.float32), int(partes[0]), int(partes[1]),
            int(partes[2]), len(crudo), float(np.mean(seq == 0)))


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="ruta a holistic_landmarker.task")
    ap.add_argument("--signs", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--min-conf", type=float, default=0.5)
    args = ap.parse_args()

    modelo = Path(args.model)
    if not modelo.exists():
        raise SystemExit(
            f"No existe {modelo}.\nDescargá holistic_landmarker.task de la "
            f"documentacion oficial de MediaPipe (Holistic landmark detection > Models).")

    vid_dir, out_dir = Path(args.videos), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(vid_dir.glob("*.mp4"))
    if args.signs:
        quiere = {f"{s:03d}" for s in args.signs}
        videos = [v for v in videos if v.stem.split("_")[0] in quiere]
    if not videos:
        raise SystemExit("No se encontraron videos. Revisá --videos y --signs.")

    workers = args.workers or max(1, cpu_count() - 1)
    print(f"Procesando {len(videos)} videos | workers={workers}")
    print(f"Productor: MediaPipe Tasks HolisticLandmarker ({modelo.name})")
    print(f"Muestreo: entero exacto idx[i]=(i*(T-1))//({FRAMES_FIJOS}-1)")

    X, y, sujetos, reps = [], [], [], []
    largos, ceros, errores = [], [], []

    with Pool(processes=workers, initializer=_init_worker,
              initargs=(str(modelo), args.min_conf)) as pool:
        for r in tqdm(pool.imap_unordered(_procesar, [str(v) for v in videos],
                                          chunksize=4), total=len(videos)):
            if r[0] != "ok":
                errores.append((r[1], r[2]))
                continue
            _, seq, sena, suj, rep, n_frames, ratio = r
            X.append(seq); y.append(sena); sujetos.append(suj); reps.append(rep)
            largos.append(n_frames); ceros.append(ratio)

    if not X:
        raise SystemExit("No se proceso ningun video correctamente.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    sujetos = np.array(sujetos)
    reps = np.array(reps)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "subjects.npy", sujetos)
    np.save(out_dir / "reps.npy", reps)

    meta = {
        "productor": "mediapipe.tasks.vision.HolisticLandmarker",
        "modeloTask": modelo.name,
        "muestreo": "entero_exacto: idx[i] = (i*(T-1))//(N-1)",
        "frames": FRAMES_FIJOS,
        "coordenadas": N_COORDS,
        "bloques": {"mano_izq": [0, 63], "mano_der": [63, 126], "pose_0_24": [126, 201]},
        "centrado": "punto medio hombros (pose 11,12) restado a x,y. z sin centrar.",
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
    print(f"  sujetos: {sorted(set(sujetos.tolist()))}")
    print(f"  largo de video: min {min(largos)}  max {max(largos)}  "
          f"medio {np.mean(largos):.1f} frames")
    print(f"  ceros: medio {np.mean(ceros):.1%}  max {max(ceros):.1%}")

    # --- controles de sanidad ---
    print("\n--- Control ---")
    por_bloque = {
        "mano_izq": float(np.mean(X[..., 0:63] == 0)),
        "mano_der": float(np.mean(X[..., 63:126] == 0)),
        "pose": float(np.mean(X[..., 126:201] == 0)),
    }
    for k, v in por_bloque.items():
        print(f"  ceros {k:9s}: {v:5.1%}")
    print(f"  rango global: [{X.min():+.3f}, {X.max():+.3f}]  std {X.std():.4f}")

    if por_bloque["pose"] > 0.5:
        print("  *** ALERTA: pose mayormente en cero -> el centrado no es valido ***")
    cortos = sum(1 for l in largos if l < FRAMES_FIJOS)
    if cortos:
        print(f"  aviso: {cortos} videos con menos de {FRAMES_FIJOS} frames (hubo padding)")
    if errores:
        print(f"\n  {len(errores)} errores:")
        for nombre, msg in errores[:5]:
            print(f"    {nombre}: {msg}")

    print(f"\n  guardado en {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())