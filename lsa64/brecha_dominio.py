#!/usr/bin/env python3
"""
brecha_dominio.py — Descarga poses de glosl-lsat (SIN videos) y mide la brecha
de dominio contra el dataset LSA64 ya procesado.

Pregunta que responde:
    ¿Cuánto se parecen los keypoints de señantes sordos reales, sin guantes y
    en condiciones no controladas, a los de los 10 señantes oyentes con guantes
    grabados en laboratorio?

Es la medición más barata y más informativa disponible ahora: no necesita
etiquetas de seña, no necesita entrenar nada, y condiciona todo lo que venga
después. Si la brecha es grande, el 0.85 medido sobre LSA64 no representa el
desempeño en uso real.

Modos:
  --inspect          descarga UN .pose y vuelca su estructura. Empezar por acá.
  --download         descarga N archivos .pose (sin videos)
  --compare          compara distribuciones contra el dataset LSA64 procesado

Uso tipico:
    pip install huggingface_hub pose-format numpy
    python brecha_dominio.py --inspect
    python brecha_dominio.py --download --limit 300
    python brecha_dominio.py --compare --lsa64 ./data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from eva_contract import COORDS, POSE_OFFSET, POSE_SOURCE_INDICES, center_sequence

REPO = "pedroodb/glosl-lsat"

# --- Contrato Eva v2 de 168 coordenadas (mismo que preprocess_tasks.py) ---
N_COORDS = COORDS
OFF_MANO_IZQ = 0
OFF_MANO_DER = 63
OFF_POSE = POSE_OFFSET
IDX_HOMBRO_IZQ = OFF_POSE
IDX_HOMBRO_DER = OFF_POSE + 3

# Nombres de componente que usa MediaPipe Holistic en el contenedor .pose.
# Se resuelven de forma tolerante porque la nomenclatura varía entre versiones.
CANDIDATOS = {
    "pose": ("POSE_LANDMARKS", "BODY", "POSE"),
    "mano_izq": ("LEFT_HAND_LANDMARKS", "LEFT_HAND"),
    "mano_der": ("RIGHT_HAND_LANDMARKS", "RIGHT_HAND"),
    "cara": ("FACE_LANDMARKS", "FACE"),
}


# --------------------------------------------------------------------------
# Lectura de .pose
# --------------------------------------------------------------------------

def abrir_pose(ruta: Path):
    from pose_format import Pose
    with open(ruta, "rb") as f:
        return Pose.read(f.read())


def mapa_componentes(pose) -> dict[str, tuple[int, int]]:
    """Devuelve {clave_logica: (offset_de_punto, cantidad_de_puntos)}."""
    offset = 0
    encontrados: dict[str, tuple[int, int]] = {}
    for comp in pose.header.components:
        nombre = comp.name.upper()
        for clave, alias in CANDIDATOS.items():
            if any(a in nombre for a in alias) and clave not in encontrados:
                encontrados[clave] = (offset, len(comp.points))
        offset += len(comp.points)
    return encontrados


def datos_numpy(pose) -> np.ndarray:
    """(frames, puntos, dims), con NaN donde el punto está enmascarado."""
    d = pose.body.data
    arr = np.ma.filled(d, np.nan) if isinstance(d, np.ma.MaskedArray) else np.asarray(d)
    if arr.ndim == 4:          # (frames, personas, puntos, dims)
        arr = arr[:, 0]
    return arr


def a_contrato_eva_v2(pose) -> np.ndarray | None:
    """Convierte un .pose de 543 puntos al vector Eva v2 de 168 coordenadas.

    IMPORTANTE: la normalización glosl-norm-v1 se aplica al CARGAR, no está
    guardada. Acá NO se invoca pose.normalize(): se toman los valores crudos y
    se aplica el centrado propio en hombros. Encadenar ambas normalizaciones
    produciría vectores mal formados sin ningún error visible.
    """
    comps = mapa_componentes(pose)
    if not all(k in comps for k in ("pose", "mano_izq", "mano_der")):
        return None

    arr = datos_numpy(pose)                 # (T, puntos, dims)
    T = arr.shape[0]
    out = np.zeros((T, N_COORDS), dtype=np.float32)

    def volcar(clave, destino, n_max):
        off, n = comps[clave]
        n = min(n, n_max)
        bloque = arr[:, off:off + n, :3]     # x, y, z
        bloque = np.nan_to_num(bloque, nan=0.0)
        out[:, destino:destino + n * 3] = bloque.reshape(T, n * 3)

    volcar("mano_izq", OFF_MANO_IZQ, 21)
    volcar("mano_der", OFF_MANO_DER, 21)
    pose_off, pose_n = comps["pose"]
    for local_index, source_index in enumerate(POSE_SOURCE_INDICES):
        if source_index >= pose_n:
            break
        point = np.nan_to_num(arr[:, pose_off + source_index, :3], nan=0.0)
        start = OFF_POSE + local_index * 3
        out[:, start:start + 3] = point

    return center_sequence(out)


# --------------------------------------------------------------------------
# Estadisticas
# --------------------------------------------------------------------------

BLOQUES = {
    "mano_izq": (0, 63),
    "mano_der": (63, 126),
    "pose": (126, 168),
}


def estadisticas(secuencias: list[np.ndarray] | np.ndarray, nombre: str) -> dict:
    if isinstance(secuencias, np.ndarray) and secuencias.ndim == 3:
        secuencias = list(secuencias)

    todo = np.concatenate([s for s in secuencias], axis=0)   # (sum_T, 168)
    d = {
        "nombre": nombre,
        "secuencias": len(secuencias),
        "frames_totales": int(todo.shape[0]),
        "frames_por_secuencia": round(float(np.mean([len(s) for s in secuencias])), 1),
        "ceros_global": round(float(np.mean(todo == 0)), 4),
        "min": round(float(np.nanmin(todo)), 4),
        "max": round(float(np.nanmax(todo)), 4),
        "std": round(float(np.nanstd(todo)), 4),
        "bloques": {},
    }
    for b, (i, j) in BLOQUES.items():
        sub = todo[:, i:j]
        d["bloques"][b] = {
            "ceros": round(float(np.mean(sub == 0)), 4),
            "std": round(float(np.nanstd(sub)), 4),
            "x_min": round(float(np.nanmin(sub[:, 0::3])), 4),
            "x_max": round(float(np.nanmax(sub[:, 0::3])), 4),
            "y_min": round(float(np.nanmin(sub[:, 1::3])), 4),
            "y_max": round(float(np.nanmax(sub[:, 1::3])), 4),
        }
    # movimiento medio entre cuadros consecutivos
    movs = [float(np.nanmean(np.abs(np.diff(s, axis=0)))) for s in secuencias if len(s) > 1]
    d["movimiento_medio"] = round(float(np.mean(movs)), 5) if movs else 0.0
    return d


def imprimir(d: dict) -> None:
    print(f"\n--- {d['nombre']} ---")
    print(f"  secuencias {d['secuencias']:,} | frames {d['frames_totales']:,} "
          f"| medio {d['frames_por_secuencia']}/sec")
    print(f"  ceros {d['ceros_global']:.1%} | rango [{d['min']:+.3f},{d['max']:+.3f}] "
          f"| std {d['std']:.4f} | movimiento {d['movimiento_medio']:.5f}")
    for b, s in d["bloques"].items():
        print(f"    {b:9s} ceros {s['ceros']:6.1%}  std {s['std']:.4f}  "
              f"x[{s['x_min']:+.2f},{s['x_max']:+.2f}] y[{s['y_min']:+.2f},{s['y_max']:+.2f}]")


def comparar(a: dict, b: dict) -> None:
    print("\n" + "=" * 66)
    print("BRECHA DE DOMINIO")
    print("=" * 66)
    print(f"{'métrica':<26}{a['nombre']:>18}{b['nombre']:>18}   señal")
    print("-" * 70)

    def fila(etiqueta, va, vb, fmt="{:.4f}", umbral=2.0):
        rel = ""
        if va and vb:
            r = max(va, vb) / max(min(va, vb), 1e-9)
            if r >= umbral:
                rel = f"  <-- {r:.1f}x"
        print(f"{etiqueta:<26}{fmt.format(va):>18}{fmt.format(vb):>18}{rel}")

    fila("ceros global", a["ceros_global"], b["ceros_global"], "{:.1%}", 1.8)
    fila("std global", a["std"], b["std"])
    fila("movimiento medio", a["movimiento_medio"], b["movimiento_medio"], "{:.5f}", 1.8)
    for blq in BLOQUES:
        fila(f"ceros {blq}", a["bloques"][blq]["ceros"],
             b["bloques"][blq]["ceros"], "{:.1%}", 1.8)
        fila(f"std {blq}", a["bloques"][blq]["std"], b["bloques"][blq]["std"])

    print("\nCómo leerlo:")
    print("  ceros muy distintos en manos -> la deteccion de manos se comporta")
    print("     distinto sin guantes; afecta directo a la entrada del modelo.")
    print("  std muy distinto             -> escalas diferentes; el modelo veria")
    print("     magnitudes fuera del rango en que fue entrenado.")
    print("  movimiento muy distinto      -> velocidad de señado diferente")
    print("     (laboratorio pausado vs señante nativo fluido).")
    print("\nDiferencias marcadas no invalidan el modelo: cuantifican cuánto")
    print("del 0.85 medido en laboratorio se sostiene fuera de él.")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=int, default=0, metavar="N",
                    help="revisa N archivos y reporta qué componentes traen datos")
    ap.add_argument("--inspect", action="store_true",
                    help="descarga un solo .pose y vuelca su estructura")
    ap.add_argument("--download", action="store_true",
                    help="descarga archivos .pose (sin videos)")
    ap.add_argument("--compare", action="store_true",
                    help="compara distribuciones contra LSA64")
    ap.add_argument("--limit", type=int, default=200,
                    help="cuántos .pose usar (default 200)")
    ap.add_argument("--poses-dir", default="./lsat_poses")
    ap.add_argument("--lsa64", default="./data/processed",
                    help="carpeta con X.npy del dataset LSA64 procesado")
    ap.add_argument("--out", default="brecha_dominio.json")
    args = ap.parse_args()

    if not (args.inspect or args.download or args.compare or args.audit):
        ap.print_help()
        return 0

    poses_dir = Path(args.poses_dir)

    # ---------- AUDIT ----------
    if args.audit:
        from huggingface_hub import list_repo_files, hf_hub_download
        import random
        archivos = list_repo_files(REPO, repo_type="dataset")
        poses = sorted(f for f in archivos
                       if f.startswith("poses/") and f.endswith(".pose"))
        random.seed(0)
        muestra = random.sample(poses, min(args.audit, len(poses)))
        print(f"Auditando {len(muestra)} archivos de {len(poses):,} ...\n")

        resumen = {"pose": [], "cara": [], "mano_izq": [], "mano_der": []}
        sin_pose = 0
        for i, f in enumerate(muestra, 1):
            try:
                ruta = hf_hub_download(REPO, f, repo_type="dataset")
                p = abrir_pose(Path(ruta))
                comps = mapa_componentes(p)
                arr = datos_numpy(p)
                fila = []
                for clave in ("pose", "cara", "mano_izq", "mano_der"):
                    if clave not in comps:
                        fila.append("--")
                        continue
                    o, n = comps[clave]
                    frac = float(np.mean(np.isnan(arr[:, o:o + n, :3])))
                    resumen[clave].append(frac)
                    fila.append(f"{clave}:{frac:5.0%}")
                if resumen["pose"] and resumen["pose"][-1] > 0.99:
                    sin_pose += 1
                print(f"  {i:3d}. {Path(f).stem[:44]:44s} " + "  ".join(fila))
            except Exception as e:
                print(f"  {i:3d}. {Path(f).stem[:44]:44s} ERROR {str(e)[:40]}")

        print("\n" + "=" * 62)
        print("RESUMEN — fracción de NaN por componente")
        print("=" * 62)
        for clave, vals in resumen.items():
            if vals:
                print(f"  {clave:9s} media {np.mean(vals):6.1%} | "
                      f"min {min(vals):5.1%} | max {max(vals):5.1%} | "
                      f"archivos 100% vacíos: {sum(1 for v in vals if v > 0.99)}/{len(vals)}")
        print(f"\nArchivos SIN datos de cuerpo: {sin_pose}/{len(muestra)}")
        if sin_pose == len(muestra):
            print("\nLa ausencia de POSE_LANDMARKS es SISTEMÁTICA en el dataset.")
            print("El centrado en hombros del contrato no es aplicable a estos datos.")
        elif sin_pose:
            print("\nLa ausencia es PARCIAL: se pueden filtrar los archivos con cuerpo.")
        else:
            print("\nHay cuerpo en todos: el archivo inspeccionado era una excepción.")
        return 0

    # ---------- INSPECT ----------
    if args.inspect:
        from huggingface_hub import list_repo_files, hf_hub_download
        print(f"Listando archivos de {REPO} ...")
        archivos = list_repo_files(REPO, repo_type="dataset")
        poses = [f for f in archivos if f.startswith("poses/") and f.endswith(".pose")]
        print(f"  .pose disponibles: {len(poses):,}")
        if not poses:
            print("  No encontré poses/. Archivos de ejemplo:")
            for f in archivos[:15]:
                print("   ", f)
            return 1

        ruta = hf_hub_download(REPO, poses[0], repo_type="dataset")
        print(f"\nDescargado: {poses[0]}")
        p = abrir_pose(Path(ruta))
        print(f"  fps: {p.body.fps}")
        print(f"  dims: {p.header.num_dims()}   puntos totales: {p.header.total_points()}")
        print("  componentes:")
        off = 0
        for c in p.header.components:
            print(f"    {c.name:26s} {len(c.points):4d} puntos  "
                  f"[{off}..{off+len(c.points)})  formato {c.format}")
            off += len(c.points)
        comps = mapa_componentes(p)
        print(f"\n  mapeo al contrato: {comps}")

        # --- diagnostico: por que puede fallar el centrado ---
        arr = datos_numpy(p)
        print("\n  --- diagnóstico de datos crudos ---")
        d = p.body.data
        if isinstance(d, np.ma.MaskedArray):
            print(f"  MaskedArray: {np.mean(np.ma.getmaskarray(d)):.1%} de puntos enmascarados")
        else:
            print("  no es MaskedArray")
        print(f"  NaN en el array: {np.mean(np.isnan(arr)):.1%}")

        off_pose = comps["pose"][0]
        hi = arr[:, off_pose + 11, :2]      # hombro izquierdo (x,y)
        hd = arr[:, off_pose + 12, :2]      # hombro derecho
        valid = ~(np.isnan(hi).any(1) | np.isnan(hd).any(1))
        nulos = ((np.nan_to_num(hi) == 0).all(1) | (np.nan_to_num(hd) == 0).all(1))
        print(f"  hombros detectados en {valid.mean():.1%} de los cuadros")
        print(f"  hombros en CERO en    {nulos.mean():.1%} de los cuadros")
        if valid.any():
            print(f"  hombro izq medio: x={np.nanmean(hi[:,0]):.3f} y={np.nanmean(hi[:,1]):.3f}")
            print(f"  hombro der medio: x={np.nanmean(hd[:,0]):.3f} y={np.nanmean(hd[:,1]):.3f}")

        for clave in ("pose", "mano_izq", "mano_der"):
            o, n = comps[clave]
            b = arr[:, o:o + n, :3]
            nan_frac = float(np.mean(np.isnan(b)))
            bn = np.nan_to_num(b)
            print(f"  {clave:9s} NaN {nan_frac:6.1%} | cero {np.mean(bn==0):6.1%} | "
                  f"x[{np.nanmin(b[...,0]):+.3f},{np.nanmax(b[...,0]):+.3f}] "
                  f"z[{np.nanmin(b[...,2]):+.3f},{np.nanmax(b[...,2]):+.3f}]")

        v = a_contrato_eva_v2(p)
        if v is None:
            print("  *** No se pudieron ubicar manos y pose. Revisar nombres. ***")
            return 1
        print(f"\n  vector resultante: {v.shape}")
        print(f"  ceros {np.mean(v==0):.1%} | rango [{v.min():+.3f},{v.max():+.3f}]")
        print(f"  hombros tras centrar: izq x={v[:,IDX_HOMBRO_IZQ].mean():+.4f} "
              f"der x={v[:,IDX_HOMBRO_DER].mean():+.4f}  (deben ser simétricos)")
        media_x = float(np.mean(v[:, 0::3]))
        print(f"  media de x: {media_x:+.4f}  "
              f"({'OK, centrado' if abs(media_x) < 0.25 else 'SOSPECHOSO: parece sin centrar'})")
        print("\nSi esto se ve razonable, seguí con --download")
        return 0

    # ---------- DOWNLOAD ----------
    if args.download:
        from huggingface_hub import list_repo_files, hf_hub_download
        poses_dir.mkdir(parents=True, exist_ok=True)
        archivos = list_repo_files(REPO, repo_type="dataset")
        poses = sorted(f for f in archivos
                       if f.startswith("poses/") and f.endswith(".pose"))[:args.limit]
        print(f"Descargando {len(poses)} archivos .pose (sin videos) ...")
        for i, f in enumerate(poses, 1):
            destino = poses_dir / Path(f).name
            if destino.exists():
                continue
            ruta = hf_hub_download(REPO, f, repo_type="dataset")
            destino.write_bytes(Path(ruta).read_bytes())
            if i % 25 == 0:
                print(f"  {i}/{len(poses)}")
        total = sum(p.stat().st_size for p in poses_dir.glob("*.pose"))
        print(f"OK: {len(list(poses_dir.glob('*.pose')))} archivos, "
              f"{total/1e6:.1f} MB en {poses_dir}")
        return 0

    # ---------- COMPARE ----------
    archivos = sorted(poses_dir.glob("*.pose"))[:args.limit]
    if not archivos:
        print(f"No hay .pose en {poses_dir}. Corré primero --download")
        return 1

    print(f"Leyendo {len(archivos)} secuencias de LSA-T ...")
    secs, fallos = [], 0
    for i, f in enumerate(archivos, 1):
        try:
            v = a_contrato_eva_v2(abrir_pose(f))
            if v is not None and len(v) > 1:
                secs.append(v)
            else:
                fallos += 1
        except Exception:
            fallos += 1
        if i % 50 == 0:
            print(f"  {i}/{len(archivos)}")
    if fallos:
        print(f"  ({fallos} archivos no se pudieron leer)")
    if not secs:
        print("No se pudo leer ninguna secuencia.")
        return 1

    st_lsat = estadisticas(secs, "LSA-T")
    imprimir(st_lsat)

    ruta_x = Path(args.lsa64) / "X.npy"
    if not ruta_x.exists():
        print(f"\nNo encontré {ruta_x}. Pasá --lsa64 con la carpeta correcta.")
        Path(args.out).write_text(json.dumps({"lsat": st_lsat}, indent=2))
        return 0

    X = np.load(ruta_x)
    st_lsa64 = estadisticas(X, "LSA64")
    imprimir(st_lsa64)
    comparar(st_lsa64, st_lsat)

    Path(args.out).write_text(
        json.dumps({"lsa64": st_lsa64, "lsat": st_lsat}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nGuardado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
