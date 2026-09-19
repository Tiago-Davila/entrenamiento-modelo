#!/usr/bin/env python3
"""
inspeccionar_lsat.py — Descarga SOLO annotations.csv de glosl-lsat y responde
la pregunta que decide toda la estrategia:

    ¿Hay glosas a nivel de seña, y traen tiempos de inicio/fin?

  - CON glosas y tiempos  -> se pueden recortar instancias y AMPLIAR el
                             clasificador de 64 señas. Es el mejor escenario.
  - CON glosas sin tiempos -> se sabe QUE señas hay en cada clip, no DONDE.
                             Sirve para supervision debil.
  - SIN glosas             -> solo subtitulos. Quedan el preentrenamiento
                             auto-supervisado y la medicion de brecha de dominio.

Descarga unos pocos MB, no los 49.8 GB del dataset completo.

Uso:
    pip install huggingface_hub pandas
    python inspeccionar_lsat.py
"""

from __future__ import annotations

import sys
import unicodedata
from collections import Counter

REPO = "pedroodb/glosl-lsat"
ARCHIVO = "annotations.csv"

# Las 64 señas de LSA64, en orden de indice de clase.
SENAS_LSA64 = [
    "Opaco", "Rojo", "Verde", "Amarillo", "Brillante", "Celeste", "Colores",
    "Rosa", "Mujer", "Enemigo", "Hijo", "Hombre", "Lejos", "Dibujar", "Nacer",
    "Aprender", "Llamar", "Skimmer", "Ubicacion", "Atrapar", "Gracias",
    "Aceptar", "Sordo", "Cuchillo", "Otro", "Ninguno", "Nombre", "Paciencia",
    "Perfume", "Deporte", "Cafe", "Uruguay", "Argentina", "Pais", "A_pesar_de",
    "Preguntar", "Cumpleanos", "Desayuno", "Foto", "Hambre", "Mapa",
    "Moneda", "Musica", "Barco", "Despues", "Duro", "Comida", "Aceite",
    "Fideos", "Pescado", "Acuerdo", "Duda", "Argolla", "Comprar", "Copa",
    "Bailar", "Novia", "Cerveza", "Guardar", "Candado", "Aguja", "Sur",
    "Aspirina", "Cruz",
]


def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn").lower()


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Falta huggingface_hub. Instalá:  pip install huggingface_hub pandas")
        return 2
    try:
        import pandas as pd
    except ImportError:
        print("Falta pandas. Instalá:  pip install pandas")
        return 2

    print(f"Descargando {ARCHIVO} de {REPO} ...")
    try:
        ruta = hf_hub_download(repo_id=REPO, filename=ARCHIVO, repo_type="dataset")
    except Exception as e:
        print(f"\nERROR al descargar: {e}\n")
        print("Si pide autenticación:  huggingface-cli login")
        print("Si el archivo no existe, mirá los nombres reales en:")
        print(f"  https://huggingface.co/datasets/{REPO}/tree/main")
        return 1

    print(f"OK -> {ruta}\n")
    df = pd.read_csv(ruta)

    print("=" * 62)
    print("ESTRUCTURA")
    print("=" * 62)
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {list(df.columns)}\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(3).to_string())
    print()

    if "split" in df.columns:
        print("Particiones:")
        for k, v in df["split"].value_counts().items():
            print(f"  {k:8s} {v:,}")
        print()

    # ---------- LA PREGUNTA QUE DECIDE TODO ----------
    print("=" * 62)
    print("¿HAY GLOSAS?")
    print("=" * 62)

    col_gloss = next((c for c in df.columns if "gloss" in c.lower()), None)
    if col_gloss is None:
        print("NO existe columna de glosas.")
        con_glosa = 0
    else:
        con_glosa = int(df[col_gloss].notna().sum())
        vacias = int((df[col_gloss].astype(str).str.strip() == "").sum())
        con_glosa -= vacias
        print(f"Columna '{col_gloss}': {con_glosa:,} de {len(df):,} "
              f"filas con contenido ({con_glosa/len(df):.1%})")
        if con_glosa:
            print("\nEjemplos de glosa:")
            for g in df[col_gloss].dropna().head(5):
                print(f"  {str(g)[:110]}")

    tiene_tiempos = all(c in df.columns for c in ("start", "end"))
    if tiene_tiempos:
        n_t = int(df[["start", "end"]].notna().all(axis=1).sum())
        print(f"\nColumnas start/end presentes: {n_t:,} filas con ambos valores")
    else:
        faltan = [c for c in ("start", "end") if c not in df.columns]
        print(f"\nNo hay columnas de tiempo: falta {faltan}")

    # ---------- Señantes ----------
    col_signer = next((c for c in df.columns if "signer" in c.lower()), None)
    if col_signer:
        print(f"\nSeñantes distintos: {df[col_signer].nunique()}")

    # ---------- Cobertura del vocabulario de 64 señas ----------
    col_texto = next((c for c in ("text", "texto", "sentence", "label")
                      if c in df.columns), None)
    if col_texto:
        print("\n" + "=" * 62)
        print("COBERTURA DE LAS 64 SEÑAS EN LOS SUBTITULOS")
        print("=" * 62)
        textos = df[col_texto].dropna().astype(str).map(sin_tildes)
        blob = " " + " ".join(textos) + " "

        hallados: list[tuple[str, int]] = []
        for sena in SENAS_LSA64:
            palabra = sin_tildes(sena.replace("_", " "))
            n = blob.count(f" {palabra} ")
            if n:
                hallados.append((sena, n))
        hallados.sort(key=lambda t: -t[1])

        print(f"Señas del vocabulario que aparecen como palabra: "
              f"{len(hallados)}/64\n")
        for sena, n in hallados[:25]:
            print(f"  {sena:14s} {n:5d} apariciones")
        if len(hallados) > 25:
            print(f"  ... y {len(hallados)-25} más")

        palabras = Counter()
        for t in textos:
            palabras.update(t.split())
        unicas = len(palabras)
        singles = sum(1 for c in palabras.values() if c == 1)
        print(f"\nVocabulario de los subtítulos: {unicas:,} palabras únicas, "
              f"{singles:,} aparecen una sola vez ({singles/unicas:.0%})")
        print(f"Largo medio de oración: "
              f"{sum(len(t.split()) for t in textos)/len(textos):.1f} palabras")

    # ---------- Veredicto ----------
    print("\n" + "=" * 62)
    print("VEREDICTO")
    print("=" * 62)
    if con_glosa and tiene_tiempos and n_t > 0:
        print("MEJOR ESCENARIO: hay glosas CON tiempos.")
        print("Se pueden recortar instancias de señas individuales y ampliar")
        print("directamente el clasificador de 64 señas.")
        print("\nSiguiente paso: revisar si las glosas usan el mismo")
        print("vocabulario/nomenclatura que LSA64, y descargar poses/.")
    elif con_glosa:
        print("ESCENARIO INTERMEDIO: hay glosas SIN tiempos.")
        print("Se sabe QUÉ señas contiene cada clip, no DÓNDE están.")
        print("Sirve para supervisión débil: correr el clasificador actual")
        print("sobre ventanas deslizantes y quedarse con las de confianza alta.")
    else:
        print("SIN GLOSAS: solo subtítulos en español (traducción, no glosa).")
        print("NO sirve para ampliar el clasificador directamente.")
        print("\nSí sirve para dos cosas de valor real:")
        print("  1. Medir la brecha de dominio contra LSA64 (103 señantes")
        print("     sordos reales sin guantes vs 10 oyentes con guantes).")
        print("  2. Preentrenamiento auto-supervisado del codificador, que")
        print("     ataca el cuello de botella de generalización entre personas.")
        print("\nAmbos requieren descargar poses/ pero NO los videos.")

    print("\nRecordatorio: los datos figuran SIN licencia explícita.")
    print("Resolver por escrito con los autores antes de entrenar y distribuir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())