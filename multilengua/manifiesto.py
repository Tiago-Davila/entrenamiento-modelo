"""Manifiesto compartido por todas las lenguas (lsa, asl, lse).

Lo escribe el entrenamiento al exportar y lo lee la app Android. Cada
manifiesto describe un modelo, su catálogo y el dataset de origen, con la
licencia y la atribución que la interfaz debe mostrar.

Formato (JSON, UTF-8):
    language            "lsa" | "asl" | "lse"
    modelVersion        versión del modelo, p. ej. "eva-lsa94-v4"
    catalogVersion      versión del catálogo que acompaña al modelo
    dataset             {"nombre": str, "version": str} del dataset de origen
    license             licencia del modelo, p. ej. "CC BY-NC-SA 4.0"
    attribution         texto de atribución que muestra la app
    modelSha256         SHA-256 del .tflite (64 caracteres hexadecimales)
    catalogSha256       SHA-256 de catalogo_senas.json (64 caracteres hex)
    numClasses          cantidad de clases de la salida
    outputsProbabilities  siempre false: la salida son logits
    frames              cuadros por ventana (40)
    coords              coordenadas por cuadro (168)
    contractVersion     versión del contrato de keypoints (3)

El modelo y el catálogo se versionan juntos: los sha256 permiten a la app
detectar un par que no corresponde.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

LENGUAS = ("lsa", "asl", "lse")

LICENCIA_CC_BY_NC_SA = "CC BY-NC-SA 4.0"
LICENCIA_CC_BY = "CC BY 4.0"

# Datos de cada dataset de origen. El manifiesto copia estos valores.
ORIGEN_LSA64 = {
    "dataset": {"nombre": "LSA64", "version": "2016"},
    "license": LICENCIA_CC_BY_NC_SA,
    "attribution": "LSA64: Ronchetti et al., LIDI, UNLP.",
}

# LSA-94 = LSA64 + grabaciones propias (Hoy, Ayer, Mañana y abecedario).
# Por lectura conservadora de la licencia de LSA64, sigue siendo NC-SA.
ORIGEN_LSA94 = {
    "dataset": {"nombre": "LSA-94 (LSA64 + grabaciones propias)", "version": "catalogo 3.0"},
    "license": LICENCIA_CC_BY_NC_SA,
    "attribution": (
        "LSA64: Ronchetti et al., LIDI, UNLP. Grabaciones propias de Helpi "
        "(Hoy, Ayer, Mañana y abecedario)."
    ),
}

ORIGEN_ISLR_POPSIGN = {
    "dataset": {"nombre": "Google ISLR + PopSign ASL", "version": "GISLR 2023"},
    "license": LICENCIA_CC_BY,
    "attribution": (
        "Google ISLR: Deaf Professional Arts Network y Georgia Tech (PopSign)."
    ),
}

ORIGEN_SWL_LSE = {
    "dataset": {"nombre": "SWL-LSE", "version": "Zenodo 13691887"},
    "license": LICENCIA_CC_BY,
    "attribution": "SWL-LSE: SignaMed. Zenodo, registro 13691887.",
}

_CAMPOS_TEXTO = ("language", "modelVersion", "catalogVersion", "license", "attribution")
_CAMPOS_SHA = ("modelSha256", "catalogSha256")
_CAMPOS_ENTEROS_POSITIVOS = ("numClasses", "frames", "coords", "contractVersion")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUERIDOS = (
    "language", "modelVersion", "catalogVersion", "dataset", "license",
    "attribution", "modelSha256", "catalogSha256", "numClasses",
    "outputsProbabilities", "frames", "coords", "contractVersion",
)


def escribir_manifiesto(destino, *, language, model_version, catalog_version,
                        blob_modelo, catalog_bytes, num_clases, dataset, license,
                        attribution, frames, coords, contract_version):
    """Calcula los sha256, valida el manifiesto y lo escribe en destino."""
    manifiesto = {
        "language": language,
        "modelVersion": model_version,
        "catalogVersion": catalog_version,
        "dataset": dict(dataset),
        "license": license,
        "attribution": attribution,
        "modelSha256": hashlib.sha256(blob_modelo).hexdigest(),
        "catalogSha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "numClasses": num_clases,
        "outputsProbabilities": False,
        "frames": frames,
        "coords": coords,
        "contractVersion": contract_version,
    }
    validar_manifiesto(manifiesto)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    return manifiesto


def validar_manifiesto(manifiesto):
    """Lanza ValueError si el manifiesto no cumple el esquema común."""
    if not isinstance(manifiesto, dict):
        raise ValueError("El manifiesto debe ser un objeto JSON.")

    faltantes = [campo for campo in _REQUERIDOS if campo not in manifiesto]
    if faltantes:
        raise ValueError(f"Faltan campos en el manifiesto: {', '.join(faltantes)}")

    if manifiesto["language"] not in LENGUAS:
        raise ValueError(
            f"language inválido: {manifiesto['language']!r} (esperado: {', '.join(LENGUAS)})")

    for campo in _CAMPOS_TEXTO:
        _exigir_texto(manifiesto, campo)

    for campo in _CAMPOS_SHA:
        valor = manifiesto[campo]
        if not isinstance(valor, str) or not _HEX_SHA256.match(valor):
            raise ValueError(f"{campo} debe ser un SHA-256 hexadecimal de 64 caracteres")

    for campo in _CAMPOS_ENTEROS_POSITIVOS:
        valor = manifiesto[campo]
        if isinstance(valor, bool) or not isinstance(valor, int) or valor <= 0:
            raise ValueError(f"{campo} debe ser un entero positivo")

    if not isinstance(manifiesto["outputsProbabilities"], bool):
        raise ValueError("outputsProbabilities debe ser booleano")

    dataset = manifiesto["dataset"]
    if not isinstance(dataset, dict):
        raise ValueError("dataset debe ser un objeto con nombre y version")
    for campo in ("nombre", "version"):
        if campo not in dataset:
            raise ValueError(f"Falta dataset.{campo}")
        _exigir_texto(dataset, campo, prefijo="dataset.")


def _exigir_texto(objeto, campo, prefijo=""):
    """Verifica que objeto[campo] sea un texto no vacío."""
    valor = objeto[campo]
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"{prefijo}{campo} debe ser un texto no vacío")
