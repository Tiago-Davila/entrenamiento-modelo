"""Carga entrenamiento-modelo/.env ANTES de importar Keras.

Keras elige el backend (tensorflow, torch) al importarse, leyendo
KERAS_BACKEND. Este módulo tiene que importarse primero en cualquier script
que entrene. Las variables ya definidas en el entorno tienen prioridad sobre
el archivo, así se puede forzar un backend puntual:

    $env:KERAS_BACKEND = "tensorflow"; python train_keras_tflite.py ...

Formato del .env: líneas CLAVE=valor; '#' inicia un comentario.
Sin dependencias (no usa python-dotenv).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def cargar_env(path: Path = ENV_PATH) -> dict[str, str]:
    leidas: dict[str, str] = {}
    if not path.exists():
        return leidas
    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip()
        if not linea or "=" not in linea:
            continue
        clave, valor = (p.strip() for p in linea.split("=", 1))
        valor = valor.strip("\"'")
        leidas[clave] = valor
        os.environ.setdefault(clave, valor)
    return leidas


cargar_env()
