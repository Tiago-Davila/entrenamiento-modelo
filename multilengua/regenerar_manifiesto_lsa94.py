"""Regenera lsa-94/exports/lsa-manifest.json con el formato multilengua.

No toca modelo_lsa.tflite ni catalogo_senas.json: los lee solo para calcular
sus sha256 y comprobar que coinciden con los del manifiesto anterior. Si algo
no coincide, aborta sin reemplazar el manifiesto.

Uso (desde la raiz del repo):
    venv/Scripts/python.exe multilengua/regenerar_manifiesto_lsa94.py
    venv/Scripts/python.exe multilengua/regenerar_manifiesto_lsa94.py --exports otra/carpeta
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from multilengua.manifiesto import ORIGEN_LSA94, escribir_manifiesto, validar_manifiesto  # noqa: E402

EXPORTS_LSA94 = RAIZ / "lsa-94" / "exports"
CONTRACT_VERSION_ESPERADA = 3


def regenerar(exports: Path) -> dict:
    """Reescribe lsa-manifest.json conservando los sha256 y los campos del anterior."""
    ruta_manifiesto = exports / "lsa-manifest.json"
    viejo = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))

    if viejo.get("contractVersion") != CONTRACT_VERSION_ESPERADA:
        raise SystemExit(f"contractVersion del manifiesto anterior es "
                         f"{viejo.get('contractVersion')!r}, se esperaba {CONTRACT_VERSION_ESPERADA}.")

    blob = (exports / "modelo_lsa.tflite").read_bytes()
    catalogo = (exports / "catalogo_senas.json").read_bytes()

    if hashlib.sha256(blob).hexdigest() != viejo["modelSha256"]:
        raise SystemExit("modelSha256 de modelo_lsa.tflite no coincide con el manifiesto anterior. Abortado.")
    if hashlib.sha256(catalogo).hexdigest() != viejo["catalogSha256"]:
        raise SystemExit("catalogSha256 de catalogo_senas.json no coincide con el manifiesto anterior. Abortado.")

    temporal = exports / "lsa-manifest.nuevo.json"
    try:
        nuevo = escribir_manifiesto(
            temporal,
            language="lsa",
            model_version=viejo["modelVersion"],
            catalog_version=viejo["catalogVersion"],
            blob_modelo=blob,
            catalog_bytes=catalogo,
            num_clases=viejo["numClasses"],
            dataset=ORIGEN_LSA94["dataset"],
            license=ORIGEN_LSA94["license"],
            attribution=ORIGEN_LSA94["attribution"],
            frames=viejo["frames"],
            coords=viejo["coords"],
            contract_version=viejo["contractVersion"],
        )
        releido = json.loads(temporal.read_text(encoding="utf-8"))
        validar_manifiesto(releido)
        for campo in ("modelSha256", "catalogSha256", "numClasses", "frames",
                      "coords", "contractVersion", "modelVersion", "catalogVersion"):
            if releido[campo] != viejo[campo]:
                raise SystemExit(f"{campo} cambio al regenerar ({viejo[campo]!r} -> {releido[campo]!r}). Abortado.")
        os.replace(temporal, ruta_manifiesto)
    finally:
        if temporal.exists():
            temporal.unlink()
    return nuevo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exports", type=Path, default=EXPORTS_LSA94,
                        help="carpeta con modelo_lsa.tflite, catalogo_senas.json y lsa-manifest.json")
    args = parser.parse_args(argv)
    nuevo = regenerar(args.exports)
    print(f"lsa-manifest.json regenerado en {args.exports.resolve()}")
    print(json.dumps(nuevo, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
