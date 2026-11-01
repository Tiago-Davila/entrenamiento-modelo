# multilengua

Manifiesto compartido por las tres lenguas de Helpi (`lsa`, `asl`, `lse`).
Una sola función lo escribe y un solo esquema lo valida, así que la app
Android lee el mismo formato para cualquier lengua.

## Archivos

- `manifiesto.py`: `escribir_manifiesto` (calcula los sha256 y escribe el
  JSON), `validar_manifiesto` (lanza `ValueError` si no cumple el esquema) y
  las constantes `ORIGEN_*` con dataset, licencia y atribución de cada origen.
- `regenerar_manifiesto_lsa94.py`: reescribe `lsa-94/exports/lsa-manifest.json`
  con el formato nuevo sin tocar el `.tflite` ni el catálogo. Aborta si los
  sha256 no coinciden con el manifiesto anterior.
- `test_manifiesto.py`: pruebas unittest del esquema.

## Formato

```json
{
  "language": "lsa",
  "modelVersion": "eva-lsa94-v4",
  "catalogVersion": "3.0",
  "dataset": { "nombre": "LSA-94 (LSA64 + grabaciones propias)", "version": "catalogo 3.0" },
  "license": "CC BY-NC-SA 4.0",
  "attribution": "LSA64: Ronchetti et al., LIDI, UNLP. Grabaciones propias de Helpi (Hoy, Ayer, Mañana y abecedario).",
  "modelSha256": "<64 hex, SHA-256 del .tflite>",
  "catalogSha256": "<64 hex, SHA-256 de catalogo_senas.json>",
  "numClasses": 94,
  "outputsProbabilities": false,
  "frames": 40,
  "coords": 168,
  "contractVersion": 3
}
```

- `language`: `lsa`, `asl` o `lse`. Cualquier otro valor es inválido.
- `dataset`: nombre y versión del dataset de origen. Ambos son texto no vacío.
- `license` y `attribution`: la app los muestra en la interfaz.
- `contractVersion`: 3 (contrato v3 de keypoints, ver `lsa-94/eva_contract.py`).
- `modelSha256` y `catalogSha256` identifican el par modelo-catálogo. El modelo
  y el catálogo se publican juntos.

Licencias: LSA64 y LSA-94 son `CC BY-NC-SA 4.0` (lectura conservadora: el
`.tflite` es obra derivada de LSA64). ISLR/PopSign y SWL-LSE son `CC BY 4.0`.

## Nombre del archivo

Decisión: **`lsa-manifest.json` se mantiene para `lsa`**. Los archivos de
`asl` y `lse` se llaman **`manifest.json`**, dentro de la carpeta de su lengua.

Motivos:

- La app ya busca `lsa/lsa-manifest.json` (`ModelBundle` en helpi). Renombrarlo
  ahora obliga a cambiar la app y el asset publicado en el mismo corte, sin
  ganancia para el usuario.
- La carpeta ya identifica la lengua, así que el nombre genérico alcanza. Para
  las lenguas nuevas no hay motivo para repetir el prefijo.

Deuda: hay dos nombres. Cuando `ModelBundle` lea `manifest.json` por carpeta,
conviene renombrar también el de `lsa` y dejar un único nombre. La función de
escritura no decide el nombre: recibe el `destino` completo.

## Uso

Desde la raíz del repo (con el venv del proyecto):

```
venv/Scripts/python.exe -m unittest multilengua.test_manifiesto
venv/Scripts/python.exe multilengua/regenerar_manifiesto_lsa94.py
```

`lsa-94/train_keras_tflite.py` llama a `escribir_manifiesto` con
`language="lsa"` y `ORIGEN_LSA94`. Ese script agrega la raíz del repo a
`sys.path` para importar `multilengua`.
