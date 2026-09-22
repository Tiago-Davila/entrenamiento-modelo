# Limpieza de datos LSA-T — Fase 1

Las muestras no se eliminaron: se movieron a una cuarentena recuperable.

## Reglas aplicadas

- `allZero`: `zeroRatio >= 1.0`.
- `longLabel`: `labelWords > 62`.

## Resultado

- Total movido: 266.
- train: 165.
- val: 47.
- test: 54.
- Destino: `/home/tiagoashe/entrenamiento-modelo/data/quarantine/phase1`.

Las muestras con cobertura baja pero no completamente vacías no fueron movidas.

## Restauración

Usar `cleaning_manifest.json` para identificar cada destino y restaurar manualmente una muestra si una revisión posterior lo requiere.
