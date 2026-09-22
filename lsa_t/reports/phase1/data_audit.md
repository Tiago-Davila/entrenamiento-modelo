# Auditoría de datos LSA-T — Fase 1

- Fuente: `/home/tiagoashe/entrenamiento-modelo/data`
- Este informe es de solo lectura: no se eliminaron ni modificaron muestras.

## Resumen

- Vocabulario: 5981 tokens.
- Muestras esperadas: 8.459.
- Muestras encontradas: 8450.
- Muestras inválidas: 0.
- Faltantes: 9.

## Por split

| Split | Esperadas | Encontradas | Inválidas | Faltantes | 100% ceros | <10% cuadros válidos | >62 palabras |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5413 | 5407 | 0 | 6 | 148 | 226 | 17 |
| val | 1354 | 1352 | 0 | 2 | 43 | 69 | 4 |
| test | 1692 | 1691 | 0 | 1 | 48 | 66 | 7 |

## Hallazgos que requieren decisión

- Las muestras con baja cobertura no se excluyen automáticamente en esta fase.
- Los IDs faltantes deben explicarse antes de regenerar el dataset.
- La longitud máxima del decoder y la política para etiquetas largas deben definirse antes de reentrenar.
- Los umbrales de calidad deben validarse con ejemplos reales, no elegirse solamente por conveniencia técnica.

## IDs faltantes

- `train`: train_01780, train_02213, train_02965, train_02967, train_02979, train_04606
- `val`: val_00329, val_00815
- `test`: test_00548

## Muestras para revisión manual

El auditor no elimina ninguna muestra. Estas son candidatas para decidir la limpieza:

| Split | 100% ceros | Cobertura <10% | ≥90% ceros | >62 palabras |
|---|---:|---:|---:|---:|
| train | 148 | 226 | 235 | 17 |
| val | 43 | 69 | 71 | 4 |
| test | 48 | 66 | 68 | 7 |

Los archivos completos y sus métricas están en `quality_candidates.json`.

## Próximo paso

Revisar este informe y aprobar los criterios de limpieza antes de ejecutar un nuevo entrenamiento.
