# Auditoría de datos LSA-T — Fase 1

- Fuente: `/home/tiagoashe/entrenamiento-modelo/data`
- Este informe es de solo lectura: no se eliminaron ni modificaron muestras.

## Resumen

- Vocabulario: 5981 tokens.
- Muestras esperadas: 8.459.
- Muestras encontradas: 8184.
- Muestras inválidas: 0.
- Faltantes: 275.

## Por split

| Split | Esperadas | Encontradas | Inválidas | Faltantes | 100% ceros | <10% cuadros válidos | >62 palabras |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5413 | 5242 | 0 | 171 | 0 | 77 | 0 |
| val | 1354 | 1305 | 0 | 49 | 0 | 26 | 0 |
| test | 1692 | 1637 | 0 | 55 | 0 | 17 | 0 |

## Hallazgos que requieren decisión

- Las muestras con baja cobertura no se excluyen automáticamente en esta fase.
- Los IDs faltantes deben explicarse antes de regenerar el dataset.
- La longitud máxima del decoder y la política para etiquetas largas deben definirse antes de reentrenar.
- Los umbrales de calidad deben validarse con ejemplos reales, no elegirse solamente por conveniencia técnica.

## IDs faltantes

- `train`: train_00241, train_00257, train_00259, train_00366, train_00413, train_00482, train_00599, train_00607, train_00608, train_00684, train_00685, train_00686, train_00687, train_00688, train_00701, train_00703, train_00706, train_00707, train_00715, train_00716, train_00718, train_00719, train_00728, train_00729, train_00730, train_00732, train_00733, train_00739, train_00746, train_00763, train_00764, train_00765, train_00766, train_00768, train_00769, train_00770, train_00780, train_00781, train_00782, train_00783, train_00784, train_00785, train_00787, train_00788, train_00789, train_00790, train_00791, train_00813, train_00814, train_00815, train_00869, train_00875, train_00889, train_00980, train_00982, train_00985, train_01024, train_01033, train_01035, train_01039, train_01040, train_01041, train_01046, train_01049, train_01050, train_01052, train_01075, train_01122, train_01123, train_01129, train_01131, train_01154, train_01229, train_01232, train_01242, train_01397, train_01399, train_01400, train_01402, train_01403, train_01405, train_01406, train_01408, train_01418, train_01423, train_01425, train_01426, train_01428, train_01429, train_01430, train_01432, train_01433, train_01655, train_01661, train_01665, train_01737, train_01739, train_01740, train_01763, train_01775, train_01780, train_01790, train_01791, train_01793, train_01934, train_02028, train_02029, train_02032, train_02033, train_02040, train_02043, train_02045, train_02053, train_02211, train_02213, train_02307, train_02310, train_02311, train_02312, train_02335, train_02336, train_02371, train_02430, train_02432, train_02965, train_02966, train_02967, train_02969, train_02979, train_03068, train_03233, train_03350, train_03351, train_03352, train_03353, train_03357, train_03440, train_03453, train_03458, train_03460, train_03461, train_03463, train_03464, train_03465, train_03469, train_03471, train_03472, train_03473, train_03474, train_03475, train_03476, train_03522, train_03523, train_03524, train_03542, train_03671, train_04479, train_04480, train_04496, train_04497, train_04498, train_04499, train_04501, train_04503, train_04506, train_04511, train_04512, train_04606, train_04905, train_04920, train_05106
- `val`: val_00059, val_00060, val_00072, val_00162, val_00177, val_00178, val_00180, val_00183, val_00187, val_00205, val_00206, val_00208, val_00228, val_00247, val_00268, val_00275, val_00329, val_00371, val_00373, val_00376, val_00378, val_00380, val_00381, val_00382, val_00383, val_00446, val_00447, val_00470, val_00480, val_00515, val_00739, val_00751, val_00815, val_00849, val_00850, val_00851, val_00854, val_00855, val_00880, val_00881, val_00888, val_01115, val_01121, val_01122, val_01123, val_01125, val_01126, val_01127, val_01230
- `test`: test_00074, test_00081, test_00088, test_00176, test_00234, test_00235, test_00240, test_00244, test_00253, test_00254, test_00255, test_00259, test_00260, test_00261, test_00281, test_00315, test_00316, test_00324, test_00332, test_00336, test_00340, test_00359, test_00386, test_00413, test_00428, test_00432, test_00435, test_00436, test_00437, test_00537, test_00548, test_00556, test_00610, test_00727, test_00728, test_00750, test_00837, test_00881, test_01020, test_01043, test_01044, test_01045, test_01046, test_01047, test_01067, test_01068, test_01087, test_01375, test_01380, test_01382, test_01383, test_01384, test_01388, test_01390, test_01544

## Muestras para revisión manual

El auditor no elimina ninguna muestra. Estas son candidatas para decidir la limpieza:

| Split | 100% ceros | Cobertura <10% | ≥90% ceros | >62 palabras |
|---|---:|---:|---:|---:|
| train | 0 | 77 | 86 | 0 |
| val | 0 | 26 | 28 | 0 |
| test | 0 | 17 | 19 | 0 |

Los archivos completos y sus métricas están en `quality_candidates.json`.

## Próximo paso

Revisar este informe y aprobar los criterios de limpieza antes de ejecutar un nuevo entrenamiento.
