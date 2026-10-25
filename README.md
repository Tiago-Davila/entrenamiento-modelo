# Entrenamiento de Helpi

Este repositorio contiene tres proyectos que no comparten modelo ni
artefactos de exportación.

| Carpeta | Proyecto | Entrada | Resultado |
|---|---|---|---|
| `lsa64/` | Eva LSA64 v3 (congelado, tag `eva-lsa64-v3`): 64 señas | 40 × 168 keypoints | una glosa de catálogo cerrado |
| `lsa-94/` | Eva LSA-94 (`eva-lsa94-v4`): 64 señas + Hoy/Ayer/Mañana + abecedario | 40 × 168 keypoints (mismo contrato v3) | una glosa de 94 |
| `lsa_t/` | LSA-T: traducción de frases | secuencias de 126 keypoints de manos | texto en español |

Cada carpeta contiene su código, datos, checkpoints, exportaciones y
documentación. No mezclar sus catálogos, fixtures ni modelos Android.

## Punto de partida

- Eva LSA-94: [`lsa-94/EVA_LSA94_ENTRENAMIENTO.md`](lsa-94/EVA_LSA94_ENTRENAMIENTO.md)
- Eva LSA64 v3: [`lsa64/EVA_V3_ENTRENAMIENTO.md`](lsa64/EVA_V3_ENTRENAMIENTO.md)
- LSA-T: [`lsa_t/ENTRENAMIENTO_WINDOWS.md`](lsa_t/ENTRENAMIENTO_WINDOWS.md)

Los directorios `*_legacy` conservan artefactos anteriores para auditoría. No
son candidatos de despliegue.
