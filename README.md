# Entrenamiento de Helpi

Este repositorio contiene dos proyectos que no comparten modelo, entrada ni
artefactos de exportación.

| Carpeta | Proyecto | Entrada | Resultado |
|---|---|---|---|
| `lsa64/` | Eva: 64 señas individuales | 40 × 168 keypoints | una glosa de catálogo cerrado |
| `lsa_t/` | LSA-T: traducción de frases | secuencias de 126 keypoints de manos | texto en español |

Cada carpeta contiene su código, datos, checkpoints, exportaciones y
documentación. No mezclar sus catálogos, fixtures ni modelos Android.

## Punto de partida

- Eva LSA64: [`lsa64/EVA_V2_ENTRENAMIENTO.md`](lsa64/EVA_V2_ENTRENAMIENTO.md)
- LSA-T: [`lsa_t/ENTRENAMIENTO_WINDOWS.md`](lsa_t/ENTRENAMIENTO_WINDOWS.md)

Los directorios `*_legacy` conservan artefactos anteriores para auditoría. No
son candidatos de despliegue.
