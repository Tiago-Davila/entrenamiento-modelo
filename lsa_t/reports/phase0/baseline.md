# Línea base de implementación — Fase 0

- Versión de artefacto: `lsa-t-seq2seq-v1-experimental`
- Capturada (UTC): `2026-09-17T02:16:30.435493+00:00`
- Repositorio: `/home/tiagoashe/entrenamiento-modelo`
- Rama: `codex/001-lsa-t-phase0`
- Commit base: `fe952d04088db0eb2bfb4a56c5a64ea3b45cbf96`

## Candidato oficial

`/home/tiagoashe/entrenamiento-modelo/checkpoints/phase_b_best.pt`

está junto al dataset raíz, coincide con la configuración vigente y es más reciente que las copias bajo lsa/checkpoints.

Las copias alternativas no se eliminan ni se reemplazan; quedan registradas para comparación.

## Artefactos y hashes

| Archivo | Existe | Bytes | SHA-256 |
|---|---:|---:|---|
| `/home/tiagoashe/entrenamiento-modelo/data/vocab.json` | sí | 69258 | `5c3e8130d1bdb5b3595a36c9a0ba84533c83f62df3cc67da3b34abf12ad85d8b` |
| `/home/tiagoashe/entrenamiento-modelo/checkpoints/phase_b_best.pt` | sí | 47578942 | `6770c4ad43003e8f3a29cf241727b9174544ec0dfca001dba5d05db577bff7c2` |
| `/home/tiagoashe/entrenamiento-modelo/checkpoints/phase_b_last.pt` | sí | 47578942 | `9417a40b401498de9e6e8ba94323d39aea2053a8e622468e1a917ef7322e25d6` |
| `/home/tiagoashe/entrenamiento-modelo/checkpoints/phase_b_history.json` | sí | 6433 | `79b3f97fe168d0c8bb46d05a7e3205f1560e6e6e3d696d9d8cfaa4e2d9bc423e` |
| `/home/tiagoashe/entrenamiento-modelo/lsa/checkpoints/phase_b_best.pt` | sí | 47578942 | `acce8ecd397b8ed34ee691fe32a547024a244e098757c343c88883fdeef8a84c` |
| `/home/tiagoashe/entrenamiento-modelo/lsa/checkpoints/phase_c_best.pt` | sí | 47578942 | `54f2be0136e4c531f722bc8a188db019352dd8649fdbcb0323b196e0382a9d43` |
| `/home/tiagoashe/entrenamiento-modelo/lsa/exports/encoder_int8.tflite` | sí | 13152536 | `1a52f3952ba3f7900930dd2347e48874fcb7fbfcc1d4aa2ba9eff70246b9a41f` |
| `/home/tiagoashe/entrenamiento-modelo/lsa/exports/decoder_int8.tflite` | sí | 22008080 | `d6c171d8e2504172f2ec7c0b7252012faa1c610355fb38d4f542c07902ee187b` |

## Estado del repositorio

El estado no se limpia automáticamente. Los archivos existentes y los cambios del usuario se conservan.

## Entorno capturado

- Python: `Python 3.12.14`
- numpy: `2.5.3`
- torch: `no instalado en el entorno activo`
- tensorflow: `no instalado en el entorno activo`
- onnx: `no instalado en el entorno activo`
- onnxruntime: `no instalado en el entorno activo`
- onnx2tf: `no instalado en el entorno activo`
- huggingface_hub: `1.30.0`
- pandas: `3.0.5`
- pose-format: `0.14.1`
- sacrebleu: `no instalado en el entorno activo`
- jiwer: `no instalado en el entorno activo`
- rouge-score: `no instalado en el entorno activo`
- tqdm: `4.70.0`

## Decisiones de Fase 0

- No se modificaron datos, checkpoints ni exportaciones.
- No se seleccionó todavía ningún modelo para Android.
- La exportación futura debe partir del candidato oficial y generar una nueva versión de artefacto.
- La carpeta `lsa/exports` actual queda fuera de la línea base candidata hasta que pase la validación LiteRT.

## Próximo paso

Continuar con la auditoría de Fase 1 y decidir los criterios de limpieza antes de reentrenar.
