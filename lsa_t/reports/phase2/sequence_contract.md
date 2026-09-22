# Contrato de entrada LSA-T — Fase 2

Este contrato pertenece exclusivamente al modelo secuencial LSA-T. No reemplaza
el contrato de Eva, que continúa usando `40×201`.

## Tensor por cuadro

Cada cuadro contiene 42 landmarks de MediaPipe Holistic:

1. Mano izquierda: landmarks 0..20.
2. Mano derecha: landmarks 0..20.

Cada landmark usa el orden intercalado `(x, y, z)`. Por lo tanto, cada cuadro
tiene `42×3 = 126` valores.

Una mano no detectada se representa con 63 ceros antes de normalizar.

## Normalización

Para cada cuadro:

1. Se calcula el promedio de `x` y `y` de los 42 landmarks, incluyendo los
   landmarks que estén en cero.
2. Se resta ese centro a todos los `x` e `y` del cuadro.
3. `z` no se centra.

Después de procesar todos los cuadros:

4. Se calcula el máximo absoluto global de `x` e `y`.
5. Se dividen todos los `x` e `y` por esa escala si es mayor que `1e-6`.
6. `z` no se escala.

Esta definición conserva el comportamiento del dataset procesado actual,
incluyendo el tratamiento de ceros. Una mejora distinta requiere reentrenar el
modelo y versionar otro contrato.

## Temporal

La salida siempre tiene 75 cuadros:

- Si `T > 75`, se usan los índices:

  ```text
  idx[i] = floor(i × (T - 1) / (75 - 1))
  ```

  La división es entera y se implementa sin `float` ni `linspace`.

- Si `T = 75`, se conserva la secuencia.
- Si `T < 75`, se agregan cuadros completamente cero al final.

El tensor final es `[75, 126]`, aplanado por filas para LiteRT.

## Implementaciones

- Python: [sequence_contract.py](/home/tiagoashe/entrenamiento-modelo/lsa_t/sequence_contract.py)
- Kotlin: [SequenceKeypointContract.kt](/home/tiagoashe/helpi/helpi/app/src/main/java/com/helpi/conversation/keypoints/SequenceKeypointContract.kt)
- Fixture: [sequence_contract_fixture.json](/home/tiagoashe/entrenamiento-modelo/lsa_t/reports/phase2/sequence_contract_fixture.json)

El fixture cubre una secuencia corta con mano ausente, una secuencia exacta de
75 cuadros y una secuencia de 79 cuadros que detecta el error de redondeo de
`linspace`.
