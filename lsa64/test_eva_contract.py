"""Pruebas rápidas del contrato que comparte el reentrenamiento Eva v2."""

import unittest

import numpy as np

from eva_contract import COORDS, FRAMES, center_frame, load_catalog, sample_indices


class EvaContractTest(unittest.TestCase):
    def test_catalogo_verificado_tiene_64_clases_en_orden(self):
        version, glosas = load_catalog()
        self.assertEqual("2.0", version)
        self.assertEqual(64, len(glosas))
        self.assertEqual("Opaco", glosas[0])
        self.assertEqual("Amargo", glosas[18])
        self.assertEqual("Encontrar", glosas[63])

    def test_contrato_descarta_cara_y_conserva_ceros_ausentes(self):
        frame = np.zeros(COORDS, dtype=np.float32)
        # hombros fuente 11/12, locales 0/1 en el bloque de pose reducido.
        frame[126:129] = [0.4, 0.5, 0.1]
        frame[129:132] = [0.6, 0.5, 0.1]
        frame[0:3] = [0.2, 0.3, 0.1]

        centered = center_frame(frame)
        self.assertEqual(168, COORDS)
        self.assertTrue(np.allclose([-0.3, -0.2, 0.1], centered[:3]))
        self.assertTrue(np.array_equal(np.zeros(3, dtype=np.float32), centered[3:6]))

    def test_muestreo_es_entero_y_fijo(self):
        self.assertEqual(40, FRAMES)
        self.assertEqual(15, sample_indices(46)[13])
        self.assertEqual(40, len(sample_indices(12)))


if __name__ == "__main__":
    unittest.main()
