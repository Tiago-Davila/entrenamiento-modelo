"""Pruebas de las piezas nuevas del entrenamiento LSA-94 (sin entrenar nada).

Importa Keras, así que tarda unos segundos más que test_eva_contract.py.
    ..\\venv\\Scripts\\python.exe test_entrenamiento.py
"""

import unittest

import numpy as np

import train_keras_tflite as tk
from eva_contract import COORDS, FRAMES, POINTS, load_groups


def mano_sintetica(rng):
    """Una secuencia v3 con ambas manos y hombros en ±0,5."""
    p = rng.normal(0.0, 0.3, size=(FRAMES, POINTS, 3)).astype(np.float32)
    p[:, 42, :2] = [0.5, 0.0]
    p[:, 43, :2] = [-0.5, 0.0]
    return p


class BalanceoTest(unittest.TestCase):
    y = np.array([0] * 54 + [1] * 54 + [2] * 4 + [3] * 4)

    def cuotas(self, alpha):
        idx = tk.indices_balanceados(self.y, alpha, np.random.default_rng(0))
        return idx, np.bincount(self.y[idx], minlength=4)

    def test_total_por_epoca_no_cambia(self):
        for alpha in (0.0, 0.5, 1.0):
            idx, _ = self.cuotas(alpha)
            self.assertEqual(len(self.y), len(idx))

    def test_alpha_uno_es_natural(self):
        _, c = self.cuotas(1.0)
        np.testing.assert_array_equal([54, 54, 4, 4], c)

    def test_alpha_cero_balancea(self):
        _, c = self.cuotas(0.0)
        self.assertLessEqual(c.max() - c.min(), 1)

    def test_alpha_medio_reduce_el_desbalance(self):
        _, c = self.cuotas(0.5)
        self.assertLess(c[0] / c[2], 54 / 4)
        self.assertGreater(c[0] / c[2], 1.0)

    def test_toda_muestra_de_clase_chica_aparece(self):
        idx, _ = self.cuotas(0.5)
        for i in np.flatnonzero(self.y >= 2):
            self.assertIn(i, idx)

    def test_perfil_marca_clases_de_un_solo_senante(self):
        y = np.array([0, 0, 1, 1])
        sujetos = np.array([1, 12, 12, 12])
        np.testing.assert_array_equal([0, 0, 1, 1], tk.perfil_por_muestra(y, sujetos))


class FeaturesForma(unittest.TestCase):
    def salida(self, p):
        capa = tk.FeaturesEva94()
        return tk.ops.convert_to_numpy(capa(p.reshape(1, FRAMES, COORDS)))[0]

    def test_forma_de_salida(self):
        rng = np.random.default_rng(0)
        out = self.salida(mano_sintetica(rng))
        self.assertEqual((FRAMES, COORDS * 2 + 21 * 3 * 2 + tk.N_FORMA * 2), out.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_mano_ausente_da_ceros(self):
        p = mano_sintetica(np.random.default_rng(1))
        p[:, 0:21] = 0.0                              # sin mano izquierda
        forma_izq = self.salida(p)[:, -2 * tk.N_FORMA:-tk.N_FORMA]
        self.assertTrue(np.all(forma_izq == 0.0))

    def test_forma_invariante_a_traslacion_y_tamano(self):
        p = mano_sintetica(np.random.default_rng(2))
        q = p.copy()
        mano = q[:, 21:42]
        q[:, 21:42] = (mano - mano[:, :1]) * 1.3 + mano[:, :1] + [0.2, -0.1, 0.0]
        a = self.salida(p)[:, -tk.N_FORMA:]
        b = self.salida(q)[:, -tk.N_FORMA:]
        # rtol: el +1e-3 de la palma (igual que FeaturesEva) deja ~0,3% de diferencia
        np.testing.assert_allclose(a, b, rtol=5e-3, atol=1e-4)

    def test_cosenos_en_rango(self):
        out = self.salida(mano_sintetica(np.random.default_rng(3)))
        cos = out[:, -15:]
        self.assertTrue(np.all(cos >= -1.0001) and np.all(cos <= 1.0001))


class FixtureTest(unittest.TestCase):
    def test_fixture_cubre_los_tres_grupos(self):
        y = np.concatenate([np.repeat(np.arange(64), 3), np.repeat(np.arange(64, 94), 1)])
        idx = tk.indices_fixture(y, load_groups(), 12)
        self.assertEqual(12, len(idx))
        etiquetas = y[idx] + 1
        self.assertTrue(any(e <= 64 for e in etiquetas))
        self.assertTrue(any(65 <= e <= 67 for e in etiquetas))
        self.assertTrue(any(e >= 68 for e in etiquetas))


class SplitTest(unittest.TestCase):
    def test_reps_y_sujetos_unicos(self):
        self.assertEqual([(12, 3), (12, 4)], tk.parse_reps(["12:3", "12:4"]))
        y = np.array([0, 0, 1])
        sujetos = np.array([1, 12, 12])
        self.assertEqual({12}, tk.sujetos_unicos(y, sujetos))
        m = tk.mascara_reps(np.array([12, 12, 9]), np.array([3, 4, 3]), [(12, 3)])
        np.testing.assert_array_equal([True, False, False], m)


if __name__ == "__main__":
    unittest.main()
