"""Pruebas sintéticas de train_lse.py (sin datos reales ni entrenamiento).

    ..\\venv\\Scripts\\python.exe test_train_lse.py
"""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

import train_lse as tl
import train_keras_tflite as tk


class CatalogoTest(unittest.TestCase):
    def test_excluye_na_y_renumera_en_orden(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "cat.json"
            ruta.write_text(json.dumps({
                "version": "9",
                "glosas": ["A", "No identificada", "C", "D"],
                "etiquetasOriginales": ["A", "#N/A", "C", "D"],
                "relevantesHelpiPorCategoria": {"dolor": [1, 4]},
            }), encoding="utf-8")
            cat = tl.cargar_catalogo(ruta)
        self.assertEqual(["A", "C", "D"], cat["glosas"])
        self.assertEqual([0, 2, 3], cat["claseIdSwl"])
        self.assertEqual(1, cat["excluida"]["claseIdSwl"])
        # relevantes 1-based del original -> etiquetas nuevas 1..3
        self.assertEqual({"dolor": [1, 3]}, cat["relevantesPorCategoria"])

    def test_necesita_exactamente_una_na(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "cat.json"
            ruta.write_text(json.dumps({
                "version": "9", "glosas": ["A", "B"],
                "etiquetasOriginales": ["A", "B"], "relevantesHelpiPorCategoria": {},
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                tl.cargar_catalogo(ruta)


class CargaTest(unittest.TestCase):
    def test_remapeo_y_exclusion_de_muestras(self):
        cat = {"claseIdSwl": [0, 2, 3]}                    # sin la clase 1 (etiqueta 2)
        n = 6
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            np.save(dd / "X.npy", np.zeros((n, tk.FRAMES, tk.COORDS), dtype=np.float32))
            np.save(dd / "y.npy", np.array([1, 2, 3, 4, 2, 4]))
            np.save(dd / "subjects.npy", np.array([0, 0, 1, 1, 2, -1]))
            np.save(dd / "split.npy", np.array([0, 0, 1, 2, 0, 3]))
            np.save(dd / "nombres.npy", np.array([f"s{i}" for i in range(n)]))
            X, y, sesion, split, nombres, excluidas = tl.cargar_datos(dd, cat)
        self.assertEqual(2, excluidas)
        np.testing.assert_array_equal([1, 2, 3, 3], y)     # etiquetas 1,3,4 -> 1,2,3
        self.assertEqual(4, len(X))
        np.testing.assert_array_equal([0, 1, 2, 3], split)   # split de las muestras que quedan


class ParticionSesionTest(unittest.TestCase):
    def datos(self, seed=0):
        rng = np.random.default_rng(seed)
        clases, sesiones = 12, 60
        filas_ses, filas_y, filas_split = [], [], []
        for s in range(sesiones):
            for _ in range(rng.integers(6, 12)):
                filas_ses.append(s)
                filas_y.append(int(rng.integers(1, clases + 1)))
                filas_split.append(0)
        for c in range(1, clases + 1):                       # cada clase al menos en 3 sesiones
            for s in rng.choice(sesiones, 3, replace=False):
                filas_ses.append(int(s))
                filas_y.append(c)
                filas_split.append(0)
        for c in range(1, clases + 1):                       # referencias
            filas_ses.append(-1)
            filas_y.append(c)
            filas_split.append(3)
        return (np.array(filas_ses), np.array(filas_y), np.array(filas_split))

    def test_sesion_no_cruza_particiones(self):
        ses, y, split = self.datos()
        asign = tl.particion_por_sesion(ses, y, split, seed=3, iteraciones=4000)
        self.assertEqual(0, tl.sesiones_en_varias_particiones(ses, asign))

    def test_referencias_van_a_train(self):
        ses, y, split = self.datos()
        asign = tl.particion_por_sesion(ses, y, split, seed=3, iteraciones=4000)
        np.testing.assert_array_equal(asign[split == 3], tl.ASIGN_REF)

    def test_tamanos_cerca_de_70_15_15(self):
        ses, y, split = self.datos()
        asign = tl.particion_por_sesion(ses, y, split, seed=3, iteraciones=20000)
        no_ref = split != 3
        n = no_ref.sum()
        val = (asign == tl.ASIGN_VAL).sum() / n
        test = (asign == tl.ASIGN_TEST).sum() / n
        self.assertLess(abs(val - 0.15), 0.08)
        self.assertLess(abs(test - 0.15), 0.08)

    def test_misma_semilla_mismo_resultado(self):
        ses, y, split = self.datos()
        a = tl.particion_por_sesion(ses, y, split, seed=7, iteraciones=3000)
        b = tl.particion_por_sesion(ses, y, split, seed=7, iteraciones=3000)
        self.assertEqual(tl.hash_particion(a), tl.hash_particion(b))

    def test_oficial_usa_split_tal_cual(self):
        split = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(split, tl.particion_oficial(split))


class MetricasTest(unittest.TestCase):
    def test_top1_top3_y_relevantes(self):
        glosas = ["a", "b", "c"]
        y0 = np.array([0, 1, 2, 2])
        logits = np.array([
            [5.0, 1.0, 0.0],     # acierta
            [0.0, 2.0, 1.5],     # acierta
            [3.0, 2.0, 1.0],     # top-3 pero no top-1 (la clase 2 es 3er puesto)
            [1.0, 0.0, 0.5],     # 2 es la 3ra? top-3 incluye todas: acierta top-3
        ])
        r, cm = tl.metricas_prediccion(logits, y0, glosas, {"dolor": [1, 3]})
        self.assertAlmostEqual(0.5, r["top1"])
        self.assertAlmostEqual(1.0, r["top3"])
        self.assertEqual(4, int(cm.sum()))
        self.assertEqual(3, r["clasesEvaluadas"])
        self.assertEqual(3, r["relevantesHelpi"]["dolor"]["n"])   # etiquetas 1 y 3 -> 3 muestras
        self.assertIn("exactitud", r["peoresClases"][0])

    def test_matriz_suma_muestras(self):
        glosas = ["a", "b"]
        r, cm = tl.metricas_prediccion(np.eye(2), np.array([0, 1]), glosas, {})
        self.assertEqual(2, int(cm.sum()))
        self.assertEqual(1.0, r["top1"])


class UmbralTest(unittest.TestCase):
    def test_elige_el_menor_umbral_que_llega_al_objetivo(self):
        probs = np.array([[0.95, 0.05], [0.6, 0.4], [0.55, 0.45], [0.5, 0.5]] * 10)
        y0 = np.array([0, 0, 1, 1] * 10)     # aciertos: 0.95 y 0.6 -> la de 0.55 falla
        curva = tl.curva_cobertura(probs, y0)
        fila = tl.elegir_umbral(curva, exactitud_objetivo=0.9, aceptadas_min=5)
        self.assertIsNotNone(fila)
        self.assertGreaterEqual(fila["exactitudAceptadas"], 0.9)
        self.assertLessEqual(fila["umbral"], 0.6)

    def test_sin_umbral_si_nunca_llega(self):
        probs = np.array([[0.5, 0.5]] * 40)
        y0 = np.array([0, 1] * 20)
        self.assertIsNone(tl.elegir_umbral(tl.curva_cobertura(probs, y0), aceptadas_min=5))

    def test_softmax_suma_uno(self):
        p = tl.softmax(np.array([[1.0, 2.0, 3.0]]))
        self.assertAlmostEqual(1.0, p.sum())


class EpocaTest(unittest.TestCase):
    def test_media_movil_ultima_epoca_de_la_ventana(self):
        curva = [0.1, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.1, 0.1]
        self.assertEqual(7, tl.epoca_sugerida(curva, ventana=5))


class TransferenciaTest(unittest.TestCase):
    def test_copia_todo_menos_la_salida(self):
        args = Namespace(arch="conv_tx", dim=16, dropout=0.1, features="forma", hidden=8)
        origen = tk.construir(args, 5)
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "pesos.weights.h5"
            origen.save_weights(ruta)
            destino = tk.construir(args, 3)
            copiadas = tk.transferir_pesos(destino, args, ruta, 5)
            self.assertGreater(copiadas, 0)
            for capa_d, capa_o in zip(destino.layers, origen.layers):
                if capa_d.name == "logits":
                    self.assertEqual(3, capa_d.get_weights()[0].shape[1])
                    continue
                for wd, wo in zip(capa_d.get_weights(), capa_o.get_weights()):
                    np.testing.assert_allclose(wd, wo)


if __name__ == "__main__":
    unittest.main()
