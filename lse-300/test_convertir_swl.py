"""Pruebas de convertir_swl.py con un fixture escrito a mano (no necesita datos descargados)."""

import io
import json
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import convertir_swl as c  # noqa: E402
import eva_contract as ec  # noqa: E402
import verificar_datasets as v  # noqa: E402


def _fixture_pose(T: int) -> np.ndarray:
    """Pose de 33 puntos: el punto k (11..24) vale (k/100, 0.5, k/100), el resto NaN."""
    pose = np.full((T, 33, 3), np.nan, dtype=np.float32)
    for t in range(T):
        for k in range(11, 25):
            pose[t, k] = (k / 100.0, 0.5, k / 100.0)
    return pose


def _fixture_manos(T: int) -> tuple[np.ndarray, np.ndarray]:
    """Mano izquierda (x=0.2+0.01j, y=0.3) y derecha (x=0.6, y=0.4+0.01j), 21 puntos cada una."""
    izq = np.full((T, 21, 3), np.nan, dtype=np.float32)
    der = np.full((T, 21, 3), np.nan, dtype=np.float32)
    for t in range(T):
        for j in range(21):
            izq[t, j] = (0.2 + 0.01 * j, 0.3, 0.0)
            der[t, j] = (0.6, 0.4 + 0.01 * j, 0.02)
    return izq, der


class FixtureDeTresCuadros(unittest.TestCase):
    """Tres cuadros: el 1 y el 3 completos; el 2 sin mano derecha y sin hombros."""

    def setUp(self):
        self.pose = _fixture_pose(3)
        self.izq, self.der = _fixture_manos(3)
        self.der[1] = np.nan
        self.pose[1, 11:13] = np.nan

    def test_mapeo_crudo_sin_correccion(self):
        crudo = c.a_vector_168(self.pose, self.izq, self.der, corregir=False)
        self.assertEqual(crudo.shape, (3, 168))
        # mano izquierda: landmark 0 y 20 en [0, 63)
        np.testing.assert_allclose(crudo[0, 0:3], [0.2, 0.3, 0.0], atol=1e-6)
        np.testing.assert_allclose(crudo[0, 60:63], [0.4, 0.3, 0.0], atol=1e-6)
        # mano derecha: landmark 0 en [63, 66) y landmark 20 en [123, 126)
        np.testing.assert_allclose(crudo[0, 63:66], [0.6, 0.4, 0.02], atol=1e-6)
        np.testing.assert_allclose(crudo[0, 123:126], [0.6, 0.6, 0.02], atol=1e-6)
        # pose local 0 = pose 11 (x=0.11), local 13 = pose 24 (x=0.24)
        np.testing.assert_allclose(crudo[0, 126:129], [0.11, 0.5, 0.11], atol=1e-6)
        np.testing.assert_allclose(crudo[0, 165:168], [0.24, 0.5, 0.24], atol=1e-6)

    def test_ausente_es_cero(self):
        crudo = c.a_vector_168(self.pose, self.izq, self.der, corregir=False)
        self.assertTrue(np.all(crudo[1, 63:126] == 0.0))
        self.assertTrue(np.all(crudo[1, 126:132] == 0.0))  # hombros 11 y 12 ausentes
        self.assertFalse(np.any(np.isnan(crudo)))

    def test_correccion_espejado_y_intercambios(self):
        crudo = c.a_vector_168(self.pose, self.izq, self.der, corregir=True)
        # bloque izquierdo nuevo = mano derecha original con x -> 1-x
        np.testing.assert_allclose(crudo[0, 0:3], [0.4, 0.4, 0.02], atol=1e-6)
        np.testing.assert_allclose(crudo[0, 60:63], [0.4, 0.6, 0.02], atol=1e-6)
        # bloque derecho nuevo = mano izquierda original con x -> 1-x
        np.testing.assert_allclose(crudo[0, 63:66], [0.8, 0.3, 0.0], atol=1e-6)
        # pose: pares intercambiados y x espejado (11 nuevo = 12 original con 1-x)
        np.testing.assert_allclose(crudo[0, 126:129], [0.88, 0.5, 0.12], atol=1e-6)
        np.testing.assert_allclose(crudo[0, 129:132], [0.89, 0.5, 0.11], atol=1e-6)  # 12 <- 11
        # 23 nuevo = 24 original espejado; 24 nuevo = 23 original espejado
        np.testing.assert_allclose(crudo[0, 162:165], [0.76, 0.5, 0.24], atol=1e-6)
        np.testing.assert_allclose(crudo[0, 165:168], [0.77, 0.5, 0.23], atol=1e-6)

    def test_entrada_v3_coincide_con_build_input_v3(self):
        crudo = c.a_vector_168(self.pose, self.izq, self.der, corregir=True)
        entrada = c.entrada_v3(crudo)
        esperada = ec.build_input_v3(crudo, 480, 480)
        self.assertEqual(entrada.shape, (ec.FRAMES, ec.COORDS))
        np.testing.assert_array_equal(entrada, esperada)

    def test_muestreo_T_menor_que_40_repite_cuadros_en_orden(self):
        crudo = c.a_vector_168(self.pose, self.izq, self.der, corregir=True)
        idx = ec.sample_indices(3)
        self.assertEqual(idx[:4], [0, 0, 0, 0])
        self.assertEqual(idx[-1], 2)
        entrada = c.entrada_v3(crudo)
        # los 40 cuadros salen de los tres originales; el cuadro 2 no tiene mano derecha original
        self.assertTrue(np.all(entrada[idx.index(1)][0:63] == 0.0))  # mano derecha original ausente -> izquierda nueva

    def test_escala_del_contrato_con_hombros_en_pixeles(self):
        """Centro y escala del contrato: cuadro 0 con hombros a 0.11 y 0.12 (x) en 480 px."""
        crudo = c.a_vector_168(self.pose, self.izq, self.der, corregir=True)
        entrada = c.entrada_v3(crudo)
        px = crudo.reshape(3, 56, 3).copy()
        px[..., 0] *= 480
        px[..., 1] *= 480
        px[..., 2] *= 480
        # hombros nuevos: 11 = (0.88, 0.5) y 12 = (0.89, 0.5) en normalizado -> distancia 0.01*480 px
        s = np.median([np.hypot(px[0, 42, 0] - px[0, 43, 0], px[0, 42, 1] - px[0, 43, 1])])
        cx = (px[0, 42, 0] + px[0, 43, 0]) / 2
        cy = (px[0, 42, 1] + px[0, 43, 1]) / 2
        # mano izquierda nueva, landmark 0: (0.4, 0.4, 0.02) -> ((0.4*480 - cx)/s, (0.4*480 - cy)/s)
        # cuadro 0 tiene hombros, así que su centro es el suyo
        self.assertAlmostEqual(float(entrada[0, 0]), (0.4 * 480 - cx) / s, places=3)
        self.assertAlmostEqual(float(entrada[0, 1]), (0.4 * 480 - cy) / s, places=3)
        # z se divide por s pero no se centra
        self.assertAlmostEqual(float(entrada[0, 2]), (0.02 * 480) / s, places=3)

    def test_sin_hombros_en_ningun_cuadro_es_invalida(self):
        pose = _fixture_pose(3)
        pose[:, 11:13] = np.nan
        izq, der = _fixture_manos(3)
        crudo = c.a_vector_168(pose, izq, der, corregir=True)
        with self.assertRaises(ValueError):
            c.entrada_v3(crudo)

    def test_sin_cuadros_es_invalida(self):
        vacio = np.zeros((0, 168), dtype=np.float32)
        with self.assertRaises(ValueError):
            c.entrada_v3(vacio)


class ValidacionesTest(unittest.TestCase):
    def test_proporcion_hombro_torso_en_fixture_conocido(self):
        pose = np.full((2, 33, 3), np.nan, dtype=np.float32)
        for t in range(2):
            pose[t, 11] = (0.4, 0.2, 0.0)
            pose[t, 12] = (0.6, 0.2, 0.0)   # hombros a 0.2 * 480 px
            pose[t, 23] = (0.45, 0.6, 0.0)
            pose[t, 24] = (0.55, 0.6, 0.0)  # caderas a 0.1 * 480 px de ancho
        izq = np.full((2, 21, 3), np.nan, dtype=np.float32)
        der = np.full((2, 21, 3), np.nan, dtype=np.float32)
        crudo = c.a_vector_168(pose, izq, der, corregir=False)
        # hombro medio (0.5, 0.2) y cadera media (0.5, 0.6): torso 0.4 * 480 px
        self.assertAlmostEqual(c.proporcion_hombro_torso(crudo), 0.2 / 0.4, places=5)

    def test_violaciones_escala(self):
        self.assertEqual(c.violaciones_escala({"train": 0.67, "val": 0.66, "test": 0.68}, 0.666), [])
        fallas = c.violaciones_escala({"train": 0.67, "test": 0.80}, 0.666)
        self.assertEqual(len(fallas), 1)
        self.assertTrue(fallas[0].startswith("test"))
        self.assertTrue(c.violaciones_escala({"train": 0.5}, None))

    def test_sesiones_cortan_por_salto_mayor_al_umbral(self):
        # microsegundos: 0 s, 10 s, 20 s | salto de 2 h | 2 h + 5 s
        base = 1_700_000_000_000_000
        ids = [base, base + 10_000_000, base + 20_000_000,
               base + 20_000_000 + 7_200_000_000, base + 20_000_000 + 7_205_000_000]
        ses = c.asignar_sesiones(ids, umbral_s=3600)
        self.assertEqual([ses[i] for i in ids], [0, 0, 0, 1, 1])
        ses_grande = c.asignar_sesiones(ids, umbral_s=10_000)
        self.assertEqual(set(ses_grande.values()), {0})

    def test_sesiones_cruzadas(self):
        ses = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
        split = {1: 0, 2: 2, 3: 0, 4: 0, 5: 3}  # sesión 0 cruza train-test; 5 es referencia
        r = c.sesiones_cruzadas(ses, split)
        self.assertEqual(r["sesiones"], 2)
        self.assertEqual(r["cruzadas"], 1)
        self.assertEqual(r["trainTest"], 1)
        self.assertEqual(r["secuenciasEnSesionesCruzadas"], 2)

    def test_catalogo_de_300_glosas_unicas(self):
        version, glosas = ec.load_catalog(AQUI / "catalogo_lse300.json", expected=300)
        self.assertEqual(version, "1.0")
        self.assertEqual(len(set(glosas)), 300)
        grupos = ec.load_groups(AQUI / "catalogo_lse300.json")
        self.assertEqual(grupos, {"swl_lse": (1, 300)})
        cat = json.loads((AQUI / "catalogo_lse300.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cat["etiquetasOriginales"]), 300)
        self.assertTrue(all(1 <= i <= 300 for i in cat["relevantesHelpi"]))
        self.assertEqual(cat["relevantesHelpi"], sorted(set(cat["relevantesHelpi"])))


class CargaSeguraDeArchivoAbiertoTest(unittest.TestCase):
    def test_cargar_pkl_acepta_archivo_abierto(self):
        datos = pickle.dumps({"holistic_legacy": {"k": 1}, "n": [1, 2]})
        self.assertEqual(v.cargar_pkl(io.BytesIO(datos)), {"holistic_legacy": {"k": 1}, "n": [1, 2]})

    def test_cargar_pkl_rechaza_clases_fuera_de_lista(self):
        import os
        datos = pickle.dumps(os.getcwd)  # referencia a una función de os
        with self.assertRaises(pickle.UnpicklingError):
            v.cargar_pkl(io.BytesIO(datos))


if __name__ == "__main__":
    unittest.main()
