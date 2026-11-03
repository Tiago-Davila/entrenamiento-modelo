"""Pruebas sintéticas de verificar_datasets.py (no necesitan datos descargados)."""

import io
import pickle
import struct
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verificar_datasets as v  # noqa: E402

MEDIAPIPE_LANDMARK = "mediapipe.tasks.python.components.containers.landmark"


def _mano_rigida(T: int, r: float, amplitud: float, seed: int = 3) -> np.ndarray:
    """Mano rígida que rota, proyectada con H/W = r (sin ruido), (T, 21, 2)."""
    rng = np.random.default_rng(seed)
    molde = rng.normal(0.0, 0.06, (21, 3))
    seq = []
    for t in range(T):
        rot = v._rotacion(amplitud * np.sin(t / 7), amplitud * np.cos(t / 5),
                          amplitud * np.sin(t / 3 + 1))
        p = molde @ rot.T
        seq.append(np.stack([p[:, 0], p[:, 1] / r], axis=1))
    return np.array(seq)


class EstimadorAspectoTest(unittest.TestCase):
    def test_recupera_r_en_mano_rigida_que_rota(self):
        for r in (0.5625, 1.7778, 0.75, 1.3333):
            with self.subTest(r=r):
                seq = _mano_rigida(40, r, amplitud=0.3)
                r_hat, _ = v.estimar_r([seq], v.ARISTAS_MANO)
                self.assertIsNotNone(r_hat)
                self.assertLessEqual(abs(r_hat - r) / r, 0.05)

    def test_ignora_cuadros_con_puntos_ausentes(self):
        seq = _mano_rigida(40, 0.75, amplitud=0.3)
        seq[::3, 4, :] = np.nan
        r_hat, _ = v.estimar_r([seq], v.ARISTAS_MANO)
        self.assertLessEqual(abs(r_hat - 0.75) / 0.75, 0.05)

    def test_sin_suficientes_cuadros_no_estima(self):
        seq = _mano_rigida(5, 0.75, amplitud=0.3)
        self.assertEqual((None, None), v.estimar_r([seq], v.ARISTAS_MANO))

    def test_grilla_cubre_0_3_a_3_5(self):
        self.assertAlmostEqual(0.3, v.GRILLA_R[0])
        self.assertAlmostEqual(3.5, v.GRILLA_R[-1])

    def test_candidata_mas_cercana(self):
        self.assertTrue(v.mas_cercano_candidata(0.56).startswith("16:9"))
        self.assertTrue(v.mas_cercano_candidata(1.78).startswith("9:16"))
        self.assertTrue(v.mas_cercano_candidata(0.98).startswith("1:1"))
        self.assertIsNone(v.mas_cercano_candidata(None))


class ProtobufTest(unittest.TestCase):
    def _landmark(self, x=None, y=None, z=None) -> bytes:
        msg = b""
        for numero, valor in ((1, x), (2, y), (3, z)):
            if valor is not None:
                msg += bytes([(numero << 3) | 5]) + struct.pack("<f", valor)
        return msg

    def test_decodifica_lista_y_campos_en_cero_son_ausentes(self):
        m1 = self._landmark(x=0.5, z=-0.25)          # y omitido = 0 (proto3)
        m2 = self._landmark()                        # todo en cero: mensaje vacío
        datos = b"\x0a" + bytes([len(m1)]) + m1 + b"\x0a" + bytes([len(m2)]) + m2
        pts = v.decodificar_landmarks(datos)
        np.testing.assert_allclose(pts, [[0.5, 0.0, -0.25], [0.0, 0.0, 0.0]], atol=1e-7)

    def test_sin_bytes_no_hay_puntos(self):
        self.assertEqual((0, 3), v.decodificar_landmarks(b"").shape)

    def test_mensaje_truncado_falla(self):
        with self.assertRaises(ValueError):
            v.decodificar_landmarks(b"\x0a\x10\x0d\x00")


class CargaSeguraTest(unittest.TestCase):
    def test_rechaza_clase_fuera_de_la_lista_blanca(self):
        # Pickle protocolo 0 que pide os.system. No debe ejecutarse ni resolverse.
        malicioso = b"cos\nsystem\n(S'echo no'\ntR."
        with self.assertRaises(pickle.UnpicklingError) as ctx:
            v._DesempaquetadorSeguro(io.BytesIO(malicioso)).load()
        self.assertIn("os.system", str(ctx.exception))

    def test_acepta_clases_de_mediapipe_como_stub_inerte(self):
        # Se serializa con un módulo sin puntos y se reemplaza el nombre en los bytes:
        # así pickle no intenta importar mediapipe real.
        nombre_falso = "mpfake_landmark"

        class NormalizedLandmark:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def __reduce__(self):
                return (NormalizedLandmark, (), dict(self.__dict__))

        NormalizedLandmark.__module__ = nombre_falso
        NormalizedLandmark.__qualname__ = "NormalizedLandmark"
        modulo = types.ModuleType(nombre_falso)
        modulo.NormalizedLandmark = NormalizedLandmark
        sys.modules[nombre_falso] = modulo
        try:
            datos = pickle.dumps([NormalizedLandmark(x=0.5, y=0.25)], protocol=0)
        finally:
            del sys.modules[nombre_falso]
        datos = datos.replace(nombre_falso.encode(), MEDIAPIPE_LANDMARK.encode())
        cargado = v._DesempaquetadorSeguro(io.BytesIO(datos)).load()
        self.assertIsInstance(cargado[0], v._Objeto)
        self.assertEqual({"x": 0.5, "y": 0.25}, cargado[0].estado)

    def test_auditor_lista_clases_sin_ejecutar(self):
        datos = b"cos\nsystem\n(S'echo no'\ntR."
        auditor = v._AuditorDeClases(io.BytesIO(datos))
        auditor.load()
        self.assertEqual({("os", "system")}, auditor.pedidas)


def _pose_base(T: int, espejada: bool) -> np.ndarray:
    """Pose con hombros y muñecas fijas; espejada invierte x como en un selfie."""
    pose = np.full((T, 33, 3), 0.5, dtype=np.float32)
    pose[:, 11] = [0.6, 0.4, 0.0]   # hombro izquierdo del sujeto
    pose[:, 12] = [0.4, 0.4, 0.0]   # hombro derecho del sujeto
    pose[:, 15] = [0.75, 0.6, 0.0]  # muñeca izquierda del sujeto
    pose[:, 16] = [0.25, 0.6, 0.0]  # muñeca derecha del sujeto
    if espejada:
        pose[..., 0] = 1.0 - pose[..., 0]
    return pose


def _mano_en(T: int, x: float, y: float, r: float) -> np.ndarray:
    """Mano de 21 puntos rígida que rota alrededor de (x, y); z = 0."""
    rigida = _mano_rigida(T, r, amplitud=0.3).astype(np.float32)
    mano = np.zeros((T, 21, 3), dtype=np.float32)
    mano[:, :, 0] = rigida[:, :, 0] + x
    mano[:, :, 1] = rigida[:, :, 1] + y
    return mano


class EspejadoLateralidadTest(unittest.TestCase):
    def test_no_espejado_y_manos_anatomicas(self):
        T, r = 30, 1.0
        pose = _pose_base(T, espejada=False)
        izq = _mano_en(T, 0.75, 0.6, r)
        der = _mano_en(T, 0.25, 0.6, r)
        res = v.analizar_arrays(pose, izq, der)
        self.assertEqual(1.0, res["espejado"]["x11_mayor_que_x12"] / res["espejado"]["cuadros_con_hombros"])
        self.assertEqual(res["lateralidad"]["izquierda"]["cuadros"],
                         res["lateralidad"]["izquierda"]["coinciden"])
        self.assertEqual(res["lateralidad"]["derecha"]["cuadros"],
                         res["lateralidad"]["derecha"]["coinciden"])

    def test_espejado_se_detecta_por_hombros(self):
        # Imagen espejada con manos según la anatomía: el espejado se ve en x11 < x12.
        T, r = 30, 1.0
        pose = _pose_base(T, espejada=True)
        izq = _mano_en(T, 0.25, 0.6, r)
        der = _mano_en(T, 0.75, 0.6, r)
        res = v.analizar_arrays(pose, izq, der)
        self.assertEqual(0, res["espejado"]["x11_mayor_que_x12"])
        self.assertEqual(res["lateralidad"]["izquierda"]["cuadros"],
                         res["lateralidad"]["izquierda"]["coinciden"])

    def test_etiquetas_intercambiadas_se_detectan(self):
        T, r = 30, 1.0
        pose = _pose_base(T, espejada=False)
        izq = _mano_en(T, 0.25, 0.6, r)   # etiquetada izquierda pero cerca de la muñeca 16
        der = _mano_en(T, 0.75, 0.6, r)
        res = v.analizar_arrays(pose, izq, der)
        self.assertEqual(0, res["lateralidad"]["izquierda"]["coinciden"])
        self.assertEqual(0, res["lateralidad"]["derecha"]["coinciden"])

    def test_faltantes_se_cuentan(self):
        T, r = 20, 1.0
        pose = _pose_base(T, espejada=False)
        izq = _mano_en(T, 0.75, 0.6, r)
        der = np.full((T, 21, 3), np.nan, dtype=np.float32)
        res = v.analizar_arrays(pose, izq, der)
        self.assertEqual(0, res["faltantes"]["sin_mano_izquierda"])
        self.assertEqual(T, res["faltantes"]["sin_mano_derecha"])
        self.assertEqual(0, res["faltantes"]["sin_ninguna_mano"])


class CocienteDispersionTest(unittest.TestCase):
    def test_mismo_cuadro_da_uno(self):
        pose = _pose_base(10, espejada=False)
        pose[:, :, :2] += np.random.default_rng(1).normal(0, 0.05, (10, 33, 2))
        self.assertAlmostEqual(1.0, v.cociente_ky_kx(pose, pose), places=6)

    def test_cuadro_con_otra_relacion_se_refleja_en_el_cociente(self):
        pose = _pose_base(10, espejada=False)
        pose[:, :, :2] += np.random.default_rng(1).normal(0, 0.05, (10, 33, 2))
        otro = pose.copy()
        otro[..., 1] *= 1.78          # mismo contenido, y estirado 1.78 en vertical
        self.assertAlmostEqual(1.78, v.cociente_ky_kx(otro, pose), places=2)


def _torso_sintetico(T: int, r: float, ancho: float = 0.2, torso: float = 0.3) -> np.ndarray:
    """Pose con hombros de ancho `ancho` y torso vertical de `torso` en W, proyectada con H/W=r."""
    pose = np.zeros((T, 33, 3), dtype=np.float32)
    pose[:, 11] = [0.5 + ancho / 2, 0.3 / r, 0]
    pose[:, 12] = [0.5 - ancho / 2, 0.3 / r, 0]
    pose[:, 23] = [0.5 + ancho / 2, (0.3 + torso) / r, 0]
    pose[:, 24] = [0.5 - ancho / 2, (0.3 + torso) / r, 0]
    return pose


class ParquetIslrTest(unittest.TestCase):
    def test_lee_esquema_islr_y_deja_nan_los_ausentes(self):
        import tempfile
        import pandas as pd

        filas = [
            {"frame": 0, "row_id": "a", "type": "pose", "landmark_index": 0, "x": 0.1, "y": 0.2, "z": 0.0},
            {"frame": 0, "row_id": "b", "type": "left_hand", "landmark_index": 0, "x": 0.3, "y": 0.4, "z": 0.5},
            {"frame": 1, "row_id": "c", "type": "pose", "landmark_index": 0, "x": 0.6, "y": 0.7, "z": 0.8},
            {"frame": 0, "row_id": "d", "type": "face", "landmark_index": 0, "x": 0.9, "y": 0.9, "z": 0.9},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "x.parquet"
            pd.DataFrame(filas).to_parquet(ruta)
            pose, izq, der = v.leer_islr(ruta)
        self.assertEqual((2, 33, 3), pose.shape)
        np.testing.assert_allclose(pose[0, 0], [0.1, 0.2, 0.0], atol=1e-6)
        np.testing.assert_allclose(izq[0, 0], [0.3, 0.4, 0.5], atol=1e-6)
        self.assertTrue(np.isnan(izq[1]).all())
        self.assertTrue(np.isnan(der).all())
        self.assertTrue(np.isnan(pose[0, 1]).all())

    def test_ratio_torso_depende_de_r(self):
        pose = _torso_sintetico(5, r=0.5625)
        self.assertAlmostEqual(0.2 / 0.3, float(np.median(v.ratio_torso(pose, 0.5625))), places=2)
        # Con r = 1 el torso vertical se estira y el cociente ancho/torso baja.
        self.assertLess(float(np.median(v.ratio_torso(pose, 1.0))),
                        float(np.median(v.ratio_torso(pose, 0.5625))))

    def test_calibracion_recupera_r_de_pose_con_otra_relacion(self):
        ref = _torso_sintetico(20, r=1.0)
        objetivo = _torso_sintetico(20, r=1.7778)
        r_hat, _ = v.calibrar_r_torso(ref, objetivo)
        self.assertLessEqual(abs(r_hat - 1.7778) / 1.7778, 0.02)


class SimulacionTest(unittest.TestCase):
    def test_sin_ruido_el_error_2d_es_chico_para_rotacion_moderada(self):
        e = v.simular_ruido(0.0, 0.0, 0.75, repeticiones=3, amplitud=0.3)
        self.assertLessEqual(abs(e["err2d_media"]), 5.0)
        self.assertLessEqual(abs(e["err3d_media"]), 1.0)


if __name__ == "__main__":
    unittest.main()
