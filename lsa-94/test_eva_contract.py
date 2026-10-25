"""Pruebas rápidas del contrato Eva LSA-94 (v3 vigente, v2 histórico)."""

import unittest
from pathlib import Path

import numpy as np

from eva_contract import (
    COORDS,
    CONTRACT_VERSION,
    FRAMES,
    POINTS,
    build_input_v3,
    center_frame,
    center_sequence,
    load_catalog,
    load_groups,
    normalize_sequence_v3,
    parse_video_name,
    sample_indices,
    v2_to_v3,
)

HL, HR = 126, 129          # offsets de hombro izquierdo y derecho


def escena(T=40, seed=0):
    """Escena física en metros: hombros a 0,40 m, manos moviéndose."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((T, POINTS, 3))
    t = np.linspace(0, 1, T)[:, None, None]
    pts[:, 0:21, :2] = [-0.25, 0.10] + rng.normal(0, 0.03, (21, 2)) + t * [0.1, -0.2]
    pts[:, 21:42, :2] = [0.20, 0.05] + rng.normal(0, 0.03, (21, 2)) - t * [0.05, 0.3]
    pts[:, 0:42, 2] = rng.normal(0, 0.02, (T, 42))
    pts[:, 42, :2] = [-0.20, 0.0]
    pts[:, 43, :2] = [0.20, 0.0]
    pts[:, 44:56, :2] = rng.normal(0, 0.2, (12, 2)) + [0.0, 0.3]
    pts[:, 42:56, 2] = rng.normal(0, 0.05, (T, 14))
    return pts


def camara(pts_m, width, height, px_por_metro, centro_px):
    """Proyecta la escena a coords normalizadas de MediaPipe para una cámara."""
    raw = np.zeros_like(pts_m)
    raw[..., 0] = (pts_m[..., 0] * px_por_metro + centro_px[0]) / width
    raw[..., 1] = (pts_m[..., 1] * px_por_metro + centro_px[1]) / height
    raw[..., 2] = pts_m[..., 2] * px_por_metro / width
    return raw.reshape(len(pts_m), COORDS).astype(np.float32)


class EvaContractTest(unittest.TestCase):
    def test_catalogo_tiene_94_clases_en_orden(self):
        version, glosas = load_catalog()
        self.assertEqual("3.0", version)
        self.assertEqual(94, len(glosas))
        # las 64 de LSA64 no se movieron
        self.assertEqual("Opaco", glosas[0])
        self.assertEqual("Amargo", glosas[18])
        self.assertEqual("Encontrar", glosas[63])
        # etiquetas de archivo 065..094 -> índice etiqueta-1
        self.assertEqual(["Hoy", "Ayer", "Mañana"], glosas[64:67])
        letras = "A B C CH D E F G H I J K L M N O P Q R S T U V W X Y Z".split()
        self.assertEqual(letras, glosas[67:94])
        self.assertEqual("CH", glosas[71 - 1])
        self.assertEqual("Z", glosas[94 - 1])

    def test_catalogo_lsa64_sigue_identico(self):
        viejo = Path(__file__).resolve().parent.parent / "lsa64" / "catalogo_lsa64.json"
        _, glosas64 = load_catalog(viejo, expected=64)
        self.assertEqual(glosas64, load_catalog()[1][:64])

    def test_grupos_cubren_todo(self):
        self.assertEqual({"lsa64": (1, 64), "temporales": (65, 67), "abecedario": (68, 94)},
                         load_groups())

    def test_nombre_de_video_estricto(self):
        self.assertEqual((65, 12, 3), parse_video_name("065_012_003"))
        self.assertEqual((1, 1, 1), parse_video_name("001_001_001"))
        for malo in ("11_012_001", "065-012-001", "065_012_01", "065_012_001_b", "abc_012_001"):
            with self.assertRaises(ValueError, msg=malo):
                parse_video_name(malo)

    def test_v2_descarta_cara_y_conserva_ceros_ausentes(self):
        frame = np.zeros(COORDS, dtype=np.float32)
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

    def test_secuencia_corta_se_estira_de_forma_pareja(self):
        idx = sample_indices(12)
        self.assertEqual(0, idx[0])
        self.assertEqual(11, idx[-1])
        self.assertEqual(list(range(12)), sorted(set(idx)))       # usa todos
        conteo = [idx.count(k) for k in range(12)]
        self.assertLessEqual(max(conteo) - min(conteo), 1)       # parejo
        self.assertEqual([0] * 40, sample_indices(1))


class ContratoV3Test(unittest.TestCase):
    def test_version(self):
        self.assertEqual(3, CONTRACT_VERSION)

    def test_invariante_a_distancia(self):
        pts = escena()
        cerca = normalize_sequence_v3(camara(pts, 1280, 720, 900, (640, 360)), 1280, 720)
        lejos = normalize_sequence_v3(camara(pts, 1280, 720, 300, (600, 300)), 1280, 720)
        np.testing.assert_allclose(cerca, lejos, atol=1e-4)

    def test_invariante_a_orientacion_y_aspecto(self):
        pts = escena()
        horizontal = normalize_sequence_v3(camara(pts, 1920, 1080, 400, (960, 500)), 1920, 1080)
        vertical = normalize_sequence_v3(camara(pts, 480, 640, 700, (240, 300)), 480, 640)
        np.testing.assert_allclose(horizontal, vertical, atol=1e-4)

    def test_hombros_quedan_en_ancho_uno_y_centrados(self):
        out = normalize_sequence_v3(camara(escena(), 960, 540, 500, (480, 270)), 960, 540)
        p = out.reshape(FRAMES, POINTS, 3)
        np.testing.assert_allclose(p[:, 42, 0] - p[:, 43, 0], -1.0, atol=1e-5)
        np.testing.assert_allclose((p[:, 42, :2] + p[:, 43, :2]) / 2, 0.0, atol=1e-5)

    def test_ausentes_siguen_en_cero(self):
        raw = camara(escena(), 960, 540, 500, (480, 270))
        raw[5:9, 0:63] = 0.0
        out = normalize_sequence_v3(raw, 960, 540)
        self.assertTrue(np.all(out[5:9, 0:63] == 0.0))
        self.assertTrue(np.all(out[10, 0:63] != 0.0))

    def test_mediana_par_promedia_los_centrales(self):
        raw = camara(escena(T=4), 1000, 1000, 1000, (500, 500))
        # anchos de hombro 0,40 / 0,40 / 0,60 / 0,80 m -> mediana 0,50 m = 500 px
        for t, ancho in enumerate([0.40, 0.40, 0.60, 0.80]):
            raw[t, HL] = (500 - ancho * 500) / 1000
            raw[t, HR] = (500 + ancho * 500) / 1000
        out = normalize_sequence_v3(raw, 1000, 1000)
        self.assertAlmostEqual(-0.8, float(out[0, HL] - out[0, HR]), places=5)

    def test_cuadro_sin_hombros_usa_el_centro_valido_mas_cercano(self):
        raw = camara(escena(), 960, 540, 500, (480, 270))
        raw[3, HL:HL + 3] = 0.0
        out = normalize_sequence_v3(raw, 960, 540)
        ref = normalize_sequence_v3(camara(escena(), 960, 540, 500, (480, 270)), 960, 540)
        # la escena tiene hombros fijos: usar el centro del cuadro 2 no cambia nada
        np.testing.assert_allclose(out[3, 0:126], ref[3, 0:126], atol=1e-5)
        self.assertTrue(np.all(out[3, HL:HL + 3] == 0.0))

    def test_sin_hombros_es_invalida(self):
        raw = camara(escena(), 960, 540, 500, (480, 270))
        raw[:, HR:HR + 3] = 0.0
        with self.assertRaises(ValueError):
            normalize_sequence_v3(raw, 960, 540)

    def test_build_input_muestrea_antes_de_normalizar(self):
        raw = camara(escena(T=46), 960, 540, 500, (480, 270))
        out = build_input_v3(raw, 960, 540)
        self.assertEqual((FRAMES, COORDS), out.shape)
        np.testing.assert_array_equal(out, normalize_sequence_v3(raw[sample_indices(46)], 960, 540))

    def test_v2_a_v3_coincide_con_v3_directo(self):
        raw = camara(escena(), 1920, 1080, 400, (900, 520))
        v2 = center_sequence(raw)
        np.testing.assert_allclose(v2_to_v3(v2, 1920, 1080),
                                   normalize_sequence_v3(raw, 1920, 1080), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
