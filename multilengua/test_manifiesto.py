"""Pruebas del manifiesto multilengua (esquema común para lsa, asl y lse)."""

import json
import tempfile
import unittest
from pathlib import Path

from multilengua.manifiesto import (
    LENGUAS,
    ORIGEN_ISLR_POPSIGN,
    ORIGEN_LSA64,
    ORIGEN_LSA94,
    ORIGEN_SWL_LSE,
    escribir_manifiesto,
    validar_manifiesto,
)

RAIZ = Path(__file__).resolve().parent.parent
MANIFIESTO_LSA94 = RAIZ / "lsa-94" / "exports" / "lsa-manifest.json"
SHA_MODELO_LSA94 = "751eab8b5cb6642556f227380dd1537154012c9341b8e26d9d593078d0e3a46b"
SHA_CATALOGO_LSA94 = "737026d776bd1e8f791b306b14166a9901ca0ebe7fcbd22f1d06d168829c356c"

CAMPOS_ESPERADOS = {
    "language", "modelVersion", "catalogVersion", "dataset", "license", "attribution",
    "modelSha256", "catalogSha256", "numClasses", "outputsProbabilities",
    "frames", "coords", "contractVersion",
}

ORIGENES = {
    "lsa": ORIGEN_LSA94,
    "asl": ORIGEN_ISLR_POPSIGN,
    "lse": ORIGEN_SWL_LSE,
}


def escribir_de_prueba(destino, language="lsa", origen=None, **cambios):
    """Escribe un manifiesto con blobs falsos y devuelve su contenido."""
    origen = origen or ORIGENES.get(language, ORIGEN_LSA94)
    args = dict(
        language=language,
        model_version=f"eva-{language}-prueba",
        catalog_version="1.0",
        blob_modelo=b"modelo de prueba",
        catalog_bytes=b'{"version": "1.0", "glosas": []}',
        num_clases=3,
        dataset=origen["dataset"],
        license=origen["license"],
        attribution=origen["attribution"],
        frames=40,
        coords=168,
        contract_version=3,
    )
    args.update(cambios)
    escribir_manifiesto(destino, **args)
    return json.loads(Path(destino).read_text(encoding="utf-8"))


class ManifiestoMultilenguaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_las_tres_lenguas_validan_contra_el_mismo_esquema(self):
        self.assertEqual(("lsa", "asl", "lse"), LENGUAS)
        for lengua in LENGUAS:
            with self.subTest(lengua=lengua):
                manifiesto = escribir_de_prueba(self.dir / lengua / "manifest.json", lengua)
                validar_manifiesto(manifiesto)
                self.assertEqual(CAMPOS_ESPERADOS, set(manifiesto))
                self.assertEqual(lengua, manifiesto["language"])

    def test_licencias_por_dataset(self):
        self.assertEqual("CC BY-NC-SA 4.0", ORIGEN_LSA64["license"])
        self.assertEqual("CC BY-NC-SA 4.0", ORIGEN_LSA94["license"])
        self.assertEqual("CC BY 4.0", ORIGEN_ISLR_POPSIGN["license"])
        self.assertEqual("CC BY 4.0", ORIGEN_SWL_LSE["license"])
        self.assertIn("Ronchetti", ORIGEN_LSA94["attribution"])
        self.assertIn("Deaf Professional Arts Network", ORIGEN_ISLR_POPSIGN["attribution"])
        self.assertIn("Georgia Tech", ORIGEN_ISLR_POPSIGN["attribution"])
        self.assertIn("13691887", ORIGEN_SWL_LSE["attribution"])

    def test_language_invalido_falla(self):
        with self.assertRaises(ValueError):
            escribir_de_prueba(self.dir / "x.json", language="xx")
        manifiesto = escribir_de_prueba(self.dir / "ok.json", "lsa")
        manifiesto["language"] = "es"
        with self.assertRaises(ValueError):
            validar_manifiesto(manifiesto)

    def test_campo_faltante_falla(self):
        for campo in sorted(CAMPOS_ESPERADOS):
            with self.subTest(campo=campo):
                manifiesto = escribir_de_prueba(self.dir / f"{campo}.json", "asl")
                del manifiesto[campo]
                with self.assertRaises(ValueError):
                    validar_manifiesto(manifiesto)

    def test_dataset_sin_version_falla(self):
        manifiesto = escribir_de_prueba(self.dir / "d.json", "lse")
        del manifiesto["dataset"]["version"]
        with self.assertRaises(ValueError):
            validar_manifiesto(manifiesto)

    def test_sha_debe_ser_hex_de_64(self):
        for campo in ("modelSha256", "catalogSha256"):
            for malo in ("abc", "G" * 64, "A" * 64 + "0", 12345):
                with self.subTest(campo=campo, valor=malo):
                    manifiesto = escribir_de_prueba(self.dir / "s.json", "lsa")
                    manifiesto[campo] = malo
                    with self.assertRaises(ValueError):
                        validar_manifiesto(manifiesto)

    def test_tipos_incorrectos_fallan(self):
        manifiesto = escribir_de_prueba(self.dir / "t.json", "lsa")
        manifiesto["numClasses"] = "94"
        with self.assertRaises(ValueError):
            validar_manifiesto(manifiesto)
        manifiesto = escribir_de_prueba(self.dir / "t2.json", "lsa")
        manifiesto["outputsProbabilities"] = 0
        with self.assertRaises(ValueError):
            validar_manifiesto(manifiesto)

    def test_escribir_no_deja_archivo_si_el_manifiesto_es_invalido(self):
        destino = self.dir / "invalido.json"
        with self.assertRaises(ValueError):
            escribir_de_prueba(destino, "lsa", contract_version=0)
        self.assertFalse(destino.exists())

    def test_contract_version_3(self):
        manifiesto = escribir_de_prueba(self.dir / "c.json", "lsa")
        self.assertEqual(3, manifiesto["contractVersion"])
        self.assertFalse(manifiesto["outputsProbabilities"])

    def test_sha256_se_calculan_sobre_los_bytes_recibidos(self):
        import hashlib
        manifiesto = escribir_de_prueba(self.dir / "h.json", "lsa")
        self.assertEqual(hashlib.sha256(b"modelo de prueba").hexdigest(), manifiesto["modelSha256"])

    def test_manifiesto_lsa94_regenerado(self):
        if not MANIFIESTO_LSA94.exists():
            self.skipTest("lsa-94/exports/lsa-manifest.json no existe (carpeta de exports ignorada)")
        manifiesto = json.loads(MANIFIESTO_LSA94.read_text(encoding="utf-8"))
        validar_manifiesto(manifiesto)
        self.assertEqual("lsa", manifiesto["language"])
        self.assertEqual("eva-lsa94-v4", manifiesto["modelVersion"])
        self.assertEqual(SHA_MODELO_LSA94, manifiesto["modelSha256"])
        self.assertEqual(SHA_CATALOGO_LSA94, manifiesto["catalogSha256"])
        self.assertEqual(3, manifiesto["contractVersion"])
        self.assertEqual("CC BY-NC-SA 4.0", manifiesto["license"])
        self.assertEqual(94, manifiesto["numClasses"])


if __name__ == "__main__":
    unittest.main()
