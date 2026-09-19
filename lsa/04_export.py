"""
04_export.py — Exporta el modelo entrenado a ONNX + INT8 + TFLite
Pipeline completo: PyTorch (.pt) -> ONNX -> ONNX INT8 -> TFLite (.tflite)

Uso:
    python 04_export.py                        # exporta phase_b_best.pt
    python 04_export.py --ckpt phase_c_best.pt
    python 04_export.py --skip-tflite          # solo ONNX + INT8
"""
import argparse
import base64
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from config import (
    CHECKPOINTS_DIR, EXPORTS_DIR, VOCAB_PATH,
    MAX_FRAMES, INPUT_DIM, CALIB_SAMPLES, MAX_DEC_LEN,
    BOS_ID, EOS_ID,
)
from model import LSAModel
from dataset import Vocabulary, LSADataset


ARTIFACT_VERSION = "lsa-t-seq2seq-v1"


def sha256(path: Path) -> str:
    """SHA-256 de un artefacto, sin cargar el modelo entero en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_base64(values: np.ndarray) -> str:
    """Serializa floats LE para un fixture compacto y exacto en Android."""
    little_endian = np.ascontiguousarray(values, dtype="<f4")
    return base64.b64encode(little_endian.tobytes()).decode("ascii")


def load_model(ckpt_name: str, vocab_size: int) -> LSAModel:
    ckpt_path = CHECKPOINTS_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} no encontrado")
        sys.exit(1)
    model = LSAModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    print(f"[export] Modelo cargado: {ckpt_path}")
    return model


# ── Exportar a ONNX ───────────────────────────────────────────────────────────

def export_onnx(model: LSAModel, vocab_size: int):
    """Exporta encoder y decoder a ONNX separadamente."""
    EXPORTS_DIR.mkdir(exist_ok=True)

    # --- Encoder ---
    dummy_kps = torch.randn(1, MAX_FRAMES, INPUT_DIM)
    enc_path  = EXPORTS_DIR / "encoder.onnx"

    torch.onnx.export(
        model.encoder,
        (dummy_kps,),
        str(enc_path),
        input_names   = ["keypoints"],
        output_names  = ["embeddings", "ctc_logits"],
        dynamic_axes  = {
            "keypoints":  {0: "batch", 1: "T"},
            "embeddings": {0: "batch", 1: "T"},
        },
        opset_version = 17,
        export_params = True,
        dynamo        = False,  # exportador dynamo/onnxscript falla en el inliner pass
    )
    size_mb = enc_path.stat().st_size / 1e6
    print(f"[onnx] Encoder exportado: {enc_path}  ({size_mb:.1f} MB)")

    # --- Decoder ---
    dummy_emb = torch.randn(1, MAX_FRAMES, 256)
    dummy_tgt = torch.zeros(1, 1, dtype=torch.long)
    dec_path  = EXPORTS_DIR / "decoder.onnx"

    torch.onnx.export(
        model.decoder,
        (dummy_tgt, dummy_emb),
        str(dec_path),
        input_names   = ["tgt_tokens", "memory"],
        output_names  = ["logits"],
        dynamic_axes  = {
            "tgt_tokens": {0: "batch", 1: "S"},
            "memory":     {0: "batch", 1: "T"},
            "logits":     {0: "batch", 1: "S"},
        },
        opset_version = 17,
        export_params = True,
        dynamo        = False,  # exportador dynamo/onnxscript falla en el inliner pass
    )
    size_mb = dec_path.stat().st_size / 1e6
    print(f"[onnx] Decoder exportado: {dec_path}  ({size_mb:.1f} MB)")

    return enc_path, dec_path


# ── Cuantizacion INT8 estatica ────────────────────────────────────────────────

def quantize_int8(enc_onnx: Path, dec_onnx: Path, vocab: Vocabulary):
    """Cuantizacion INT8 estatica con calibracion sobre el val set de LSA-T."""
    try:
        from onnxruntime.quantization import (
            quantize_static, CalibrationDataReader, QuantType, QuantFormat
        )
    except ImportError:
        print("[quant] ERROR: onnxruntime no instalado.")
        print("        pip install onnxruntime")
        return None, None

    print("[quant] Cargando muestras de calibracion (val set)...")
    val_ds  = LSADataset("val", vocab, augment=False)
    n_calib = min(CALIB_SAMPLES, len(val_ds))
    calib_kps = np.stack([
        val_ds[i]["keypoints"].numpy() for i in range(n_calib)
    ])  # [N, T, 126]
    print(f"[quant] {n_calib} muestras de calibracion")

    # --- Calibrador encoder ---
    class EncoderCalibReader(CalibrationDataReader):
        def __init__(self):
            self.idx = 0

        def get_next(self):
            if self.idx >= n_calib:
                return None
            kps = calib_kps[self.idx : self.idx + 1]  # [1, T, 126]
            self.idx += 1
            return {"keypoints": kps}

    enc_int8 = EXPORTS_DIR / "encoder_int8.onnx"
    quantize_static(
        str(enc_onnx),
        str(enc_int8),
        calibration_data_reader = EncoderCalibReader(),
        quant_format            = QuantFormat.QDQ,
        weight_type             = QuantType.QInt8,
        per_channel             = False,
    )
    size_mb = enc_int8.stat().st_size / 1e6
    print(f"[quant] Encoder INT8: {enc_int8}  ({size_mb:.1f} MB)")

    # --- Calibrador decoder ---
    class DecoderCalibReader(CalibrationDataReader):
        """Calibra el decoder con embeddings reales y prefijos reales.

        Calibrar con memoria toda-cero mide rangos que no ocurren en
        inferencia y degrada el decoder al cuantizarlo.
        """
        def __init__(self):
            self.idx = 0
            self.encoder = ort.InferenceSession(str(enc_int8))

        def get_next(self):
            if self.idx >= n_calib:
                return None
            item = val_ds[self.idx]
            kps = item["keypoints"].numpy()[None]
            embeddings = self.encoder.run(
                ["embeddings"],
                {"keypoints": kps},
            )[0]
            tokens = item["tgt_in"].numpy()[None].astype(np.int64)
            self.idx += 1
            return {
                "tgt_tokens": tokens,
                "memory":     embeddings,
            }

    dec_int8 = EXPORTS_DIR / "decoder_int8.onnx"
    quantize_static(
        str(dec_onnx),
        str(dec_int8),
        calibration_data_reader = DecoderCalibReader(),
        quant_format            = QuantFormat.QDQ,
        weight_type             = QuantType.QInt8,
        per_channel             = False,
    )
    size_mb = dec_int8.stat().st_size / 1e6
    print(f"[quant] Decoder INT8: {dec_int8}  ({size_mb:.1f} MB)")

    return enc_int8, dec_int8


# ── Verificar precision post-cuantizacion ────────────────────────────────────

def verify_quantization(model: LSAModel, enc_int8: Path, vocab: Vocabulary, n: int = 50):
    """Mide la diferencia de embeddings del encoder FP32 frente al INT8."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] Instalar onnxruntime para verificacion")
        return

    session = ort.InferenceSession(str(enc_int8))
    val_ds  = LSADataset("val", vocab, augment=False)

    fp32_embs, int8_embs = [], []
    model.eval()

    for i in range(min(n, len(val_ds))):
        kps_np = val_ds[i]["keypoints"].numpy()[None]  # [1, T, 126]
        # FP32
        with torch.no_grad():
            fp32_out, _ = model.encoder(torch.tensor(kps_np))
        fp32_embs.append(fp32_out.numpy())
        # INT8
        int8_out = session.run(["embeddings"], {"keypoints": kps_np})[0]
        int8_embs.append(int8_out)

    # Diferencia media de embeddings
    diffs = [np.abs(f - i8).mean() for f, i8 in zip(fp32_embs, int8_embs)]
    print(f"[verify] Diferencia media de embeddings FP32 vs INT8: {np.mean(diffs):.5f}")
    print(f"         (< 0.01 es excelente, < 0.05 es aceptable)")


# ── Convertir a TFLite y validar el artefacto ─────────────────────────────────

def _tensor_spec(detail: dict) -> dict:
    return {
        "name": str(detail["name"]),
        "dtype": np.dtype(detail["dtype"]).name,
        "shape": [int(v) for v in detail["shape"]],
        "shapeSignature": [int(v) for v in detail["shape_signature"]],
    }


def inspect_tflite(path: Path) -> dict:
    """Lee la interfaz real del FlatBuffer, no el nombre supuesto del archivo."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow es necesario para validar el artefacto TFLite") from exc

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return {
        "inputs": [_tensor_spec(d) for d in interpreter.get_input_details()],
        "outputs": [_tensor_spec(d) for d in interpreter.get_output_details()],
    }


def select_float32_tflite(
    generated_dir: Path,
    destination: Path,
    input_dtypes: list[str],
    output_dtypes: list[str],
) -> dict:
    """Elige por interfaz el único FlatBuffer F32 válido y lo copia al destino.

    onnx2tf suele dejar varias variantes en el directorio (float16, float32 y
    alguna de depuración). Elegir ``files[0]`` produjo antes un encoder F16 y
    un decoder F32, un par que Android no podía conectar.
    """
    candidates = sorted(generated_dir.glob("*.tflite"))
    matches: list[tuple[Path, dict]] = []
    rejected: list[str] = []
    for candidate in candidates:
        try:
            interface = inspect_tflite(candidate)
        except Exception as exc:
            rejected.append(f"{candidate.name}: ilegible ({exc})")
            continue
        actual_inputs = [t["dtype"] for t in interface["inputs"]]
        actual_outputs = [t["dtype"] for t in interface["outputs"]]
        if actual_inputs == input_dtypes and actual_outputs == output_dtypes:
            matches.append((candidate, interface))
        else:
            rejected.append(
                f"{candidate.name}: entradas={actual_inputs}, salidas={actual_outputs}"
            )

    if len(matches) != 1:
        detail = "\n  ".join(rejected) or "sin archivos .tflite"
        raise RuntimeError(
            f"Se esperaba exactamente una variante TFLite compatible para {destination.name}; "
            f"se encontraron {len(matches)}.\n  {detail}"
        )

    source, interface = matches[0]
    shutil.copy2(source, destination)
    size_mb = destination.stat().st_size / 1e6
    print(f"[tflite] {destination.name} <- {source.name} ({size_mb:.1f} MB)")
    return interface


def _run_onnx2tf(onnx_path: Path, out_dir: Path) -> None:
    import subprocess

    out_dir.mkdir(parents=True)
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(out_dir),
        "--non_verbose",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"onnx2tf falló para {onnx_path.name}:\n{result.stderr[:1000]}"
        )


def _run_encoder_tflite(model_path: Path, keypoints: np.ndarray) -> np.ndarray:
    """Ejecuta el encoder con la transposición que exige el FlatBuffer final."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_detail = interpreter.get_input_details()[0]
    # El exportador ONNX->TF transforma [B,T,C] en [B,C,T].
    model_input = np.transpose(keypoints[None].astype(np.float32), (0, 2, 1))
    interpreter.resize_tensor_input(input_detail["index"], model_input.shape, strict=False)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_detail["index"], model_input)
    interpreter.invoke()

    outputs = interpreter.get_output_details()
    embedding_detail = next(
        detail for detail in outputs if int(detail["shape"][-1]) == 256
    )
    embeddings_bct = interpreter.get_tensor(embedding_detail["index"])
    return np.transpose(embeddings_bct, (0, 2, 1))  # vuelve a [B,T,256]


def _run_decoder_tflite(
    model_path: Path,
    token_ids: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Ejecuta una iteración del decoder con IDs INT64 y memoria [B,256,T]."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    token_detail, memory_detail = interpreter.get_input_details()
    if np.dtype(token_detail["dtype"]).kind not in {"i", "u"}:
        token_detail, memory_detail = memory_detail, token_detail

    memory_bdt = np.transpose(embeddings.astype(np.float32), (0, 2, 1))
    interpreter.resize_tensor_input(token_detail["index"], token_ids.shape, strict=False)
    interpreter.resize_tensor_input(memory_detail["index"], memory_bdt.shape, strict=False)
    interpreter.allocate_tensors()
    interpreter.set_tensor(token_detail["index"], token_ids.astype(np.int64))
    interpreter.set_tensor(memory_detail["index"], memory_bdt)
    interpreter.invoke()
    logits_bsv = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    return logits_bsv.astype(np.float32)


def write_android_fixture(
    encoder_model: Path,
    decoder_model: Path,
    vocab: Vocabulary,
    samples: int = 3,
) -> Path:
    """Genera casos reales para la etapa 1 de la integración Android."""
    val_ds = LSADataset("val", vocab, augment=False)
    records = []
    for i in range(min(samples, len(val_ds))):
        item = val_ds[i]
        keypoints = item["keypoints"].numpy()
        embeddings = _run_encoder_tflite(encoder_model, keypoints)

        tokens = np.array([[BOS_ID]], dtype=np.int64)
        first_logits = _run_decoder_tflite(decoder_model, tokens, embeddings)
        generated: list[int] = []
        for _ in range(MAX_DEC_LEN):
            logits = _run_decoder_tflite(decoder_model, tokens, embeddings)
            next_token = int(np.argmax(logits[0, -1]))
            generated.append(next_token)
            if next_token == EOS_ID:
                break
            tokens = np.concatenate([tokens, [[next_token]]], axis=1)

        records.append({
            "label": item["label"],
            "keypoints": {
                "layout": "B,T,126",
                "shape": [1, MAX_FRAMES, INPUT_DIM],
                "f32leBase64": float32_base64(keypoints[None]),
            },
            "expected": {
                "encoderEmbeddings": {
                    "layout": "B,T,256",
                    "shape": [1, MAX_FRAMES, 256],
                    "f32leBase64": float32_base64(embeddings),
                },
                "firstDecoderLogits": {
                    "layout": "B,S,V",
                    "shape": [int(v) for v in first_logits.shape],
                    "f32leBase64": float32_base64(first_logits),
                },
                "tokenIds": generated,
            },
        })

    if not records:
        raise RuntimeError("No hay muestras de validación para generar fixture_android.json")

    fixture = EXPORTS_DIR / "fixture_android.json"
    fixture.write_text(
        json.dumps({"artifactVersion": ARTIFACT_VERSION, "samples": records}),
        encoding="utf-8",
    )
    print(f"[fixture] {fixture} ({len(records)} secuencias reales)")
    return fixture


def write_artifact_manifest(
    encoder_model: Path,
    decoder_model: Path,
    vocab_file: Path,
    fixture: Path,
    encoder_interface: dict,
    decoder_interface: dict,
) -> Path:
    """Describe el contrato que Android debe validar antes de usar el modelo."""
    manifest = {
        "artifactVersion": ARTIFACT_VERSION,
        "preprocessing": {
            "extractor": "MediaPipe Holistic: left hand followed by right hand",
            "sourceLandmarkRanges": {"left": [501, 522], "right": [522, 543]},
            "inputLayoutBeforeExport": "B,T,126",
            "normalization": (
                "per-frame mean of all 42 hand landmarks over x/y, including zeros; "
                "global max absolute x/y scale; z unchanged"
            ),
            "temporal": (
                "integer floor sampling idx[i]=(i*(T-1))//(75-1); "
                "zero-pad at end to 75 frames"
            ),
        },
        "encoder": {
            "file": encoder_model.name,
            "sha256": sha256(encoder_model),
            "androidInputLayout": "B,126,T",
            **encoder_interface,
        },
        "decoder": {
            "file": decoder_model.name,
            "sha256": sha256(decoder_model),
            "androidMemoryLayout": "B,256,T",
            **decoder_interface,
        },
        "vocabulary": {
            "file": vocab_file.name,
            "sha256": sha256(vocab_file),
            "size": len(json.loads(vocab_file.read_text(encoding="utf-8"))),
            "specialTokenIds": {"pad": 0, "bos": BOS_ID, "eos": EOS_ID, "unk": 3},
        },
        "decoderPolicy": {"algorithm": "greedy", "maxTokens": MAX_DEC_LEN},
        "fixture": {"file": fixture.name, "sha256": sha256(fixture)},
    }
    destination = EXPORTS_DIR / "android-manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] {destination}")
    return destination


def convert_tflite(enc_int8: Path, dec_int8: Path, vocab: Vocabulary) -> None:
    """Convierte, selecciona y verifica un par TFLite de precisión float32."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="onnx2tf-", dir=EXPORTS_DIR) as temp_dir:
        work_dir = Path(temp_dir)
        enc_tflite_dir = work_dir / "encoder"
        dec_tflite_dir = work_dir / "decoder"

        print("[tflite] Convirtiendo encoder...")
        _run_onnx2tf(enc_int8, enc_tflite_dir)
        encoder_model = EXPORTS_DIR / "encoder_int8.tflite"
        encoder_interface = select_float32_tflite(
            enc_tflite_dir,
            encoder_model,
            input_dtypes=["float32"],
            output_dtypes=["float32", "float32"],
        )

        print("[tflite] Convirtiendo decoder...")
        _run_onnx2tf(dec_int8, dec_tflite_dir)
        decoder_model = EXPORTS_DIR / "decoder_int8.tflite"
        decoder_interface = select_float32_tflite(
            dec_tflite_dir,
            decoder_model,
            input_dtypes=["int64", "float32"],
            output_dtypes=["float32"],
        )

    vocab_file = EXPORTS_DIR / "vocab.json"
    shutil.copy2(VOCAB_PATH, vocab_file)
    fixture = write_android_fixture(encoder_model, decoder_model, vocab)
    write_artifact_manifest(
        encoder_model,
        decoder_model,
        vocab_file,
        fixture,
        encoder_interface,
        decoder_interface,
    )


# ── Resumen final ─────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 55)
    print("RESUMEN DE ARCHIVOS PARA ANDROID")
    print("=" * 55)
    files_info = [
        ("encoder_int8.tflite", "Modelo encoder INT8"),
        ("decoder_int8.tflite", "Modelo decoder INT8"),
        ("vocab.json",          "Vocabulario"),
        ("android-manifest.json", "Contrato para Android"),
        ("fixture_android.json",  "Fixture de integración"),
    ]
    total_mb = 0
    for fname, desc in files_info:
        fpath = EXPORTS_DIR / fname
        if fpath.exists():
            mb = fpath.stat().st_size / 1e6
            total_mb += mb
            print(f"  {fname:30s}  {mb:6.1f} MB  {desc}")
        else:
            print(f"  {fname:30s}  (no generado)")
    print(f"  {'TOTAL':30s}  {total_mb:6.1f} MB")
    print(f"\nEntregar estos cinco archivos como un único artefacto versionado.")
    print("Android ya utiliza HolisticLandmarker; no agregar HandLandmarker.")
    print(f"APK overhead estimado (sin el extractor ya existente): ~{total_mb:.0f} MB")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportar modelo LSA a TFLite")
    parser.add_argument("--ckpt",         default="phase_b_best.pt")
    parser.add_argument("--skip-tflite",  action="store_true",
                        help="Solo exportar ONNX + INT8, sin conversion a TFLite")
    parser.add_argument("--skip-quant",   action="store_true",
                        help="Solo exportar ONNX FP32")
    args = parser.parse_args()

    vocab = Vocabulary()
    model = load_model(args.ckpt, len(vocab))

    # 1. ONNX
    print("\n--- Paso 1: Exportar a ONNX ---")
    enc_onnx, dec_onnx = export_onnx(model, len(vocab))

    if args.skip_quant:
        print_summary()
        sys.exit(0)

    # 2. INT8
    print("\n--- Paso 2: Cuantizacion INT8 ---")
    enc_int8, dec_int8 = quantize_int8(enc_onnx, dec_onnx, vocab)

    if enc_int8:
        verify_quantization(model, enc_int8, vocab)

    if args.skip_tflite or enc_int8 is None:
        print_summary()
        sys.exit(0)

    # 3. TFLite
    print("\n--- Paso 3: Convertir a TFLite ---")
    convert_tflite(enc_int8, dec_int8, vocab)

    print_summary()
    print("\nSiguiente paso: integrar en el proyecto Android.")
