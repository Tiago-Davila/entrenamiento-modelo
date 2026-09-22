"""
model.py — Arquitectura Encoder-Decoder Transformer para LSA
Compatible con torch.xpu (Intel Arc via IPEX), torch.cuda y CPU.

Encoder: Linear(126, 256) + RoPE + 6x TransformerEncoderLayer + CTC head
Decoder: Token embedding + 3x TransformerDecoderLayer + output projection
Total:   ~13-15M parametros
"""
import math
import torch
import torch.nn as nn


# ── Device helper ─────────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Detecta el mejor device disponible: xpu (Intel Arc) > cuda > cpu.
    Imprime advertencias utiles si XPU no esta disponible.
    """
    # Intentar XPU (Intel Arc 140V via torch.xpu nativo, PyTorch 2.7+)
    try:
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            available = torch.xpu.is_available()
        if available:
            name = torch.xpu.get_device_name(0)
            print(f"[device] Intel XPU: {name}")
            return "xpu"
        elif caught:
            # Mostrar advertencia util en lugar del traceback de torch
            msg = str(caught[0].message)
            if "device count is zero" in msg:
                print("[device] XPU: Arc 140V detectada por Level Zero pero torch.xpu no puede acceder.")
                print("         Posible fix: ZES_ENABLE_SYSMAN=1 o actualizar intel-compute-runtime.")
    except Exception as e:
        print(f"[device] XPU no disponible: {e}")

    # Fallback a CUDA (Kaggle, Colab)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[device] CUDA: {name}")
        return "cuda"

    # CPU: Core Ultra 7 258V tiene AVX-512, util para smoke tests
    print("[device] CPU — AVX-512 disponible en Core Ultra 7 258V")
    print("         Para GPU real: usar Kaggle (T4 gratis, ~10h para plan completo)")
    return "cpu"


# ── Rotary Position Embedding ─────────────────────────────────────────────────

class LiteRtLayerNorm(nn.Module):
    """LayerNorm expresado con operaciones elementales exportables a LiteRT.

    Conserva los nombres ``weight`` y ``bias`` de ``nn.LayerNorm``. Por eso un
    checkpoint entrenado con la arquitectura original se carga sin migrarlo.
    La forma de normalización del modelo es siempre el último eje (D=256).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, tuple):
            if len(normalized_shape) != 1:
                raise ValueError("LiteRtLayerNorm solo admite un eje normalizado")
            normalized_shape = normalized_shape[0]
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


def replace_layer_norms_for_litert(module: nn.Module) -> None:
    """Sustituye recursivamente LayerNorm sin cambiar las claves del state dict."""
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            if not child.elementwise_affine:
                raise ValueError("El modelo requiere LayerNorm con weight y bias")
            replacement = LiteRtLayerNorm(child.normalized_shape, child.eps)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_layer_norms_for_litert(child)

class RoPE(nn.Module):
    """Rotary Position Embedding. Mas estable que PE absoluto para T variable."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        assert dim % 2 == 0, "dim debe ser par"
        theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        pos   = torch.arange(max_len).float()
        angles = torch.outer(pos, theta)          # [max_len, dim/2]
        self.register_buffer("cos_cache", torch.cos(angles))
        self.register_buffer("sin_cache", torch.sin(angles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] — aplica rotacion en las ultimas D dimensiones."""
        T   = x.size(1)
        cos = self.cos_cache[:T].unsqueeze(0)     # [1, T, D/2]
        sin = self.sin_cache[:T].unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]       # pares de dimensiones
        x_rot = torch.stack(
            [x1 * cos - x2 * sin,
             x1 * sin + x2 * cos], dim=-1
        ).flatten(-2)                              # [B, T, D]
        return x_rot


# ── Encoder ───────────────────────────────────────────────────────────────────

class LSAEncoder(nn.Module):
    """
    Encoder Transformer sobre secuencias de keypoints de manos.

    Input:  [B, T, input_dim]  donde input_dim = 126 (21 kp * 2 manos * 3 coords)
    Output: embeddings [B, T, d_model] + ctc_logits [B, T, vocab_size+1]
    """

    def __init__(
        self,
        input_dim: int   = 126,
        d_model: int     = 256,
        nhead: int       = 8,
        num_layers: int  = 6,
        dim_ffn: int     = 1024,
        dropout: float   = 0.1,
        vocab_size: int  = 7000,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.rope        = RoPE(d_model)
        self.dropout     = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ffn,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # Pre-norm: mas estable en entrenamiento largo
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        replace_layer_norms_for_litert(self.transformer)
        self.norm        = LiteRtLayerNorm(d_model)

        # CTC head: vocab_size + 1 (indice 0 = blank token)
        self.ctc_head = nn.Linear(d_model, vocab_size + 1)

    def forward(
        self,
        x: torch.Tensor,                              # [B, T, input_dim]
        src_key_padding_mask: torch.Tensor | None = None,  # [B, T] True = ignorar
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(self.rope(self.input_proj(x)))
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x, self.ctc_head(x)   # embeddings, ctc_logits


# ── Decoder ───────────────────────────────────────────────────────────────────

class LSADecoder(nn.Module):
    """
    Decoder autorregresivo: genera la traduccion token a token.

    tgt:    [B, S]    tokens de entrada (con BOS, sin EOS)
    memory: [B, T, d_model]
    Output: [B, S, vocab_size] logits
    """

    def __init__(
        self,
        vocab_size: int  = 7000,
        d_model: int     = 256,
        nhead: int       = 8,
        num_layers: int  = 3,
        dim_ffn: int     = 512,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale     = math.sqrt(d_model)
        self.dropout   = nn.Dropout(dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ffn,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        replace_layer_norms_for_litert(self.transformer)
        self.norm        = LiteRtLayerNorm(d_model)
        self.out_proj    = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,      # [B, S]
        memory: torch.Tensor,   # [B, T, d_model]
        tgt_mask: torch.Tensor | None            = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.dropout(self.embedding(tgt) * self.scale)
        out = self.transformer(
            emb, memory,
            tgt_mask                 = tgt_mask,
            tgt_key_padding_mask     = tgt_key_padding_mask,
            memory_key_padding_mask  = memory_key_padding_mask,
        )
        return self.out_proj(self.norm(out))    # [B, S, vocab_size]

    @torch.no_grad()
    def greedy_decode(
        self,
        memory: torch.Tensor,   # [B, T, d_model]
        bos_id: int,
        eos_id: int,
        max_len: int = 64,
    ) -> list[list[int]]:
        """Decodificacion greedy para inferencia rapida."""
        B      = memory.size(0)
        device = memory.device
        tokens  = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done    = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            S        = tokens.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
            logits   = self.forward(tokens, memory, tgt_mask=tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1)    # [B]
            tokens   = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            done    |= (next_tok == eos_id)
            if done.all():
                break

        results = []
        for i in range(B):
            seq = tokens[i, 1:].tolist()   # sin BOS
            if eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            results.append(seq)
        return results


# ── Modelo completo ────────────────────────────────────────────────────────────

class LSAModel(nn.Module):
    """Wrapper del modelo completo: Encoder + Decoder."""

    def __init__(self, vocab_size: int = 7000, **enc_kwargs):
        super().__init__()
        self.encoder = LSAEncoder(vocab_size=vocab_size, **enc_kwargs)
        self.decoder = LSADecoder(vocab_size=vocab_size)

    def forward(
        self,
        keypoints: torch.Tensor,            # [B, T, 126]
        tgt_tokens: torch.Tensor,           # [B, S]
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          dec_logits:  [B, S, vocab_size]   para CrossEntropy
          ctc_logits:  [B, T, vocab_size+1] para CTC Loss (opcional)
        """
        embeddings, ctc_logits = self.encoder(keypoints, src_padding_mask)

        S        = tgt_tokens.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            S, device=keypoints.device
        )
        dec_logits = self.decoder(
            tgt_tokens, embeddings,
            tgt_mask                = tgt_mask,
            tgt_key_padding_mask    = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask,
        )
        return dec_logits, ctc_logits

    def count_params(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        enc   = sum(p.numel() for p in self.encoder.parameters())
        dec   = sum(p.numel() for p in self.decoder.parameters())
        return (
            f"Parametros totales: {total/1e6:.1f}M  "
            f"(Encoder: {enc/1e6:.1f}M | Decoder: {dec/1e6:.1f}M)"
        )


if __name__ == "__main__":
    import json
    from config import VOCAB_PATH

    vocab_size = 7000
    if VOCAB_PATH.exists():
        vocab_size = len(json.loads(VOCAB_PATH.read_text()))

    model = LSAModel(vocab_size=vocab_size)
    print(model.count_params())

    # Smoke test
    B, T, S = 2, 75, 12
    kps     = torch.randn(B, T, 126)
    tgt     = torch.randint(0, vocab_size, (B, S))
    dec_out, ctc_out = model(kps, tgt)
    print(f"Decoder output: {dec_out.shape}")   # [2, 12, vocab_size]
    print(f"CTC output:     {ctc_out.shape}")   # [2, 75, vocab_size+1]
    print("OK")
