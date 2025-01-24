# type: ignore

"""

Simple implementation of Multi Latent Attention from the Deepseek V2 paper
https://arxiv.org/abs/2405.04434

Done without reference to the source code, so may not be faithful to the original
work. Basic idea is to compress the KV cache into a latent vector.

This has no direct impact on model perplexity or performance, however it
significantly reduces the size of the KV cache (by up to 93.3 %) which provides
substantial memory efficiency benefits during inference.

"""

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.register_buffer("inv_freq", (1 / base) ** (torch.arange(0, dim, 2) / dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        t = torch.arange(T, device=x.device)
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


def causal_mod(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


class MLAttention(nn.Module):
    """Multi Latent Attention module - compresses key/value cache into latent vectors for memory efficiency.

    Implementation inspired by the Deepseek V2 paper. Projects Q/K/V through latent dimensions and supports rotary
    embeddings to enable significant reduction in memory usage with minimal performance impact."""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_head
        self.d_head = d_model // n_head
        self.d_q = d_model // 3
        self.d_kv = d_model // 2
        self.d_qk_n = self.d_head // 2
        self.d_qk_r = self.d_head // 2

        self.W_d_kv = nn.Linear(self.d_model, self.d_kv + self.d_qk_r)
        self.W_u_kv = nn.Linear(self.d_kv, self.d_model + (self.n_heads * self.d_qk_n))
        self.kv_norm = nn.LayerNorm(self.d_kv)

        self.W_d_q = nn.Linear(self.d_model, self.d_q)
        self.W_u_q = nn.Linear(self.d_q, self.d_model)
        self.q_norm = nn.LayerNorm(self.d_q)

        self.rotary = Rotary(self.d_qk_r)

        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Q Projections
        c_q = self.W_d_q(x)
        c_q = self.q_norm(c_q)
        q = self.W_u_q(c_q)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, q_r = torch.split(q, [self.d_qk_n, self.d_qk_r], dim=-1)

        # Apply rotary to q_rope
        q_r = self.rotary(q_r)

        # KV Projections
        c_kv = self.W_d_kv(x)
        kv_n, k_r = torch.split(c_kv, [self.d_kv, self.d_qk_r], dim=-1)

        # Project KV and split
        kv_n = self.kv_norm(kv_n)
        kv = self.W_u_kv(kv_n)
        kv = kv.view(B, T, self.n_heads, self.d_head + self.d_qk_n).transpose(1, 2)
        k, v = torch.split(kv, [self.d_qk_n, self.d_head], dim=-1)

        # Run rotary for K
        k_r = k_r.view(B, T, 1, self.d_qk_r).transpose(1, 2)
        k_r = self.rotary(k_r)
        k_r = k_r.repeat(1, self.n_heads, 1, 1)

        # Split to heads
        q = torch.cat([q, q_r], dim=-1)
        k = torch.cat([k, k_r], dim=-1)

        # Scale and Attention
        causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=x.device), diagonal=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask
        )
        y = y.transpose(1, 2).reshape(B, T, C)

        # Project output
        out = self.W_o(y)

        return out

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Configuration
    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    D_MODEL = 8192
    N_HEAD = D_MODEL // 128
    T = 4096

    PREFILL_LEN = 2048
    GEN_LEN = 2048

    model = MLAttention(
        d_model=D_MODEL,
        n_head=N_HEAD,
    ).to(DEVICE).to(DTYPE).eval()
    model = torch.compile(model)

    # Test parameters
    prompt = torch.randn((1, PREFILL_LEN, D_MODEL), device=DEVICE, dtype=DTYPE)

    # Warmup model
    print("\nWarming up model...")
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(25):
            out = model(prompt)
    torch.cuda.synchronize()
    print(f"Output shape: {out.shape}")
