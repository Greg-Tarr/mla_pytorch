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

Tensor = torch.Tensor


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, False)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.to(input.dtype))


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.register_buffer("inv_freq", (1 / base) ** (torch.arange(0, dim, 2) / dim))

    def forward(self, x: Tensor, cache_len: int):
        T = x.shape[1]
        t = torch.arange(T, device=x.device) + cache_len
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


def causal_mod(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


class MLAttention(nn.Module):
    """ """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Configuration
    DEVICE = "cuda"
    D_MODEL = 8192
    N_HEAD = D_MODEL // 128
    T = 4096

    PREFILL_LEN = 2048
    GEN_LEN = 2048

    model = MLAttention().to(DEVICE).bfloat16().eval()
    model = torch.compile(model)

    # Test parameters
    prompt = torch.randn((1, PREFILL_LEN, GEN_LEN), device=DEVICE, dtype=torch.float32)

    # Warmup model
    print("\nWarming up model...")
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(25):
            out = model(prompt)
    torch.cuda.synchronize()
    print(f"Output shape: {out.shape}")
