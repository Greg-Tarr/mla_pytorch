# type: ignore

"""

Simple implementation of Multi Latent Attention from the Deepseek V2 paper
https://arxiv.org/abs/2405.04434

Done without reference to the source code, so may not be faithful to the original
work. Basic idea is to compress the KV cache into a latent vector.

This has a very minor impact on model perplexity or performance, but it
significantly reduces the size of the KV cache (by up to 93.3 %) which provides
substantial memory efficiency benefits during inference.

> nvidia-smi

    Fri Jan 24 15:06:46 2025
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce RTX 4090        On  |   00000000:0A:00.0 Off |                  Off |
    | 31%   48C    P8             20W /  450W |       2MiB /  24564MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

> uv run mla_pytorch/mla.py

    Warming up model...
    Output shape: torch.Size([1, 2048, 8192])

    Benchmarking With Cache prefill...
    Avg time per step: 12.58ms
    Tokens per second (prefill): 162853.3

    Testing chunked/autoregressive generation with With Cache...
    Maximum difference between full and chunked: 0.00017948
    Mean difference between full and chunked: 0.00000391
    Outputs match: True

    Testing generation with With Cache...

    Total Time: 12649.47ms (161.9 tok/sec)
    Time per token: 6.18ms
    Max throughput: 161.9 tok/sec
    With Cache: 12649.47ms (161.9 tokens/sec)

    Warming up model...
    Output shape: torch.Size([1, 2048, 8192])

    Benchmarking No Cache prefill...
    Avg time per step: 13.05ms
    Tokens per second (prefill): 156937.1

    Testing generation with No Cache...

    Total Time: 44057.54ms (46.5 tok/sec)
    Time per token: 21.51ms
    Max throughput: 46.5 tok/sec
    No Cache: 44057.54ms (46.5 tokens/sec)

    Comparing with and without cache speeds.

    Testing generation with No Cache...

    Total Time: 44062.65ms (46.5 tok/sec)
    Time per token: 21.51ms
    Max throughput: 46.5 tok/sec

    Testing generation with With Cache...

    Total Time: 11014.80ms (185.9 tok/sec)
    Time per token: 5.38ms
    Max throughput: 185.9 tok/sec

    Speedup from KV cache: 4.00x
"""

import torch
from torch import nn

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)


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


class NoKVCache(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = 0

    def zero(self) -> None:
        self.offset = 0

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        self.offset += c_kv.size(1)
        return c_kv


class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple) -> None:
        super().__init__()
        self.register_buffer('data', torch.zeros(kv_cache_shape))
        self.zero()

    def zero(self) -> None:
        self.offset = 0
        self.data.zero_()

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        assert self.offset + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(c_kv.dtype)
        self.data[
            :, self.offset : self.offset + c_kv.size(1), :
        ] = c_kv
        self.offset += c_kv.size(1)

        return self.data[:, :self.offset]


def create_causal_mod(cache_len: int):
    def causal_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + cache_len

    return causal_mod


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

    def forward(self, x: torch.Tensor, kv_cache: KVCache | NoKVCache) -> tuple[torch.Tensor, torch.Tensor]:
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
        c_kv[..., :self.d_kv] = self.kv_norm(c_kv[..., :self.d_kv]) # pre-cache norm
        c_kv = kv_cache(c_kv)
        kv_n, k_r = torch.split(c_kv, [self.d_kv, self.d_qk_r], dim=-1)

        # Project KV and split
        kv = self.W_u_kv(kv_n)
        kv = kv.view(B, -1, self.n_heads, self.d_head + self.d_qk_n).transpose(1, 2)
        k, v = torch.split(kv, [self.d_qk_n, self.d_head], dim=-1)

        # Run rotary for K
        k_r = k_r.view(B, -1, 1, self.d_qk_r).transpose(1, 2)
        k_r = self.rotary(k_r)
        k_r = k_r.repeat(1, self.n_heads, 1, 1)

        # Split to heads
        q = torch.cat([q, q_r], dim=-1)
        k = torch.cat([k, k_r], dim=-1)

        # Mask & Attention
        causal_mod = create_causal_mod(kv_cache.offset - T)
        block_mask = create_block_mask(
            causal_mod, B=None, H=None, Q_LEN=q.size(2), KV_LEN=k.size(2)
        )
        y = flex_attention(q, k, v, block_mask=block_mask, kernel_options={
            # for a <40GB card (gpu poor)
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_M1": 16,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 16,
        })
        y = y.transpose(1, 2).contiguous().view_as(x)

        # Project output
        out = self.W_o(y)

        return out

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Model configuration
    D_MODEL = 8192
    N_HEAD = D_MODEL // 128
    MAX_SEQ_LEN = 4096
    DTYPE = torch.float32
    DEVICE = "cuda"
    COMPILE = True

    # Test prompt
    PROMPT_LEN = 2048
    GEN_LEN = 2048
    prompt = torch.randn((1, PROMPT_LEN, D_MODEL), device=DEVICE, dtype=DTYPE)

    # Create both non-cached and cached models
    model_nocache = (
        MLAttention(
            d_model=D_MODEL,
            n_head=N_HEAD
        )
        .to(DEVICE)
        .to(DTYPE)
        .eval()
    )
    no_cache = NoKVCache()

    model_cache = (
        MLAttention(
            d_model=D_MODEL,
            n_head=N_HEAD
        )
        .to(DEVICE)
        .to(DTYPE)
        .eval()
    )
    kv_cache = KVCache((1, MAX_SEQ_LEN, model_cache.d_kv + model_cache.d_qk_r)).to(DEVICE)

    if compile:
        model_nocache = torch.compile(model_nocache)
        model_cache = torch.compile(model_cache)

    def warmup(model: MLAttention, cache: KVCache | NoKVCache):
        print("\nWarming up model...")
        with torch.no_grad():
            for _ in range(25):
                cache.zero()
                out = model(prompt, cache)
        torch.cuda.synchronize()
        print(f"Output shape: {out.shape}")
        return out

    def benchmark_prefill(model: MLAttention, name: str, cache: KVCache | NoKVCache) -> torch.Tensor:
        print(f"\nBenchmarking {name} prefill...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        with torch.no_grad():
            for _ in range(1000):
                cache.zero()
                out = model(prompt, cache)
        end_time.record()

        torch.cuda.synchronize()
        elapsed_ms = start_time.elapsed_time(end_time)
        tokens_per_sec = (prompt.size(1) * 1000 * 1000) / elapsed_ms
        print(f"Avg time per step: {elapsed_ms/1000:.2f}ms")
        print(f"Tokens per second (prefill): {tokens_per_sec:.1f}")
        return out

    def test_chunked_generation(model: MLAttention, name: str, cache: KVCache | NoKVCache) -> tuple[torch.Tensor, torch.Tensor]:
        print(f"\nTesting chunked/autoregressive generation with {name}...")

        # Test with full prompt
        cache.zero()
        with torch.no_grad():
            out_full = model(prompt, cache)

        # Test with chunked prompt
        split_idx = PROMPT_LEN // 2
        prompt_chunk1 = prompt[:, :split_idx]
        prompt_chunk2 = prompt[:, split_idx:]

        cache.zero()
        with torch.no_grad():
            out_chunk1 = model(prompt_chunk1, cache)
            out_chunk2 = model(prompt_chunk2, cache)
            out_chunked = torch.cat([out_chunk1, out_chunk2], dim=1)

        # Compare results
        max_diff = torch.max(torch.abs(out_full - out_chunked))
        mean_diff = torch.mean(torch.abs(out_full - out_chunked))
        print(f"Maximum difference between full and chunked: {max_diff:.8f}")
        print(f"Mean difference between full and chunked: {mean_diff:.8f}")
        print(f"Outputs match: {mean_diff < 1e-5}")
        return out_full, out_chunked

    def test_generation(model: MLAttention, name: str, cache: KVCache | NoKVCache) -> tuple[float, float]:
        print(f"\nTesting generation with {name}...")
        tokens = prompt.clone()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        cache.zero()

        start_time.record()
        with torch.no_grad():
            tokens = model(tokens, cache)
            pred = tokens

            for i in range(GEN_LEN):
                if isinstance(cache, KVCache):  # Using KVCache
                    pred = model(pred[:, -1:], cache)
                else:  # Using NoKVCache
                    cache.zero()
                    pred = model(tokens, cache)
                    tokens = torch.cat([tokens, pred[:, -1:]], dim=1)

        end_time.record()
        torch.cuda.synchronize()
        elapsed_ms = start_time.elapsed_time(end_time)
        tokens_per_sec = (GEN_LEN * 1000) / elapsed_ms
        print(f"\nTotal Time: {elapsed_ms:.2f}ms ({GEN_LEN/elapsed_ms*1000:.1f} tok/sec)")
        print(f"Time per token: {elapsed_ms/GEN_LEN:.2f}ms")
        print(f"Max throughput: {min(GEN_LEN/elapsed_ms*1000,PROMPT_LEN*1000/elapsed_ms):.1f} tok/sec")
        return elapsed_ms, tokens_per_sec

    # Test both models
    for model, name, cache in [(model_cache, "With Cache", kv_cache), (model_nocache, "No Cache", no_cache)]:
        # Warmup
        warmup(model, cache)

        # Benchmark prefill
        benchmark_prefill(model, name, cache)

        # Test chunked generation
        if name == "With Cache":  # Only test autoregression with cache addition
            test_chunked_generation(model, name, cache)

        # Test generation
        time_ms, tps = test_generation(model, name, cache)
        print(f"{name}: {time_ms:.2f}ms ({tps:.1f} tokens/sec)")

    # Compare speedup
    print("\nComparing with and without cache speeds.")
    nocache_time, _ = test_generation(model_nocache, "No Cache", no_cache)
    cache_time, _ = test_generation(model_cache, "With Cache", kv_cache)
    speedup = nocache_time / cache_time
    print(f"\nSpeedup from KV cache: {speedup:.2f}x")
