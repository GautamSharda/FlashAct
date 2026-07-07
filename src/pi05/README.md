# π0.5 Megakernel — fastest verified end-to-end π0.5 on RTX 5090

The entire 10-step flow-matching denoise loop of π0.5's action expert (18-layer
Gemma-300M: adaRMS time conditioning, MQA attention, MLP, Euler updates) runs as
**one cooperative CUDA kernel launch** at FP8-W8A16. Combined with FlashRT's FP8
vision/encoder front-end via a zero-cost in-kernel KV handoff, it is the fastest
end-to-end π0.5 implementation we know of on a single RTX 5090 — verified for
both latency and closed-loop task success.

## Verified results (RTX 5090, CUDA 13, pi05_libero, 2 views, 10 steps)

| stack | e2e wall p50 | LIBERO spatial (10 tasks × 50 trials, same harness/seed) |
|---|---|---|
| FlashRT full FP8 (fastest shipped config) | 19.84 ms | 493/500 = 98.6% (reproduces their published 98.3%) |
| **Hybrid: FlashRT front-end + this megakernel** | **19.06 ms (−3.9%)** | **491/500 = 98.2%** (statistically equivalent) |

Both stacks measured back-to-back in the same process, identical checkpoint,
prompts, seeds, and measurement boundary. Loop fidelity vs the bf16 openpi
reference: cosine 0.99995. The standalone loop (9.5 ms) also beats FlashRT's
FP8 action-decoder path (10.2 ms) with tighter numerics.

## Files

| file | what it is |
|---|---|
| `mk6.cu` | the megakernel: full denoise loop in one `cudaLaunchCooperativeKernel`, FP8 weights dequantized in-register (`cvt.rn.f16x2.e4m3x2`), `mma.m16n8k16` tensor cores, `cp.async` double-buffered weight tiles, grid-wide sync between stages; prologue ingests FlashRT's KV cache (RoPE de-interleave + V transpose + bf16→f16) in ~10 µs |
| `binding.cpp` | PyTorch extension binding (launch signature, dynamic smem) |
| `pack_weights.py` | weight packing/quantization + RoPE tables + KV compaction |
| `hybrid_model.py` | the deployable hybrid: FlashRT-compatible `predict(images, prompt)`; captures their vision+encoder as an encoder-only CUDA graph, hands enc_K/enc_V to the kernel |

Benchmarks and the LIBERO eval live in `benchmarks/pi05/`:

| file | what it is |
|---|---|
| `benchmarks/pi05/hybrid_e2e.py` | e2e latency benchmark: hybrid vs FlashRT full-FP8, same process, CUDA-synced wall p50 |
| `benchmarks/pi05/eval_numerics.py` | standalone loop benchmark + correctness gate vs bf16 openpi reference (cos 0.99995) |
| `benchmarks/pi05/eval_libero.py` | LIBERO eval harness (FlashRT's `eval_libero.py` + `HYBRID=1` switch, `--trial_offset`, per-step stall watchdog, osmesa) |
| `benchmarks/pi05/chunked_eval.sh` | eval driver: 15 trials per subprocess (LIBERO's sim hangs after ~18 env resets per process on x86+osmesa), resumable per-chunk JSONs, final aggregate |

## Running

Requirements: RTX 5090 (sm_120), CUDA 13, torch 2.10+cu130, ninja, FlashRT built
in `third_party/FlashRT`, openpi in `third_party/openpi`, LIBERO + robosuite 1.4.1 +
mujoco 3.1.6 for the eval. Data paths (override via env):

- `MK_WEIGHTS` — merged fp32 state dict of pi05_libero (default points to our volume)
- `MK_CHECKPOINT` — FlashRT-format pi05_libero checkpoint dir (safetensors + assets)
- `MK_PYTHON` — python for eval subprocesses (needs the full stack above)

```bash
cd ../../benchmarks/pi05  # benchmarks live here

# e2e latency (idle GPU): prints hybrid + FlashRT wall p50
python hybrid_e2e.py

# standalone loop benchmark + correctness vs bf16 reference
python eval_numerics.py

# LIBERO spatial, 500 trials per stack
bash chunked_eval.sh flashrt            # baseline
HYBRID=1 bash chunked_eval.sh hybrid    # ours
```

## Key handoff detail

FlashRT stores K with interleaved RoPE pairs `(2d, 2d+1)`; openpi/HF use split-half
`(d, d+128)`. The kernel prologue de-interleaves (verified against bf16 ground truth,
cos 0.9943), transposes V, and converts bf16→f16 — on-GPU, inside the same launch,
replacing ~1 ms of torch-side transforms. Decoder RoPE positions start at
`512 + prompt_len`. The encoder-only CUDA graph must include FlashRT's
`_copy_lang_embeds_to_encoder_x` staging step — omitting it silently corrupts the
language K rows.
