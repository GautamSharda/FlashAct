# MolmoAct2 Action-Expert Megakernel — RTX 5090

The entire 10-step flow-matching loop of the MolmoAct2 continuous action expert —
36 DiT blocks (adaLN modulation, self-attention over the action chunk,
cross-attention to per-layer projected VLM KV, silu-gated MLP), final layer and
Euler updates — as **ONE cooperative CUDA kernel launch** per action chunk.

Model: `allenai/MolmoAct2-SO100_101` (5B Molmo2-ER backbone + 36-layer/768-hidden
action expert), bf16. Scenario: 2 RealSense views, SO100 6-dim state, ~493 valid
VLM context tokens, horizon 30, 10 flow steps, batch 1.

## Results (RTX 5090, sm_120, 170 SMs)

Flow loop (batch 1, same-run pair from `eval_numerics.py`):

| impl | loop p50 | speedup | cos vs eager | max abs diff |
|---|---|---|---|---|
| HF eager loop (model's own cached modulations) | 201.4 ms | 1.0x | -- | -- |
| **megakernel** | **30.75 ms** | **6.5x** | 0.999997 | 0.0055 |

The eager loop is host-dispatch-bound (148-590 ms across pods depending on CPU);
the megakernel is GPU-bound and stable (~30.8 ms on every full 5090 tested).

End-to-end `predict_action` (preprocessing + backbone prefill + flow loop +
unnormalize, identical inputs, back-to-back on the same box) from `eval_e2e.py`,
which replaces only `_run_action_flow_loop` with the megakernel:

| path | e2e p50 | speedup | cos vs eager |
|---|---|---|---|
| HF eager | 204.9 ms | 1.0x | -- |
| authors' CUDA-graph (their best published technique) | 139.3 ms | 1.47x | 1.000000 |
| **megakernel** | **90.9 ms** | **2.26x** (**1.53x vs CUDA-graph**) | 0.9999978 |

Raw outputs: `benchmarks/molmoact2/results.json`, `benchmarks/molmoact2/e2e_results.json`.
The launch grid auto-clamps to cooperative capacity, so the kernel also runs on
smaller sm_120 parts (verified on an 82-SM RTX PRO 4500: 1.34x vs CUDA-graph).

Only the flow loop is our contribution — the backbone prefill runs the model's
unmodified HF code in all paths.

## Files

| file | what it is |
|---|---|
| `mk_ma2.cu` | the megakernel: 36 DiT blocks × 10 steps in one launch; mma.m16n8k16 + cp.async double-buffered weight tiles; adaLN (shift/scale/gate × self/cross/mlp) from the model's own modulation cache; per-layer cross-KV compacted to valid tokens, V transposed; qk-norm + RoPE + 30-token self-attention fused per mini-stage; biases folded into mma epilogues; action-dim padding masked in the Euler update |
| `binding.cpp` | PyTorch extension binding |
| `../../benchmarks/molmoact2/eval_numerics.py` | packs weights, verifies vs HF eager reference (cos ~0.9999975), times the loop |
| `../../benchmarks/molmoact2/eval_e2e.py` | full `predict_action` wall-clock: eager vs authors' CUDA-graph vs megakernel (monkeypatches `_run_action_flow_loop` only; per-call cross-KV packing inside the timed region) |

## Running

```bash
export MA2_MODEL=/path/to/MolmoAct2-SO100_101   # HF snapshot with remote code
cd benchmarks/molmoact2
python eval_numerics.py   # correctness gate + loop timing
python eval_e2e.py        # three-way e2e comparison -> e2e_results.json
```

Requires: RTX 5090 (sm_120), torch cu13x, ninja, transformers ≥ 5.x (the pi0.5 work
pins 4.53.2 — use a separate venv).
