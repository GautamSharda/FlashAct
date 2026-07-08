"""End-to-end MolmoAct2 predict_action benchmark: eager vs the authors' CUDA-graph
path vs the megakernel flow loop.

Boundary: full ``model.predict_action`` wall-clock (preprocessing + backbone prefill +
flow loop + unnormalize), identical inputs for all three paths. The megakernel path
monkeypatches ``_run_action_flow_loop`` only — everything else is the model's own
code. Per-call cross-KV packing runs inside the timed region; modulation packing is
cached across calls exactly like the model's own modulation cache (fixed schedule).
"""
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.cpp_extension import load
from transformers import AutoModelForImageTextToText, AutoProcessor

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERN = os.path.join(_HERE, "../../src/molmoact2")
MODEL = os.environ.get("MA2_MODEL", "/network_volume/megakernels/molmoact2/model")
device = "cuda"
D, HEADS, HD2, F, NL, T, AD = 768, 8, 96, 3072, 36, 32, 32
NB, NT = 170, 256

ext = load(name="mk_ma2", sources=[f"{_KERN}/binding.cpp", f"{_KERN}/mk_ma2.cu"],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120,code=sm_120"],
           verbose=False)
print("ext built", flush=True)

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True,
                                                    dtype=torch.bfloat16).to(device).eval()
core = model.model
expert = core._require_action_expert()
imgs = [Image.open(f"{MODEL}/assets/sample_realsense_{v}_rgb.png").convert("RGB") for v in ("side", "top")]
state = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.5], dtype=np.float32)
task = "pick up the cube and place it on the pink pad"

# ---- pack weights once (deployment-time setup, analogous to the baseline's graph capture) ----
def cat_layers(fn, dtype=torch.bfloat16):
    return torch.stack([fn(b).to(dtype) for b in expert.blocks]).contiguous()

w_qkv = cat_layers(lambda b: b.self_attn.qkv.weight)
b_qkv = cat_layers(lambda b: b.self_attn.qkv.bias, torch.float32)
w_so = cat_layers(lambda b: b.self_attn.out_proj.weight)
b_so = cat_layers(lambda b: b.self_attn.out_proj.bias, torch.float32)
w_cq = cat_layers(lambda b: b.cross_attn.q_proj.weight)
b_cq = cat_layers(lambda b: b.cross_attn.q_proj.bias, torch.float32)
w_co = cat_layers(lambda b: b.cross_attn.out_proj.weight)
b_co = cat_layers(lambda b: b.cross_attn.out_proj.bias, torch.float32)
wg = torch.stack([b.mlp.gate_proj.weight for b in expert.blocks])
wu = torch.stack([b.mlp.up_proj.weight for b in expert.blocks])
w_gu = torch.stack([wg, wu], dim=2).reshape(NL, 2 * F, D).to(torch.bfloat16).contiguous()
bg = torch.stack([b.mlp.gate_proj.bias for b in expert.blocks])
bu = torch.stack([b.mlp.up_proj.bias for b in expert.blocks])
b_gu = torch.stack([bg, bu], dim=2).reshape(NL, 2 * F).float().contiguous()
w_dn = cat_layers(lambda b: b.mlp.down_proj.weight)
b_dn = cat_layers(lambda b: b.mlp.down_proj.bias, torch.float32)
w_ain = expert.action_embed.weight.float().contiguous()
b_ain = expert.action_embed.bias.float().contiguous()
w_aout = expert.final_layer.linear.weight.to(torch.bfloat16).contiguous()
b_aout = expert.final_layer.linear.bias.float().contiguous()
ws = [w_qkv, w_so, w_cq, w_co, w_gu, w_dn]
bs = [b_qkv, b_so, b_cq, b_co, b_gu, b_dn]

x_t = torch.zeros(T, AD, dtype=torch.float32, device=device)
x = torch.zeros(T, D, dtype=torch.float32, device=device)
xp = torch.zeros(3, T, D, dtype=torch.float32, device=device)
bufs = [torch.zeros(T, D, dtype=torch.bfloat16, device=device) for _ in range(8)]
bufs.append(torch.zeros(T, F, dtype=torch.bfloat16, device=device))
sc_cyc = torch.zeros(16, dtype=torch.int64, device=device)
cosf = torch.zeros(T, 48, dtype=torch.float32, device=device)
sinf = torch.zeros(T, 48, dtype=torch.float32, device=device)
_rope_done = {}
_mods_cache = {}
_kv_state = {}


def _pack_mods(modulations, steps):
    key = (id(modulations), steps)
    if key in _mods_cache:
        return _mods_cache[key]
    mods = torch.zeros(steps, NL, 3, 3, D, dtype=torch.float32, device=device)
    modf = torch.zeros(steps, 2, D, dtype=torch.float32, device=device)
    for s in range(steps):
        for l in range(NL):
            (sh_msa, sc_msa, g_msa, sh_mca, sc_mca, g_mca,
             sh_mlp, sc_mlp, g_mlp) = modulations[s].block_modulations[l]
            for n, (scv, shv, gv) in enumerate([(sc_msa, sh_msa, g_msa),
                                                (sc_mca, sh_mca, g_mca),
                                                (sc_mlp, sh_mlp, g_mlp)]):
                mods[s, l, n, 0] = scv[0].float()
                mods[s, l, n, 1] = shv[0].float()
                mods[s, l, n, 2] = gv[0].float()
        fsh, fsc = modulations[s].final_modulation
        modf[s, 0] = fsc[0].float()
        modf[s, 1] = fsh[0].float()
    _mods_cache.clear()
    _mods_cache[key] = (mods, modf)
    return mods, modf


def mk_flow_loop(self, inputs, steps):
    ctx = inputs.context
    cm = ctx.cross_mask
    if cm is None:
        L_full = ctx.kv_contexts[0][0].shape[1]
        vmask = torch.ones(L_full, dtype=torch.bool, device=device)
    else:
        cmv = cm[0, 0, 0].float()
        vmask = cmv == cmv.max()
    Lk = int(vmask.sum().item())
    Lkmax = (Lk + 15) // 16 * 16
    if _kv_state.get("Lkmax") != Lkmax:
        _kv_state["kctx"] = torch.zeros(NL, Lkmax, D, dtype=torch.bfloat16, device=device)
        _kv_state["vtctx"] = torch.zeros(NL, D, Lkmax, dtype=torch.bfloat16, device=device)
        _kv_state["scores"] = torch.zeros(HEADS, T, Lkmax, dtype=torch.float32, device=device)
        _kv_state["probs"] = torch.zeros(HEADS, T, Lkmax, dtype=torch.bfloat16, device=device)
        _kv_state["Lkmax"] = Lkmax
    kctx, vtctx = _kv_state["kctx"], _kv_state["vtctx"]
    for l, (kc, vc) in enumerate(ctx.kv_contexts):
        kctx[l, :Lk] = kc[0][vmask].reshape(Lk, D).to(torch.bfloat16)
        vtctx[l, :, :Lk] = vc[0][vmask].reshape(Lk, D).t().to(torch.bfloat16)

    traj = inputs.trajectory
    Tv = traj.shape[1]
    if not _rope_done:
        rope = expert.blocks[0].self_attn.rope.build_cache(seq_len=Tv, device=torch.device(device),
                                                           dtype=torch.bfloat16)
        cosf[:Tv] = rope[0][0, 0].float()
        sinf[:Tv] = rope[1][0, 0].float()
        _rope_done["done"] = True

    mods, modf = _pack_mods(inputs.modulations, int(steps))
    adp = inputs.action_dim_is_pad
    adim = int((~adp[0].bool()).sum().item()) if adp is not None else AD
    x_t.zero_()
    x_t[:Tv] = traj[0].float()
    ext.launch(ws, bs, mods, modf, cosf, sinf, w_ain, b_ain, w_aout, b_aout,
               kctx, vtctx, Lk, Lkmax, Tv, adim, int(steps), x_t, x, xp, sc_cyc,
               bufs, _kv_state["scores"], _kv_state["probs"], NB, NT)
    return x_t[:Tv].to(traj.dtype)[None]


def predict(graph):
    return model.predict_action(processor=processor, images=imgs, task=task, state=state,
                                norm_tag="so100_so101_molmoact2",
                                inference_action_mode="continuous",
                                generator=torch.Generator(device=device).manual_seed(42),
                                enable_cuda_graph=graph)


def timeit(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def actions_of(out):
    if isinstance(out, dict):
        for k in ("actions", "action"):
            if k in out:
                out = out[k]
                break
    if torch.is_tensor(out):
        out = out.detach().float().cpu()
    return np.asarray(out, dtype=np.float32)


with torch.no_grad():
    out_eager = predict(False)
    t_eager = timeit(lambda: predict(False), iters=5)

    core._run_action_flow_loop = mk_flow_loop.__get__(core)
    out_mk = predict(False)
    t_mk = timeit(lambda: predict(False), iters=10)
    core._run_action_flow_loop = type(core)._run_action_flow_loop.__get__(core)

    out_graph = predict(True)
    t_graph = timeit(lambda: predict(True), iters=10)

a_e, a_m, a_g = (actions_of(o).ravel() for o in (out_eager, out_mk, out_graph))
cos_mk = float(a_e @ a_m / (np.linalg.norm(a_e) * np.linalg.norm(a_m) + 1e-9))
cos_g = float(a_e @ a_g / (np.linalg.norm(a_e) * np.linalg.norm(a_g) + 1e-9))
res = {
    "e2e_eager_ms_p50": float(np.median(t_eager)),
    "e2e_cudagraph_ms_p50": float(np.median(t_graph)),
    "e2e_megakernel_ms_p50": float(np.median(t_mk)),
    "speedup_vs_eager": float(np.median(t_eager) / np.median(t_mk)),
    "speedup_vs_cudagraph": float(np.median(t_graph) / np.median(t_mk)),
    "cos_mk_vs_eager": cos_mk,
    "cos_cudagraph_vs_eager": cos_g,
    "actions_eager_row0": actions_of(out_eager).reshape(-1, actions_of(out_eager).shape[-1])[0][:6].tolist(),
    "actions_mk_row0": actions_of(out_mk).reshape(-1, actions_of(out_mk).shape[-1])[0][:6].tolist(),
}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{_HERE}/e2e_results.json", "w"), indent=1)
