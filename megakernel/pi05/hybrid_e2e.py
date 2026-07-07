"""Hybrid fastest-e2e pi0.5: FlashRT FP8 front-end (vision+encoder, CUDA graph)
feeding OUR single-kernel FP8 megakernel denoise loop. Measured like their wall."""
_HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
import ctypes, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, _HERE)
sys.path.insert(0, _HERE)
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
NB, NT = 170, 256
D, H, HD, F, NL, T, AD, QD = 1024, 8, 256, 4096, 18, 16, 32, 2048
TV = 10
device = torch.device("cuda")

ext = load(name="mk_v6", sources=[f"{HERE}/binding.cpp", f"{HERE}/mk6.cu"],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120,code=sm_120"], verbose=False)

# ---- pack our loop weights directly from the merged state dict ----
sd = torch.load(__import__('os').environ.get("MK_WEIGHTS", "/network_volume/megakernels/pi05/weights/pi05_libero_merged_fp32.pt"), map_location="cpu", weights_only=True)
P = "paligemma_with_expert.gemma_expert.model.layers."
def L(i, n): return sd[P + str(i) + "." + n].to(device)
def stack(n): return torch.stack([L(i, n) for i in range(NL)])
wq, wk, wv = stack("self_attn.q_proj.weight"), stack("self_attn.k_proj.weight"), stack("self_attn.v_proj.weight")
wo = stack("self_attn.o_proj.weight").half().contiguous()
wg, wu = stack("mlp.gate_proj.weight"), stack("mlp.up_proj.weight")
wdn = stack("mlp.down_proj.weight").half().contiguous()
p_ = torch.arange(1024, device=device); hh, jj = p_ // 128, p_ % 128
qsrc = torch.stack([hh * 256 + jj, hh * 256 + jj + 128], 1).reshape(-1)
jk = torch.arange(128, device=device); ksrc = torch.stack([jk, jk + 128], 1).reshape(-1)
w1 = torch.cat([wq[:, qsrc], wk[:, ksrc], wv], dim=1).half().contiguous()
w2 = torch.stack([wg, wu], dim=2).reshape(NL, 2 * F, D).half().contiguous()
def quant(w):
    s = (w.float().abs().amax(dim=-1).clamp(min=1e-8) / 448.0)
    q = (w.float() / s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).contiguous(), s.float().contiguous()
w1q, s1 = quant(w1); woq, so_ = quant(wo); w2q, s2 = quant(w2); w3q, s3 = quant(wdn)
del wq, wk, wv, wg, wu, w1, w2, wo, wdn

# modulations for the 10-step schedule from time_mlp + norm dense weights
sys.path.insert(0, __import__('os').path.join(_HERE, "../../third_party/openpi/src"))
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
S = 10
tt = torch.tensor([1.0 - i * 0.1 for i in range(S)], dtype=torch.float32, device=device)
temb = create_sinusoidal_pos_embedding(tt, D, min_period=4e-3, max_period=4.0, device=device).float()
def lin(x, wname, bname):
    return torch.nn.functional.linear(x, sd[wname].float().to(device), sd[bname].float().to(device))
x_ = torch.nn.functional.silu(lin(temb, "time_mlp_in.weight", "time_mlp_in.bias"))
cond = torch.nn.functional.silu(lin(x_, "time_mlp_out.weight", "time_mlp_out.bias"))
mods = torch.zeros(S, NL, 2, 3072, dtype=torch.float32, device=device)
for i in range(NL):
    mods[:, i, 0] = lin(cond, P + f"{i}.input_layernorm.dense.weight", P + f"{i}.input_layernorm.dense.bias")
    mods[:, i, 1] = lin(cond, P + f"{i}.post_attention_layernorm.dense.weight", P + f"{i}.post_attention_layernorm.dense.bias")
mods = mods.contiguous()
modf = lin(cond, "paligemma_with_expert.gemma_expert.model.norm.dense.weight",
           "paligemma_with_expert.gemma_expert.model.norm.dense.bias").contiguous()
w_ain = sd["action_in_proj.weight"].float().to(device).contiguous()
b_ain = sd["action_in_proj.bias"].float().to(device).contiguous()
w_aout = sd["action_out_proj.weight"].float().to(device).contiguous()
b_aout = sd["action_out_proj.bias"].float().to(device).contiguous()
del sd
print("our weights packed", flush=True)

# ---- FlashRT front-end with decoder no-oped (encoder-only graph) ----
import flash_rt
model = flash_rt.load_model(__import__('os').environ.get("MK_CHECKPOINT", "/network_volume/megakernels/pi05_libero_pt"), framework="torch", config="pi05",
                            hardware="auto", num_views=2, num_steps=10, cache_frames=1, use_fp8=True)
np.random.seed(0)
images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
prompt = "pick up the red block and place it in the tray"
state = np.zeros(8, dtype=np.float32)
out0 = model.predict(images, prompt, state=state)
torch.cuda.synchronize()

# ---- re-capture their full graph as ENCODER-ONLY (calibration already done) ----
from flash_rt.core.cuda_graph import CUDAGraph
import ctypes as _ct
fe = getattr(model, "_frontend", model)
if not hasattr(fe, "pipeline"):
    fe = getattr(model, "_pipe", model)
_pipe0 = fe.pipeline if hasattr(fe, "pipeline") else None
if _pipe0 is None or not hasattr(_pipe0, "vision_encoder"):
    cands = [getattr(o, a) for o in (model, fe) for a in dir(o)
             if not a.startswith("__") and hasattr(getattr(o, a, None), "vision_encoder")]
    _pipe0 = cands[0]
tstream = getattr(fe, "_graph_torch_stream", None) or torch.cuda.Stream()
with torch.cuda.stream(tstream):
    si = tstream.cuda_stream
    for _ in range(3):
        _pipe0._copy_lang_embeds_to_encoder_x(stream=si)
        _pipe0.vision_encoder(stream=si)
        _pipe0.transformer_encoder(stream=si)
    torch.cuda.synchronize()
    g = CUDAGraph()
    handle = _ct.c_void_p(si)
    g.begin_capture(handle)
    _pipe0._copy_lang_embeds_to_encoder_x(stream=si)
    _pipe0.vision_encoder(stream=si)
    _pipe0.transformer_encoder(stream=si)
    g.end_capture(handle)
    torch.cuda.synchronize()
_pipe0._graph = g
_pipe0._graph_stream = handle
if getattr(_pipe0, "_use_exec", False):
    _pipe0._exec_full.adopt(0, g._graph_exec.value)
print("encoder-only graph swapped in", flush=True)
pipe = _pipe0
print("pipe:", type(pipe).__name__, flush=True)
prompt_len = None
for obj in (model, fe, pipe):
    for a in ("current_prompt_len", "prompt_len", "_prompt_len"):
        v = getattr(obj, a, None)
        if v: prompt_len = int(v)
if not prompt_len:
    cands = {a: getattr(fe, a) for a in dir(fe) if "prompt" in a.lower() and isinstance(getattr(fe, a, None), int)}
    print("prompt attrs:", cands)
    prompt_len = [v for v in cands.values() if v > 0][0]
print("prompt_len:", prompt_len, "vision_seq_enc:", pipe.vision_seq_enc, flush=True)
rope_start = int(pipe.vision_seq_enc) + int(prompt_len)
Lp = int(pipe.encoder_seq_len)  # padded, matching their decoder attention span
Lmax = (Lp + T + 15) // 16 * 16
kstride = int(pipe._enc_kv_layer_stride)  # bytes per layer
kbase = int(pipe._attn_ptrs["enc_K"]); vbase = int(pipe._attn_ptrs["enc_V"])
rows_per_layer = kstride // (HD * 2)
print("Lp", Lp, "rows_per_layer", rows_per_layer, flush=True)

cudart = ctypes.CDLL("libcudart.so")
cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
kraw = torch.empty(NL, rows_per_layer, HD, dtype=torch.bfloat16, device=device)
vraw = torch.empty(NL, rows_per_layer, HD, dtype=torch.bfloat16, device=device)
def grab_kv(stream_ptr):
    cudart.cudaMemcpyAsync(ctypes.c_void_p(kraw.data_ptr()), ctypes.c_void_p(kbase), NL * kstride, 3, stream_ptr)
    cudart.cudaMemcpyAsync(ctypes.c_void_p(vraw.data_ptr()), ctypes.c_void_p(vbase), NL * kstride, 3, stream_ptr)

import pack_weights as pw
cos, sin = pw.rope_table(rope_start, device="cuda")
kcache = torch.zeros(NL, Lmax, HD, dtype=torch.float16, device=device)
vtcache = torch.zeros(NL, HD, Lmax, dtype=torch.float16, device=device)
x_t = torch.zeros(T, AD, dtype=torch.float32, device=device)
xb = torch.zeros(2 * T, D, dtype=torch.float32, device=device)
xp = torch.zeros(4, T, D, dtype=torch.float32, device=device)
xn = torch.zeros(T, D, dtype=torch.float16, device=device)
qb = torch.zeros(T, QD, dtype=torch.float16, device=device)
attnb = torch.zeros(T, QD, dtype=torch.float16, device=device)
hmlp = torch.zeros(T, F, dtype=torch.float16, device=device)
scores = torch.zeros(H, T, Lmax, dtype=torch.float32, device=device)
probs = torch.zeros(H, T, Lmax, dtype=torch.float16, device=device)
sc_cyc = torch.zeros(16, dtype=torch.int64, device=device)
noise_t = torch.from_numpy(np.random.default_rng(7).standard_normal((TV, AD)).astype(np.float32)).to(device)

def mk():
    ext.launch(w1q, woq, w2q, w3q, s1, so_, s2, s3, mods, modf, cos, sin,
               w_ain, b_ain, w_aout, b_aout,
               kraw, vraw, rows_per_layer,
               kcache, vtcache, Lp, Lmax, 10, TV,
               x_t, xb, xp, sc_cyc, xn, qb, attnb, hmlp, scores, probs, NB, NT)

from flash_rt.core.utils.actions import unnormalize_actions
cur = torch.cuda.current_stream().cuda_stream
def hybrid_predict():
    model.predict(images, prompt, state=state)      # staging + encoder-only graph
    grab_kv(ctypes.c_void_p(cur))
    x_t[:TV] = noise_t
    mk()
    raw = x_t[:TV, :].float().cpu().numpy()
    return unnormalize_actions(raw, fe.norm_stats)

torch.manual_seed(42)
acts = hybrid_predict()
torch.cuda.synchronize()
print("hybrid actions shape", np.asarray(acts).shape, flush=True)
ts = []
for _ in range(10): hybrid_predict()
torch.cuda.synchronize()
for _ in range(50):
    t0 = time.perf_counter(); hybrid_predict(); torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1000)
ts.sort()
res = {"hybrid_wall_p50_ms": ts[len(ts)//2], "hybrid_min_ms": ts[0], "Lp": Lp}
print(json.dumps(res, indent=1))
np.save(f"{HERE}/hybrid_actions.npy", np.asarray(acts))

# ---- verification + their-full-pipeline benchmark in identical conditions ----
with torch.cuda.stream(tstream):
    si = tstream.cuda_stream
    for _ in range(2):
        _pipe0.run_pipeline(stream=si)
    torch.cuda.synchronize()
    gf = CUDAGraph()
    hf = _ct.c_void_p(si)
    gf.begin_capture(hf)
    _pipe0.run_pipeline(stream=si)
    gf.end_capture(hf)
    torch.cuda.synchronize()
_pipe0._graph = gf
_pipe0._graph_stream = hf
if getattr(_pipe0, "_use_exec", False):
    _pipe0._exec_full.adopt(0, gf._graph_exec.value)
torch.manual_seed(42)
theirs = model.predict(images, prompt, state=state)
torch.cuda.synchronize()
ts2 = []
for _ in range(10): model.predict(images, prompt, state=state)
torch.cuda.synchronize()
for _ in range(50):
    t0 = time.perf_counter(); model.predict(images, prompt, state=state); torch.cuda.synchronize()
    ts2.append((time.perf_counter() - t0) * 1000)
ts2.sort()
a_t = np.asarray(theirs["actions"] if isinstance(theirs, dict) else theirs)
nd = a_t.shape[-1]
a_h = np.asarray(acts).reshape(10, -1)[:, :nd].ravel()
a_t = a_t.ravel()
cosv = float(np.dot(a_h, a_t) / (np.linalg.norm(a_h) * np.linalg.norm(a_t) + 1e-9))
res["flashrt_full_fp8_wall_p50_ms"] = ts2[len(ts2)//2]
res["cos_hybrid_vs_flashrt_full"] = cosv
res["max_diff"] = float(np.abs(a_h - a_t).max())
print(json.dumps(res, indent=1))
json.dump(res, open(f"{HERE}/hybrid_results.json", "w"), indent=1)
