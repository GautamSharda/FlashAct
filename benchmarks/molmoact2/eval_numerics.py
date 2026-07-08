"""Numerical correctness bench + standalone loop timing for the MolmoAct2 megakernel.

Packs the action-expert weights, runs the full 10-step flow loop as ONE cooperative
kernel launch, checks outputs against the HF eager reference loop (expected cos
~0.9999975), and reports loop p50 for both. Closed-loop/e2e timing: eval_e2e.py."""
import json, os, sys, time
import numpy as np
import torch
from torch.utils.cpp_extension import load
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL = os.environ.get("MA2_MODEL", "/network_volume/megakernels/molmoact2/model")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/molmoact2")
device = "cuda"
D, HEADS, HD2, F, NL, T, AD = 768, 8, 96, 3072, 36, 32, 32
NB, NT = 170, 256

ext = load(name="mk_ma2", sources=[f"{HERE}/binding.cpp", f"{HERE}/mk_ma2.cu"],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120,code=sm_120"], verbose=False)
print("ext built", flush=True)

torch.manual_seed(0)
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True,
                                                    dtype=torch.bfloat16).to(device).eval()
core = model.model
expert = core._require_action_expert()
imgs = [Image.open(f"{MODEL}/assets/sample_realsense_{v}_rgb.png").convert("RGB") for v in ("side", "top")]
state = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.5], dtype=np.float32)
task = "pick up the cube and place it on the pink pad"

captured = {}
orig = core.generate_actions_from_inputs
def capture(**kw):
    captured.update(kw); return orig(**kw)
core.generate_actions_from_inputs = capture
with torch.no_grad():
    model.predict_action(processor=processor, images=imgs, task=task, state=state,
                         norm_tag="so100_so101_molmoact2", inference_action_mode="continuous",
                         generator=torch.Generator(device=device).manual_seed(42), enable_cuda_graph=False)
core.generate_actions_from_inputs = orig

steps, Tv = 10, 30
adim_pad = captured.get("action_dim_is_pad")
adim = int((~adim_pad[0].bool()).sum().item()) if adim_pad is not None else 6
print("valid action dims:", adim)

with torch.no_grad():
    outputs = core(input_ids=captured["input_ids"], pixel_values=captured.get("pixel_values"),
                   image_token_pooling=captured.get("image_token_pooling"),
                   image_grids=captured.get("image_grids"), image_num_crops=captured.get("image_num_crops"),
                   attention_mask=captured.get("attention_mask"),
                   token_type_ids=captured.get("token_type_ids"), use_cache=True)
    kv_states = core._extract_kv_states(outputs.past_key_values)
    enc_mask = core._get_encoder_attention_mask(captured["input_ids"], captured.get("attention_mask"))
    ctx = expert.prepare_context(encoder_kv_states=kv_states, encoder_attention_mask=enc_mask,
                                 batch_size=1, seq_len=Tv, device=torch.device(device), dtype=torch.bfloat16)
    fts = [torch.full((1,), i / steps, device=device, dtype=torch.float32) for i in range(steps)]
    modc = expert.prepare_modulation_cache(fts)

    noise = torch.randn(1, Tv, AD, device=device, dtype=torch.bfloat16,
                        generator=torch.Generator(device=device).manual_seed(42))
    if adim_pad is not None:
        noise = noise * (~adim_pad[0].bool()).to(noise.dtype)[None, None, :]

    def ref_loop():
        x = noise.clone()
        for i in range(steps):
            v = expert.forward_with_context(x, modc[i].conditioning, context=ctx, modulation=modc[i])
            if adim_pad is not None:
                v = v * (~adim_pad[0].bool()).to(v.dtype)[None, None, :]
            x = x + (1.0 / steps) * v
        return x
    ref = ref_loop()

    # ---- pack ----
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

    # modulations: [S][NL][3 norms][3][768] as (scale, shift, gate)
    mods = torch.zeros(steps, NL, 3, 3, D, dtype=torch.float32, device=device)
    modf = torch.zeros(steps, 2, D, dtype=torch.float32, device=device)
    for s in range(steps):
        for l in range(NL):
            (sh_msa, sc_msa, g_msa, sh_mca, sc_mca, g_mca, sh_mlp, sc_mlp, g_mlp) = modc[s].block_modulations[l]
            for n, (scv, shv, gv) in enumerate([(sc_msa, sh_msa, g_msa), (sc_mca, sh_mca, g_mca), (sc_mlp, sh_mlp, g_mlp)]):
                mods[s, l, n, 0] = scv[0].float(); mods[s, l, n, 1] = shv[0].float(); mods[s, l, n, 2] = gv[0].float()
        fsh, fsc = modc[s].final_modulation
        modf[s, 0] = fsc[0].float(); modf[s, 1] = fsh[0].float()
    # NOTE final_modulation order is (shift, scale) per chunk(2): checked below
    # ActionExpertFinalLayer: shift, scale = modulation -> _modulate(norm, shift, scale)
    # so chunk0=shift chunk1=scale; kernel wants mod[0]=scale mod[D]=shift:
    # we stored modf[s,0]=chunk1(scale)? fsh,fsc = final_modulation -> fsh=chunk0=shift, fsc=chunk1=scale ✓

    rope = expert.blocks[0].self_attn.rope.build_cache(seq_len=Tv, device=torch.device(device), dtype=torch.bfloat16)
    cosf = torch.zeros(T, 48, dtype=torch.float32, device=device)
    sinf = torch.zeros(T, 48, dtype=torch.float32, device=device)
    cosf[:Tv] = rope[0][0, 0].float(); sinf[:Tv] = rope[1][0, 0].float()

    w_ain = expert.action_embed.weight.float().contiguous()
    b_ain = expert.action_embed.bias.float().contiguous()
    w_aout = expert.final_layer.linear.weight.to(torch.bfloat16).contiguous()
    b_aout = expert.final_layer.linear.bias.float().contiguous()

    # cross KV: compact valid tokens
    vmask = enc_mask[0].bool()
    Lk = int(vmask.sum().item())
    Lkmax = (Lk + 15) // 16 * 16
    kctx = torch.zeros(NL, Lkmax, D, dtype=torch.bfloat16, device=device)
    vtctx = torch.zeros(NL, D, Lkmax, dtype=torch.bfloat16, device=device)
    for l, (kc, vc) in enumerate(ctx.kv_contexts):
        kv = kc[0][vmask].reshape(Lk, D)      # [Lk][8*96]
        vv = vc[0][vmask].reshape(Lk, D)
        kctx[l, :Lk] = kv.to(torch.bfloat16)
        vtctx[l, :, :Lk] = vv.t().to(torch.bfloat16)
    print(f"Lk={Lk} Lkmax={Lkmax}")

    x_t = torch.zeros(T, AD, dtype=torch.float32, device=device)
    x_t[:Tv] = noise[0].float()
    x = torch.zeros(T, D, dtype=torch.float32, device=device)
    xp = torch.zeros(3, T, D, dtype=torch.float32, device=device)
    bufs = [torch.zeros(T, D, dtype=torch.bfloat16, device=device) for _ in range(8)]
    bufs.append(torch.zeros(T, F, dtype=torch.bfloat16, device=device))
    scores = torch.zeros(HEADS, T, Lkmax, dtype=torch.float32, device=device)
    probs = torch.zeros(HEADS, T, Lkmax, dtype=torch.bfloat16, device=device)
    sc_cyc = torch.zeros(16, dtype=torch.int64, device=device)
    ws = [w_qkv, w_so, w_cq, w_co, w_gu, w_dn]
    bs = [b_qkv, b_so, b_cq, b_co, b_gu, b_dn]

    def mk():
        ext.launch(ws, bs, mods, modf, cosf, sinf, w_ain, b_ain, w_aout, b_aout,
                   kctx, vtctx, Lk, Lkmax, Tv, adim, steps, x_t, x, xp, sc_cyc,
                   bufs, scores, probs, NB, NT)

    mk(); torch.cuda.synchronize()
    mk_out = x_t[:Tv].clone()
    r = ref[0].float()
    diff = (mk_out[:, :adim] - r[:, :adim]).abs()
    cosv = torch.nn.functional.cosine_similarity(mk_out[:, :adim].ravel(), r[:, :adim].ravel(), dim=0)
    print(f"ref[0,:6] {r[0,:6].cpu().numpy()}")
    print(f"mk [0,:6] {mk_out[0,:6].cpu().numpy()}")
    print(f"max_abs_diff={diff.max().item():.5f} mean={diff.mean().item():.5f} cos={cosv.item():.6f}")

    sc_cyc.zero_(); x_t[:Tv] = noise[0].float(); mk(); torch.cuda.synchronize()
    names = ["embed","norm1","qkv","selfattn","selfout","norm2","crossq","qnorm","scores","softmax","pv","crossout","norm3","gateup","down","final"]
    cyc = sc_cyc.cpu().numpy()
    for n, c in zip(names, cyc):
        print(f"  {n:9s} {c/2.9/1000:8.1f} us ({100.0*c/max(cyc.sum(),1):.1f}%)")

    def bench(fn, iters=30, warmup=3):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(iters):
            x_t[:Tv] = noise[0].float()
            ev0.record(); fn(); ev1.record(); torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        return np.array(ts)
    t_mk = bench(mk)
    t_ref = bench(lambda: ref_loop(), iters=10)
    res = {"megakernel_ms": {"p50": float(np.median(t_mk)), "min": float(t_mk.min())},
           "reference_loop_ms": {"p50": float(np.median(t_ref)), "min": float(t_ref.min())},
           "speedup_p50": float(np.median(t_ref) / np.median(t_mk)),
           "max_abs_diff": float(diff.max().item()), "cos": float(cosv.item()),
           "Lk": Lk, "Tv": Tv, "adim": adim}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{os.path.dirname(os.path.abspath(__file__))}/results.json", "w"), indent=1)
