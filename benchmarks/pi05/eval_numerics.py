"""mk6 FP8 megakernel on pi05_libero (horizon 10), FlashRT-comparable scenario."""
_HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
_KERN = __import__('os').path.join(_HERE, "../../src/pi05")
import json, os, sys, time
sys.path.insert(0, __import__('os').path.join(_HERE, "../../third_party/openpi/src"))
sys.path.insert(0, _KERN)
sys.path.insert(0, _KERN)
import numpy as np
import torch
from torch.utils.cpp_extension import load
import openpi.models.pi0_config as pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
import pack_weights as pw

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/pi05")
NB, NT = 170, 256
D, H, HD, F, NL, T, AD, QD = 1024, 8, 256, 4096, 18, 16, 32, 2048
TV = 10

ext = load(name="mk_v6", sources=[f"{HERE}/binding.cpp", f"{HERE}/mk6.cu"],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120,code=sm_120"], verbose=False)
device = torch.device("cuda")
cfg = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=TV, discrete_state_input=False,
                           paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
                           dtype="bfloat16", pytorch_compile_mode=None)
model = PI0Pytorch(cfg)
sd = torch.load(__import__('os').environ.get("MK_WEIGHTS", "/network_volume/megakernels/pi05/weights/pi05_libero_merged_fp32.pt"), map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"missing={len(missing)} unexpected={len(unexpected)}")
model.to(device).eval()

d = np.load("/network_volume/megakernels/pi05/reference/libero_inputs.npz")
class Obs:
    token_ar_mask = None
    token_loss_mask = None
obs = Obs()
obs.images = {k: torch.from_numpy(d[k]).permute(0, 3, 1, 2).contiguous().to(device) for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
obs.image_masks = {k: torch.from_numpy(d[f"image_mask_{k}"]).to(device) for k in obs.images}
obs.state = torch.from_numpy(d["state"]).to(device)
obs.tokenized_prompt = torch.from_numpy(d["tokenized_prompt"]).to(device)
obs.tokenized_prompt_mask = torch.from_numpy(d["tokenized_prompt_mask"]).to(device)
noise = torch.from_numpy(d["noise"]).to(device)
m = model

sys.path.insert(0, HERE)
from run_mk import pack_v4

with torch.no_grad():
    images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(obs, train=False)
    def prefill():
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        att4d = m._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, pkv = m.paligemma_with_expert.forward(attention_mask=att4d, position_ids=pos,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
        return pkv, prefix_pad_masks
    past_key_values, prefix_pad_masks = prefill()

    def ref_loop():
        dt = torch.tensor(-0.1, dtype=torch.float32, device=device)
        x_t = noise.clone()
        tt = torch.tensor(1.0, dtype=torch.float32, device=device)
        while tt >= -dt / 2:
            v_t = m.denoise_step(state, prefix_pad_masks, past_key_values, x_t, tt.expand(1))
            x_t = x_t + dt * v_t
            tt += dt
        return x_t
    ref_actions = ref_loop()

    num_steps = 10
    packed = pack_v4(model, num_steps=num_steps, device="cuda")
    packed = {k: (v.to(torch.float16).contiguous() if v.dtype == torch.bfloat16 else v) for k, v in packed.items()}
    def quant_fp8(w):
        s = (w.float().abs().amax(dim=-1).clamp(min=1e-8) / 448.0)
        q = (w.float() / s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return q.view(torch.uint8).contiguous(), s.float().contiguous()
    scales = {}
    for wk in ("w1", "wo", "w2", "w3"):
        packed[wk], scales[wk] = quant_fp8(packed[wk])
    Lp = int(prefix_pad_masks[0].sum().item())
    Lmax = (Lp + T + 15) // 16 * 16
    kv_prefix = pw.compact_kv(past_key_values, prefix_pad_masks, device="cuda").to(torch.float16)
    kcache = torch.zeros(NL, Lmax, HD, dtype=torch.float16, device=device)
    vtcache = torch.zeros(NL, HD, Lmax, dtype=torch.float16, device=device)
    kcache[:, :Lp] = kv_prefix[:, 0]
    vtcache[:, :, :Lp] = kv_prefix[:, 1].transpose(1, 2)
    cos, sin = pw.rope_table(Lp, device="cuda")
    x_t = torch.zeros(T, AD, dtype=torch.float32, device=device)
    x_t[:TV] = noise[0].float()
    x = torch.zeros(2 * T, D, dtype=torch.float32, device=device)
    xp = torch.zeros(4, T, D, dtype=torch.float32, device=device)
    xn = torch.zeros(T, D, dtype=torch.float16, device=device)
    qb = torch.zeros(T, QD, dtype=torch.float16, device=device)
    attnb = torch.zeros(T, QD, dtype=torch.float16, device=device)
    hmlp = torch.zeros(T, F, dtype=torch.float16, device=device)
    scores = torch.zeros(H, T, Lmax, dtype=torch.float32, device=device)
    probs = torch.zeros(H, T, Lmax, dtype=torch.float16, device=device)
    sc_cyc = torch.zeros(16, dtype=torch.int64, device=device)

    def mk():
        ext.launch(packed["w1"], packed["wo"], packed["w2"], packed["w3"],
                   scales["w1"], scales["wo"], scales["w2"], scales["w3"],
                   packed["mods"], packed["mod_final"], cos, sin,
                   packed["w_ain"], packed["b_ain"], packed["w_aout"], packed["b_aout"],
                   torch.empty(0, dtype=torch.bfloat16, device=device), torch.empty(0, dtype=torch.bfloat16, device=device), 0, kcache, vtcache, Lp, Lmax, num_steps, TV, x_t, x, xp, sc_cyc,
                   xn, qb, attnb, hmlp, scores, probs, NB, NT)
    mk(); torch.cuda.synchronize()
    mk_actions = x_t[:TV].clone()
    sc_cyc.zero_(); x_t[:TV] = noise[0].float(); mk(); torch.cuda.synchronize()
    names = ["embed","norm1","qkv","scores","softmax","pv","o_proj","norm2","gateup","down","final"]
    cyc = sc_cyc.cpu().numpy()
    print("stage breakdown (us/layer-ish):")
    for n, c in zip(names, cyc):
        print(f"  {n:8s} {c/2.9/1000:8.1f} us ({100.0*c/max(cyc.sum(),1):.1f}%)")
    ref = ref_actions[0].float()
    diff = (mk_actions - ref).abs()
    cosv = torch.nn.functional.cosine_similarity(mk_actions.ravel(), ref.ravel(), dim=0)
    print(f"max_abs_diff={diff.max().item():.5f} cos={cosv.item():.6f}")

    def bench(fn, iters=50, warmup=5):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(iters):
            x_t[:TV] = noise[0].float()
            ev0.record(); fn(); ev1.record(); torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        return np.array(ts)
    t_mk = bench(mk)
    t_ref = bench(lambda: ref_loop(), iters=15)
    # policy-compute boundary: prefill + loop, inputs pre-staged on GPU
    def policy():
        pkv, ppm = prefill()
        kvp = pw.compact_kv(pkv, ppm, device="cuda").to(torch.float16)
        kcache[:, :Lp] = kvp[:, 0]
        vtcache[:, :, :Lp] = kvp[:, 1].transpose(1, 2)
        x_t[:TV] = noise[0].float()
        mk()
    t_pol = bench(policy, iters=20)
    res = {"mk_loop_ms_p50": float(np.median(t_mk)), "ref_loop_ms_p50": float(np.median(t_ref)),
           "policy_prefill_plus_loop_ms_p50": float(np.median(t_pol)),
           "cos": float(cosv.item()), "max_abs_diff": float(diff.max().item()), "Lp": Lp, "Tv": TV}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{HERE}/results_libero.json", "w"), indent=1)
