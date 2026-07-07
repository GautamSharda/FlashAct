"""Pack pi0.5 action-expert weights + precomputed tables into flat GPU buffers for the megakernel."""
_HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
import sys
sys.path.insert(0, __import__('os').path.join(_HERE, "../../third_party/openpi/src"))
sys.path.insert(0, _HERE)
import numpy as np
import torch

D = 1024; H = 8; HD = 256; F = 4096; NL = 18; T = 16; AD = 32

def pack(model, num_steps=10, device="cuda"):
    """Returns dict of flat tensors on device. Weights [out,in] row-major bf16."""
    exp = model.paligemma_with_expert.gemma_expert.model
    out = {}
    per_layer = []
    for i in range(NL):
        L = exp.layers[i]
        per_layer.append(dict(
            wq=L.self_attn.q_proj.weight, wk=L.self_attn.k_proj.weight,
            wv=L.self_attn.v_proj.weight, wo=L.self_attn.o_proj.weight,
            wgate=L.mlp.gate_proj.weight, wup=L.mlp.up_proj.weight,
            wdown=L.mlp.down_proj.weight))
    for name in ("wq","wk","wv","wo","wgate","wup","wdown"):
        out[name] = torch.stack([pl[name].to(torch.bfloat16) for pl in per_layer]).contiguous().to(device)

    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
    dt = -1.0 / num_steps
    ts = [1.0 + i * dt for i in range(num_steps)]
    tt = torch.tensor(ts, dtype=torch.float32, device=device)
    temb = create_sinusoidal_pos_embedding(tt, D, min_period=4e-3, max_period=4.0,
                                           device=torch.device(device)).to(torch.float32)
    with torch.no_grad():
        tmi = model.time_mlp_in.to(device); tmo = model.time_mlp_out.to(device)
        x = torch.nn.functional.silu(torch.nn.functional.linear(temb, tmi.weight.float(), tmi.bias.float()))
        cond = torch.nn.functional.silu(torch.nn.functional.linear(x, tmo.weight.float(), tmo.bias.float()))
        mods = []
        for i in range(NL):
            L = exp.layers[i]
            m_in = torch.nn.functional.linear(cond, L.input_layernorm.dense.weight.float(), L.input_layernorm.dense.bias.float())
            m_post = torch.nn.functional.linear(cond, L.post_attention_layernorm.dense.weight.float(), L.post_attention_layernorm.dense.bias.float())
            mods.append(torch.stack([m_in, m_post], dim=1))
        out["mods"] = torch.stack(mods, dim=1).contiguous()  # [S, NL, 2, 3072] fp32 (stack dim order: S first)
        out["mods"] = out["mods"].permute(0,1,2,3).contiguous() if out["mods"].shape[0]==num_steps else out["mods"]
        # note: torch.stack(mods, dim=1) gives [S, NL, 2, 3072] since each mods[i] is [S,2,3072]
        out["mod_final"] = torch.nn.functional.linear(cond, exp.norm.dense.weight.float(), exp.norm.dense.bias.float()).contiguous()  # [S,3072]
    out["cond"] = cond

    out["w_ain"] = model.action_in_proj.weight.float().to(device).contiguous()
    out["b_ain"] = model.action_in_proj.bias.float().to(device).contiguous()
    out["w_aout"] = model.action_out_proj.weight.float().to(device).contiguous()
    out["b_aout"] = model.action_out_proj.bias.float().to(device).contiguous()
    return out

def rope_table(prefix_valid_len, num=T, device="cuda"):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32, device=device) / HD))
    pos = torch.arange(prefix_valid_len, prefix_valid_len + num, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().contiguous(), freqs.sin().contiguous()

def compact_kv(past_key_values, prefix_pad_masks, device="cuda"):
    mask = prefix_pad_masks[0].bool()
    ks, vs = [], []
    for i in range(NL):
        try:
            k = past_key_values.layers[i].keys; v = past_key_values.layers[i].values
        except AttributeError:
            k = past_key_values.key_cache[i]; v = past_key_values.value_cache[i]
        ks.append(k[0, 0][mask]); vs.append(v[0, 0][mask])
    K = torch.stack(ks); V = torch.stack(vs)
    return torch.stack([K, V], dim=1).to(torch.bfloat16).contiguous().to(device)
