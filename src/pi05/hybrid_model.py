"""Hybrid pi0.5 model with a FlashRT-compatible ``predict`` API.

The default arguments preserve the LIBERO deployment.  SO100 uses the same
implementation with a 16-token action horizon, raw normalized outputs, and a
checkpoint-specific action-expert weight source.
"""
_HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
import ctypes, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, _HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, __import__('os').path.join(_HERE, "../../third_party/openpi/src"))

D, H, HD, F, NL, T, AD, QD = 1024, 8, 256, 4096, 18, 16, 32, 2048
TV = 10
NB, NT = 170, 256
device = torch.device("cuda")


class HybridPi05:
    def __init__(self, checkpoint, *, weights=None, chunk_size=10, num_steps=10,
                 output_dim=7, return_raw=False, initial_noise_mode="gaussian",
                 initial_noise_scale=1.0, state_in_prompt=True):
        self.chunk_size = int(chunk_size)
        self.num_steps = int(num_steps)
        self.output_dim = int(output_dim)
        self.return_raw = bool(return_raw)
        self.initial_noise_mode = str(initial_noise_mode)
        self.initial_noise_scale = float(initial_noise_scale)
        self.state_in_prompt = bool(state_in_prompt)
        if not 1 <= self.chunk_size <= T:
            raise ValueError(f"chunk_size must be in [1, {T}], got {self.chunk_size}")
        if self.initial_noise_mode not in {"gaussian", "zero", "scale"}:
            raise ValueError(f"unsupported initial noise mode: {self.initial_noise_mode}")
        checkpoint = pathlib.Path(checkpoint)
        self.weight_source = pathlib.Path(
            weights or __import__('os').environ.get("MK_WEIGHTS", checkpoint / "model.safetensors")
        )
        from torch.utils.cpp_extension import load as _load
        HERE = _HERE
        cuda_include = (
            pathlib.Path(torch.__file__).resolve().parent.parent
            / "nvidia" / "cu13" / "include"
        )
        extra_include_paths = [str(cuda_include)] if cuda_include.exists() else []
        self.ext = _load(name="mk_v6", sources=[f"{HERE}/binding.cpp", f"{HERE}/mk6.cu"],
                         extra_cuda_cflags=["-O3", "--use_fast_math",
                                            "-gencode=arch=compute_120,code=sm_120"],
                         extra_include_paths=extra_include_paths,
                         verbose=False)
        self._pack_weights()
        from flash_rt.api import VLAModel
        from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
        frontend = Pi05TorchFrontendRtx(
            checkpoint,
            num_views=2,
            chunk_size=self.chunk_size,
            num_steps=self.num_steps,
            cache_frames=1,
            use_fp8=True,
            hardware="rtx_sm120",
        )
        self.inner = VLAModel(frontend, "torch")
        self.fe = getattr(self.inner, "_frontend", self.inner)
        if not hasattr(self.fe, "pipeline"):
            self.fe = getattr(self.inner, "_pipe", self.inner)
        self._cudart = ctypes.CDLL("libcudart.so")
        self._cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self._pipe_id = None
        self.norm_stats = None
        self.latency_records = []

    def _pack_weights(self):
        from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
        if self.weight_source.suffix == ".safetensors":
            from safetensors import safe_open
            handle = safe_open(str(self.weight_source), framework="pt", device="cpu")

            class SafeTensorWeights:
                def __getitem__(self, key):
                    return handle.get_tensor(key)

            sd = SafeTensorWeights()
        else:
            sd = torch.load(
                self.weight_source,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        P = "paligemma_with_expert.gemma_expert.model.layers."
        def stack(n): return torch.stack([sd[P + str(i) + "." + n].to(device) for i in range(NL)])
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
        self.w1q, self.s1 = quant(w1); self.woq, self.so_ = quant(wo)
        self.w2q, self.s2 = quant(w2); self.w3q, self.s3 = quant(wdn)
        tt = torch.tensor(
            [1.0 - i / self.num_steps for i in range(self.num_steps)],
            dtype=torch.float32,
            device=device,
        )
        temb = create_sinusoidal_pos_embedding(tt, D, min_period=4e-3, max_period=4.0, device=device).float()
        def lin(x, wn, bn):
            return torch.nn.functional.linear(x, sd[wn].float().to(device), sd[bn].float().to(device))
        xa = torch.nn.functional.silu(lin(temb, "time_mlp_in.weight", "time_mlp_in.bias"))
        cond = torch.nn.functional.silu(lin(xa, "time_mlp_out.weight", "time_mlp_out.bias"))
        mods = torch.zeros(self.num_steps, NL, 2, 3072, dtype=torch.float32, device=device)
        for i in range(NL):
            mods[:, i, 0] = lin(cond, P + f"{i}.input_layernorm.dense.weight", P + f"{i}.input_layernorm.dense.bias")
            mods[:, i, 1] = lin(cond, P + f"{i}.post_attention_layernorm.dense.weight",
                                P + f"{i}.post_attention_layernorm.dense.bias")
        self.mods = mods.contiguous()
        self.modf = lin(cond, "paligemma_with_expert.gemma_expert.model.norm.dense.weight",
                        "paligemma_with_expert.gemma_expert.model.norm.dense.bias").contiguous()
        self.w_ain = sd["action_in_proj.weight"].float().to(device).contiguous()
        self.b_ain = sd["action_in_proj.bias"].float().to(device).contiguous()
        self.w_aout = sd["action_out_proj.weight"].float().to(device).contiguous()
        self.b_aout = sd["action_out_proj.bias"].float().to(device).contiguous()
        del sd

    def _setup_for_pipeline(self):
        """(Re)build encoder-only graph + our buffers for the current pipeline/prompt."""
        from flash_rt.core.cuda_graph import CUDAGraph
        pipe = self.fe.pipeline
        tstream = getattr(self.fe, "_graph_torch_stream", None) or torch.cuda.Stream()
        with torch.cuda.stream(tstream):
            si = tstream.cuda_stream
            for _ in range(2):
                pipe._copy_lang_embeds_to_encoder_x(stream=si)
                pipe.vision_encoder(stream=si)
                pipe.transformer_encoder(stream=si)
            torch.cuda.synchronize()
            g = CUDAGraph()
            h = ctypes.c_void_p(si)
            g.begin_capture(h)
            pipe._copy_lang_embeds_to_encoder_x(stream=si)
            pipe.vision_encoder(stream=si)
            pipe.transformer_encoder(stream=si)
            g.end_capture(h)
            torch.cuda.synchronize()
        pipe._graph = g
        pipe._graph_stream = h
        if getattr(pipe, "_use_exec", False):
            pipe._exec_full.adopt(0, g._graph_exec.value)
        # geometry
        import pack_weights as pw
        prompt_len = int(getattr(self.fe, "current_prompt_len", 0) or pipe.max_prompt_len)
        self.Lp = int(pipe.encoder_seq_len)
        rope_start = int(pipe.vision_seq_enc) + prompt_len
        self.Lmax = (self.Lp + T + 15) // 16 * 16
        self.kstride = int(pipe._enc_kv_layer_stride)
        self.rows = self.kstride // (HD * 2)
        self.kbase = int(pipe._attn_ptrs["enc_K"])
        self.vbase = int(pipe._attn_ptrs["enc_V"])
        self.cos, self.sin = pw.rope_table(rope_start, device="cuda")
        self.kraw = torch.empty(NL, self.rows, HD, dtype=torch.bfloat16, device=device)
        self.vraw = torch.empty(NL, self.rows, HD, dtype=torch.bfloat16, device=device)
        self.kcache = torch.zeros(NL, self.Lmax, HD, dtype=torch.float16, device=device)
        self.vtcache = torch.zeros(NL, HD, self.Lmax, dtype=torch.float16, device=device)
        self.x_t = torch.zeros(T, AD, dtype=torch.float32, device=device)
        self.xb = torch.zeros(2 * T, D, dtype=torch.float32, device=device)
        self.xp = torch.zeros(4, T, D, dtype=torch.float32, device=device)
        self.xn = torch.zeros(T, D, dtype=torch.float16, device=device)
        self.qb = torch.zeros(T, QD, dtype=torch.float16, device=device)
        self.attnb = torch.zeros(T, QD, dtype=torch.float16, device=device)
        self.hmlp = torch.zeros(T, F, dtype=torch.float16, device=device)
        self.scores = torch.zeros(H, T, self.Lmax, dtype=torch.float32, device=device)
        self.probs = torch.zeros(H, T, self.Lmax, dtype=torch.float16, device=device)
        self.sc_cyc = torch.zeros(16, dtype=torch.int64, device=device)
        self.nstage = torch.empty(self.chunk_size, AD, dtype=torch.bfloat16, device=device)
        self.norm_stats = self.fe.norm_stats
        self._pipe_id = (id(pipe), prompt_len)

    def predict(self, images, prompt, state=None, **kw):
        import time as _t
        t0 = _t.perf_counter()
        if state is None and self.state_in_prompt:
            state = np.zeros(8, dtype=np.float32)
        out = self.inner.predict(images, prompt, state=state)
        pipe = self.fe.pipeline
        pl = int(getattr(self.fe, "current_prompt_len", 0) or 0)
        if self._pipe_id != (id(pipe), pl):
            self._setup_for_pipeline()
            # re-run staging+encoder under the new graph
            out = self.inner.predict(images, prompt, state=state)
        cur = torch.cuda.current_stream().cuda_stream
        cp = ctypes.c_void_p(cur)
        self._cudart.cudaMemcpyAsync(ctypes.c_void_p(self.kraw.data_ptr()),
                                     ctypes.c_void_p(self.kbase), NL * self.kstride, 3, cp)
        self._cudart.cudaMemcpyAsync(ctypes.c_void_p(self.vraw.data_ptr()),
                                     ctypes.c_void_p(self.vbase), NL * self.kstride, 3, cp)
        self._cudart.cudaMemcpyAsync(ctypes.c_void_p(self.nstage.data_ptr()),
                                     ctypes.c_void_p(int(pipe.bufs["diffusion_noise"].ptr.value)),
                                     self.chunk_size * AD * 2, 3, cp)
        Lp = self.Lp
        self.x_t.zero_()
        if self.initial_noise_mode == "gaussian":
            self.x_t[:self.chunk_size] = self.nstage.float()
        elif self.initial_noise_mode == "scale":
            self.x_t[:self.chunk_size] = self.nstage.float() * self.initial_noise_scale
        self.ext.launch(self.w1q, self.woq, self.w2q, self.w3q, self.s1, self.so_, self.s2, self.s3,
                        self.mods, self.modf, self.cos, self.sin,
                        self.w_ain, self.b_ain, self.w_aout, self.b_aout,
                        self.kraw, self.vraw, self.rows,
                        self.kcache, self.vtcache, Lp, self.Lmax, self.num_steps, self.chunk_size,
                        self.x_t, self.xb, self.xp, self.sc_cyc,
                        self.xn, self.qb, self.attnb, self.hmlp, self.scores, self.probs, NB, NT)
        raw = self.x_t[:self.chunk_size].float().cpu().numpy()
        self.latency_records.append((_t.perf_counter() - t0) * 1000)
        if self.return_raw:
            return raw
        from flash_rt.core.utils.actions import unnormalize_actions, LIBERO_ACTION_DIM
        unnorm = np.asarray(unnormalize_actions(raw, self.norm_stats))
        return unnorm[:, :self.output_dim]


def load_model(checkpoint, **kw):
    return HybridPi05(checkpoint)
