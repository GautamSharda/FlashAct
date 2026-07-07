#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

using bf16 = __half;

#include <cuda_bf16.h>
struct MKParams {
  const unsigned char *w1, *wo, *w2, *w3;
  const float *s1, *so_, *s2, *s3;
  const float *mods, *mod_final, *rope_cos, *rope_sin, *w_ain, *b_ain, *w_aout, *b_aout;
  const __nv_bfloat16 *kvsrc_k, *kvsrc_v;
  int srows;
  bf16 *kcache, *vtcache;
  int Lp, Lmax, num_steps, Tv;
  float *x_t, *x, *xp;
  unsigned long long* stage_cycles;
  bf16 *xn, *q, *attn, *hmlp;
  float* scores;
  bf16* probs;
};

extern "C" __global__ void pi05_megakernel(MKParams p);

static int g_smem = 0;

void launch(torch::Tensor w1, torch::Tensor wo, torch::Tensor w2, torch::Tensor w3,
            torch::Tensor s1, torch::Tensor so_, torch::Tensor s2, torch::Tensor s3,
            torch::Tensor mods, torch::Tensor mod_final, torch::Tensor rope_cos, torch::Tensor rope_sin,
            torch::Tensor w_ain, torch::Tensor b_ain, torch::Tensor w_aout, torch::Tensor b_aout,
            torch::Tensor kvsrc_k, torch::Tensor kvsrc_v, int64_t srows,
            torch::Tensor kcache, torch::Tensor vtcache, int64_t Lp, int64_t Lmax, int64_t num_steps, int64_t Tv,
            torch::Tensor x_t, torch::Tensor x, torch::Tensor xp, torch::Tensor stage_cycles,
            torch::Tensor xn, torch::Tensor q, torch::Tensor attn, torch::Tensor hmlp,
            torch::Tensor scores, torch::Tensor probs,
            int64_t nblocks, int64_t nthreads) {
  MKParams p;
  p.w1 = (const unsigned char*)w1.data_ptr(); p.wo = (const unsigned char*)wo.data_ptr();
  p.w2 = (const unsigned char*)w2.data_ptr(); p.w3 = (const unsigned char*)w3.data_ptr();
  p.s1 = s1.data_ptr<float>(); p.so_ = so_.data_ptr<float>();
  p.s2 = s2.data_ptr<float>(); p.s3 = s3.data_ptr<float>();
  p.mods = mods.data_ptr<float>(); p.mod_final = mod_final.data_ptr<float>();
  p.rope_cos = rope_cos.data_ptr<float>(); p.rope_sin = rope_sin.data_ptr<float>();
  p.w_ain = w_ain.data_ptr<float>(); p.b_ain = b_ain.data_ptr<float>();
  p.w_aout = w_aout.data_ptr<float>(); p.b_aout = b_aout.data_ptr<float>();
  p.kvsrc_k = srows > 0 ? (const __nv_bfloat16*)kvsrc_k.data_ptr() : nullptr;
  p.kvsrc_v = srows > 0 ? (const __nv_bfloat16*)kvsrc_v.data_ptr() : nullptr;
  p.srows = (int)srows;
  p.kcache = (bf16*)kcache.data_ptr(); p.vtcache = (bf16*)vtcache.data_ptr();
  p.Lp = (int)Lp; p.Lmax = (int)Lmax; p.num_steps = (int)num_steps; p.Tv = (int)Tv;
  p.x_t = x_t.data_ptr<float>(); p.x = x.data_ptr<float>(); p.xp = xp.data_ptr<float>();
  p.stage_cycles = stage_cycles.numel() ? (unsigned long long*)stage_cycles.data_ptr() : nullptr;
  p.xn = (bf16*)xn.data_ptr(); p.q = (bf16*)q.data_ptr();
  p.attn = (bf16*)attn.data_ptr(); p.hmlp = (bf16*)hmlp.data_ptr();
  p.scores = scores.data_ptr<float>(); p.probs = (bf16*)probs.data_ptr();

  int smem = 2 * 32 * (1024 + 16) + 8 * 32 * 4 * 4;
  if (smem != g_smem) {
    cudaFuncSetAttribute((const void*)pi05_megakernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    g_smem = smem;
  }
  void* args[] = {&p};
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t err = cudaLaunchCooperativeKernel((const void*)pi05_megakernel,
      dim3(nblocks), dim3(nthreads), args, smem, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "coop launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
