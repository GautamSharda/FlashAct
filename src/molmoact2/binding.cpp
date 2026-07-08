#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;

struct MAParams {
  const bf16 *w_qkv; const float *b_qkv;
  const bf16 *w_so; const float *b_so;
  const bf16 *w_cq; const float *b_cq;
  const bf16 *w_co; const float *b_co;
  const bf16 *w_gu; const float *b_gu;
  const bf16 *w_dn; const float *b_dn;
  const float *mods, *mod_final, *rope_cos, *rope_sin, *w_ain, *b_ain;
  const bf16 *w_aout; const float *b_aout;
  const bf16 *kctx, *vtctx;
  int Lk, Lkmax, Tv, adim, num_steps;
  float *x_t, *x, *xp;
  unsigned long long* stage_cycles;
  bf16 *xn, *qb, *kb, *vb, *sattn, *qc, *qcn, *attn, *hmlp;
  float* scores;
  bf16* probs;
};

extern "C" __global__ void ma2_megakernel(MAParams p);
static bool g_attr = false;

void launch(std::vector<torch::Tensor> ws, std::vector<torch::Tensor> bs,
            torch::Tensor mods, torch::Tensor mod_final, torch::Tensor rope_cos, torch::Tensor rope_sin,
            torch::Tensor w_ain, torch::Tensor b_ain, torch::Tensor w_aout, torch::Tensor b_aout,
            torch::Tensor kctx, torch::Tensor vtctx,
            int64_t Lk, int64_t Lkmax, int64_t Tv, int64_t adim, int64_t num_steps,
            torch::Tensor x_t, torch::Tensor x, torch::Tensor xp, torch::Tensor stage_cycles,
            std::vector<torch::Tensor> bufs, torch::Tensor scores, torch::Tensor probs,
            int64_t nblocks, int64_t nthreads) {
  MAParams p;
  p.w_qkv = (const bf16*)ws[0].data_ptr(); p.w_so = (const bf16*)ws[1].data_ptr();
  p.w_cq = (const bf16*)ws[2].data_ptr(); p.w_co = (const bf16*)ws[3].data_ptr();
  p.w_gu = (const bf16*)ws[4].data_ptr(); p.w_dn = (const bf16*)ws[5].data_ptr();
  p.b_qkv = bs[0].data_ptr<float>(); p.b_so = bs[1].data_ptr<float>();
  p.b_cq = bs[2].data_ptr<float>(); p.b_co = bs[3].data_ptr<float>();
  p.b_gu = bs[4].data_ptr<float>(); p.b_dn = bs[5].data_ptr<float>();
  p.mods = mods.data_ptr<float>(); p.mod_final = mod_final.data_ptr<float>();
  p.rope_cos = rope_cos.data_ptr<float>(); p.rope_sin = rope_sin.data_ptr<float>();
  p.w_ain = w_ain.data_ptr<float>(); p.b_ain = b_ain.data_ptr<float>();
  p.w_aout = (const bf16*)w_aout.data_ptr(); p.b_aout = b_aout.data_ptr<float>();
  p.kctx = (const bf16*)kctx.data_ptr(); p.vtctx = (const bf16*)vtctx.data_ptr();
  p.Lk = (int)Lk; p.Lkmax = (int)Lkmax; p.Tv = (int)Tv; p.adim = (int)adim; p.num_steps = (int)num_steps;
  p.x_t = x_t.data_ptr<float>(); p.x = x.data_ptr<float>(); p.xp = xp.data_ptr<float>();
  p.stage_cycles = stage_cycles.numel() ? (unsigned long long*)stage_cycles.data_ptr() : nullptr;
  p.xn = (bf16*)bufs[0].data_ptr(); p.qb = (bf16*)bufs[1].data_ptr();
  p.kb = (bf16*)bufs[2].data_ptr(); p.vb = (bf16*)bufs[3].data_ptr();
  p.sattn = (bf16*)bufs[4].data_ptr(); p.qc = (bf16*)bufs[5].data_ptr();
  p.qcn = (bf16*)bufs[6].data_ptr(); p.attn = (bf16*)bufs[7].data_ptr();
  p.hmlp = (bf16*)bufs[8].data_ptr();
  p.scores = scores.data_ptr<float>(); p.probs = (bf16*)probs.data_ptr();

  int smem = 2 * 8 * (1024 + 8) * 2 + 8 * 32 * 8 * 4;
  if (!g_attr) {
    cudaFuncSetAttribute((const void*)ma2_megakernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    g_attr = true;
  }
  void* args[] = {&p};
  auto stream = at::cuda::getCurrentCUDAStream();
  // clamp to cooperative-launch capacity (toolchain-dependent occupancy); grid-stride loops accept any block count
  int max_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_per_sm, (const void*)ma2_megakernel, nthreads, smem);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int cap = max_per_sm * sms;
  static bool g_diag = false;
  if (!g_diag) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, (const void*)ma2_megakernel);
    fprintf(stderr, "[mk_ma2] max_per_sm=%d sms=%d cap=%d req=%d regs=%d smem_static=%zu smem_dyn=%d\n",
            max_per_sm, sms, cap, nblocks, fa.numRegs, fa.sharedSizeBytes, smem);
    g_diag = true;
  }
  if (cap > 0 && nblocks > cap) nblocks = cap;
  cudaError_t err = cudaLaunchCooperativeKernel((const void*)ma2_megakernel,
      dim3(nblocks), dim3(nthreads), args, smem, stream.stream());
  TORCH_CHECK(err == cudaSuccess, "coop launch failed: ", cudaGetErrorString(err));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
