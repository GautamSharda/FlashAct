// pi0.5 denoise-loop megakernel v5 — RTX 5090 (sm_120)
//
// ONE cooperative launch runs the entire 10-step flow-matching denoise loop of the
// pi0.5 action expert (18-layer Gemma-300M, adaRMS, MQA attention over the frozen
// PaliGemma prefix KV cache).
//
// v5 engine:
//   - all projections on tensor cores (mma.m16n8k16 bf16->fp32); the 16 suffix
//     tokens are the mma m-dim
//   - weights stream through smem via cp.async double-buffered tiles;
//     qkv/o_proj/down use uniform [8 rows x 1024-k] tiles (k-splits become extra
//     tiles, so every block pipelines several tiles); gate/up uses [16 x 1024]
//   - k-split partial sums land in per-slice buffers and are combined for free
//     inside the next norm mini-stage (deterministic, no atomics, no extra barrier)
//   - attention = scores mma (Q@K^T) -> softmax mini-stage -> P@V mma with a
//     transposed V cache; MQA prefix K/V compacted once per inference
//   - adaRMS modulations precomputed for the fixed schedule; RoPE pairs and
//     gate/up rows pre-interleaved at pack time so mma epilogues stay in-lane
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define D 1024
#define H 8
#define HD 256
#define F 4096
#define NL 18
#define T 16
#define AD 32
#define QD 2048
#define EPS 1e-6f
#define ROWS16 16
#define ROWS8 8

using bf16 = __half;
using w8 = unsigned char;

struct MKParams {
  const w8* __restrict__ w1;      // fp8 e4m3 [NL][2560][1024]
  const w8* __restrict__ wo;      // fp8 [NL][1024][2048]
  const w8* __restrict__ w2;      // fp8 [NL][8192][1024]
  const w8* __restrict__ w3;      // fp8 [NL][1024][4096]
  const float* __restrict__ s1;   // per-row scales [NL][2560]
  const float* __restrict__ so_;  // [NL][1024]
  const float* __restrict__ s2;   // [NL][8192]
  const float* __restrict__ s3;   // [NL][1024]
  const float* __restrict__ mods;
  const float* __restrict__ mod_final;
  const float* __restrict__ rope_cos;
  const float* __restrict__ rope_sin;
  const float* __restrict__ w_ain;
  const float* __restrict__ b_ain;
  const float* __restrict__ w_aout;
  const float* __restrict__ b_aout;
  const __nv_bfloat16* __restrict__ kvsrc_k;  // their enc_K [NL][srows][256] bf16 interleaved-rope (optional)
  const __nv_bfloat16* __restrict__ kvsrc_v;
  int srows;                                   // their rows per layer (0 = handoff disabled)
  bf16* __restrict__ kcache;   // [NL][Lmax][256]
  bf16* __restrict__ vtcache;  // [NL][256][Lmax]
  int Lp, Lmax, num_steps, Tv;
  float* __restrict__ x_t;
  float* __restrict__ x;    // [T][D]
  float* __restrict__ xp;   // [4][T][D] k-slice partials (down uses 4, o_proj 2)
  unsigned long long* __restrict__ stage_cycles;
  bf16* __restrict__ xn;    // [T][D]
  bf16* __restrict__ q;     // [T][QD]
  bf16* __restrict__ attn;  // [T][QD]
  bf16* __restrict__ hmlp;  // [T][F]
  float* __restrict__ scores;  // [H][T][Lmax]
  bf16* __restrict__ probs;    // [H][T][Lmax]
};

__device__ __forceinline__ float bf2f(bf16 v) { return __half2float(v); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float gelu_tanh(float x) {
  const float k0 = 0.7978845608028654f;
  return 0.5f * x * (1.0f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  unsigned saddr = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ unsigned cvt_e4m3x2(unsigned short v) {
  unsigned r;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(v));
  return r;
}
__device__ __forceinline__ void mma16816(float c[4], const unsigned a[4], const unsigned b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void load_a(unsigned a[4], const bf16* __restrict__ act, int stride,
                                       int k0, int gid, int tig) {
  const bf16* ar0 = &act[gid * stride + k0 + tig * 2];
  const bf16* ar1 = &act[(gid + 8) * stride + k0 + tig * 2];
  a[0] = *reinterpret_cast<const unsigned*>(ar0);
  a[1] = *reinterpret_cast<const unsigned*>(ar1);
  a[2] = *reinterpret_cast<const unsigned*>(ar0 + 8);
  a[3] = *reinterpret_cast<const unsigned*>(ar1 + 8);
}

extern "C" __global__ void __launch_bounds__(256, 1)
pi05_megakernel(MKParams p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem_raw[];
  bf16* tiles = reinterpret_cast<bf16*>(smem_raw);  // up to [2][16][1032]
  float* parts = reinterpret_cast<float*>(smem_raw + 2 * 32 * (D + 16));  // [8][32][4]

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int nwarp = blockDim.x >> 5;
  const int gwarp = blockIdx.x * nwarp + warp;
  const int ngwarp = gridDim.x * nwarp;
  const int gtid = blockIdx.x * blockDim.x + tid;
  const int ngtid = gridDim.x * blockDim.x;
  const int gid = lane >> 2, tig = lane & 3;
  const int Lp = p.Lp, Ltot = p.Lp + p.Tv, Lmax = p.Lmax;
  const float dt = -1.0f / p.num_steps;
  unsigned long long _t0 = clock64();
#define MARK(idx)                                                        \
  if (p.stage_cycles && blockIdx.x == 0 && threadIdx.x == 0) {           \
    unsigned long long now = clock64();                                  \
    p.stage_cycles[idx] += now - _t0;                                    \
    _t0 = now;                                                           \
  }

  // ---- norm mini-stage: block t handles token t. Optionally combines nsl k-slice
  // partials (xp) scaled by gate[] into x first. Writes xn = adaRMS(x). ----
  auto mini_norm = [&](const float* srcx, float* dstx, const float* mod, const float* gate, int nsl) {
    if (blockIdx.x >= T * 8) return;
    int t = blockIdx.x >> 3, ch = blockIdx.x & 7;
    float* row = reinterpret_cast<float*>(tiles);  // 4KB block-local
    __shared__ float red[8];
    float ss = 0.f;
    for (int d = tid; d < D; d += blockDim.x) {
      float v = srcx[t * D + d];
      for (int s = 0; s < nsl; s++) v += p.xp[(s * T + t) * D + d] * gate[d];
      row[d] = v;
      ss += v * v;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w2i = 0; w2i < 8; w2i++) tot += red[w2i];
    float r = rsqrtf(tot / D + EPS);
    if (tid < 128) {
      int d = ch * 128 + tid;
      float v = row[d];
      p.xn[t * D + d] = __float2half(v * r * (1.f + mod[d]) + mod[D + d]);
      if (nsl > 0) dstx[t * D + d] = v;
    }
  };

  // ---- unified rows8 x k1024 mma tile pipeline ----
  // weight rows at `wbase`, row stride `wstride` elems; tile -> (rowblk, kwin).
  // A acts at `act` (+ kwin*1024 k-offset), row stride astride.
  // kind 0: qkv packed epilogue (uses Klp/VTlp); kind 2: store partial to xp[kwin].
  auto rows8_mma = [&](const w8* wbase, const float* scal, int wstride, int nrowblk, int nkwin,
                       const bf16* act, int astride, int kind, bf16* Klp, bf16* VTlp) {
    // 16-row fp8 tiles: nrowblk counts 16-row blocks; warps = 2 n8-blocks x 4 k-quarters
    const int TROWB = D + 16;
    w8* tiles8 = reinterpret_cast<w8*>(tiles);
    const int tiles_total = nrowblk * nkwin;
    const int nblk = warp >> 2, kq = warp & 3;
    auto issue = [&](int tileidx, int buf) {
      int rowblk = tileidx % nrowblk, kwin = tileidx / nrowblk;
      const w8* src = wbase + (size_t)rowblk * 16 * wstride + kwin * D;
      w8* dst = tiles8 + (size_t)buf * 16 * TROWB;
      for (int i = tid; i < 16 * D / 16; i += blockDim.x) {
        int r = i >> 6, c = i & 63;
        cp16(&dst[r * TROWB + c * 16], &src[(size_t)r * wstride + c * 16]);
      }
      cp_commit();
    };
    int t0 = blockIdx.x;
    if (t0 < tiles_total) issue(t0, 0);
    int buf = 0;
    for (int tile = t0; tile < tiles_total; tile += gridDim.x) {
      int next = tile + gridDim.x;
      cp_wait<0>();
      __syncthreads();
      if (next < tiles_total) issue(next, buf ^ 1);
      int rowblk = tile % nrowblk, kwin = tile / nrowblk;
      const w8* tp = tiles8 + (size_t)buf * 16 * TROWB;
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const w8* brow = &tp[(nblk * 8 + gid) * TROWB];
#pragma unroll
      for (int ks = 0; ks < 16; ks++) {
        int k0 = kq * 256 + ks * 16;
        unsigned a[4], b[2];
        load_a(a, act + kwin * D, astride, k0, gid, tig);
        b[0] = cvt_e4m3x2(*reinterpret_cast<const unsigned short*>(&brow[k0 + tig * 2]));
        b[1] = cvt_e4m3x2(*reinterpret_cast<const unsigned short*>(&brow[k0 + tig * 2 + 8]));
        mma16816(c, a, b);
      }
      float* pp = &parts[((nblk * 4 + kq) * 32 + lane) * 4];
      pp[0] = c[0]; pp[1] = c[1]; pp[2] = c[2]; pp[3] = c[3];
      __syncthreads();
      if (kq == 0) {
        float r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#pragma unroll
        for (int qq = 0; qq < 4; qq++) {
          const float* s = &parts[((nblk * 4 + qq) * 32 + lane) * 4];
          r0 += s[0]; r1 += s[1]; r2 += s[2]; r3 += s[3];
        }
        int gr = rowblk * 16 + nblk * 8 + tig * 2;
        float sc0 = scal[gr], sc1 = scal[gr + 1];
        r0 *= sc0; r1 *= sc1; r2 *= sc0; r3 *= sc1;
        if (kind == 0) {
          if (gr < 2048) {
            int pr = gr >> 1, h = pr >> 7, j = pr & 127;
            float c0 = p.rope_cos[gid * 128 + j], s0 = p.rope_sin[gid * 128 + j];
            float c1 = p.rope_cos[(gid + 8) * 128 + j], s1 = p.rope_sin[(gid + 8) * 128 + j];
            p.q[gid * QD + h * HD + j] = __float2half(r0 * c0 - r1 * s0);
            p.q[gid * QD + h * HD + j + 128] = __float2half(r1 * c0 + r0 * s0);
            p.q[(gid + 8) * QD + h * HD + j] = __float2half(r2 * c1 - r3 * s1);
            p.q[(gid + 8) * QD + h * HD + j + 128] = __float2half(r3 * c1 + r2 * s1);
          } else if (gr < 2304) {
            int j = (gr - 2048) >> 1;
            float c0 = p.rope_cos[gid * 128 + j], s0 = p.rope_sin[gid * 128 + j];
            float c1 = p.rope_cos[(gid + 8) * 128 + j], s1 = p.rope_sin[(gid + 8) * 128 + j];
            Klp[(size_t)(Lp + gid) * HD + j] = __float2half(r0 * c0 - r1 * s0);
            Klp[(size_t)(Lp + gid) * HD + j + 128] = __float2half(r1 * c0 + r0 * s0);
            Klp[(size_t)(Lp + gid + 8) * HD + j] = __float2half(r2 * c1 - r3 * s1);
            Klp[(size_t)(Lp + gid + 8) * HD + j + 128] = __float2half(r3 * c1 + r2 * s1);
          } else {
            int o = gr - 2304;
            VTlp[(size_t)o * Lmax + Lp + gid] = __float2half(r0);
            VTlp[(size_t)(o + 1) * Lmax + Lp + gid] = __float2half(r1);
            VTlp[(size_t)o * Lmax + Lp + gid + 8] = __float2half(r2);
            VTlp[(size_t)(o + 1) * Lmax + Lp + gid + 8] = __float2half(r3);
          }
        } else {
          float* xps = p.xp + (size_t)kwin * T * D;
          xps[gid * D + gr] = r0;
          xps[gid * D + gr + 1] = r1;
          xps[(gid + 8) * D + gr] = r2;
          xps[(gid + 8) * D + gr + 1] = r3;
        }
      }
      __syncthreads();
      buf ^= 1;
    }
  };

  // ---- KV handoff prologue: de-interleave their K, transpose V, bf16->f16 ----
  if (p.srows > 0) {
    int total = NL * Lp;
    for (int i = gtid; i < total; i += ngtid) {
      int l = i / Lp, r = i % Lp;
      const __nv_bfloat16* ks = p.kvsrc_k + ((size_t)l * p.srows + r) * HD;
      const __nv_bfloat16* vs = p.kvsrc_v + ((size_t)l * p.srows + r) * HD;
      bf16* kd = p.kcache + ((size_t)l * Lmax + r) * HD;
      bf16* vtd = p.vtcache + (size_t)l * HD * Lmax;
#pragma unroll 4
      for (int d = 0; d < 128; d++) {
        kd[d] = __float2half(__bfloat162float(ks[2 * d]));
        kd[d + 128] = __float2half(__bfloat162float(ks[2 * d + 1]));
      }
#pragma unroll 4
      for (int d = 0; d < HD; d++)
        vtd[(size_t)d * Lmax + r] = __float2half(__bfloat162float(vs[d]));
    }
    grid.sync();
  }

  for (int step = 0; step < p.num_steps; step++) {
    // ---- embed ----
    for (int i = gtid; i < T * D; i += ngtid) {
      int t = i / D, d = i % D;
      float acc = p.b_ain[d];
      const float* w = &p.w_ain[d * AD];
#pragma unroll
      for (int a = 0; a < AD; a++) acc = fmaf(w[a], p.x_t[t * AD + a], acc);
      p.x[t * D + d] = acc;
    }
    grid.sync();
    MARK(0)

    float* xcur = p.x;
    float* xalt = p.x + T * D;
    const float* prev_gate = nullptr;
    for (int l = 0; l < NL; l++) {
      const float* mod_in = &p.mods[((size_t)(step * NL + l) * 2 + 0) * 3072];
      const float* mod_post = &p.mods[((size_t)(step * NL + l) * 2 + 1) * 3072];
      bf16* Kl = p.kcache + (size_t)l * Lmax * HD;
      bf16* VTl = p.vtcache + (size_t)l * HD * Lmax;

      // ---- norm1 (combines previous layer's down partials) ----
      mini_norm(xcur, xalt, mod_in, prev_gate, prev_gate ? 4 : 0);
      if (prev_gate) { float* tmp = xcur; xcur = xalt; xalt = tmp; }
      grid.sync();
      MARK(1)

      // ---- QKV + rope: rows8 tiles over W1 (320 tiles) ----
      rows8_mma(p.w1 + (size_t)l * 2560 * D, p.s1 + (size_t)l * 2560, D, 2560 / 16, 1, p.xn, D, 0, Kl, VTl);
      grid.sync();
      MARK(2)

      // ---- attention scores ----
      {
        int jblocks = (Ltot + 7) / 8;
        for (int jb = blockIdx.x; jb < jblocks; jb += gridDim.x) {
          int h = warp;
          float c[4] = {0.f, 0.f, 0.f, 0.f};
          const bf16* brow = &Kl[(size_t)(jb * 8 + gid) * HD];
          bool valid = (jb * 8 + gid) < Ltot;
          unsigned a[4], b[2], an[4], bn[2];
          load_a(a, p.q + h * HD, QD, 0, gid, tig);
          b[0] = valid ? *reinterpret_cast<const unsigned*>(&brow[tig * 2]) : 0u;
          b[1] = valid ? *reinterpret_cast<const unsigned*>(&brow[tig * 2 + 8]) : 0u;
#pragma unroll
          for (int ks = 0; ks < 16; ks++) {
            if (ks + 1 < 16) {
              int kn = (ks + 1) * 16;
              load_a(an, p.q + h * HD, QD, kn, gid, tig);
              bn[0] = valid ? *reinterpret_cast<const unsigned*>(&brow[kn + tig * 2]) : 0u;
              bn[1] = valid ? *reinterpret_cast<const unsigned*>(&brow[kn + tig * 2 + 8]) : 0u;
            }
            mma16816(c, a, b);
            a[0] = an[0]; a[1] = an[1]; a[2] = an[2]; a[3] = an[3];
            b[0] = bn[0]; b[1] = bn[1];
          }
          int j0 = jb * 8 + tig * 2;
          float* sc = p.scores + (size_t)h * T * Lmax;
          if (j0 < Ltot) sc[gid * Lmax + j0] = c[0] * 0.0625f;
          if (j0 + 1 < Ltot) sc[gid * Lmax + j0 + 1] = c[1] * 0.0625f;
          if (j0 < Ltot) sc[(gid + 8) * Lmax + j0] = c[2] * 0.0625f;
          if (j0 + 1 < Ltot) sc[(gid + 8) * Lmax + j0 + 1] = c[3] * 0.0625f;
        }
      }
      grid.sync();
      MARK(3)

      // ---- softmax mini-stage ----
      for (int u = blockIdx.x; u < H * T; u += gridDim.x) {
        int h = u >> 4, t = u & 15;
        const float* sc = p.scores + ((size_t)h * T + t) * Lmax;
        bf16* pb = p.probs + ((size_t)h * T + t) * Lmax;
        __shared__ float red2[8];
        float lmax = -1e30f;
        for (int j = tid; j < Ltot; j += blockDim.x) lmax = fmaxf(lmax, sc[j]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
        if (lane == 0) red2[warp] = lmax;
        __syncthreads();
        float gmax = -1e30f;
#pragma unroll
        for (int w2i = 0; w2i < 8; w2i++) gmax = fmaxf(gmax, red2[w2i]);
        __syncthreads();
        float lsum = 0.f;
        for (int j = tid; j < Ltot; j += blockDim.x) lsum += __expf(sc[j] - gmax);
        lsum = warp_sum(lsum);
        if (lane == 0) red2[warp] = lsum;
        __syncthreads();
        float gsum = 0.f;
#pragma unroll
        for (int w2i = 0; w2i < 8; w2i++) gsum += red2[w2i];
        float inv = 1.0f / gsum;
        for (int j = tid; j < Lmax; j += blockDim.x)
          pb[j] = __float2half(j < Ltot ? __expf(sc[j] - gmax) * inv : 0.f);
        __syncthreads();
      }
      grid.sync();
      MARK(4)

      // ---- P@V: block per (h, dblk-pair): 48 units; warps = 2 dblk x 4 k-slices ----
      {
        int ktot = Lmax / 16;
        int kper = (ktot + 3) / 4;
        for (int u = blockIdx.x; u < H * 16; u += gridDim.x) {
          int h = u >> 4, dp = u & 15;                // dp covers 2 of 32 dblks
          int dblk = dp * 2 + (warp >> 2);            // 32 dblks per head
          int kq = warp & 3;
          float c[4] = {0.f, 0.f, 0.f, 0.f};
          const bf16* brow = &VTl[(size_t)(dblk * 8 + gid) * Lmax];
          const bf16* arow = p.probs + (size_t)h * T * Lmax;
#pragma unroll 4
          for (int ks = 0; ks < kper; ks++) {
            int kstep = kq * kper + ks;
            if (kstep >= ktot) break;
            int k0 = kstep * 16;
            unsigned a[4], b[2];
            load_a(a, arow, Lmax, k0, gid, tig);
            b[0] = *reinterpret_cast<const unsigned*>(&brow[k0 + tig * 2]);
            b[1] = *reinterpret_cast<const unsigned*>(&brow[k0 + tig * 2 + 8]);
            mma16816(c, a, b);
          }
          float* pp = &parts[(warp * 32 + lane) * 4];
          pp[0] = c[0]; pp[1] = c[1]; pp[2] = c[2]; pp[3] = c[3];
          __syncthreads();
          if (kq == 0) {
            float r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#pragma unroll
            for (int qq = 0; qq < 4; qq++) {
              const float* s = &parts[(((warp >> 2) * 4 + qq) * 32 + lane) * 4];
              r0 += s[0]; r1 += s[1]; r2 += s[2]; r3 += s[3];
            }
            int d0 = dblk * 8 + tig * 2;
            p.attn[gid * QD + h * HD + d0] = __float2half(r0);
            p.attn[gid * QD + h * HD + d0 + 1] = __float2half(r1);
            p.attn[(gid + 8) * QD + h * HD + d0] = __float2half(r2);
            p.attn[(gid + 8) * QD + h * HD + d0 + 1] = __float2half(r3);
          }
          __syncthreads();
        }
      }
      grid.sync();
      MARK(5)

      // ---- o_proj: rows8 x k1024, 2 k-windows -> xp[0..1] ----
      rows8_mma(p.wo + (size_t)l * D * QD, p.so_ + (size_t)l * D, QD, D / 16, 2, p.attn, QD, 2, nullptr, nullptr);
      grid.sync();
      MARK(6)

      // ---- norm2: combine o_proj partials (gate = mod_in[2048..]) ----
      mini_norm(xcur, xalt, mod_post, mod_in + 2048, 2);
      { float* tmp = xcur; xcur = xalt; xalt = tmp; }
      grid.sync();
      MARK(7)

      // ---- gate/up: ROWS16 tiles over W2 ----
      {
        const w8* wmat = p.w2 + (size_t)l * 2 * F * D;
        const float* s2l = p.s2 + (size_t)l * 2 * F;
        const int tiles_total = 2 * F / 32;
        const int nblk = warp >> 1, kq = warp & 1;
        const int TROWB = D + 16;
        w8* tiles8 = reinterpret_cast<w8*>(tiles);
        auto issue = [&](int tileidx, int buf) {
          const w8* src = wmat + (size_t)tileidx * 32 * D;
          w8* dst = tiles8 + (size_t)buf * 32 * TROWB;
          for (int i = tid; i < 32 * D / 16; i += blockDim.x) {
            int r = i >> 6, c = i & 63;
            cp16(&dst[r * TROWB + c * 16], &src[r * D + c * 16]);
          }
          cp_commit();
        };
        int t0 = blockIdx.x;
        if (t0 < tiles_total) issue(t0, 0);
        int buf = 0;
        for (int tile = t0; tile < tiles_total; tile += gridDim.x) {
          int next = tile + gridDim.x;
          cp_wait<0>();
          __syncthreads();
          if (next < tiles_total) issue(next, buf ^ 1);
          const w8* tp = tiles8 + (size_t)buf * 32 * TROWB;
          float c[4] = {0.f, 0.f, 0.f, 0.f};
          const w8* brow = &tp[(nblk * 8 + gid) * TROWB];
#pragma unroll 4
          for (int ks = 0; ks < 32; ks++) {
            int k0 = kq * 512 + ks * 16;
            unsigned a[4], b[2];
            load_a(a, p.xn, D, k0, gid, tig);
            b[0] = cvt_e4m3x2(*reinterpret_cast<const unsigned short*>(&brow[k0 + tig * 2]));
            b[1] = cvt_e4m3x2(*reinterpret_cast<const unsigned short*>(&brow[k0 + tig * 2 + 8]));
            mma16816(c, a, b);
          }
          float* pp = &parts[((nblk * 2 + kq) * 32 + lane) * 4];
          pp[0] = c[0]; pp[1] = c[1]; pp[2] = c[2]; pp[3] = c[3];
          __syncthreads();
          if (kq == 0) {
            float r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#pragma unroll
            for (int qq = 0; qq < 2; qq++) {
              const float* s = &parts[((nblk * 2 + qq) * 32 + lane) * 4];
              r0 += s[0]; r1 += s[1]; r2 += s[2]; r3 += s[3];
            }
            int gr = tile * 32 + nblk * 8 + tig * 2;
            r0 *= s2l[gr]; r1 *= s2l[gr + 1]; r2 *= s2l[gr]; r3 *= s2l[gr + 1];
            int f = gr >> 1;
            p.hmlp[gid * F + f] = __float2half(gelu_tanh(r0) * r1);
            p.hmlp[(gid + 8) * F + f] = __float2half(gelu_tanh(r2) * r3);
          }
          __syncthreads();
          buf ^= 1;
        }
      }
      grid.sync();
      MARK(8)

      // ---- down: rows8 x k1024, 4 k-windows -> xp[0..3] ----
      rows8_mma(p.w3 + (size_t)l * D * F, p.s3 + (size_t)l * D, F, D / 16, 4, p.hmlp, F, 2, nullptr, nullptr);
      grid.sync();
      MARK(9)

      prev_gate = mod_post + 2048;
    }  // layers

    // ---- final norm (combines last down) + aout + Euler ----
    {
      const float* modf = &p.mod_final[(size_t)step * 3072];
      mini_norm(xcur, xalt, modf, prev_gate, 4);
      grid.sync();
      for (int u = gwarp; u < T * AD; u += ngwarp) {
        int t = u / AD, a = u % AD;
        float acc = 0.f;
        const float* w = &p.w_aout[a * D];
        for (int d = lane; d < D; d += 32) acc = fmaf(w[d], bf2f(p.xn[t * D + d]), acc);
        acc = warp_sum(acc);
        if (lane == 0) p.x_t[t * AD + a] += dt * (acc + p.b_aout[a]);
      }
    }
    grid.sync();
    MARK(10)
  }  // steps
}
