// MolmoAct2 flow-matching action-expert megakernel v1 — RTX 5090 (sm_120)
//
// ONE cooperative launch runs the entire 10-step flow loop of the MolmoAct2 action
// expert: 36 DiT blocks (adaLN-modulated self-attention over the 30 action tokens,
// cross-attention to per-layer projected VLM KV, silu-gated MLP), final layer and
// Euler updates. Engine ported from the pi0.5 megakernel v5: tensor-core mma with
// cp.async double-buffered weight tiles, mini-stages for norms/softmax, k-window
// splits with partials folded into the next norm stage.
//
// Dims: D=768, HEADS=8, HD2=96, F=3072, NL=36, T=32 padded (Tv=30 valid), AD=32.
// adaLN modulations for the fixed schedule are precomputed on the host with the
// model's own modulation cache (bit-identical conditioning).
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define D 768
#define HEADS 8
#define HD2 96
#define F 3072
#define NL 36
#define T 32
#define AD 32
#define QKVR 2304
#define EPS 1e-6f
#define ROWS8 8

using bf16 = __nv_bfloat16;

struct MAParams {
  // weights, [rows][in] bf16 per layer, plus fp32 biases
  const bf16* __restrict__ w_qkv;   // [NL][2304][768]
  const float* __restrict__ b_qkv;  // [NL][2304]
  const bf16* __restrict__ w_so;    // [NL][768][768] self out
  const float* __restrict__ b_so;   // [NL][768]
  const bf16* __restrict__ w_cq;    // [NL][768][768] cross q
  const float* __restrict__ b_cq;   // [NL][768]
  const bf16* __restrict__ w_co;    // [NL][768][768] cross out
  const float* __restrict__ b_co;   // [NL][768]
  const bf16* __restrict__ w_gu;    // [NL][6144][768] gate/up interleaved
  const float* __restrict__ b_gu;   // [NL][6144] interleaved
  const bf16* __restrict__ w_dn;    // [NL][768][3072]
  const float* __restrict__ b_dn;   // [NL][768]
  const float* __restrict__ mods;      // [S][NL][3][3][768] (scale|shift|gate per norm)
  const float* __restrict__ mod_final; // [S][2][768] (scale|shift)
  const float* __restrict__ rope_cos;  // [T][48]
  const float* __restrict__ rope_sin;  // [T][48]
  const float* __restrict__ w_ain;     // [768][32]
  const float* __restrict__ b_ain;     // [768]
  const bf16* __restrict__ w_aout;     // [32][768]
  const float* __restrict__ b_aout;    // [32]
  const bf16* __restrict__ kctx;   // [NL][Lkmax][768] projected+normed cross K
  const bf16* __restrict__ vtctx;  // [NL][768][Lkmax] cross V transposed
  int Lk, Lkmax, Tv, adim, num_steps;
  float* __restrict__ x_t;   // [T][AD]
  float* __restrict__ x;     // [T][D]
  float* __restrict__ xp;    // [3][T][D]
  unsigned long long* __restrict__ stage_cycles;
  bf16* __restrict__ xn;     // [T][D]
  bf16* __restrict__ qb;     // [T][768]
  bf16* __restrict__ kb;     // [T][768]
  bf16* __restrict__ vb;     // [T][768]
  bf16* __restrict__ sattn;  // [T][768]
  bf16* __restrict__ qc;     // [T][768]  cross q raw
  bf16* __restrict__ qcn;    // [T][768]  cross q normed
  bf16* __restrict__ attn;   // [T][768]
  bf16* __restrict__ hmlp;   // [T][F]
  float* __restrict__ scores;  // [HEADS][T][Lkmax]
  bf16* __restrict__ probs;    // [HEADS][T][Lkmax]
};

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  unsigned saddr = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void mma16816(float c[4], const unsigned a[4], const unsigned b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void load_a(unsigned a[4], const bf16* __restrict__ act, int stride,
                                       int k0, int row0, int tig) {
  const bf16* ar0 = &act[row0 * stride + k0 + tig * 2];
  const bf16* ar1 = &act[(row0 + 8) * stride + k0 + tig * 2];
  a[0] = *reinterpret_cast<const unsigned*>(ar0);
  a[1] = *reinterpret_cast<const unsigned*>(ar1);
  a[2] = *reinterpret_cast<const unsigned*>(ar0 + 8);
  a[3] = *reinterpret_cast<const unsigned*>(ar1 + 8);
}

extern "C" __global__ void __launch_bounds__(256, 1)
ma2_megakernel(MAParams p) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ unsigned char smem_raw[];
  bf16* tiles = reinterpret_cast<bf16*>(smem_raw);  // [2][8][1032] max
  float* parts = reinterpret_cast<float*>(smem_raw + 2 * ROWS8 * (1024 + 8) * 2);  // [8][32][4] x2 mtiles

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int nwarp = blockDim.x >> 5;
  const int gwarp = blockIdx.x * nwarp + warp;
  const int ngwarp = gridDim.x * nwarp;
  const int gtid = blockIdx.x * blockDim.x + tid;
  const int ngtid = gridDim.x * blockDim.x;
  const int gid = lane >> 2, tig = lane & 3;
  const int Lk = p.Lk, Lkmax = p.Lkmax, Tv = p.Tv;
  const float dt = 1.0f / p.num_steps;
  unsigned long long _t0 = clock64();
#define MARK(idx)                                                        \
  if (p.stage_cycles && blockIdx.x == 0 && threadIdx.x == 0) {           \
    unsigned long long now = clock64();                                  \
    p.stage_cycles[idx] += now - _t0;                                    \
    _t0 = now;                                                           \
  }

  // ---- norm mini-stage: block t < Tv; combines nsl xp slices with gate,
  //      then writes xn = modulate(rms(x), scale, shift). mod = {scale,shift,...}
  auto mini_norm = [&](const float* mod, const float* gate, int nsl, const float* bias) {
    if (blockIdx.x >= Tv) return;
    int t = blockIdx.x;
    __shared__ float red[8];
    if (nsl > 0) {
      for (int d = tid; d < D; d += blockDim.x) {
        float add = (bias != nullptr) ? bias[d] : 0.f;
        for (int s = 0; s < nsl; s++) add += p.xp[(s * T + t) * D + d];
        p.x[t * D + d] += add * gate[d];
      }
      __syncthreads();
    }
    float ss = 0.f;
    for (int d = tid; d < D; d += blockDim.x) {
      float v = p.x[t * D + d];
      ss += v * v;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w2i = 0; w2i < 8; w2i++) tot += red[w2i];
    float r = rsqrtf(tot / D + EPS);
    for (int d = tid; d < D; d += blockDim.x) {
      float v = p.x[t * D + d] * r * (1.f + mod[d]) + mod[D + d];
      p.xn[t * D + d] = __float2bfloat16(v);
    }
  };

  // ---- rows8 mma tile pipeline, m=32 tokens (2 m-tiles) ----
  // kind: 0 = qkv split writes (qb/kb/vb), 1 = plain bf16 out buffer,
  //       2 = xp[kwin] partial (bias folded into kwin 0), 3 = gate/up silu pairs
  auto rows8_mma = [&](const bf16* wbase, int wstride, int kwlen, int nrowblk, int nkwin,
                       const bf16* act, int astride, int kind, bf16* outbuf, int outstride,
                       const float* bias) {
    const int TROW = kwlen + 8;
    const int tiles_total = nrowblk * nkwin;
    const int kslice = kwlen / 8;        // per-warp k length (96 or 128)
    const int ksteps = kslice / 16;      // 6 or 8
    auto issue = [&](int tileidx, int buf) {
      int rowblk = tileidx % nrowblk, kwin = tileidx / nrowblk;
      const bf16* src = wbase + (size_t)rowblk * ROWS8 * wstride + kwin * kwlen;
      bf16* dst = tiles + (size_t)buf * ROWS8 * TROW;
      int chunks = kwlen / 8;
      for (int i = tid; i < ROWS8 * chunks; i += blockDim.x) {
        int r = i / chunks, c = i % chunks;
        cp16(&dst[r * TROW + c * 8], &src[(size_t)r * wstride + c * 8]);
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
      const bf16* tp = tiles + (size_t)buf * ROWS8 * TROW;
      float c0[4] = {0.f, 0.f, 0.f, 0.f};  // tokens gid, gid+8
      float c1[4] = {0.f, 0.f, 0.f, 0.f};  // tokens gid+16, gid+24
      const bf16* brow = &tp[gid * TROW];
      const bf16* actw = act + kwin * kwlen;
      for (int ks = 0; ks < ksteps; ks++) {
        int k0 = warp * kslice + ks * 16;
        unsigned a[4], b[2];
        b[0] = *reinterpret_cast<const unsigned*>(&brow[k0 + tig * 2]);
        b[1] = *reinterpret_cast<const unsigned*>(&brow[k0 + tig * 2 + 8]);
        load_a(a, actw, astride, k0, gid, tig);
        mma16816(c0, a, b);
        load_a(a, actw, astride, k0, gid + 16, tig);
        mma16816(c1, a, b);
      }
      float* pp = &parts[(warp * 32 + lane) * 8];
      pp[0] = c0[0]; pp[1] = c0[1]; pp[2] = c0[2]; pp[3] = c0[3];
      pp[4] = c1[0]; pp[5] = c1[1]; pp[6] = c1[2]; pp[7] = c1[3];
      __syncthreads();
      if (warp == 0) {
        float r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int qq = 0; qq < 8; qq++) {
          const float* s = &parts[(qq * 32 + lane) * 8];
#pragma unroll
          for (int m = 0; m < 8; m++) r[m] += s[m];
        }
        int gr = rowblk * ROWS8 + tig * 2;
        float bv0 = 0.f, bv1 = 0.f;
        if (bias != nullptr && (kind != 2 || kwin == 0)) { bv0 = bias[gr]; bv1 = bias[gr + 1]; }
        // token rows: gid, gid+8, gid+16, gid+24; cols gr, gr+1
        int trow[4] = {gid, gid + 8, gid + 16, gid + 24};
        float vals[4][2] = {{r[0] + bv0, r[1] + bv1}, {r[2] + bv0, r[3] + bv1},
                            {r[4] + bv0, r[5] + bv1}, {r[6] + bv0, r[7] + bv1}};
        if (kind == 0) {
          // qkv: rows [0,768) q, [768,1536) k, [1536,2304) v
          bf16* dst = (gr < 768) ? p.qb : (gr < 1536 ? p.kb : p.vb);
          int col = gr - ((gr < 768) ? 0 : (gr < 1536 ? 768 : 1536));
#pragma unroll
          for (int m = 0; m < 4; m++) {
            dst[trow[m] * D + col] = __float2bfloat16(vals[m][0]);
            dst[trow[m] * D + col + 1] = __float2bfloat16(vals[m][1]);
          }
        } else if (kind == 1) {
#pragma unroll
          for (int m = 0; m < 4; m++) {
            outbuf[trow[m] * outstride + gr] = __float2bfloat16(vals[m][0]);
            outbuf[trow[m] * outstride + gr + 1] = __float2bfloat16(vals[m][1]);
          }
        } else if (kind == 2) {
          float* xps = p.xp + (size_t)kwin * T * D;
#pragma unroll
          for (int m = 0; m < 4; m++) {
            xps[trow[m] * D + gr] = vals[m][0];
            xps[trow[m] * D + gr + 1] = vals[m][1];
          }
        } else {  // gate/up interleaved: even row gate, odd up; f = gr>>1
          int f = gr >> 1;
#pragma unroll
          for (int m = 0; m < 4; m++)
            p.hmlp[trow[m] * F + f] = __float2bfloat16(silu(vals[m][0]) * vals[m][1]);
        }
      }
      __syncthreads();
      buf ^= 1;
    }
  };

  for (int step = 0; step < p.num_steps; step++) {
    // ---- embed: x = action_embed(x_t) ----
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

    const float* prev_gate = nullptr;
    const float* prev_bias = nullptr;
    for (int l = 0; l < NL; l++) {
      const float* modl = &p.mods[(((size_t)step * NL + l) * 3) * 3 * D];
      const float* mod_msa = modl;              // {scale, shift, gate}
      const float* mod_mca = modl + 3 * D;
      const float* mod_mlp = modl + 6 * D;
      const bf16* kcl = p.kctx + (size_t)l * Lkmax * D;
      const bf16* vtl = p.vtctx + (size_t)l * D * Lkmax;

      // ---- norm1 (combine prev layer mlp partials) ----
      mini_norm(mod_msa, prev_gate, prev_gate ? 3 : 0, prev_bias);
      grid.sync();
      MARK(1)

      // ---- fused QKV GEMM ----
      rows8_mma(p.w_qkv + (size_t)l * QKVR * D, D, D, QKVR / ROWS8, 1, p.xn, D, 0,
                nullptr, 0, p.b_qkv + (size_t)l * QKVR);
      grid.sync();
      MARK(2)

      // ---- self-attention mini-stage: block per (h,t), qk-norm + rope + attn ----
      for (int u = blockIdx.x; u < HEADS * Tv; u += gridDim.x) {
        int h = u / Tv, t = u % Tv;
        // smem: qv[96], kv-row scratch, scores[Tv], pacc[nwarp][96]
        float* qv = parts;             // reuse parts region (>= 8KB): qv 96
        float* sc = qv + 128;          // scores up to 32
        float* pacc = sc + 32;         // [8][96]
        // q: norm + rope
        float ss = 0.f;
        for (int d2 = tid; d2 < HD2; d2 += blockDim.x) {
          float v = bf2f(p.qb[t * D + h * HD2 + d2]);
          qv[d2] = v;
          ss += v * v;
        }
        __syncthreads();
        // block reduce ss
        ss = warp_sum(ss);
        __shared__ float red6[8];
        if (lane == 0) red6[warp] = ss;
        __syncthreads();
        float tot = 0.f;
#pragma unroll
        for (int w2i = 0; w2i < 8; w2i++) tot += red6[w2i];
        float rq = rsqrtf(tot / HD2 + EPS);
        __syncthreads();
        if (tid < 48) {
          float x1 = qv[tid] * rq, x2 = qv[tid + 48] * rq;
          float c = p.rope_cos[t * 48 + tid], s = p.rope_sin[t * 48 + tid];
          qv[tid] = x1 * c - x2 * s;
          qv[tid + 48] = x1 * s + x2 * c;
        }
        __syncthreads();
        // scores: warp per key j
        for (int j = warp; j < Tv; j += nwarp) {
          // k_j: norm + rope, computed lane-cooperatively (3 elems per lane)
          float kss = 0.f;
          float kvv[3];
#pragma unroll
          for (int m = 0; m < 3; m++) {
            int d2 = lane + m * 32;
            kvv[m] = bf2f(p.kb[j * D + h * HD2 + d2]);
            kss += kvv[m] * kvv[m];
          }
          kss = warp_sum(kss);
          float rk = rsqrtf(kss / HD2 + EPS);
          // rope on k: element d2 pairs with d2+-48
          float acc = 0.f;
#pragma unroll
          for (int m = 0; m < 3; m++) {
            int d2 = lane + m * 32;
            float kn = kvv[m] * rk;
            float other;
            // fetch pair element (normed) via shfl: pair index d2^?? not aligned; recompute:
            int pd = (d2 < 48) ? d2 + 48 : d2 - 48;
            // reload pair raw (cheap, L1):
            other = bf2f(p.kb[j * D + h * HD2 + pd]) * rk;
            float c = p.rope_cos[j * 48 + (d2 < 48 ? d2 : pd)];
            float s = p.rope_sin[j * 48 + (d2 < 48 ? d2 : pd)];
            float kr = (d2 < 48) ? (kn * c - other * s) : (other * s + kn * c);
            // careful: for d2>=48: rotated = x1*sin + x2*cos where x1=other, x2=kn
            acc = fmaf(kr, qv[d2], acc);
          }
          acc = warp_sum(acc) * 0.10206207261596577f;  // 1/sqrt(96)
          if (lane == 0) sc[j] = acc;
        }
        __syncthreads();
        // softmax over Tv (single warp)
        if (warp == 0) {
          float m = -1e30f;
          for (int j = lane; j < Tv; j += 32) m = fmaxf(m, sc[j]);
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
          float ssum = 0.f;
          for (int j = lane; j < Tv; j += 32) {
            float e = __expf(sc[j] - m);
            sc[j] = e;
            ssum += e;
          }
          ssum = warp_sum(ssum);
          float inv = 1.0f / ssum;
          for (int j = lane; j < Tv; j += 32) sc[j] *= inv;
        }
        __syncthreads();
        // out[d] = sum_j p_j v_j[d]: warp per d-slice
        for (int d2 = tid; d2 < HD2; d2 += blockDim.x) {
          float acc = 0.f;
          for (int j = 0; j < Tv; j++) acc = fmaf(sc[j], bf2f(p.vb[j * D + h * HD2 + d2]), acc);
          p.sattn[t * D + h * HD2 + d2] = __float2bfloat16(acc);
        }
        __syncthreads();
      }
      grid.sync();
      MARK(3)

      // ---- self out proj -> xp[0] ----
      rows8_mma(p.w_so + (size_t)l * D * D, D, D, D / ROWS8, 1, p.sattn, D, 2,
                nullptr, 0, p.b_so + (size_t)l * D);
      grid.sync();
      MARK(4)

      // ---- norm2 (combine self-attn, gate_msa) ----
      mini_norm(mod_mca, mod_msa + 2 * D, 1, nullptr);
      grid.sync();
      MARK(5)

      // ---- cross q proj ----
      rows8_mma(p.w_cq + (size_t)l * D * D, D, D, D / ROWS8, 1, p.xn, D, 1,
                p.qc, D, p.b_cq + (size_t)l * D);
      grid.sync();
      MARK(6)

      // ---- cross q norm mini-stage: block per t: 8 head-norms ----
      if (blockIdx.x < Tv) {
        int t = blockIdx.x;
        // warp per head
        for (int h = warp; h < HEADS; h += nwarp) {
          float ss2 = 0.f;
          float v3[3];
#pragma unroll
          for (int m = 0; m < 3; m++) {
            int d2 = lane + m * 32;
            v3[m] = bf2f(p.qc[t * D + h * HD2 + d2]);
            ss2 += v3[m] * v3[m];
          }
          ss2 = warp_sum(ss2);
          float r = rsqrtf(ss2 / HD2 + EPS);
#pragma unroll
          for (int m = 0; m < 3; m++)
            p.qcn[t * D + h * HD2 + lane + m * 32] = __float2bfloat16(v3[m] * r);
        }
      }
      grid.sync();
      MARK(7)

      // ---- cross scores mma: block per 8-key block, warp = head ----
      {
        int jblocks = (Lk + 7) / 8;
        for (int jb = blockIdx.x; jb < jblocks; jb += gridDim.x) {
          int h = warp;
          float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
          const bf16* brow = &kcl[(size_t)(jb * 8 + gid) * D + h * HD2];
          bool valid = (jb * 8 + gid) < Lk;
#pragma unroll
          for (int ks = 0; ks < 6; ks++) {
            int k0 = ks * 16;
            unsigned a[4], b[2];
            b[0] = valid ? *reinterpret_cast<const unsigned*>(&brow[k0 + tig * 2]) : 0u;
            b[1] = valid ? *reinterpret_cast<const unsigned*>(&brow[k0 + tig * 2 + 8]) : 0u;
            load_a(a, p.qcn + h * HD2, D, k0, gid, tig);
            mma16816(c0, a, b);
            load_a(a, p.qcn + h * HD2, D, k0, gid + 16, tig);
            mma16816(c1, a, b);
          }
          int j0 = jb * 8 + tig * 2;
          float* sc = p.scores + (size_t)h * T * Lkmax;
          const float scl = 0.10206207261596577f;
          if (j0 < Lk) {
            sc[gid * Lkmax + j0] = c0[0] * scl;
            sc[(gid + 8) * Lkmax + j0] = c0[2] * scl;
            sc[(gid + 16) * Lkmax + j0] = c1[0] * scl;
            sc[(gid + 24) * Lkmax + j0] = c1[2] * scl;
          }
          if (j0 + 1 < Lk) {
            sc[gid * Lkmax + j0 + 1] = c0[1] * scl;
            sc[(gid + 8) * Lkmax + j0 + 1] = c0[3] * scl;
            sc[(gid + 16) * Lkmax + j0 + 1] = c1[1] * scl;
            sc[(gid + 24) * Lkmax + j0 + 1] = c1[3] * scl;
          }
        }
      }
      grid.sync();
      MARK(8)

      // ---- cross softmax mini-stage: block per (h,t) ----
      for (int u = blockIdx.x; u < HEADS * Tv; u += gridDim.x) {
        int h = u / Tv, t = u % Tv;
        const float* sc = p.scores + ((size_t)h * T + t) * Lkmax;
        bf16* pb = p.probs + ((size_t)h * T + t) * Lkmax;
        __shared__ float red2[8];
        float lmax = -1e30f;
        for (int j = tid; j < Lk; j += blockDim.x) lmax = fmaxf(lmax, sc[j]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
        if (lane == 0) red2[warp] = lmax;
        __syncthreads();
        float gmax = -1e30f;
#pragma unroll
        for (int w2i = 0; w2i < 8; w2i++) gmax = fmaxf(gmax, red2[w2i]);
        __syncthreads();
        float lsum = 0.f;
        for (int j = tid; j < Lk; j += blockDim.x) lsum += __expf(sc[j] - gmax);
        lsum = warp_sum(lsum);
        if (lane == 0) red2[warp] = lsum;
        __syncthreads();
        float gsum = 0.f;
#pragma unroll
        for (int w2i = 0; w2i < 8; w2i++) gsum += red2[w2i];
        float inv = 1.0f / gsum;
        for (int j = tid; j < Lkmax; j += blockDim.x)
          pb[j] = __float2bfloat16(j < Lk ? __expf(sc[j] - gmax) * inv : 0.f);
        __syncthreads();
      }
      grid.sync();
      MARK(9)

      // ---- P@V mma: block per (h, dtri of 4 dblocks): 8*3=24 blocks x 8... ----
      // 96 dims per head = 12 dblocks of 8; use warp-unit (h, dblk): 96 units
      {
        int ksteps2 = Lkmax / 16;
        for (int u = gwarp; u < HEADS * 12; u += ngwarp) {
          int h = u / 12, dblk = u % 12;
          float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
          const bf16* brow = &vtl[(size_t)(h * HD2 + dblk * 8 + gid) * Lkmax];
          const bf16* arow = p.probs + (size_t)h * T * Lkmax;
#pragma unroll 4
          for (int ks = 0; ks < ksteps2; ks++) {
            int k0 = ks * 16;
            unsigned a[4], b[2];
            b[0] = *reinterpret_cast<const unsigned*>(&brow[k0 + tig * 2]);
            b[1] = *reinterpret_cast<const unsigned*>(&brow[k0 + tig * 2 + 8]);
            load_a(a, arow, Lkmax, k0, gid, tig);
            mma16816(c0, a, b);
            load_a(a, arow, Lkmax, k0, gid + 16, tig);
            mma16816(c1, a, b);
          }
          int d0 = h * HD2 + dblk * 8 + tig * 2;
          p.attn[gid * D + d0] = __float2bfloat16(c0[0]);
          p.attn[gid * D + d0 + 1] = __float2bfloat16(c0[1]);
          p.attn[(gid + 8) * D + d0] = __float2bfloat16(c0[2]);
          p.attn[(gid + 8) * D + d0 + 1] = __float2bfloat16(c0[3]);
          p.attn[(gid + 16) * D + d0] = __float2bfloat16(c1[0]);
          p.attn[(gid + 16) * D + d0 + 1] = __float2bfloat16(c1[1]);
          p.attn[(gid + 24) * D + d0] = __float2bfloat16(c1[2]);
          p.attn[(gid + 24) * D + d0 + 1] = __float2bfloat16(c1[3]);
        }
      }
      grid.sync();
      MARK(10)

      // ---- cross out proj -> xp[0] ----
      rows8_mma(p.w_co + (size_t)l * D * D, D, D, D / ROWS8, 1, p.attn, D, 2,
                nullptr, 0, p.b_co + (size_t)l * D);
      grid.sync();
      MARK(11)

      // ---- norm3 (combine cross-attn, gate_mca) ----
      mini_norm(mod_mlp, mod_mca + 2 * D, 1, nullptr);
      grid.sync();
      MARK(12)

      // ---- gate/up (interleaved, silu) -> hmlp ----
      rows8_mma(p.w_gu + (size_t)l * 2 * F * D, D, D, 2 * F / ROWS8, 1, p.xn, D, 3,
                nullptr, 0, p.b_gu + (size_t)l * 2 * F);
      grid.sync();
      MARK(13)

      // ---- down: k=3072 in 3 windows of 1024 -> xp[0..2] ----
      rows8_mma(p.w_dn + (size_t)l * D * F, F, 1024, D / ROWS8, 3, p.hmlp, F, 2,
                nullptr, 0, p.b_dn + (size_t)l * D);
      grid.sync();
      MARK(14)

      prev_gate = mod_mlp + 2 * D;
      prev_bias = nullptr;  // down bias folded into xp[0] via kind==2 kwin==0
    }  // layers

    // ---- final: norm (combine mlp) + modulate + linear + Euler ----
    {
      const float* modf = &p.mod_final[(size_t)step * 2 * D];
      mini_norm(modf, prev_gate, 3, nullptr);
      grid.sync();
      for (int u = gwarp; u < Tv * AD; u += ngwarp) {
        int t = u / AD, a = u % AD;
        if (a >= p.adim) continue;  // padded action dims stay zero
        float acc = 0.f;
        const bf16* w = &p.w_aout[a * D];
        for (int d = lane; d < D; d += 32) acc = fmaf(bf2f(w[d]), bf2f(p.xn[t * D + d]), acc);
        acc = warp_sum(acc);
        if (lane == 0) p.x_t[t * AD + a] += dt * (acc + p.b_aout[a]);
      }
    }
    grid.sync();
    MARK(15)
  }  // steps
}
