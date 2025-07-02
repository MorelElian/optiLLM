#ifndef FORWARD_H
#define FORWARD_H
#include <stddef.h>
#include "types.h"
static inline int MINDEX0() { return 0; }

static inline int MINDEX1(int N1, int i1) { return i1; }

static inline int MINDEX2(int N1, int N2, int i1, int i2) { return i1 * N2 + i2; }

static inline int MINDEX3(int N1, int N2, int N3, int i1, int i2, int i3) {
  return i1 * N2 * N3 + i2 * N3 + i3;
}

static inline int MINDEX4(int N1, int N2, int N3, int N4, int i1, int i2, int i3,
                   int i4) {
  return i1 * N2 * N3 * N4 + i2 * N3 * N4 + i3 * N4 + i4;
}
static inline size_t MSIZE0() { return 1; }

static inline size_t MSIZE1(int N1) { return (size_t)N1; }

static inline size_t MSIZE2(int N1, int N2) { return (size_t)N1 * (size_t)N2; }

static inline size_t MSIZE3(int N1, int N2, int N3) {
  return (size_t)N1 * (size_t)N2 * (size_t)N3;
}

static inline size_t MSIZE4(int N1, int N2, int N3, int N4) {
  return (size_t)N1 * (size_t)N2 * (size_t)N3 * (size_t)N4;
}

#define CALLOC0(T) (T *)calloc(MSIZE0(), sizeof(T))
#define CALLOC1(T, N1) (T *)calloc(MSIZE1(N1), sizeof(T))
#define CALLOC2(T, N1, N2) (T *)calloc(MSIZE2(N1, N2), sizeof(T))
#define CALLOC3(T, N1, N2, N3) (T *)calloc(MSIZE3(N1, N2, N3), sizeof(T))
#define CALLOC4(T, N1, N2, N3, N4)                                             \
  (T *)calloc(MSIZE4(N1, N2, N3, N4), sizeof(T))

void forward(int token, int vocabulary_len,
               int context_len, int layer_count, int q_head_count,
               int kv_head_count, int q_head_per_kv_head_count,
               int embedding_dim, int head_dim, int q_dim, int kv_dim,
               int hidden_dim,
               float epsilon,
               float *embedding_weight, float *mha_norm_weight,
               float *mha_q_weight, float *mha_k_weight, float *mha_v_weight,
               float *mha_out_weight, float *ffn_norm_weight,
               float *ffn_fc_weight, float *ffn_up_weight,
               float *ffn_out_weight, float *out_norm_weight, float *out_weight,
               float *k_cache, float *v_cache, float *logits, int pos,
               int logits_count);

void generate_prompt_proc(transformer_t *transformer_t, int * sequence, int sequence_len);
void softmax(int col_count, int col_stride, float *x) ;
void driver(transformer_t *transformer, int token, int pos,
              int logits_count);
#endif