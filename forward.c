#include "forward.h"
#include "dumper.h"
inline float *forward(int token, int vocabulary_len,
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
               int logits_count) {

  float *embedding = CALLOC1(float, embedding_dim);
  float *mha_norm = CALLOC1(float, embedding_dim);
  float *mha_q = CALLOC2(float, q_head_count, head_dim);
  float *mha_score = CALLOC2(float, q_head_count, context_len);
  float *mha_blend = CALLOC2(float, q_head_count, head_dim);
  float *mha_att = CALLOC1(float, embedding_dim);
  float *mha_out = CALLOC1(float, embedding_dim);
  float *ffn_norm = CALLOC1(float, embedding_dim);
  float *ffn_fc = CALLOC1(float, hidden_dim);
  float *ffn_up = CALLOC1(float, hidden_dim);
  float *ffn_out = CALLOC1(float, embedding_dim);

  // Get embedding representation of each token in the token sequence
  for (int e = 0; e < embedding_dim; e++) {
    embedding[MINDEX1(embedding_dim, e)] =
        embedding_weight[MINDEX2(vocabulary_len, embedding_dim, token, e)];
  }
  COND_DUMP(fwrite_array_float("./dump/embedding_start", &embedding[0],
                               embedding_dim,0),
            pos == 0)

  // forward all the layers
  for (int l = 0; l < layer_count; l++) {

    // attention rmsnorm
    rmsnorm(embedding_dim, mha_norm, embedding,
            &(mha_norm_weight[MINDEX2(layer_count,embedding_dim, l,0)]), epsilon);

    COND_DUMP(
        fwrite_array_float("./dump/mha_norm_init", &mha_norm[0], embedding_dim,l),
        pos == 0)
    // qkv matmuls for this position
    for (int q = 0; q < q_head_count; q++) {
      matmul(head_dim, embedding_dim,
             &mha_q[MINDEX2(q_head_count, head_dim, q, 0)], mha_norm,
             &mha_q_weight[MINDEX4(layer_count, q_head_count, head_dim,
                                   embedding_dim, l, q, 0, 0)]);
    }

    COND_DUMP(
        fwrite_array_float("./dump/mha_q_before_rope", &mha_q[0], head_dim,l),
        pos == 0)
    COND_DUMP(
        fwrite_array_float("./dump/mha_q_before_rope_second_head_group", &mha_q[q_head_per_kv_head_count*head_dim], head_dim,l),
        pos == 0)
    for (int h = 0; h < kv_head_count; h++) {
      matmul(head_dim, embedding_dim,
             &k_cache[MINDEX4(layer_count, kv_head_count, context_len, head_dim,
                              l, h, pos, 0)],
             mha_norm,
             &mha_k_weight[MINDEX4(layer_count, kv_head_count, head_dim,
                                   embedding_dim, l, h, 0, 0)]);
    }
    COND_DUMP(fwrite_array_float("./dump/mha_k_weight", mha_k_weight,
                                 embedding_dim * head_dim,l),
              pos == 0)
    COND_DUMP(
        fwrite_array_float("./dump/k_cache_before_rope", &k_cache[MINDEX4(layer_count, kv_head_count, head_dim,
                                   embedding_dim, l, 0, 0, 0)], head_dim,l),
        pos == 0)
    COND_DUMP(
        fwrite_array_float("./dump/k_cache_before_rope_second_head_group", &k_cache[MINDEX4(layer_count,kv_head_count,context_len,head_dim,l,1,pos,0)], head_dim,l),
        pos == 0)
    for (int h = 0; h < kv_head_count; h++) {
      matmul(head_dim, embedding_dim,
             &v_cache[MINDEX4(layer_count, kv_head_count, context_len, head_dim,
                              l, h, pos, 0)],
             mha_norm,
             &mha_v_weight[MINDEX4(layer_count, kv_head_count, head_dim,
                                   embedding_dim, l, h, 0, 0)]);
    }
    COND_DUMP(fwrite_array_float("./dump/v_cache", &v_cache[MINDEX4(layer_count, kv_head_count, context_len, head_dim,
                              l, 0, pos, 0)], head_dim,l),
              pos == 0)

    // RoPE q: complex-valued rotate q in each head
    for (int q = 0; q < q_head_count; q++) {
      rope(head_dim, &mha_q[MINDEX2(q_head_count, head_dim, q, 0)], pos);
    }
    COND_DUMP(
        fwrite_array_float("./dump/mha_q_after_rope", &mha_q[0], head_dim,l),
        pos == 0)

    // RoPE k: complex-valued rotate k in each head
    for (int h = 0; h < kv_head_count; h++) {
      rope(head_dim,
           &k_cache[MINDEX4(layer_count, kv_head_count, context_len, head_dim,
                            l, h, pos, 0)],
           pos);
    }
    COND_DUMP(
        fwrite_array_float("./dump/k_cache_after_rope", &k_cache[MINDEX4(layer_count, kv_head_count, context_len, head_dim,
                              l, 0, pos, 0)], head_dim,l),
        pos == 0)
    // multihead attention. iterate over all heads

    for (int q = 0; q < q_head_count; q++) {
      for (int p = 0; p <= pos; p++) {
        mha_score[MINDEX2(q_head_count, context_len, q, p)] =
            0.0f;
        for (int e = 0; e < head_dim; e++) {
          mha_score[MINDEX2(q_head_count, context_len, q, p)] +=
              mha_q[MINDEX2(q_head_count, head_dim, q, e)] *
              k_cache[MINDEX4(layer_count, kv_head_count, context_len, head_dim,
                              l, q / q_head_per_kv_head_count, p, e)];
        }
        mha_score[MINDEX2(q_head_count, context_len, q, p)] /= sqrtf(head_dim);
      }


      // softmax the scores to get attention weights
      softmax(pos + 1, context_len,
              &mha_score[MINDEX2(q_head_count, context_len, q, 0)]);

      // weighted sum of the values
      for (int e = 0; e < head_dim; e++) {
        mha_blend[MINDEX2(q_head_count, head_dim, q, e)] = 0.0f;
      }

      for (int p = 0; p <= pos; p++) {
        for (int e = 0; e < head_dim; e++) {
          mha_blend[MINDEX2(q_head_count, head_dim, q, e)] +=
              mha_score[MINDEX2(q_head_count, context_len, q, p)] *
              v_cache[MINDEX4(layer_count, kv_head_count, context_len, head_dim,
                              l, q / q_head_per_kv_head_count, p, e)];
        }
      }

    }
    COND_DUMP(fwrite_array_float("./dump/mha_blend", &mha_blend[0], head_dim,l),pos == 0)

    for (int q = 0; q < q_head_count; q++) {
      for (int e = 0; e < head_dim; e++) {
        mha_att[MINDEX1(embedding_dim, q * head_dim + e)] =
            mha_blend[MINDEX2(q_head_count, head_dim, q, e)];
      }
    }
    COND_DUMP(fwrite_array_float("./dump/mha_att", &mha_att[0], embedding_dim,l),
             pos == 0)

    // final matmul to get the output of the attention
    matmul(embedding_dim, embedding_dim, mha_out, mha_att,
           &mha_out_weight[MINDEX3(layer_count, embedding_dim, embedding_dim, l,
                                   0, 0)]);
    COND_DUMP(fwrite_array_float("./dump/mha_out", &mha_out[0], embedding_dim,l),
             pos == 0)

    // residual connection back into x

    for (int e = 0; e < embedding_dim; e++) {
      embedding[MINDEX1(embedding_dim, e)] +=
          mha_out[MINDEX1(embedding_dim, e)];
    }
    COND_DUMP(fwrite_array_float("./dump/embedding_mha", &embedding[0], embedding_dim,l),pos == 0)

    // ffn rmsnormRhum 3L
    rmsnorm(embedding_dim, ffn_norm, embedding,
            &ffn_norm_weight[MINDEX2(layer_count, embedding_dim,l,0)], epsilon);
    COND_DUMP(fwrite_array_float("./dump/ffn_rmsnorm", &ffn_norm[0], embedding_dim,l),
              pos == 0)
    matmul(hidden_dim, embedding_dim, ffn_fc, ffn_norm,
           &ffn_fc_weight[MINDEX3(layer_count, hidden_dim,embedding_dim,l,0,0)]);

    COND_DUMP(fwrite_array_float("./dump/ffn_fc", &ffn_fc[0], embedding_dim,l),pos == 0)
    matmul(hidden_dim, embedding_dim, ffn_up, ffn_norm,
           &ffn_up_weight[MINDEX3(layer_count, hidden_dim,embedding_dim,l, 0,0)]);
    COND_DUMP(fwrite_array_float("./dump/ffn_up", &ffn_up[0], hidden_dim,l),
              pos == 0)

    // SwiGLU non-linearity
    for (int e = 0; e < hidden_dim; e++) {
      ffn_fc[MINDEX1(hidden_dim, e)] *=
          (1.0f / (1.0f + expf(-ffn_fc[MINDEX1(hidden_dim, e)])));
      ffn_fc[MINDEX1(hidden_dim, e)] *= ffn_up[MINDEX1(hidden_dim, e)];
    }
    COND_DUMP(fwrite_array_float("./dump/ffn_fc_after_swiglu", &ffn_fc[0], hidden_dim,l),
              pos == 0)

    // final matmul to get the output of the ffn
    matmul(embedding_dim, hidden_dim, ffn_out, ffn_fc,
           &ffn_out_weight[MINDEX3(layer_count,embedding_dim,hidden_dim, l,0,0)]);
    COND_DUMP(fwrite_array_float("./dump/ffn_out", &ffn_out[0], embedding_dim,l),
              pos == 0)
    // residual connection
    for (int e = 0; e < embedding_dim; e++) {
      embedding[MINDEX1(embedding_dim, e)] +=
          ffn_out[MINDEX1(embedding_dim, e)];
    }
    COND_DUMP(fwrite_array_float("./dump/embedding_layer_end", &embedding[0], embedding_dim,l),
              pos == 0)
  }
  // final rmsnorm

  DUMP(fwrite_array_float("./dump/embedding_before_final_rmsnorm", &embedding[0], embedding_dim,0))
  rmsnorm(embedding_dim, embedding, embedding, out_norm_weight, epsilon);
  DUMP(fwrite_array_float("./dump/embedding_final", &embedding[0], embedding_dim,0))
  // classifier into logits
  matmul(vocabulary_len, embedding_dim, logits,
         embedding , out_weight);

  return &logits[0];
}