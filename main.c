/* Inference for Llama-2 transformer_t model in pure C */

#include <ctype.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "forward.h"
#include "dumper.h"
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SEQUENCE_CHUNK_MAX_LEN 512

// ----------------------------------------------------------------------------
// transformer_t model

// Print a vector with some statistics for debug purpose
void vector_print(size_t size, float *vector, size_t sample_size, char *name) {
  if (vector == NULL || size == 0) {
    fprintf(stderr, "Empty or invalid vector.\n");
    return;
  }

  // Print the first sample_size elements
  size_t end = (sample_size < size) ? sample_size : size;

  fprintf(stderr, "%6s: ", name ? name : "vector");
  for (size_t i = 0; i < end; i++) {
    fprintf(stderr, "%7.3f ", vector[i]);
  }

  // Print ellipsis if middle elements are skipped
  if (sample_size * 2 < size) {
    fprintf(stderr, "... ");
  }

  // Print the last sample_size elements
  size_t start_tail = (sample_size < size) ? size - sample_size : end;
  for (size_t i = start_tail; i < size; i++) {
    fprintf(stderr, "%7.3f ", vector[i]);
  }

  fprintf(stderr, "-- ");

  // Compute statistics
  float min = FLT_MAX;
  float max = -FLT_MAX;
  float sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    if (vector[i] < min)
      min = vector[i];
    if (vector[i] > max)
      max = vector[i];
    sum += vector[i];
  }

  float mean = sum / size;

  // Print statistics
  fprintf(stderr, "Min: %7.3f, Max: %7.3f, Mean: %7.3f, Sum: %f\n", min, max,
          mean, sum);
}

typedef struct {
  int embedding_dim;  // Token representation (embedding) dimension
  int hidden_dim;     // Intermediate representation dimension in the FFN
  int layer_count;    // Number of decoder layers
  int q_head_count;   // Number of query heads
  int kv_head_count;  // Number of key/value heads
  int vocabulary_len; // Vocabulary size
  int context_len;    // Maximum sequence length
} configuration_t;

typedef struct {
  // Embedding parameter set
  float *embedding_weight; // [vocabulary_len][embedding_dim]
  // Decoder parameter set
  // - Multi-head attention
  float *mha_norm_weight; // [layer_count][embedding_dim]
  float *mha_q_weight;    // (layer, dim, q_head_count * head_dim)
  float *mha_k_weight;    // (layer, dim, kv_head_count * head_dim)
  float *mha_v_weight;    // (layer, dim, kv_head_count * head_dim)
  float *mha_out_weight;  // (layer, q_head_count * head_dim, embedding_dim)
  // - Feed-forward network
  float *ffn_norm_weight; // (layer, dim) ffn_norm_weight
  float *ffn_fc_weight;   // (layer, hidden_dim, dim)
  float *ffn_up_weight;   // (layer, hidden_dim, dim)
  float *ffn_out_weight;  // (layer, dim, hidden_dim)
  // Output parameter set
  float *out_norm_weight; // (dim,)
  float *out_weight;      // (vocabulary_len, dim)
} parameter_set_t;

typedef struct {
  // Activations
  float *embedding;
  float *mha_norm;
  float *mha_q;
  float *mha_k_act;
  float *mha_v_act;
  float *mha_score; // buffer for scores/attention values (q_head_count,
                    // context_len)
  float *mha_blend;
  float *mha_att;
  float *mha_out;
  float *ffn_norm;
  float *ffn_fc;
  float *ffn_up;
  float *ffn_out;
  float *logits; // output logits
  // KV-cache
  float *k_cache; // (layer, context_len, dim)
  float *v_cache; // (layer, context_len, dim)
} state_t;

typedef struct {
  configuration_t config; // Hyperparameters
  parameter_set_t params; // Weights
  state_t state;          // Activations
  int fd;                 // file descriptor for memory mapping
  float *data;            // memory mapped data pointer
  ssize_t file_size;      // size of the checkpoint file in bytes
} transformer_t;

void state_malloc(state_t *s, configuration_t *p) {
  size_t kv_dim = (p->embedding_dim * p->kv_head_count) / p->q_head_count;
  size_t embed_len = SEQUENCE_CHUNK_MAX_LEN * p->embedding_dim;
  size_t hidden_len = SEQUENCE_CHUNK_MAX_LEN * p->hidden_dim;
  size_t score_len = p->q_head_count * p->context_len * p->context_len;
  size_t cache_len = p->context_len * p->layer_count * kv_dim;
  size_t logits_len = SEQUENCE_CHUNK_MAX_LEN * p->vocabulary_len;

  s->embedding = calloc(embed_len, sizeof(*s->embedding));
  s->mha_norm = calloc(embed_len, sizeof(*s->mha_norm));
  s->mha_q = calloc(embed_len, sizeof(*s->mha_q));
  s->mha_score = calloc(score_len, sizeof(*s->mha_score));
  s->mha_blend = calloc(embed_len, sizeof(*s->mha_blend));
  s->mha_att = calloc(embed_len, sizeof(*s->mha_att));
  s->mha_out = calloc(embed_len, sizeof(*s->mha_out));
  s->ffn_norm = calloc(embed_len, sizeof(*s->ffn_norm));
  s->ffn_fc = calloc(hidden_len, sizeof(*s->ffn_fc));
  s->ffn_up = calloc(hidden_len, sizeof(*s->ffn_up));
  s->ffn_out = calloc(embed_len, sizeof(*s->ffn_out));
  s->k_cache = calloc(cache_len, sizeof(*s->k_cache));
  s->v_cache = calloc(cache_len, sizeof(*s->v_cache));
  s->logits = calloc(logits_len, sizeof(float));

  // ensure all mallocs went fine
  if (!s->embedding || !s->mha_norm || !s->mha_q || !s->mha_score ||
      !s->mha_blend || !s->mha_out || !s->ffn_norm || !s->ffn_fc ||
      !s->ffn_up || !s->ffn_out || !s->logits || !s->k_cache || !s->v_cache) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(state_t *s) {
  free(s->embedding);
  free(s->mha_norm);
  free(s->mha_q);
  free(s->mha_score);
  free(s->mha_blend);
  free(s->mha_out);
  free(s->ffn_norm);
  free(s->ffn_fc);
  free(s->ffn_up);
  free(s->ffn_out);
  free(s->logits);
  free(s->k_cache);
  free(s->v_cache);
}

void memory_map_params(parameter_set_t *w, configuration_t *p, float *ptr,
                       int shared_params) {
  int head_dim = p->embedding_dim / p->q_head_count;
  // make sure the multiplications below are done in 64bit to fit the parameter
  // counts of 13B+ models
  unsigned long long layer_count = p->layer_count;
  w->embedding_weight = ptr;
  ptr += p->vocabulary_len * p->embedding_dim;
  w->mha_norm_weight = ptr;
  ptr += layer_count * p->embedding_dim;
  w->mha_q_weight = ptr;
  ptr += layer_count * p->embedding_dim * (p->q_head_count * head_dim);
  w->mha_k_weight = ptr;
  ptr += layer_count * p->embedding_dim * (p->kv_head_count * head_dim);
  w->mha_v_weight = ptr;
  ptr += layer_count * p->embedding_dim * (p->kv_head_count * head_dim);
  w->mha_out_weight = ptr;
  ptr += layer_count * (p->q_head_count * head_dim) * p->embedding_dim;
  w->ffn_norm_weight = ptr;
  ptr += layer_count * p->embedding_dim;
  w->ffn_fc_weight = ptr;
  ptr += layer_count * p->embedding_dim * p->hidden_dim;
  w->ffn_out_weight = ptr;
  ptr += layer_count * p->hidden_dim * p->embedding_dim;
  w->ffn_up_weight = ptr;
  ptr += layer_count * p->embedding_dim * p->hidden_dim;
  w->out_norm_weight = ptr;
  ptr += p->embedding_dim;
  ptr += p->context_len * head_dim /
         2; // skip what used to be freq_cis_real (for RoPE)
  ptr += p->context_len * head_dim /
         2; // skip what used to be freq_cis_imag (for RoPE)
  w->out_weight = shared_params ? w->embedding_weight : ptr;
}

void read_checkpoint(char *checkpoint, configuration_t *config,
                     parameter_set_t *params, int *fd, float **data,
                     ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in the config header
  if (fread(config, sizeof(configuration_t), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // negative vocab size is hacky way of signaling unshared params. bit yikes.
  int shared_params = config->vocabulary_len > 0 ? 1 : 0;
  config->vocabulary_len = abs(config->vocabulary_len);
  // figure out the file size
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
  fclose(file);
  // memory map the transformer_t params into the data pointer
  *fd = open(checkpoint, O_RDONLY); // open in read only mode
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  float *params_ptr = *data + sizeof(configuration_t) / sizeof(float);
  memory_map_params(params, config, params_ptr, shared_params);
}

void build_transformer(transformer_t *t, char *checkpoint_path) {
  // read in the configuration_t and the params from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->params, &t->fd, &t->data,
                  &t->file_size);
  // allocate the state_t buffers
  state_malloc(&t->state, &t->config);
}

void free_transformer(transformer_t *t) {
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // free the state_t buffers
  free_run_state(&t->state);
}

void print_vector(size_t size, float *vector, size_t sample_size, char *name) {
  if (vector == NULL || size == 0) {
    printf("Empty or invalid vector.\n");
    return;
  }

  // Print the first sample_size elements
  size_t i;
  size_t end = (sample_size < size) ? sample_size : size;

  printf("%6s: ", name ? name : "vector");
  for (i = 0; i < end; ++i) {
    printf("%7.3f ", vector[i]);
  }

  // Print ellipsis if middle elements are skipped
  if (sample_size * 2 < size) {
    printf("... ");
  }

  // Print the last sample_size elements
  size_t start_tail = (sample_size < size) ? size - sample_size : end;
  for (i = start_tail; i < size; ++i) {
    printf("%7.3f ", vector[i]);
  }

  printf("-- ");

  // Compute statistics
  float min = FLT_MAX;
  float max = -FLT_MAX;
  float sum = 0.0f;

  for (i = 0; i < size; ++i) {
    if (vector[i] < min)
      min = vector[i];
    if (vector[i] > max)
      max = vector[i];
    sum += vector[i];
  }

  float mean = sum / size;

  // Print statistics
  printf("Min: %7.3f, Max: %7.3f, Mean: %7.3f, Sum: %f\n", min, max, mean, sum);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the transformer_t

void rmsnorm(int col_count, float *y, float *x, float *w, float epsilon) {

  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < col_count; j++) {
    ss += x[MINDEX1(col_count, j)] * x[MINDEX1(col_count, j)];
  }
  ss /= col_count;
  ss += epsilon;
  ss = 1.0f / sqrtf(ss);

  // normalize and scale
  for (int j = 0; j < col_count; j++) {
    y[MINDEX1(col_count, j)] =
        w[MINDEX1(col_count, j)] * (ss * x[MINDEX1(col_count, j)]);
  }
}

void softmax(int col_count, int col_stride, float *x) {

  // find max value (for numerical stability)
  float max_val = x[MINDEX1(col_count, 0)];
  for (int j = 1; j < col_count; j++) {
    if (x[MINDEX1(col_count, j)] > max_val) {
      max_val = x[MINDEX1(col_count, j)];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int j = 0; j < col_count; j++) {
    x[MINDEX1(col_count, j)] = expf(x[MINDEX1(col_count, j)] - max_val);
    sum += x[MINDEX1(col_count, j)];
  }
  // normalize
  for (int j = 0; j < col_count; j++) {
    x[MINDEX1(col_count, j)] /= sum;
  }
}

void matmul(int col_count, int red_count, float *y, float *x, float *w) {
  for (int j = 0; j < col_count; j++) {
    y[MINDEX1(col_count, j)] = 0.0f;
    for (int k = 0; k < red_count; k++) {
      y[MINDEX1(col_count, j)] +=
          x[MINDEX1(red_count, k)] * w[MINDEX2(col_count, red_count, j, k)];
    }
  }
}

void rope(int col_count, float *x, int pos) {

  for (int j = 0; j < col_count; j += 2) {
    float freq = 1.0f / powf(500000.0f, j / (float)col_count);
    float val = (pos)*freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float v0 = x[MINDEX1(col_count, j)];
    float v1 = x[MINDEX1(col_count, j + 1)];
    x[MINDEX1(col_count, j)] = v0 * fcr - v1 * fci;
    x[MINDEX1(col_count, j + 1)] = v0 * fci + v1 * fcr;
  }
}

void print_emb(char *qual, float *emb, int size) {
  printf("%s ", qual);
  for (int i = 0; i < size; i++) {
    printf("%f ", emb[i]);
  }
  printf("\n");
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

float *driver(transformer_t *transformer, int token, int pos,
              int logits_count) {
  configuration_t *c = &transformer->config;
  parameter_set_t *p = &transformer->params;
  state_t *s = &transformer->state;
  int vocabulary_len = c->vocabulary_len;
  int context_len = c->context_len;
  int layer_count = c->layer_count;
  int q_head_count = c->q_head_count;
  int kv_head_count = c->kv_head_count;
  int q_head_per_kv_head_count = q_head_count / kv_head_count;
  int embedding_dim = c->embedding_dim;
  int head_dim = embedding_dim / q_head_count;
  int q_dim = head_dim * q_head_count;
  int kv_dim = head_dim * kv_head_count;
  int hidden_dim = c->hidden_dim;

  return forward(token, vocabulary_len, context_len, layer_count, q_head_count,
                 kv_head_count, q_head_per_kv_head_count, embedding_dim,
                 head_dim, q_dim, kv_dim, hidden_dim, 1e-5f,
                 p->embedding_weight, p->mha_norm_weight, p->mha_q_weight,
                 p->mha_k_weight, p->mha_v_weight, p->mha_out_weight,
                 p->ffn_norm_weight, p->ffn_fc_weight, p->ffn_up_weight,
                 p->ffn_out_weight, p->out_norm_weight, p->out_weight,
                 s->k_cache, s->v_cache, s->logits, pos, logits_count);
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];

  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars
  // or whitespace because some of the other bytes can be various control codes,
  // backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1
  // if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex),
                            compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens,
            int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[]
  // array bos != 0 means prepend the BOS token (=1), eos != 0 means append the
  // EOS token (=2)
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }

  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two
  // consecutive tokens *2 for concat, +1 for null terminator +2 for UTF8 (in
  // case max_token_length is 1)
  size_t str_buffer_size = (t->max_token_length * 2 + 1 + 2) * sizeof(char);
  char *str_buffer = malloc(str_buffer_size);
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=128000) token, if desired
  if (bos)
    tokens[(*n_tokens)++] = 128000;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have
  // the energy to read more of the sentencepiece code to figure out what it's
  // doing

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++) {

    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the
    // rest 0x80 is 10000000 in UTF-8, all continuation bytes start with "10" in
    // first two bits so in English this is: "if this byte is not a continuation
    // byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char
      // (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] =
        *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning
    // str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (int i = 0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair or triple each iteration, according to the
  // scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    int best_len =
        2; // length of the best merge sequence (2 for pair, 3 for triple)

    // first, try to find the best pair to merge
    for (int i = 0; i < (*n_tokens - 1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      snprintf(str_buffer, str_buffer_size, "%s%s", t->vocab[tokens[i]],
               t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    // if no pair was found, try to find the best triple to merge
    if (best_idx == -1) {
      for (int i = 0; i < (*n_tokens - 2); i++) {
        // check if we can merge the triple (tokens[i], tokens[i+1],
        // tokens[i+2])
        snprintf(str_buffer, str_buffer_size, "%s%s%s", t->vocab[tokens[i]],
                 t->vocab[tokens[i + 1]], t->vocab[tokens[i + 2]]);
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1 && t->vocab_scores[id] > best_score) {
          // this merge triple exists in vocab! record its score and position
          best_score = t->vocab_scores[id];
          best_id = id;
          best_idx = i;
          best_len = 3;
        }
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs or triples to merge, so we're
             // done
    }

    // merge the consecutive pair or triple (best_idx, best_idx+1[, best_idx+2])
    // into new token best_id
    tokens[best_idx] = best_id;
    // delete token(s) at position best_idx+1 (and optionally best_idx+2), shift
    // the entire sequence back
    for (int i = best_idx + 1; i < (*n_tokens - best_len + 1); i++) {
      tokens[i] = tokens[i + best_len - 1];
    }
    (*n_tokens) -=
        (best_len -
         1); // token length decreased by the number of merged tokens minus one
  }

  // add optional EOS (=128001) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = 128001;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocabulary_len;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex,
                float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocabulary_len, float temperature,
                   float topp, unsigned long long rng_seed) {
  sampler->vocabulary_len = vocabulary_len;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc(sampler->vocabulary_len * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocabulary_len);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocabulary_len; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(sampler->vocabulary_len, sampler->vocabulary_len,
            (float (*)[sampler->vocabulary_len])logits);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocabulary_len, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocabulary_len, sampler->topp,
                         sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(transformer_t *transformer, Tokenizer *tokenizer,
              Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }
  printf("IN GENERATE\n");
  // encode the (string) prompt into tokens sequence
  int sequence_len = 0;
  int *sequence =
      malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, sequence, &sequence_len);
  if (sequence_len < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // Print the prompt tokens
  printf("Sequence (%d tokens):\n", sequence_len);
  for (int i = 0; i < sequence_len; i++) {
    printf("%d ", sequence[i]);
  }
  printf("\n");

  // Print the prompt string
  printf("\033[1;34m");
  while (*prompt) {
    printf("%c", *prompt);
    prompt++;
  }
  printf("\033[0m");

  int next; // will store the next token in the sequence
  float *warmup = driver(transformer, sequence[0], 1, 1); // cache warmup

  size_t generated_count = 0; // number of tokens generated so far
  size_t past = 0;            // number of tokens already processed

  // Input token sequence must not be empty
  if (sequence_len == 0) {
    fprintf(stderr, "prompt must not be empty\n");
    exit(EXIT_FAILURE);
  }

  // Timing
  long start = 0;
  long end = 0;
  double prefill_time = 0.0;
  double decode_time = 0.0;

  // First process the input token sequence by chunks
  int *chunk = sequence;
  size_t remaining_sequence_len = sequence_len;
  // size_t chunk_len = MIN(remaining_sequence_len, SEQUENCE_CHUNK_MAX_LEN);
  size_t chunk_len = 1;
  start = time_in_ms();
  while (remaining_sequence_len != 0) {
    // We only compute the very last logits
    int logits_count = (remaining_sequence_len - chunk_len == 0) ? 1 : 0;

    // Run the transformer to fill the KV-cache and get final logits
    float *logits = driver(transformer, *chunk, past, logits_count);

    remaining_sequence_len -= chunk_len;
    chunk += chunk_len;
    past += chunk_len;

    // First token generation after the prompt has been processed
    if (remaining_sequence_len == 0) {
      end = time_in_ms();
      prefill_time = (end - start) / 1000.0;

      next = sample(sampler, logits);
      // print the token as string, decode it with the Tokenizer object
      int current = sequence[sequence_len - 1];
      char *piece = decode(tokenizer, current, next);
      safe_printf(piece); // safe printf("%s", piece)
      fflush(stdout);
      generated_count++;
    }
  }

  // Then generate tokens one by one
  start = time_in_ms();
  while (generated_count < steps) {
    // forward the transformer to get logits for the next token
    int current = next;
    float *logits = driver(transformer, current, past, 1);

    end = time_in_ms();
    decode_time += (end - start) / 1000.0;
    start = end;

    next = sample(sampler, logits);
    generated_count++;
    past++;

    // data-dependent terminating condition: the BOS (=1) token delimits
    // sequences
    if (next == 1) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, current, next);
    safe_printf(piece);
    fflush(stdout);
  }
  printf("\n");

  // report achieved tok/s
  if (past > 2) {
    fprintf(stderr, "Prefill: %f tok/s\n", sequence_len / prefill_time);
    fprintf(stderr, "Decode:  %f tok/s\n", (generated_count - 1) / decode_time);
  }

  free(sequence);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] "
                  "default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = "
                  "max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  // default parameters
  char *checkpoint_path = NULL; // e.g. out/model.bin
  char *tokenizer_path = "tokenizer.bin";
  float temperature =
      1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9
                                   // mha_out_weightrks well, but slower
  int steps = 256;                 // number of steps to run for
  char *prompt = NULL;             // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default

  // poor man's C argparse so we can override the defaults above from the
  // command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage();
    } // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else {
      error_usage();
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
  if (steps < 0)
    steps = 0;

  // build the transformer_t via the model .bin file
  transformer_t transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.context_len)
    steps = transformer.config.context_len; // override to ~max length

  // Print configuration_t data
  fprintf(stderr, "transformer_t configuration:\n");
  fprintf(stderr, "- embedding_dim:     %d\n",
          transformer.config.embedding_dim);
  fprintf(stderr, "- hidden_dim:    %d\n", transformer.config.hidden_dim);
  fprintf(stderr, "- layer_count:   %d\n", transformer.config.layer_count);
  fprintf(stderr, "- q_head_count:  %d\n", transformer.config.q_head_count);
  fprintf(stderr, "- kv_head_count: %d\n", transformer.config.kv_head_count);
  fprintf(stderr, "- vocabulary_len:     %d\n",
          transformer.config.vocabulary_len);
  fprintf(stderr, "- context_len:   %d\n", transformer.config.context_len);

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path,
                  transformer.config.vocabulary_len);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocabulary_len, temperature, topp,
                rng_seed);

  // run!
  generate(&transformer, &tokenizer, &sampler, prompt, steps);

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}
#endif
