// ---------------------------------------------------------------------------
//  llm_core.h - INT8 Transformer Inferenz fuer ESP32 (laeuft auch auf dem PC)
//
//  Architektur (aus model_int8.h abgeleitet):
//      x = token_embedding[tok] + position_embedding[pos]
//      pro Block:  Causal Multi-Head-Attention + Residual + LayerNorm
//                  FeedForward 128 -> 512 -> 128 + Residual + LayerNorm
//      x = LayerNorm(x);  logits = x * token_embedding^T   (weight tying)
//
//  Quantisierungsformat von model_int8.h:
//      Matrizen  INT8, eine float-Skala PRO ZEILE:
//                y[o] = bias[o] + scale[o] * sum_i(q[o*in + i] * x[i])
//      Vektoren  (LayerNorm-Gewichte, alle Biases) sind float32.
//      Attention ist in q_proj/k_proj/v_proj/out_proj aufgeteilt
//      (kein fusioniertes in_proj), FeedForward heisst ff.fc_in/ff.fc_out.
//
//  Saemtliche Stellschrauben stehen in llm_config.h.
// ---------------------------------------------------------------------------
#pragma once

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#include "llm_config.h"
#include "model_int8.h"
#include "tokenizer.h"

// --------------------------- Abmessungen -----------------------------------
#ifndef LLM_HEAD_DIM
#define LLM_HEAD_DIM  (LLM_EMBEDDING_DIM / LLM_NUM_HEADS)   // 128 / 4 = 32
#endif

// --------------------------- Konfiguration pruefen -------------------------
// Lieber hier ein klarer Fehler als spaeter Kauderwelsch im Serial Monitor.
#if LLM_CONTEXT_LEN > LLM_BLOCK_SIZE
  #error "LLM_CONTEXT_LEN groesser als LLM_BLOCK_SIZE: so viele Positions-Embeddings hat das Modell nicht gelernt."
#endif
#if LLM_CONTEXT_LEN < 2
  #error "LLM_CONTEXT_LEN muss mindestens 2 sein."
#endif
#if LLM_GEN_RESERVE >= LLM_CONTEXT_LEN
  #error "LLM_GEN_RESERVE muss kleiner als LLM_CONTEXT_LEN sein, sonst bleibt kein Platz fuer den Prompt."
#endif
#if (LLM_EMBEDDING_DIM % LLM_NUM_HEADS) != 0
  #error "LLM_EMBEDDING_DIM ist nicht durch LLM_NUM_HEADS teilbar."
#endif
#if LLM_FF_ACT != LLM_ACT_GELU && LLM_FF_ACT != LLM_ACT_RELU && LLM_FF_ACT != LLM_ACT_SILU
  #error "LLM_FF_ACT muss LLM_ACT_GELU, LLM_ACT_RELU oder LLM_ACT_SILU sein."
#endif
#if LLM_KV_MODE != LLM_KV_INT8 && LLM_KV_MODE != LLM_KV_INT16 && LLM_KV_MODE != LLM_KV_FLOAT
  #error "LLM_KV_MODE muss LLM_KV_INT8, LLM_KV_INT16 oder LLM_KV_FLOAT sein."
#endif

// LLM_FF_DIM gegen die exportierten Gewichte pruefen (Groessen stehen dort
// als const int32_t, das geht erst hier und nicht im Praeprozessor).
#ifdef __cplusplus
static_assert(LLM_FF_DIM == blocks_0_ff_fc_in_weight_rows,
              "LLM_FF_DIM passt nicht zu model_int8.h");
static_assert(LLM_FF_DIM == blocks_0_ff_fc_out_weight_cols,
              "LLM_FF_DIM passt nicht zu model_int8.h");
#endif

// Parameterzahl (nur zur Anzeige). lm_head teilt sich die Gewichte mit dem
// Token-Embedding und zaehlt deshalb nicht noch einmal mit.
#ifndef LLM_PARAMETER_COUNT
#define LLM_PARAMETER_COUNT ((int)(                                            \
      (long)LLM_VOCAB_SIZE * LLM_EMBEDDING_DIM                                 \
    + (long)LLM_BLOCK_SIZE * LLM_EMBEDDING_DIM                                 \
    + (long)LLM_NUM_LAYERS * ( 4L * LLM_EMBEDDING_DIM * LLM_EMBEDDING_DIM      \
                             + 4L * LLM_EMBEDDING_DIM                          \
                             + 2L * LLM_EMBEDDING_DIM * LLM_FF_DIM             \
                             + LLM_FF_DIM + LLM_EMBEDDING_DIM                  \
                             + 4L * LLM_EMBEDDING_DIM )                        \
    + 2L * LLM_EMBEDDING_DIM ))
#endif

// --------------------------- Gewichts-Tabelle ------------------------------
// w = INT8-Matrix (row-major), s = Skala je Zeile, b = float-Bias
typedef struct {
    const float  *norm1_w;   const float *norm1_b;
    const int8_t *q_w;       const float *q_s;    const float *q_b;    // (E, E)
    const int8_t *k_w;       const float *k_s;    const float *k_b;    // (E, E)
    const int8_t *v_w;       const float *v_s;    const float *v_b;    // (E, E)
    const int8_t *out_w;     const float *out_s;  const float *out_b;  // (E, E)
    const float  *norm2_w;   const float *norm2_b;
    const int8_t *ff1_w;     const float *ff1_s;  const float *ff1_b;  // (FF, E)
    const int8_t *ff2_w;     const float *ff2_s;  const float *ff2_b;  // (E, FF)
} LlmLayer;

static const LlmLayer llm_layers[LLM_NUM_LAYERS] = {
    {
        blocks_0_norm1_weight,              blocks_0_norm1_bias,
        blocks_0_attention_q_proj_weight,   blocks_0_attention_q_proj_weight_scale,
        blocks_0_attention_q_proj_bias,
        blocks_0_attention_k_proj_weight,   blocks_0_attention_k_proj_weight_scale,
        blocks_0_attention_k_proj_bias,
        blocks_0_attention_v_proj_weight,   blocks_0_attention_v_proj_weight_scale,
        blocks_0_attention_v_proj_bias,
        blocks_0_attention_out_proj_weight, blocks_0_attention_out_proj_weight_scale,
        blocks_0_attention_out_proj_bias,
        blocks_0_norm2_weight,              blocks_0_norm2_bias,
        blocks_0_ff_fc_in_weight,           blocks_0_ff_fc_in_weight_scale,
        blocks_0_ff_fc_in_bias,
        blocks_0_ff_fc_out_weight,          blocks_0_ff_fc_out_weight_scale,
        blocks_0_ff_fc_out_bias,
    },
    {
        blocks_1_norm1_weight,              blocks_1_norm1_bias,
        blocks_1_attention_q_proj_weight,   blocks_1_attention_q_proj_weight_scale,
        blocks_1_attention_q_proj_bias,
        blocks_1_attention_k_proj_weight,   blocks_1_attention_k_proj_weight_scale,
        blocks_1_attention_k_proj_bias,
        blocks_1_attention_v_proj_weight,   blocks_1_attention_v_proj_weight_scale,
        blocks_1_attention_v_proj_bias,
        blocks_1_attention_out_proj_weight, blocks_1_attention_out_proj_weight_scale,
        blocks_1_attention_out_proj_bias,
        blocks_1_norm2_weight,              blocks_1_norm2_bias,
        blocks_1_ff_fc_in_weight,           blocks_1_ff_fc_in_weight_scale,
        blocks_1_ff_fc_in_bias,
        blocks_1_ff_fc_out_weight,          blocks_1_ff_fc_out_weight_scale,
        blocks_1_ff_fc_out_bias,
    },
};

// --------------------------- Laufzeit-Speicher -----------------------------
// KV-Cache fuer LLM_CONTEXT_LEN Positionen. Als float waeren das bei 64 Token
//     2 Layer * 64 Pos * 128 * 4 Byte * 2 (K+V) = 128 KB.
// Deshalb standardmaessig INT8 mit einer Skala je Gruppe und Position:
//     Werte  2 * 2 * 64 * 128          = 32 KB
//     Skalen 2 * 2 * 64 *   4 * 4 Byte =  4 KB
// Der Cache liegt auf dem Heap (nicht im statischen DRAM) und wird pro Layer
// einzeln angelegt - kleine Bloecke sind gegenueber Fragmentierung robuster.
//
// Elementtyp und Quantisierung kommen aus llm_config.h (LLM_KV_MODE).

#if LLM_KV_MODE == LLM_KV_FLOAT
  #define LLM_KV_ELEM      float
  #define LLM_KV_QUANT     0
  #define LLM_KV_MODE_NAME "FLOAT"
#elif LLM_KV_MODE == LLM_KV_INT16
  #define LLM_KV_ELEM      int16_t
  #define LLM_KV_QUANT     1
  #define LLM_KV_QMAX      32767.0f
  #define LLM_KV_MODE_NAME "INT16"
#else
  #define LLM_KV_ELEM      int8_t
  #define LLM_KV_QUANT     1
  #define LLM_KV_QMAX      127.0f
  #define LLM_KV_MODE_NAME "INT8"
#endif

// Klartextname der Aktivierung fuer /info
#if LLM_FF_ACT == LLM_ACT_RELU
  #define LLM_FF_ACT_NAME "ReLU"
#elif LLM_FF_ACT == LLM_ACT_SILU
  #define LLM_FF_ACT_NAME "SiLU"
#else
  #define LLM_FF_ACT_NAME "GELU"
#endif

// Eine Skala pro Kopf oder eine pro ganzer Position
#if LLM_KV_SCALE_PER_HEAD
  #define LLM_KV_GROUPS     LLM_NUM_HEADS
#else
  #define LLM_KV_GROUPS     1
#endif
#define LLM_KV_GROUP_DIM    (LLM_EMBEDDING_DIM / LLM_KV_GROUPS)
// zu welcher Skala gehoert Kopf h?
#define LLM_KV_GROUP_OF(h)  (((h) * LLM_HEAD_DIM) / LLM_KV_GROUP_DIM)

#define LLM_KV_VALUES ((size_t)LLM_CONTEXT_LEN * LLM_EMBEDDING_DIM)
#define LLM_KV_SCALES ((size_t)LLM_CONTEXT_LEN * LLM_KV_GROUPS)

static LLM_KV_ELEM *llm_k_q[LLM_NUM_LAYERS] = { NULL };
static LLM_KV_ELEM *llm_v_q[LLM_NUM_LAYERS] = { NULL };
#if LLM_KV_QUANT
static float *llm_k_s[LLM_NUM_LAYERS] = { NULL };   // Skala je Gruppe und Position
static float *llm_v_s[LLM_NUM_LAYERS] = { NULL };
#endif

// Key/Value von Layer l an Position p
#define LLM_K(l, p) (llm_k_q[l] + (size_t)(p) * LLM_EMBEDDING_DIM)
#define LLM_V(l, p) (llm_v_q[l] + (size_t)(p) * LLM_EMBEDDING_DIM)
#define LLM_KS(l, p) (llm_k_s[l] + (size_t)(p) * LLM_KV_GROUPS)
#define LLM_VS(l, p) (llm_v_s[l] + (size_t)(p) * LLM_KV_GROUPS)

static float llm_x[LLM_EMBEDDING_DIM];         // Residual-Strom
static float llm_h[LLM_EMBEDDING_DIM];         // Zwischenpuffer
static float llm_qkv[3 * LLM_EMBEDDING_DIM];   // Q|K|V des aktuellen Tokens
static float llm_att[LLM_CONTEXT_LEN];         // Attention-Scores eines Heads
static float llm_ao[LLM_EMBEDDING_DIM];        // Attention-/FF-Ausgabe
static float llm_ff[LLM_FF_DIM];               // FeedForward-Hidden
static float llm_logits[LLM_VOCAB_SIZE];       // Ausgabe-Logits

static int llm_pos = 0;   // naechste freie Position im KV-Cache

// Schreibt einen Embedding-Vektor in den Cache. Quantisiert gruppenweise
// symmetrisch: je Gruppe wird der groesste Betrag auf LLM_KV_QMAX abgebildet.
static void llm_kv_store(LLM_KV_ELEM *dst, float *scales, const float *src)
{
#if !LLM_KV_QUANT
    (void)scales;
    memcpy(dst, src, sizeof(float) * LLM_EMBEDDING_DIM);
#else
    const int GD = LLM_KV_GROUP_DIM;

    for (int g = 0; g < LLM_KV_GROUPS; ++g) {
        const float *v = src + g * GD;
        LLM_KV_ELEM *d = dst + g * GD;

        float amax = 0.0f;
        for (int i = 0; i < GD; ++i) {
            float a = fabsf(v[i]);
            if (a > amax) amax = a;
        }

        if (amax < 1e-20f) {              // reiner Nullvektor
            for (int i = 0; i < GD; ++i) d[i] = 0;
            scales[g] = 0.0f;
            continue;
        }

        const float s   = amax / LLM_KV_QMAX;
        const float inv = LLM_KV_QMAX / amax;
        const long  lim = (long)LLM_KV_QMAX;
        for (int i = 0; i < GD; ++i) {
            long q = lrintf(v[i] * inv);
            if (q >  lim) q =  lim;
            if (q < -lim) q = -lim;
            d[i] = (LLM_KV_ELEM)q;
        }
        scales[g] = s;
    }
#endif
}

// Legt den KV-Cache an. Muss einmal vor llm_forward() aufgerufen werden.
// Rueckgabe: 1 = ok, 0 = zu wenig Speicher.
static void *llm_alloc(size_t bytes)
{
#if LLM_USE_PSRAM && defined(ESP32)
  #if LLM_KV_PSRAM_FIRST
    void *p = ps_malloc(bytes);     // PSRAM zuerst, internen DRAM schonen
    if (!p) p = malloc(bytes);
  #else
    void *p = malloc(bytes);
    if (!p) p = ps_malloc(bytes);   // notfalls externes PSRAM
  #endif
    return p;
#else
    return malloc(bytes);
#endif
}

static int llm_init(void)
{
    const size_t val_bytes = LLM_KV_VALUES * sizeof(LLM_KV_ELEM);
    int ok = 1;

    for (int l = 0; l < LLM_NUM_LAYERS; ++l) {
        if (!llm_k_q[l]) llm_k_q[l] = (LLM_KV_ELEM *)llm_alloc(val_bytes);
        if (!llm_v_q[l]) llm_v_q[l] = (LLM_KV_ELEM *)llm_alloc(val_bytes);
        if (!llm_k_q[l] || !llm_v_q[l]) ok = 0;
#if LLM_KV_QUANT
        const size_t sc_bytes = LLM_KV_SCALES * sizeof(float);
        if (!llm_k_s[l]) llm_k_s[l] = (float *)llm_alloc(sc_bytes);
        if (!llm_v_s[l]) llm_v_s[l] = (float *)llm_alloc(sc_bytes);
        if (!llm_k_s[l] || !llm_v_s[l]) ok = 0;
#endif
    }

    if (!ok) {
        for (int l = 0; l < LLM_NUM_LAYERS; ++l) {
            free(llm_k_q[l]);  llm_k_q[l] = NULL;
            free(llm_v_q[l]);  llm_v_q[l] = NULL;
#if LLM_KV_QUANT
            free(llm_k_s[l]);  llm_k_s[l] = NULL;
            free(llm_v_s[l]);  llm_v_s[l] = NULL;
#endif
        }
        return 0;
    }
    return 1;
}

static size_t llm_kv_bytes(void)
{
    size_t b = 2 * LLM_NUM_LAYERS * LLM_KV_VALUES * sizeof(LLM_KV_ELEM);
#if LLM_KV_QUANT
    b += 2 * LLM_NUM_LAYERS * LLM_KV_SCALES * sizeof(float);
#endif
    return b;
}

// --------------------------- Bausteine -------------------------------------

// out[i] = bias[i] + scale[i] * sum_j W[i][j] * in[j]
// W row-major INT8, scale je Zeile, bias float (darf NULL sein).
static void llm_matvec(float *out, const float *in,
                       const int8_t *w, const float *w_s,
                       const float *b,
                       int n_out, int n_in)
{
    for (int i = 0; i < n_out; ++i) {
        const int8_t *row = w + (size_t)i * n_in;
        float acc = 0.0f;
        for (int j = 0; j < n_in; ++j) {
            acc += (float)row[j] * in[j];
        }
        out[i] = acc * w_s[i] + (b ? b[i] : 0.0f);
    }
}

// LayerNorm ueber die Embedding-Dimension (wie torch.nn.LayerNorm, eps 1e-5).
// Gain und Bias liegen als float32 im Flash.
static void llm_layernorm(float *out, const float *in,
                          const float *g, const float *b)
{
    float mean = 0.0f;
    for (int i = 0; i < LLM_EMBEDDING_DIM; ++i) mean += in[i];
    mean /= (float)LLM_EMBEDDING_DIM;

    float var = 0.0f;
    for (int i = 0; i < LLM_EMBEDDING_DIM; ++i) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)LLM_EMBEDDING_DIM;

    float inv = 1.0f / sqrtf(var + LLM_LN_EPS);
    for (int i = 0; i < LLM_EMBEDDING_DIM; ++i) {
        out[i] = (in[i] - mean) * inv * g[i] + b[i];
    }
}

static inline float llm_act(float v)
{
#if LLM_FF_ACT == LLM_ACT_RELU
    return v > 0.0f ? v : 0.0f;
#elif LLM_FF_ACT == LLM_ACT_SILU
    return v / (1.0f + expf(-v));                 // x * sigmoid(x)
#else
    // GELU (tanh-Approximation)
    const float c = 0.7978845608f;   // sqrt(2/pi)
    return 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
#endif
}

static void llm_softmax(float *v, int n)
{
    float mx = v[0];
    for (int i = 1; i < n; ++i) if (v[i] > mx) mx = v[i];

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        v[i] = expf(v[i] - mx);
        sum += v[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

// --------------------------- Forward-Pass ----------------------------------

static void llm_reset_context(void) { llm_pos = 0; }
static int  llm_context_len(void)   { return llm_pos; }

// Verarbeitet EIN Token an der naechsten freien Position und liefert die
// Logits fuer das Folge-Token. NULL, wenn der Kontext voll ist.
static const float *llm_forward(int token)
{
    if (!llm_k_q[0] || !llm_v_q[0]) return NULL;   // llm_init() vergessen?
    if (llm_pos >= LLM_CONTEXT_LEN) return NULL;
    if (token < 0 || token >= LLM_VOCAB_SIZE) token = 0;

    const int E  = LLM_EMBEDDING_DIM;
    const int HD = LLM_HEAD_DIM;
    const int t  = llm_pos;

    // 1) Token- + Positions-Embedding (Skala je Zeile = je Token/Position)
    const int8_t *te = token_embedding_weight + (size_t)token * E;
    const int8_t *pe = position_embedding_weight + (size_t)t * E;
    const float te_s = token_embedding_weight_scale[token];
    const float pe_s = position_embedding_weight_scale[t];
    for (int i = 0; i < E; ++i) {
        llm_x[i] = (float)te[i] * te_s + (float)pe[i] * pe_s;
    }

    // 2) Transformer-Bloecke
    for (int l = 0; l < LLM_NUM_LAYERS; ++l) {
        const LlmLayer *L = &llm_layers[l];
        const float *attn_in;

#if LLM_PRE_NORM
        llm_layernorm(llm_h, llm_x, L->norm1_w, L->norm1_b);
        attn_in = llm_h;
#else
        attn_in = llm_x;
#endif

        // --- Q, K, V des aktuellen Tokens (drei getrennte Projektionen) ---
        llm_matvec(llm_qkv,         attn_in, L->q_w, L->q_s, L->q_b, E, E);
        llm_matvec(llm_qkv + E,     attn_in, L->k_w, L->k_s, L->k_b, E, E);
        llm_matvec(llm_qkv + 2 * E, attn_in, L->v_w, L->v_s, L->v_b, E, E);

#if LLM_KV_QUANT
        llm_kv_store(LLM_K(l, t), LLM_KS(l, t), llm_qkv + E);
        llm_kv_store(LLM_V(l, t), LLM_VS(l, t), llm_qkv + 2 * E);
#else
        llm_kv_store(LLM_K(l, t), NULL, llm_qkv + E);
        llm_kv_store(LLM_V(l, t), NULL, llm_qkv + 2 * E);
#endif

        // --- Causal Self-Attention ueber Positionen 0..t ---
        const float scale = 1.0f / sqrtf((float)HD);
        for (int hh = 0; hh < LLM_NUM_HEADS; ++hh) {
            const int off  = hh * HD;
            const float *q = llm_qkv + off;
#if LLM_KV_QUANT
            const int    gr = LLM_KV_GROUP_OF(hh);   // Skala dieses Kopfes
#endif

            for (int p = 0; p <= t; ++p) {
                const LLM_KV_ELEM *k = LLM_K(l, p) + off;
                float dot = 0.0f;
                for (int d = 0; d < HD; ++d) dot += q[d] * (float)k[d];
#if LLM_KV_QUANT
                llm_att[p] = dot * LLM_KS(l, p)[gr] * scale;
#else
                llm_att[p] = dot * scale;
#endif
            }
            llm_softmax(llm_att, t + 1);

            for (int d = 0; d < HD; ++d) llm_ao[off + d] = 0.0f;
            for (int p = 0; p <= t; ++p) {
#if LLM_KV_QUANT
                const float a = llm_att[p] * LLM_VS(l, p)[gr];
#else
                const float a = llm_att[p];
#endif
                const LLM_KV_ELEM *v = LLM_V(l, p) + off;
                for (int d = 0; d < HD; ++d) llm_ao[off + d] += a * (float)v[d];
            }
        }

        // --- Output-Projektion + Residual ---
        llm_matvec(llm_h, llm_ao, L->out_w, L->out_s, L->out_b, E, E);
        for (int i = 0; i < E; ++i) llm_x[i] += llm_h[i];

        const float *ff_in;
#if LLM_PRE_NORM
        llm_layernorm(llm_h, llm_x, L->norm2_w, L->norm2_b);
        ff_in = llm_h;
#else
        // Post-Norm: x = norm1(x + attn)
        llm_layernorm(llm_x, llm_x, L->norm1_w, L->norm1_b);
        ff_in = llm_x;
#endif

        // --- FeedForward 128 -> 512 -> 128 ---
        llm_matvec(llm_ff, ff_in, L->ff1_w, L->ff1_s, L->ff1_b, LLM_FF_DIM, E);
        for (int i = 0; i < LLM_FF_DIM; ++i) llm_ff[i] = llm_act(llm_ff[i]);

        llm_matvec(llm_ao, llm_ff, L->ff2_w, L->ff2_s, L->ff2_b, E, LLM_FF_DIM);

        for (int i = 0; i < E; ++i) llm_x[i] += llm_ao[i];

#if !LLM_PRE_NORM
        // Post-Norm: x = norm2(x + ff)
        llm_layernorm(llm_x, llm_x, L->norm2_w, L->norm2_b);
#endif
    }

    // 3) Abschluss-LayerNorm + Sprachkopf.
    //    lm_head teilt sich die Gewichte mit dem Token-Embedding
    //    (weight tying) und hat keinen Bias.
    llm_layernorm(llm_h, llm_x, norm_weight, norm_bias);
    llm_matvec(llm_logits, llm_h,
               token_embedding_weight, token_embedding_weight_scale, NULL,
               LLM_VOCAB_SIZE, LLM_EMBEDDING_DIM);

    llm_pos++;
    return llm_logits;
}

// --------------------------- Sampling --------------------------------------

static float llm_rand01(void)
{
#ifdef ARDUINO
    return (float)random(0, 1000000) / 1000000.0f;
#else
    return (float)rand() / ((float)RAND_MAX + 1.0f);
#endif
}

// temperature <= 0 -> greedy;  top_k <= 0 -> kein Top-k
// repeat_penalty > 1 daempft Tokens, die in "recent" schon vorkamen
static int llm_sample(const float *logits, float temperature, int top_k,
                      const int *recent, int n_recent, float repeat_penalty)
{
    static float probs[LLM_VOCAB_SIZE];
#if LLM_ENABLE_TOP_K
    static float tmp[LLM_VOCAB_SIZE];
#endif
    memcpy(probs, logits, sizeof(float) * LLM_VOCAB_SIZE);

    probs[LLM_PAD_ID] = -1e30f;   // <PAD> nie erzeugen
    probs[LLM_UNK_ID] = -1e30f;   // <UNK> nie erzeugen

    if (repeat_penalty > 1.0f && recent) {
        for (int i = 0; i < n_recent; ++i) {
            int id = recent[i];
            if (id < 2 || id >= LLM_VOCAB_SIZE) continue;
            probs[id] = probs[id] > 0.0f ? probs[id] / repeat_penalty
                                         : probs[id] * repeat_penalty;
        }
    }

    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < LLM_VOCAB_SIZE; ++i)
            if (probs[i] > probs[best]) best = i;
        return best;
    }

    // Top-k: alles ausserhalb der k groessten Logits abschneiden
#if LLM_ENABLE_TOP_K
    if (top_k > 0 && top_k < LLM_VOCAB_SIZE) {
        memcpy(tmp, probs, sizeof(float) * LLM_VOCAB_SIZE);
        float kth = -1e30f;
        for (int i = 0; i < top_k; ++i) {
            int best = 0;
            for (int j = 1; j < LLM_VOCAB_SIZE; ++j)
                if (tmp[j] > tmp[best]) best = j;
            kth = tmp[best];
            tmp[best] = -1e30f;
        }
        for (int i = 0; i < LLM_VOCAB_SIZE; ++i)
            if (probs[i] < kth) probs[i] = -1e30f;
    }
#else
    (void)top_k;
#endif

    for (int i = 0; i < LLM_VOCAB_SIZE; ++i) probs[i] /= temperature;
    llm_softmax(probs, LLM_VOCAB_SIZE);

    float r = llm_rand01();
    float cum = 0.0f;
    for (int i = 0; i < LLM_VOCAB_SIZE; ++i) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return LLM_VOCAB_SIZE - 1;
}
