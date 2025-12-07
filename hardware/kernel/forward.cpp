#include "forward.h"
#include "config.h"
#include <cstring>
// #include "hls_math.h" // Use HLS math for synthesis

// ... [Keep your rmsnorm, softmax, matmul templates EXACTLY as provided] ...
template <int S>
void rmsnorm(float o[S], float x[S], float weight[S])
{
    constexpr auto array_size = S * sizeof(float);
    float ss = 0.0f;
    float x_buff[S];
    float weight_buff[S];
    float out_buff[S];
#pragma HLS array_partition variable = x_buff type = cyclic factor = 128
#pragma HLS array_partition variable = weight_buff type = cyclic factor = 64
#pragma HLS array_partition variable = out_buff type = cyclic factor = 64
    // Note: for HLS, memcpy usually needs constant size. 
    // If this worked in your base code, we keep it. 
    // Ideally, replace with loops for pure hardware safety, but sticking to your request to keep base logic.
    std::memcpy(x_buff, x, array_size);
    std::memcpy(weight_buff, weight, array_size);

    sum_of_squares:
    for (int j = 0; j < S; j++) {
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor = 128 skip_exit_check
        float x_j = x_buff[j];
        ss += x_j * x_j;
    }
    ss /= S; 
    ss += 1e-5f; 
    // !!!!!!!!!!!!!!!!
    ss = 1.0f / sqrt(ss); // Use hls::sqrt for synthesis

    norm_and_scale:
    for (int j = 0; j < S; j++) 
    {
    #pragma HLS PIPELINE
    #pragma HLS UNROLL factor = 64
        float weight_j = weight_buff[j];
        float x_j = x_buff[j];
        out_buff[j] = weight_j * (ss * x_j);
    }
    std::memcpy(o, out_buff, array_size);
}

template <int MAXSIZE>
void softmax(float *x, int size) {
    float buffer[MAXSIZE];
    float max_val = x[0];
    max:
    for (int i = 1; i < size; i++) {
        #pragma HLS loop_tripcount min = 0 max = 257 avg = 129
        #pragma HLS PIPELINE
        float x_i = x[i];
        if (x_i > max_val) max_val = x_i;
    }
    
    exp_loop:
    for (int i = 0; i < size; i++) {
        #pragma HLS loop_tripcount min = 0 max = 257 avg = 129
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor = 16
        // !!!!!!!!!!!!!!!!!!!
        float x_i = exp(x[i] - max_val); // Use hls::exp
        buffer[i] = x_i;
    }
    float sum = 0.0f;
    sum_loop:
    for (int i = 0; i < size; i++) {
        #pragma HLS loop_tripcount min = 0 max = 257 avg = 129
        sum += buffer[i];
    }
    const float inv_sum = 1.0f / sum;
    
    norm_loop:
    for (int i = 0; i < size; i++) {
        #pragma HLS loop_tripcount min = 0 max = 257 avg = 129
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor = 16
        x[i] = buffer[i] * inv_sum;
    }
}

template <int N, int D>
void matmul_old(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws)
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  // inputs to this function are both quantized

  // wq - quantized weight matrix
  // ws - scaling factor for each row of wq
  // xq - quantized input vector
  // xs - scaling factor for xq
  // xout - output vector

  static int8_t x_buffer[N];
  static float xs_buffer[N / GS];
  // float out_buffer[D];
  int8_t w_buffer[N * D];
  float ws_buffer[N * D / GS];

#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 128
#pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = 32
//
x_buff:
  for (int i = 0; i < N; i++)
  {
#pragma HLS UNROLL factor = 16
    x_buffer[i] = xq[i];
  }
xs_buff:
  for (int j = 0; j <= N - GS; j += GS)
  {
#pragma HLS UNROLL factor = 4
    xs_buffer[j / GS] = xs[j / GS];
  }

w_buff:
  for (int i = 0; i < N * D; i++)
  {
#pragma HLS UNROLL factor = 128
    w_buffer[i] = wq[i];
  }

ws_buff:
  for (int i = 0; i < N * D / GS; i++)
  {
#pragma HLS UNROLL factor = 32
    ws_buffer[i] = ws[i];
  }

  int i;
  for (i = 0; i < D; i++)
  {
#pragma HLS PIPEPLINE
    float val = 0.0f;
    // start index of row i
    const int in = i * N;
    // matmul1:
    // for (int j = 0; j < N; j++)
    //{
    // #pragma HLS UNROLL
    //  w_buffer[j] = wq[j + in];
    //}
    // matmul2:
    const int in_s = i * N / GS;
    // const int groups = N / GS;
    // for (int j = 0; j < groups; j++)
    //  {
    // #pragma HLS UNROLL
    //  //  ws_buffer[j] = ws[in_s + j];
    // }

    // do the matmul in groups of GS

    int j;
  matmul3:
    for (j = 0; j <= N - GS; j += GS)
    {
#pragma HLS UNROLL
      int32_t ival = 0;
    matmul4:
      for (int k = 0; k < GS; k++)
      {
#pragma HLS UNROLL
        ival += ((int32_t)x_buffer[j + k]) * ((int32_t)w_buffer[in + j + k]);
      }
      val += ((float)ival) * ws_buffer[in_s + j / GS] * xs_buffer[j / GS];
    }
    xout[i] = val;
  }
}

template <int N, int D>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws)
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  // inputs to this function are both quantized

  // wq - quantized weight matrix
  // ws - scaling factor for each row of wq
  // xq - quantized input vector
  // xs - scaling factor for xq
  // xout - output vector

  static int8_t x_buffer[N];
  static float xs_buffer[N / GS];
  // float out_buffer[D];

#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = 4
//
x_buff:
  for (int i = 0; i < N; i++)
  {
#pragma HLS UNROLL factor = 16
    x_buffer[i] = xq[i];
  }
xs_buff:
  for (int j = 0; j <= N - GS; j += GS)
  {
#pragma HLS UNROLL factor = 4
    xs_buffer[j / GS] = xs[j / GS];
  }

  int i;
  for (i = 0; i < D; i++)
  {
#pragma HLS PIPELINE
    float val = 0.0f;
    int8_t w_buffer[N];
    float ws_buffer[N / GS];
#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 32
#pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = 32
    // start index of row i
    const int in = i * N;
  matmul1:
    for (int j = 0; j < N; j++)
    {
      // #pragma HLS UNROLL factor
      w_buffer[j] = wq[j + in];
    }
  matmul2:
    const int in_s = i * N / GS;
    const int groups = N / GS;
    for (int j = 0; j < groups; j++)
    {
      // #pragma HLS UNROLL factor
      ws_buffer[j] = ws[in_s + j];
    }

    // do the matmul in groups of GS
    int j;
  matmul3:
    for (j = 0; j <= N - GS; j += GS)
    {
      // #pragma HLS UNROLL
      int32_t ival = 0;
    matmul4:
      for (int k = 0; k < GS; k++)
      {
        // #pragma HLS UNROLL
        ival += ((int32_t)x_buffer[j + k]) * ((int32_t)w_buffer[j + k]);
      }
      val += ((float)ival) * ws_buffer[j / GS] * xs_buffer[j / GS];
    }
    xout[i] = val;
  }
}


extern "C" void forward(
    Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, 
    int token, 
    int pos, 
    float key_cache[n_layers * cache_budget * ((dim * n_kv_heads) / n_heads)], 
    float value_cache[n_layers * cache_budget * ((dim * n_kv_heads) / n_heads)], 
    float accumulated_scores[n_layers * cache_budget], 
    int cache_timestamps[n_layers * cache_budget],     
    float *out) 
{
  // --- HLS INTERFACES (Mandatory for Synthesis) ---
  #pragma HLS INTERFACE m_axi port=transformer offset=slave bundle=gmem0 depth=1
  #pragma HLS INTERFACE m_axi port=key_cache offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=value_cache offset=slave bundle=gmem1
  // New ports map to gmem2 to distribute bandwidth
  #pragma HLS INTERFACE m_axi port=accumulated_scores offset=slave bundle=gmem2
  #pragma HLS INTERFACE m_axi port=cache_timestamps offset=slave bundle=gmem2
  #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0 depth=vocab_size
  

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  #pragma HLS INTERFACE s_axilite port=token
  #pragma HLS INTERFACE s_axilite port=pos
  #pragma HLS INTERFACE s_axilite port=return

  auto w = &transformer->weights;
  constexpr int UNROLL_FACTOR = 16;
  
  static float x[config.dim];                                        
  static float xb[config.dim];                                       
  static float xb2[config.dim];                                      
  static float hb[config.hidden_dim];                                
  static float hb2[config.hidden_dim];                               
  static QuantizedTensor<config.dim> xq;                             
  static QuantizedTensor<config.hidden_dim> hq;                      
  static float q[config.dim];                                        
  static float k[(config.dim * config.n_kv_heads) / config.n_heads]; 
  static float v[(config.dim * config.n_kv_heads) / config.n_heads]; 
  
  // Attention buffer size fix
  static float att[config.n_heads * cache_budget];                 

  // Array Partitions for Parallelism
  #pragma HLS ARRAY_PARTITION variable=q cyclic factor=UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable=k cyclic factor=UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable=v cyclic factor=UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable=att cyclic factor=UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable = hq.q cyclic factor = UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable = hq.s cyclic factor = UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable = xq.q cyclic factor = UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable = xq.s cyclic factor = UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable = hb type = cyclic factor = UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable = hb2 type = cyclic factor = UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable = x type = cyclic factor = UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable = xb type = cyclic factor = UNROLL_FACTOR
  #pragma HLS ARRAY_PARTITION variable = xb2 type = cyclic factor = UNROLL_FACTOR

  constexpr int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
  constexpr int kv_mul = config.n_heads / config.n_kv_heads; 
  constexpr int head_size = dim / config.n_heads;

  // Load Embedding
  // Explicit loop for synthesis safety
   std :: memcpy(x,w -> token_embedding_table + token * dim, dim * sizeof(float));

  main_forward_loop: for (int l = 0; l < config.n_layers; l++) {
    rmsnorm<dim>(xb, x, w->rms_att_weight + l * dim);

    quantize(&xq, xb, GS);
    matmul<dim, dim>(q, xq.q, xq.s, (w->wq + l)->q, (w->wq + l)->s);
    matmul<dim, kv_dim>(k, xq.q, xq.s, (w->wk + l)->q, (w->wk + l)->s);
    matmul<dim, kv_dim>(v, xq.q, xq.s, (w->wv + l)->q, (w->wv + l)->s);

    rotation1:
    // Rotation for both query and key vectors (i < kv_dim)
    for (int i = 0; i < kv_dim; i += 2)
    {
#pragma HLS UNROLL factor = UNROLL_FACTOR
#pragma HLS PIPELINE
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);

      // Rotate the query vector
      float v0_q = q[i];
      float v1_q = q[i + 1];
      q[i] = v0_q * fcr - v1_q * fci;
      q[i + 1] = v0_q * fci + v1_q * fcr;

      // Rotate the key vector
      float v0_k = k[i];
      float v1_k = k[i + 1];
      k[i] = v0_k * fcr - v1_k * fci;
      k[i + 1] = v0_k * fci + v1_k * fcr;
    }
  rotation2:
    // Rotation for only the query vector (i >= kv_dim)
    for (int i = kv_dim; i < dim; i += 2)
    {
#pragma HLS PIPELINE
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);

      // Rotate only the query vector
      float v0 = q[i];
      float v1 = q[i + 1];
      q[i] = v0 * fcr - v1 * fci;
      q[i + 1] = v0 * fci + v1 * fcr;
    }

    // ========================================================================
    // H2O LOGIC STEP 1: Synthesizable Load-Compute-Store
    // ========================================================================
    
    int target_slot = -1;
    int layer_offset = l * cache_budget; 

    // 1. Burst Load H2O State into Local BRAM
    // We MUST buffer this to avoid random reads to AXI during the scan loop
    float local_acc_scores[cache_budget];
    int local_timestamps[cache_budget];
    #pragma HLS ARRAY_PARTITION variable=local_acc_scores cyclic factor=8
    
    h2o_read_burst: for(int i=0; i<cache_budget; i++) {
        #pragma HLS PIPELINE II=1
        local_acc_scores[i] = accumulated_scores[layer_offset + i];
        local_timestamps[i] = cache_timestamps[layer_offset + i];
    }

    if (pos < cache_budget) {
        target_slot = pos;
    } else {
        float min_score = 1e9f; 
        int victim_idx = -1;

        h2o_scan: for (int i = 0; i < cache_budget; i++) {
            #pragma HLS PIPELINE II=1
            // int stored_pos = cache_timestamps[layer_offset + i];
            int stored_pos = local_timestamps[i];
            
            // RULE 1: Protect Recent
            if (stored_pos > pos - recent_window) continue;

            // RULE 2: Find Weakest
            // float score = accumulated_scores[layer_offset + i];
            float score = local_acc_scores[i];
            if (score < min_score) {
                min_score = score;
                victim_idx = i;
            }
        }
        if (victim_idx == -1) victim_idx = 0;
        target_slot = victim_idx;
    }

    // ========================================================================
    // H2O LOGIC STEP 2: UPDATE CACHE
    // ========================================================================
    
    int kv_cache_offset = l * cache_budget * kv_dim;
    float *key_cache_row = key_cache + kv_cache_offset + target_slot * kv_dim;
    float *value_cache_row = value_cache + kv_cache_offset + target_slot * kv_dim;

    std::memcpy(key_cache_row, k, kv_dim * sizeof(*key_cache_row));
    std::memcpy(value_cache_row, v, kv_dim * sizeof(*value_cache_row));

    // update local state
    local_acc_scores[target_slot] = 0.0f;
    local_timestamps[target_slot] = pos;

    // // Reset score for the new token (it starts fresh)
    // accumulated_scores[layer_offset + target_slot] = 0.0f;
    // // Record that this slot now holds token 'pos'
    // cache_timestamps[layer_offset + target_slot] = pos;


    // ========================================================================
    // H2O LOGIC STEP 3: ATTENTION WITH ACCUMULATION
    // ========================================================================

    // ========================================================================
    // ATTENTION LOOP
    // ========================================================================

    int current_cache_size = (pos < cache_budget) ? (pos + 1) : cache_budget;

    attn_heads: for (int h = 0; h < n_heads; h++) {
      const int q_offset = h * head_size;
      const int att_offset = h * cache_budget; 

      // Score Calculation
      // This loop reads random locations from DDR (key_cache). 
      // Performance will be memory bound here, but logic is synthesizable.
      attn_calc: for (int t = 0; t < current_cache_size; t++) {
        #pragma HLS PIPELINE
        #pragma HLS LOOP_TRIPCOUNT min=0 max=cache_budget avg=cache_budget
        int k_base = l * cache_budget * kv_dim + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        
        dot_prod: for (int i = 0; i < head_size; i++) {
        #pragma HLS unroll
          score += q[i + q_offset] * key_cache[k_base + i];
        }
        // !!!!!!!!!!!!!!!!!!!
        score /= sqrt((float)head_size);
        att[t + att_offset] = score;
      }

      // Softmax
      softmax<cache_budget>(att + att_offset, current_cache_size);

      const int xb_offset = h * head_size;
      memset(xb + xb_offset, 0, head_size * sizeof(float));

      // Accumulate Scores into Local Buffer
      // Use DEPENDENCE pragma to assert that writes to local_acc_scores[t] don't conflict
      accum_loop: for (int t = 0; t < current_cache_size; t++) {
          #pragma HLS PIPELINE
          #pragma HLS DEPENDENCE variable=local_acc_scores inter false
          #pragma HLS LOOP_TRIPCOUNT min=0 max=cache_budget avg=cache_budget
    
        int v_base = kv_cache_offset + t * kv_dim + (h / kv_mul) * head_size;
        float a = att[t + att_offset];
        // accumulated_scores[layer_offset + t] += a;
        local_acc_scores[t]+=a;
        
        val_acc: for (int i = 0; i < head_size; i++) {
          #pragma HLS UNROLL
          xb[i + xb_offset] += a * value_cache[v_base + i];
        }
      }
    }

    // 3. Burst Write H2O State Back to DDR
    // We do this after all heads are processed
    h2o_write_burst: for(int i=0; i<cache_budget; i++) {
        #pragma HLS PIPELINE II=1
        accumulated_scores[layer_offset + i] = local_acc_scores[i];
        cache_timestamps[layer_offset + i] = local_timestamps[i];
    }

    // FFN & Residuals (Standard Logic)
    quantize(&xq, xb, GS);
    matmul<dim, dim>(xb2, xq.q, xq.s, (w->wo + l)->q, (w->wo + l)->s);

    res1: for (int i = 0; i < dim; i++) 
    {
 #pragma HLS PIPELINE II=1
    x[i] += xb2[i];
    }

    rmsnorm<dim>(xb, x, w->rms_ffn_weight + l * dim);
    quantize(&xq, xb, GS);
    matmul<dim, hidden_dim>(hb, xq.q, xq.s, (w->w1 + l)->q, (w->w1 + l)->s);
    matmul<dim, hidden_dim>(hb2, xq.q, xq.s, (w->w3 + l)->q, (w->w3 + l)->s);
    float hb_out[hidden_dim];
#pragma HLS array_partition variable=hb_out type=cyclic factor=16
    silu: for (int i = 0; i < hidden_dim; i++) {
      #pragma HLS PIPELINE
      #pragma HLS UNROLL factor = 4
      float val = hb[i];
      // !!!!!!!!!!!!!!!!
      val *= (1.0f / (1.0f + exp(-val))); 
      val *= hb2[i];
      hb_out[i] = val;
    }
    std::memcpy(hb, hb_out, hidden_dim * sizeof(float));

    quantize(&hq, hb, GS);
    matmul<hidden_dim, dim>(xb, hq.q, hq.s, (w->w2 + l)->q, (w->w2 + l)->s);

    res2: for (int i = 0; i < dim; i++)
    {
    #pragma HLS UNROLL factor = 16 skip_exit_check
    x[i] += xb[i];
    }
  }

  rmsnorm<dim>(x, x, w->rms_final_weight);
  quantize(&xq, x, GS);
  matmul<dim, vocab_size>(out, xq.q, xq.s, w->wcls->q, w->wcls->s);
}