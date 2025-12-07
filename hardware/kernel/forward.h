#include "typedefs.h"
#include "config.h"
#include <math.h>
#include <cstring>

extern "C" void forward(
  Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, 
  int token, 
  int pos, 
  // KV Cache is now sized by 'cache_budget', not 'seq_len'
  float key_cache[n_layers * cache_budget * ((dim * n_kv_heads) / n_heads)], 
  float value_cache[n_layers * cache_budget * ((dim * n_kv_heads) / n_heads)], 
  // New H2O State Arrays
  float accumulated_scores[n_layers * cache_budget], // Tracks importance
  int cache_timestamps[n_layers * cache_budget],     // Tracks logical pos of each slot
  float *out
);


template <int S>
void dequantize(QuantizedTensor<S> *qx, float x[S], int GS)
{
  for (int i = 0; i < S; i++)
  {
    x[i] = qx->q[i] * qx->s[i / GS];
  }
}

template <int S>
void quantize(QuantizedTensor<S> *qx, float x[S], int GS)
{
  constexpr int num_groups = S / 64;
  constexpr float Q_MAX = 127.0f;
  float scale_buffer[num_groups];
  int8_t quantized_buffer[S];
//#pragma HLS ARRAY_PARTITION variable = x type=cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = quantized_buffer type=cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = scale_buffer type=cyclic factor = 16


main_loop:
  for (int group = 0; group < num_groups; group++)
  {
#pragma HLS UNROLL factor = 8
#pragma HLS PIPELINE
    float wmax = 0.0;
    int base_idx = group * GS;

    // Calculate the max absolute value in the current group
    max:
    for (int i = 0; i < GS; i++)
    {
#pragma HLS PIPELINE
      float val = fabs(x[base_idx + i]);
      if (val > wmax)
      {
        wmax = val;
      }
    }

    // Calculate and write the scaling factor
    float scale = wmax / Q_MAX;
    scale_buffer[group] = scale;

    // Calculate and write the quantized values
    for (int i = 0; i < GS; i++)
    {
//#pragma HLS UNROLL factor=8 skip_exit_check
#pragma HLS PIPELINE
      float quant_value = x[base_idx + i] / scale;   // scale
      int8_t quantized = (int8_t)round(quant_value); // round and clamp
      quantized_buffer[base_idx + i] = quantized;
    }
  }

  std::memcpy(qx->q, quantized_buffer, S * sizeof(int8_t));
  std::memcpy(qx->s, scale_buffer, num_groups * sizeof(float));
}
