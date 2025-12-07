#pragma once
#include "typedefs.h"

static constexpr int dim = 288;           // Stories 15M
static constexpr int hidden_dim = 768;
static constexpr int n_layers = 6;
static constexpr int n_heads = 6;
static constexpr int n_kv_heads = 6;
static constexpr int vocab_size = 32000;
static constexpr int seq_len = 256;      // Max Logical Sequence Length
static constexpr int GS = 64;

// --- H2O Configuration ---
// The physical cache size (k in the paper)
static constexpr int cache_budget = 32;  
// The number of recent tokens to protect (local window)
static constexpr int recent_window = 16; 

constexpr Config config = {
    .dim = dim,
    .hidden_dim = hidden_dim,
    .n_layers = n_layers,
    .n_heads = n_heads,
    .n_kv_heads = n_kv_heads,
    .vocab_size = vocab_size,
    .seq_len = seq_len,
    .GS = GS,
};


// #pragma once
// #include "typedefs.h"

// // --- TINY CONFIG FOR FAST SIMULATION ---
// static constexpr int dim = 16;            // Tiny dimension
// static constexpr int hidden_dim = 32;     // Tiny FFN
// static constexpr int n_layers = 2;        // Just 2 layers
// static constexpr int n_heads = 4;         // 4 Heads
// static constexpr int n_kv_heads = 4;
// static constexpr int vocab_size = 256;    // Tiny vocab
// static constexpr int seq_len = 128;       // Logical limit
// static constexpr int GS = 4;              // Tiny Group Size

// // --- H2O Configuration ---
// static constexpr int cache_budget = 16;   // Small budget to trigger eviction fast
// static constexpr int recent_window = 4;   // Keep last 4 tokens

// constexpr Config config = {
//     .dim = dim,
//     .hidden_dim = hidden_dim,
//     .n_layers = n_layers,
//     .n_heads = n_heads,
//     .n_kv_heads = n_kv_heads,
//     .vocab_size = vocab_size,
//     .seq_len = seq_len,
//     .GS = GS,
// };