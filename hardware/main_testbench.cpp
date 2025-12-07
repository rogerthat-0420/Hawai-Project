#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iomanip>
#include <string>

// Include your project headers
#include "kernel/config.h"
#include "kernel/typedefs.h"
#include "kernel/forward.h" 

// ==================================================================================
// 1. TOKENIZER UTILITIES 
// ==================================================================================
struct Tokenizer {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
    unsigned char byte_pieces[512]; 

    Tokenizer(const char* path, int size) {
        vocab_size = size;
        vocab = new char*[vocab_size];
        vocab_scores = new float[vocab_size];

        FILE* file = fopen(path, "rb");
        if (!file) { std::cerr << "Error: could not open tokenizer.bin\n"; exit(1); }

        float max_token_len;
        if (fread(&max_token_len, sizeof(float), 1, file) != 1) { exit(1); }

        for (int i = 0; i < vocab_size; i++) {
            if (fread(&vocab_scores[i], sizeof(float), 1, file) != 1) { exit(1); }
            int len;
            if (fread(&len, sizeof(int), 1, file) != 1) { exit(1); }
            vocab[i] = new char[len + 1];
            if (fread(vocab[i], len, 1, file) != 1) { exit(1); }
            vocab[i][len] = '\0';
        }
        fclose(file);
    }

    const char* decode(int token, int prev_token) {
        if (token < 0 || token >= vocab_size) return "";
        return vocab[token];
    }
};

// ==================================================================================
// 2. CPU-SIDE QUANTIZATION 
// ==================================================================================
template <int SIZE>
void cpu_quantize_block(float* src_buffer, QuantizedTensor<SIZE>& dest) {
    constexpr int num_groups = SIZE / GS;
    constexpr float Q_MAX = 127.0f;

    for (int g = 0; g < num_groups; g++) {
        float wmax = 0.0f;
        int base_idx = g * GS;
        for (int i = 0; i < GS; i++) {
            float val = fabsf(src_buffer[base_idx + i]);
            if (val > wmax) wmax = val;
        }

        float scale = wmax / Q_MAX;
        if (scale < 1e-8f) scale = 1e-8f; 
        dest.s[g] = scale;

        float inv_scale = 1.0f / scale;
        for (int i = 0; i < GS; i++) {
            float val = src_buffer[base_idx + i];
            float quant_val = roundf(val * inv_scale);
            if (quant_val > 127.0f) quant_val = 127.0f;
            if (quant_val < -127.0f) quant_val = -127.0f;
            dest.q[base_idx + i] = (int8_t)quant_val;
        }
    }
}

// ==================================================================================
// 3. MODEL LOADER
// ==================================================================================
template<typename T>
void read_floats(FILE* f, T* buffer, size_t count) {
    if (fread(buffer, sizeof(float), count, f) != count) {
        std::cerr << "Error reading model weights." << std::endl;
        exit(1);
    }
}

void load_and_quantize_weights(const char* checkpoint_path, 
    Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>* t) 
{
    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) { std::cerr << "Error: could not open model file\n"; exit(1); }

    int header[7];
    if (fread(header, sizeof(int), 7, file) != 7) exit(1);
    std::cout << "Loading weights..." << std::endl;

    auto& w = t->weights;

    // 1. Token Embeddings
    read_floats(file, w.token_embedding_table, vocab_size * dim);

    // 2. RMS Attn Weights
    std::vector<float> temp_rms(n_layers * dim);
    read_floats(file, temp_rms.data(), n_layers * dim);
    for(int i=0; i<n_layers; i++) {
        memcpy(&w.rms_att_weight[i*dim], &temp_rms[i*dim], dim * sizeof(float));
    }

    // 3. WQ
    {
        int tensor_size = dim * (n_heads * (dim / n_heads)); 
        std::vector<float> temp_w(tensor_size); 
        for(int i=0; i<n_layers; i++) {
            read_floats(file, temp_w.data(), tensor_size);
            cpu_quantize_block(temp_w.data(), w.wq[i]);
        }
    }

    // 4. WK
    {
        int tensor_size = dim * (n_kv_heads * (dim / n_heads)); 
        std::vector<float> temp_w(tensor_size);
        for(int i=0; i<n_layers; i++) {
            read_floats(file, temp_w.data(), tensor_size);
            cpu_quantize_block(temp_w.data(), w.wk[i]);
        }
    }

    // 5. WV
    {
        int tensor_size = dim * (n_kv_heads * (dim / n_heads)); 
        std::vector<float> temp_w(tensor_size);
        for(int i=0; i<n_layers; i++) {
            read_floats(file, temp_w.data(), tensor_size);
            cpu_quantize_block(temp_w.data(), w.wv[i]);
        }
    }

    // 6. WO
    {
        int tensor_size = dim * (n_heads * (dim / n_heads)); 
        std::vector<float> temp_w(tensor_size);
        for(int i=0; i<n_layers; i++) {
            read_floats(file, temp_w.data(), tensor_size);
            cpu_quantize_block(temp_w.data(), w.wo[i]);
        }
    }

    // 7. RMS FFN
    read_floats(file, temp_rms.data(), n_layers * dim);
    for(int i=0; i<n_layers; i++) {
        memcpy(&w.rms_ffn_weight[i*dim], &temp_rms[i*dim], dim * sizeof(float));
    }

    // 8. W1
    {
        int tensor_size = dim * hidden_dim;
        std::vector<float> temp_w(tensor_size);
        for(int i=0; i<n_layers; i++) {
            read_floats(file, temp_w.data(), tensor_size);
            cpu_quantize_block(temp_w.data(), w.w1[i]);
        }
    }

    // 9. W2
    {
        int tensor_size = dim * hidden_dim;
        std::vector<float> temp_w(tensor_size);
        for(int i=0; i<n_layers; i++) {
            read_floats(file, temp_w.data(), tensor_size);
            cpu_quantize_block(temp_w.data(), w.w2[i]);
        }
    }

    // 10. W3
    {
        int tensor_size = dim * hidden_dim;
        std::vector<float> temp_w(tensor_size);
        for(int i=0; i<n_layers; i++) {
            read_floats(file, temp_w.data(), tensor_size);
            cpu_quantize_block(temp_w.data(), w.w3[i]);
        }
    }

    // 11. Final RMS
    read_floats(file, w.rms_final_weight, dim);

    // 12. Skip RoPE
    int head_size = dim / n_heads;
    long rope_skip = seq_len * (head_size / 2) * 2 * sizeof(float);
    fseek(file, rope_skip, SEEK_CUR);

    // 13. Classifier
    cpu_quantize_block(w.token_embedding_table, w.wcls[0]);

    fclose(file);
}

// ==================================================================================
// 4. VISUALIZATION ENGINE
// ==================================================================================

struct ScorePair {
    int pos;
    float score;
    int slot_idx;
};

// Comparator for sorting to find Heavy Hitters
bool compareScorePairs(const ScorePair& a, const ScorePair& b) {
    return a.score > b.score;
}

// Helper to get a clean string for a token position from history
std::string get_word_at_pos(int target_pos, const std::vector<int>& history, Tokenizer& tok) {
    if (target_pos < 0 || target_pos >= history.size()) return "<?> ";
    int token = history[target_pos];
    // Simple heuristic: pass token as prev_token just to get the string content
    std::string s = tok.decode(token, token); 
    // Escape newlines for clean printing
    std::string clean_s = "";
    for (char c : s) {
        if (c == '\n') clean_s += "\\n";
        else if (c == '\r') clean_s += "\\r";
        else clean_s += c;
    }
    return "\"" + clean_s + "\"";
}

void visualize_h2o_step(
    int current_step, 
    int budget, 
    int num_layers,
    int* old_timestamps, int* new_timestamps, 
    float* old_scores, float* new_scores,
    const std::vector<int>& token_history,
    Tokenizer& tokenizer
) {
    bool eviction_occurred_anywhere = false;
    
    // Detect if we are in the eviction phase (cache is full)
    if (current_step < budget) return; // Skip visualization during cache fill-up phase

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  [H2O VISUALIZER] Step " << current_step << " | Processing Token: " 
              << get_word_at_pos(current_step, token_history, tokenizer) << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (int l = 0; l < num_layers; l++) {
        int offset = l * budget;
        
        int victim_slot = -1;
        int evicted_pos = -1;
        float victim_score_before = 0.0f;

        // 1. Detect Eviction in this layer
        for (int i = 0; i < budget; i++) {
            int idx = offset + i;
            if (old_timestamps[idx] != new_timestamps[idx]) {
                // Slot changed. If it wasn't empty, it was an eviction.
                if (old_timestamps[idx] != -1) {
                    victim_slot = i;
                    evicted_pos = old_timestamps[idx];
                    victim_score_before = old_scores[idx];
                    eviction_occurred_anywhere = true;
                    break; 
                }
            }
        }

        // 2. Gather stats for this layer
        std::vector<ScorePair> active_tokens;
        for(int i=0; i<budget; i++) {
            int idx = offset + i;
            if (new_timestamps[idx] != -1) {
                active_tokens.push_back({new_timestamps[idx], new_scores[idx], i});
            }
        }
        std::sort(active_tokens.begin(), active_tokens.end(), compareScorePairs);

        // 3. Print Layer Info
        if (victim_slot != -1) {
            std::cout << "  LAYER " << std::setw(2) << l << " | ";
            
            // Print Eviction Details
            std::string evicted_word = get_word_at_pos(evicted_pos, token_history, tokenizer);
            
            // Colorize Eviction (ANSI Red for evicted, Green for new)
            std::cout << "\033[1;31m[EVICTED]\033[0m Pos " << std::setw(4) << evicted_pos 
                      << " " << std::setw(10) << evicted_word 
                      << " (Score: " << std::fixed << std::setprecision(5) << victim_score_before << ")"
                      << "  --> Replaced by Pos " << current_step << "\n";

            // Print Top 3 Heavy Hitters (Survivors)
            std::cout << "           | \033[1;32m[HEAVY HITTERS]\033[0m Top Retained Tokens:\n";
            for(int k=0; k<std::min((int)active_tokens.size(), 3); k++) {
                ScorePair& sp = active_tokens[k];
                // Skip the one we just inserted (usually has score 0 or close to it initially)
                if (sp.pos == current_step) continue; 

                std::string word = get_word_at_pos(sp.pos, token_history, tokenizer);
                std::cout << "           |    Rank " << k+1 << ": Pos " << std::setw(4) << sp.pos 
                          << " " << std::setw(10) << word 
                          << " | Score: " << sp.score << "\n";
            }
            std::cout << "\n";
        }
    }

    if (!eviction_occurred_anywhere) {
        std::cout << "  (No evictions this step - Filling cache or Static)\n";
    }
}

// ==================================================================================
// 5. MAIN
// ==================================================================================

int main(int argc, char** argv) {
    const char* model_path = "/home/kushang.agarwal/h2o_project/stories15M.bin";
    const char* tokenizer_path = "/home/kushang.agarwal/h2o_project/tokenizer.bin";
    
    std::cout << "Loading Tokenizer..." << std::endl;
    Tokenizer tokenizer(tokenizer_path, vocab_size);

    typedef Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> ModelType;
    ModelType* model = new ModelType();

    load_and_quantize_weights(model_path, model);

    // H2O Allocation
    constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
    constexpr int cache_size_floats = n_layers * cache_budget * kv_dim; 
    constexpr int state_size = n_layers * cache_budget;

    float* key_cache = new float[cache_size_floats];
    float* val_cache = new float[cache_size_floats];
    
    // H2O Live State
    float* acc_scores = new float[state_size];
    int* timestamps = new int[state_size];

    // H2O Snapshots (For "Before" vs "After" comparison)
    int* old_timestamps = new int[state_size];
    float* old_scores = new float[state_size];

    std::memset(key_cache, 0, cache_size_floats * sizeof(float));
    std::memset(val_cache, 0, cache_size_floats * sizeof(float));
    std::memset(acc_scores, 0, state_size * sizeof(float));
    std::memset(timestamps, -1, state_size * sizeof(int));

    // Prompt and History
    std::vector<int> prompt_tokens = {1, 338, 3764, 263, 931}; // "Once upon a time"
    
    // This vector stores ALL tokens generated/prompted so we can look up words by position
    std::vector<int> token_history; 
    
    int token = prompt_tokens[0]; 
    int pos = 0;
    std::vector<float> logits(vocab_size);

    std::string full_generated_text = "";

    std::cout << "\n--- CONFIGURATION ---\n";
    std::cout << "Cache Budget: " << cache_budget << " tokens\n";
    std::cout << "Sequence Limit: " << seq_len << "\n";
    std::cout << "---------------------\n";

    // Initial history population
    // Note: The loop logic processes `token` then finds `next_token`. 
    // We push `token` to history at the start of the loop.

    int total_steps = 150; // Increased steps to ensure we see plenty of evictions

    while (pos < total_steps) {
        
        // 1. Record History
        token_history.push_back(token);

        // 2. Snapshot State (BEFORE Forward/Eviction)
        std::memcpy(old_timestamps, timestamps, state_size * sizeof(int));
        std::memcpy(old_scores, acc_scores, state_size * sizeof(float));

        // 3. Run HLS Kernel
        forward(model, token, pos, key_cache, val_cache, acc_scores, timestamps, logits.data());

        // 4. Visualize Changes (Compare Old Snapshot vs New State)
        visualize_h2o_step(pos, cache_budget, n_layers, old_timestamps, timestamps, old_scores, acc_scores, token_history, tokenizer);

        // 5. Token Sampling & Advance
        int next_token = 0;
        const char* piece = nullptr;
        if (pos < prompt_tokens.size() - 1) {
            next_token = prompt_tokens[pos + 1];
            piece = tokenizer.decode(next_token, token);
        } else {
            float max_val = -1e9; 
            for (int i = 0; i < vocab_size; i++) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    next_token = i;
                }
            }
            piece = tokenizer.decode(next_token, token);
            
            // Simple print for progress (detailed view is in visualize_h2o_step)
            std::cout << piece << std::flush; 
        }

        if (piece != nullptr) {
            full_generated_text += piece;
        }

        token = next_token;
        pos++;
    }
    
    std::cout << "\n\n" << std::string(50, '#') << "\n";
    std::cout << " FINAL GENERATED STORY OUTPUT \n";
    std::cout << std::string(50, '#') << "\n";
    // Reconstruct start to ensure it looks nice
    std::cout << "Once" << full_generated_text << "\n"; 
    std::cout << std::string(50, '#') << "\n";

    delete model;
    delete[] key_cache;
    delete[] val_cache;
    delete[] acc_scores;
    delete[] timestamps;
    delete[] old_timestamps;
    delete[] old_scores;
    return 0;
}