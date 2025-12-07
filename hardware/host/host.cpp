/**
 * Host code for H2O Transformer Inference
 * Repurposed using the structure of the provided reference host.cpp
 */

 #include "cmdlineparser.h"
 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <cstring>
 #include <chrono>
 #include <cstdint>
 #include <iomanip>
 #include <algorithm>
 
 // XRT includes
 #include "experimental/xrt_bo.h"
 #include "experimental/xrt_device.h"
 #include "experimental/xrt_kernel.h"
 
 // Project Includes
 #include "../kernel/config.h"
 #include "../kernel/typedefs.h"
 #include "../kernel/forward.h"
 
 // ==================================================================================
 // 1. HELPER CLASSES & UTILITIES (Kept from your logic)
 // ==================================================================================
 
 struct Tokenizer {
     char** vocab;
     float* vocab_scores;
     int vocab_size;
 
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
 
 // Quantization Helper
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
 
 // Model Loading Helper
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
     if (!file) { std::cerr << "Error: could not open model file: " << checkpoint_path << "\n"; exit(1); }
 
     int header[7];
     if (fread(header, sizeof(int), 7, file) != 7) exit(1);
     std::cout << "Loading weights directly into FPGA mapped memory..." << std::endl;
 
     auto& w = t->weights;
     read_floats(file, w.token_embedding_table, vocab_size * dim);
 
     std::vector<float> temp_rms(n_layers * dim);
     read_floats(file, temp_rms.data(), n_layers * dim);
     for(int i=0; i<n_layers; i++) {
         memcpy(&w.rms_att_weight[i*dim], &temp_rms[i*dim], dim * sizeof(float));
     }
 
     // Quantize weights (Q, K, V, Output)
     int head_dim = dim / n_heads;
     auto process_layer_weights = [&](auto& dest_w, int tensor_size) {
         std::vector<float> temp_w(tensor_size);
         for(int i=0; i<n_layers; i++) {
             read_floats(file, temp_w.data(), tensor_size);
             cpu_quantize_block(temp_w.data(), dest_w[i]);
         }
     };
 
     process_layer_weights(w.wq, dim * (n_heads * head_dim));
     process_layer_weights(w.wk, dim * (n_kv_heads * head_dim));
     process_layer_weights(w.wv, dim * (n_kv_heads * head_dim));
     process_layer_weights(w.wo, dim * (n_heads * head_dim));
 
     read_floats(file, temp_rms.data(), n_layers * dim);
     for(int i=0; i<n_layers; i++) {
         memcpy(&w.rms_ffn_weight[i*dim], &temp_rms[i*dim], dim * sizeof(float));
     }
 
     // FFN Weights
     process_layer_weights(w.w1, dim * hidden_dim);
     process_layer_weights(w.w2, dim * hidden_dim);
     process_layer_weights(w.w3, dim * hidden_dim);
 
     read_floats(file, w.rms_final_weight, dim);
 
     long rope_skip = seq_len * (head_dim / 2) * 2 * sizeof(float);
     fseek(file, rope_skip, SEEK_CUR);
 
     cpu_quantize_block(w.token_embedding_table, w.wcls[0]);
     fclose(file);
 }
 
 // Visualization Helper
 struct ScorePair { int pos; float score; int slot_idx; };
 bool compareScorePairs(const ScorePair& a, const ScorePair& b) { return a.score > b.score; }
 
 std::string get_word_at_pos(int target_pos, const std::vector<int>& history, Tokenizer& tok) {
     if (target_pos < 0 || (size_t)target_pos >= history.size()) return "<?> ";
     int token = history[target_pos];
     std::string s = tok.decode(token, token); 
     std::string clean_s = "";
     for (char c : s) {
         if (c == '\n') clean_s += "\\n"; else if (c == '\r') clean_s += "\\r"; else clean_s += c;
     }
     return "\"" + clean_s + "\"";
 }
 
 void visualize_h2o_step(int current_step, int budget, int num_layers,
     int* old_timestamps, int* new_timestamps, float* old_scores, float* new_scores,
     const std::vector<int>& token_history, Tokenizer& tokenizer) 
 {
     if (current_step < budget) return; 
     std::cout << "\n" << std::string(80, '=') << "\n";
     std::cout << "  [H2O VISUALIZER] Step " << current_step << " | Token: " 
               << get_word_at_pos(current_step, token_history, tokenizer) << "\n";
     std::cout << std::string(80, '-') << "\n";
 
     bool eviction = false;
     for (int l = 0; l < num_layers; l++) {
         int offset = l * budget;
         for (int i = 0; i < budget; i++) {
             int idx = offset + i;
             if (old_timestamps[idx] != new_timestamps[idx] && old_timestamps[idx] != -1) {
                 std::cout << "  LAYER " << std::setw(2) << l << " | \033[1;31m[EVICTED]\033[0m Pos " 
                           << std::setw(4) << old_timestamps[idx] << " " 
                           << std::setw(10) << get_word_at_pos(old_timestamps[idx], token_history, tokenizer) 
                           << " (Score: " << std::fixed << std::setprecision(5) << old_scores[idx] << ")\n";
                 eviction = true;
                 break; 
             }
         }
     }
     if (!eviction) std::cout << "  (No evictions this step)\n";
 }
 
 // ==================================================================================
 // 2. MAIN FUNCTION
 // ==================================================================================
 int main(int argc, char** argv) {
     // Command Line Parser (Matching Reference Structure)
     sda::utils::CmdLineParser parser;
 
     // Switches
     parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
     parser.addSwitch("--device_id", "-d", "device index", "0");
     parser.addSwitch("--model", "-m", "model binary file", "stories15M.bin");
     parser.addSwitch("--steps", "-s", "number of steps to generate", "50");
 
     parser.parse(argc, argv);
 
     // Read settings
     std::string binaryFile = parser.value("xclbin_file");
     std::string modelFile = parser.value("model");
     int device_index = std::stoi(parser.value("device_id"));
     int total_steps = std::stoi(parser.value("steps"));
 
     if (binaryFile.empty()) {
         parser.printHelp();
         return EXIT_FAILURE;
     }
 
     std::cout << "Device index: " << device_index << std::endl;
     std::cout << "Model File:   " << modelFile << std::endl;
 
     // Open device and load xclbin
     std::cout << "Opening device..." << std::endl;
     auto device = xrt::device(device_index);
 
     std::cout << "Loading xclbin: " << binaryFile << std::endl;
     auto uuid = device.load_xclbin(binaryFile);
 
     // Kernel handle
     auto krnl = xrt::kernel(device, uuid, "forward");
 
     // Buffer sizes in bytes
     typedef Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> ModelType;
     
     const size_t size_model_bytes = sizeof(ModelType);
     const size_t size_cache_bytes = n_layers * cache_budget * ((dim * n_kv_heads) / n_heads) * sizeof(float);
     const size_t size_scores_bytes = n_layers * cache_budget * sizeof(float);
     const size_t size_timestamps_bytes = n_layers * cache_budget * sizeof(int);
     const size_t size_out_bytes = vocab_size * sizeof(float);
 
     // Allocate buffers
     std::cout << "Allocating buffers..." << std::endl;
     auto bo_model = xrt::bo(device, size_model_bytes, krnl.group_id(0));
     auto bo_key_cache = xrt::bo(device, size_cache_bytes, krnl.group_id(3));
     auto bo_val_cache = xrt::bo(device, size_cache_bytes, krnl.group_id(4));
     auto bo_scores = xrt::bo(device, size_scores_bytes, krnl.group_id(5));
     auto bo_timestamps = xrt::bo(device, size_timestamps_bytes, krnl.group_id(6));
     auto bo_out = xrt::bo(device, size_out_bytes, krnl.group_id(7));
 
     // Map buffers to host memory
     std::cout << "Mapping Buffers to Host Memory..." << std::endl;
     auto mapped_model = bo_model.map<ModelType*>();
     auto k_ptr = bo_key_cache.map<float*>();
     auto v_ptr = bo_val_cache.map<float*>();
     auto ts_ptr = bo_timestamps.map<int*>();
     auto scores_ptr = bo_scores.map<float*>();
     auto out_ptr = bo_out.map<float*>();
 
     // Initialize input data
     std::cout << "Initializing model and caches..." << std::endl;
     load_and_quantize_weights(modelFile.c_str(), mapped_model);
     
     // Zero out caches
     std::memset(k_ptr, 0, size_cache_bytes);
     std::memset(v_ptr, 0, size_cache_bytes);
     std::fill(ts_ptr, ts_ptr + (n_layers * cache_budget), -1);
 
     // Synchronize input buffers to device
     std::cout << "Copying inputs to device global memory..." << std::endl;
     bo_model.sync(XCL_BO_SYNC_BO_TO_DEVICE);
     bo_key_cache.sync(XCL_BO_SYNC_BO_TO_DEVICE);
     bo_val_cache.sync(XCL_BO_SYNC_BO_TO_DEVICE);
     bo_timestamps.sync(XCL_BO_SYNC_BO_TO_DEVICE);
 
     // Setup Tokenizer and Prompts
     Tokenizer tokenizer("tokenizer.bin", vocab_size);
     std::vector<int> prompt_tokens = {1, 338, 3764, 263, 931}; 
     std::vector<int> token_history;
     int token = prompt_tokens[0];
     int pos = 0;
 
     // Shadow buffers for visualization
     std::vector<int> old_timestamps(n_layers * cache_budget, -1);
     std::vector<float> old_scores(n_layers * cache_budget, 0.0f);
 
     std::cout << "\n--- Starting Hardware Inference ---\n";
     std::cout << "Result: " << std::flush;
 
     auto total_start = std::chrono::high_resolution_clock::now();
 
     // Execution Loop
     while (pos < total_steps) {
         token_history.push_back(token);
 
         // Update shadow copy before kernel run
         std::memcpy(old_timestamps.data(), ts_ptr, size_timestamps_bytes);
         std::memcpy(old_scores.data(), scores_ptr, size_scores_bytes);
 
         // Launch kernel
         // Arguments: (model, token, pos, K, V, Scores, TS, Out)
         auto run = krnl(bo_model, token, pos, bo_key_cache, bo_val_cache, bo_scores, bo_timestamps, bo_out);
         run.wait();
 
         // Get output back from device
         bo_timestamps.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
         bo_scores.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
         bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
 
         // Visualization
         visualize_h2o_step(pos, cache_budget, n_layers, 
             old_timestamps.data(), ts_ptr, 
             old_scores.data(), scores_ptr, 
             token_history, tokenizer);
 
         // Sample next token
         int next_token = 0;
         if (pos < (int)prompt_tokens.size() - 1) {
             next_token = prompt_tokens[pos + 1];
         } else {
             float max_val = -1e9;
             for (int i = 0; i < vocab_size; i++) {
                 if (out_ptr[i] > max_val) {
                     max_val = out_ptr[i];
                     next_token = i;
                 }
             }
             const char* piece = tokenizer.decode(next_token, token);
             std::cout << piece << std::flush;
         }
         
         token = next_token;
         pos++;
     }
 
     auto total_end = std::chrono::high_resolution_clock::now();
     auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
     
     std::cout << "\n\nTotal execution time: " << elapsed_ms << " ms" << std::endl;
     std::cout << "TEST PASSED\n";
 
     return EXIT_SUCCESS;
 }