# Mini-SGLang

A simple version of sglang project. For study purpose.

## Roadmap

### Basic Architecture
- [x] Tokenizer & Detokenizer & Scheduler procs
- [x] Model Runner forward
- [x] Zmq IPC for worker/control reqs
- [ ] Server & APIs
  
### Memory Management
- [x] PageManager
  - [x] page size = 1
  - [x] page size > 1
- [x] KVCache
- [ ] RadixCache
  - [x] Insert Prefix
  - [x] Prefix Matching
  - [ ] Evict

### Scheduler
- [x] FSFS/Random
- [ ] Cache Aware(LPM)
- [ ] Aggressive max_new_tokens prediction & Retracting
- [ ] Chunked Prefill

### Backend
- [x] Torch Native kernels
- [ ] FA3 support

### Distributed Support
- [x] Tensor Parallelism
- [ ] Data Parallel Attention(for MLA)
- [ ] MoE support
  - [ ] FusedMoE
  - [ ] EPMoE

### Models
- [x] Llama 3
- [ ] Mixtral MoE

### Optimizations
- [ ] Stream Output
- [ ] CUDA graph forward
- [ ] Overlap Scheduling
- [ ] Unit tests
