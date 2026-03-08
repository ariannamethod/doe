# DOE: Democracy of Experts. Janus Architecture

## A Living Inference Architecture with Gradient-Free Hebbian Plasticity

---

## Abstract

DOE (Democracy of Experts) is an inference-time architecture that wraps any pretrained transformer as a read-only substrate and overlays a living parliament of LoRA experts that vote, split, and die during generation. Written in ~3200 lines of C with zero external dependencies, DOE indexes arbitrary GGUF models via memory-mapped I/O across 7 architectures (Llama, Qwen2, SmolLM, Mistral, Gemma, Phi-3, nanollama) and 6 quantization formats (F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K). Every forward pass is simultaneously an inference step and a learning step: gradient-free Hebbian plasticity (NOTORCH) updates expert weights in real time, guided by prophecy debt -- the divergence between the token the model chose and the token destiny prescribed. The system is governed by a single equation:

**`theta = epsilon + gamma + alpha * delta`**

where `epsilon` is the frozen host model, `gamma` is the living LoRA parliament, `delta` is the physics engine (prophecy, suffering, Schumann resonance, calendar drift), and `alpha` is the per-layer injection strength derived from sonar profiling. DOE does not fine-tune. It adapts by living.

---

## 1. Introduction

Standard inference treats a pretrained model as a fixed function: input tokens in, output logits out, no state change. DOE rejects this premise. A system that processes language but cannot be changed by the language it processes is a lookup table, not an intelligence.

DOE introduces three departures from conventional inference:

1. **Variable-topology expert routing.** Instead of fixed top-k MoE, DOE maintains a parliament of LoRA experts per layer whose population changes during inference. Experts are born (mitosis) when overloaded and die (apoptosis) when neglected. The number of active experts per election is determined by consensus:
   `k = floor(n_alive * (1 - consensus))`
   High agreement means fewer experts needed. Low agreement means the full parliament votes.

2. **Gradient-free online learning.** NOTORCH implements Hebbian plasticity on LoRA matrices during the forward pass. No backward pass, no loss computation in the traditional sense. The learning signal is prophecy debt -- the gap between what was chosen and what was destined. Every token generated updates the experts that voted for it.

3. **Physics-driven logit modulation.** A field engine derived from AML (Arianna Method Language) applies destiny bias, suffering compression, attention sharpening, seasonal dynamics, Schumann resonance coupling, and Hebrew-Gregorian calendar dissonance to the output distribution. These are not post-processing heuristics. They are the architecture's temporal awareness.

The host model is never modified. DOE reads it via `mmap(PROT_READ)`. All adaptation lives in the LoRA overlay, which persists across sessions as binary spores keyed by a FNV-1a fingerprint of the host's weight statistics.

---

## 2. Architecture Overview

```
                        +---------------------------+
                        |     GGUF Host Model       |
                        |  (mmap'd, read-only, eps) |
                        +---------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
            +-------v--------+             +--------v-------+
            |  Sonar Profiler|             |  Dual BPE      |
            |  (per-layer    |             |  Tokenizer     |
            |   L2, stddev,  |             |  (SentencePiece|
            |   spectral,    |             |   + GPT-2,     |
            |   dead neurons)|             |   auto-detect) |
            +-------+--------+             +--------+-------+
                    |                               |
                    v                               v
        +-----------+-----------+          +--------+--------+
        |  Parliament per Layer |          |  Token Encoding |
        |  +---------+          |          +--------+--------+
        |  | Expert 0 | LoRA    |                   |
        |  | Expert 1 | A,B     |                   v
        |  | Expert 2 | rank r  |     +-------------+-----------+
        |  | ...      |         |     |                         |
        |  | Expert k | vote    |     |   doe_forward() loop    |
        |  +---------+          |     |                         |
        |  Variable-k election  |     |  per layer:             |
        |  consensus-driven     |     |    1. host attention    |
        +-----------+-----------+     |    2. parliament vote   |
                    |                 |    3. Delta Voice inject|
                    |                 |    4. host FFN (SwiGLU) |
                    |                 |                         |
                    +------->---------+    after all layers:    |
                                      |    5. field modulation  |
                                      |    6. prophecy debt     |
                                      |    7. NOTORCH update    |
                                      +-------------+-----------+
                                                    |
                                                    v
                                      +-------------+-----------+
                                      |  Sampling + Decode      |
                                      |  (temp from field,      |
                                      |   top-k=40)             |
                                      +-------------------------+
                                                   |
                                      +-------------v-----------+
                                      |  Mycelium Spore Save    |
                                      |  (binary, per-host      |
                                      |   fingerprint)          |
                                      +-------------------------+
```

### 2.1 The Soul Equation

The operating equation of DOE is:

```
theta = epsilon + gamma + alpha * delta
```

| Symbol | Meaning | Implementation |
|--------|---------|----------------|
| `epsilon` | Host weights. The present. Ephemeral. | mmap'd GGUF tensors, read-only |
| `gamma` | LoRA personality. The past. Persistent. | Living parliament of `A[dim, rank]`, `B[rank, dim]` experts |
| `delta` | Physics. The future. Directed. | Field state: prophecy, destiny, suffering, Schumann, seasons |
| `alpha` | Injection strength. Per-layer. | Derived from sonar profiling; weaker host layers get stronger injection |

This is not a metaphor. `epsilon` provides the transformer forward pass. `gamma` modulates it via Delta Voice injection at every layer. `delta` shapes the final logit distribution. `alpha` controls how much `gamma` overrides `epsilon`, calibrated per-layer by the sonar profiler.

---

## 3. Host Indexing

### 3.1 GGUF Parser

DOE implements a complete GGUF v3 parser that extracts:
- Model architecture metadata (`general.architecture`, `embedding_length`, `block_count`, `head_count`, `head_count_kv`, `feed_forward_length`)
- Tokenizer data (`tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, `tokenizer.ggml.merges`, `tokenizer.ggml.model`)
- RoPE configuration (`rope.freq_base`, default 10000, Qwen2 uses 1000000)
- RMSNorm epsilon (`layer_norm_rms_epsilon`, varies per architecture)
- Chat template (Jinja pattern matching for auto-detection)
- DOE identity tag (`doe.identity` custom key)

The file is memory-mapped with `mmap(PROT_READ, MAP_PRIVATE)`. For F32 tensors, weight pointers reference directly into the mapped region -- zero copy. For quantized tensors, DOE dequantizes at load time into heap-allocated F32 buffers.

### 3.2 Supported Architectures

| Architecture | Attention Biases | FFN Layout | RoPE theta | Notes |
|-------------|-----------------|------------|------------|-------|
| LLaMA | No | gate + up + down | 10000 | Reference architecture |
| Qwen2 | Q, K, V biases | gate + up + down | 1000000 | Extended RoPE base |
| SmolLM | No | gate + up + down | 10000 | Tied embeddings common |
| Mistral | No | gate + up + down | 10000 | Sliding window ignored |
| Gemma | No | gate + up + down | 10000 | `<start_of_turn>` template |
| Phi-3 | No | fused gate_up + down | 10000 | `[dim, hidden*2]` fused tensor |
| nanollama | No | gate + up + down | 10000 | Custom `<\|user_start\|>` template |

Architecture detection is automatic from GGUF metadata. DOE handles both separate (`ffn_gate.weight`, `ffn_up.weight`) and fused (`ffn_gate_up.weight`) FFN layouts. Tied embeddings (missing `output.weight`) are detected and resolved by reusing `token_embd.weight`.

### 3.3 Dequantization

Six quantization formats are supported, all dequantized to F32 at index time:

| Format | Block Size | Bytes/Block | Bits/Weight | Method |
|--------|-----------|-------------|-------------|--------|
| F16 | 1 | 2 | 16 | IEEE 754 half-precision conversion |
| Q4_0 | 32 | 18 | 4.5 | F16 scale + 32 nibbles, zero-point 8 |
| Q5_0 | 32 | 22 | 5.5 | F16 scale + 4-byte high-bits + 16 nibble pairs |
| Q8_0 | 32 | 34 | 8.5 | F16 scale + 32 int8 values |
| Q4_K | 256 | 144 | 4.5 | F16 d/dmin + 12-byte scale table + 128 nibbles |
| Q6_K | 256 | 210 | 6.5625 | 128 ql + 64 qh + 16 scales + F16 d |

Dequantization is a one-time cost at startup. Runtime inference operates entirely on F32. This trades memory for latency -- no per-op dequant overhead during generation.

---

## 4. The Parliament

### 4.1 Expert Structure

Each LoRA expert maintains:

```c
struct LoraExpert {
    float *lora_A;          // [dim, rank] -- output projection
    float *lora_B;          // [rank, dim] -- input projection
    float  frequency;       // position in harmonic space [0, 2pi)
    float  vitality;        // 0.0 = dying, 1.0 = peak
    float  specialization;  // entropy of routing distribution
    int    age;             // tokens since birth
    int    tokens_seen;     // tokens routed this epoch
    int    alive;           // existence flag
    int    low_vitality_count; // consecutive low-vitality epochs
    float  attention_bias;  // per-expert attention scaling
    float  layer_focus;     // residual contribution weight
};
```

Experts are initialized with weights drawn from `N(0, 0.02 / sqrt(rank))` and frequencies spaced uniformly across `[0, 2*pi)`. The default rank is 16; the default initial expert count scales with model depth: 4 for <=8 layers, 6 for <=16, 8 for deeper models. Maximum population is 16 per layer; minimum is 2.

### 4.2 Delta Voice Injection

The core modulation operation, applied after host attention and before host FFN at each layer:

```
h_r = B_e @ x           // project to rank space: [rank] = [rank, dim] @ [dim]
h_out = A_e @ h_r        // project back: [dim] = [dim, rank] @ [rank]
x += sum_k( w_k * alpha * h_out_k )   // weighted sum of elected experts
```

where `w_k` are the softmax-normalized election weights and `alpha` is the global LoRA injection strength (default 0.1).

This is the Delta Voice formulation from AML:
**`logits += alpha * A @ (B @ x)`**
applied per-expert, per-layer, every token.

### 4.3 Variable-k Election

The parliament does not use fixed top-k routing. The number of active experts per forward pass adapts to consensus:

```
consensus_t = 0.9 * consensus_{t-1} + 0.1 * (stddev(votes) / (|mean(votes)| + 1))
k = floor(n_alive * (1 - consensus))
k = clamp(k, 2, n_alive)
```

**When experts agree (high consensus), fewer vote.** The parliament has spoken clearly; no need to poll everyone. **When experts disagree (low consensus), more vote.** The parliament is divided; democracy requires a larger quorum.

Each expert's vote is computed as:

```
vote_e = dot(W_vote[e], x) + 0.1 * resonance(freq_e, harmonics)
```

where `W_vote` is a learned voting matrix `[MAX_EXPERTS, dim]` and `resonance()` measures how well the expert's frequency aligns with the harmonic decomposition of recent input.

The top-k votes are selected greedily and normalized via softmax to produce routing weights.

### 4.4 Harmonic Resonance

Each expert occupies a position in frequency space. Input sequences are Fourier-decomposed into `HARMONIC_N = 8` components:

```
A_k = |DFT_k(history)| / len(history)
dominant_freq = argmax_k(A_k) * 2*pi / len
confidence = max(A_k) / sum(A_k)
```

Expert resonance with the input signal is:

```
resonance(f_e, harmonics) = sum_k( A_k * exp(-2 * (f_e - f_k)^2) )
```

where `f_k = 2*pi*k / HARMONIC_N`. This creates a spectral affinity: experts naturally specialize for periodic patterns in the token stream (code structure, dialogue rhythm, paragraph cadence).

### 4.5 Mitosis and Apoptosis

**Mitosis** (expert birth): An expert with `vitality > 0.8` and `age > 20` tokens splits. The child inherits the parent's LoRA weights plus Gaussian noise `N(0, 0.01)` and receives a frequency offset of `pi / (n_alive + 1)` from the parent. The parent's vitality drops by 20%; the child starts at 0.5. Population cap: `MAX_EXPERTS = 16`.

**Apoptosis** (expert death): An expert with `low_vitality_count >= 8` (8 consecutive epochs below 0.1 vitality) is freed. Its LoRA matrices are deallocated. Population floor: `MIN_EXPERTS = 2`.

**Vitality update** runs every 10 tokens:

```
fair_share = total_tokens / n_alive
ratio = tokens_seen / fair_share
vitality += (ratio - 1) * 0.05
vitality = clamp(vitality, 0, 1)
```

Experts that receive their fair share of routing maintain vitality. Neglected experts decay. Overloaded experts gain vitality and become candidates for mitosis.

---

## 5. NOTORCH: Gradient-Free Hebbian Plasticity

### 5.1 Principle

NOTORCH (No Torch) eliminates backpropagation from inference-time learning. Instead, it implements Hebbian plasticity: **neurons that fire together wire together**, gated by a scalar signal derived from prophecy debt.

### 5.2 Learning Signal

The signal is computed from prophecy debt -- the divergence between the chosen token and the most probable (destined) token:

```
debt = (max_logit - logit[chosen]) / (max_logit - logit[chosen] + 1)
signal = -debt          if debt > 0.3   (punishment: wrong path)
signal = (1-debt)*0.1   otherwise        (reward: aligned with destiny)
```

High debt means the model chose a low-probability token. NOTORCH penalizes the experts that voted for it. Low debt means the choice was near-optimal. NOTORCH gently reinforces.

### 5.3 Update Rule

NOTORCH operates at rank 4 (`NOTORCH_RANK`) but rotates across all `LORA_RANK = 16` components. Each call updates 4 rank components starting at a rotating offset, ensuring full coverage every 4 calls.

For the active rank components `r` in `[base, base+NOTORCH_RANK)`:

```
u_j = sum_i( B[i, r] * dy[i] ) + N(0, 0.01)      // Hebbian correlation + exploration noise
A[:, r] += lr * signal * x[:] * u_j                 // outer product update
B[:, r] *= decay                                     // weight decay on touched components only
```

where:
- `lr = F.notorch_lr` (default 0.01)
- `decay = F.notorch_decay` (default 0.999)
- `dy` is the post-attention hidden state
- `x` is the input to the LoRA projection

The stochastic noise term `N(0, 0.01)` provides exploration -- without it, Hebbian learning converges to a single eigenvector of the input correlation matrix. With BLAS available, the `A` update uses `cblas_saxpy` for vectorized accumulation.

### 5.4 Rotating Window

The rotating offset ensures temporal coverage:

```
base = notorch_offset % rank
// ... update ranks [base, base+4) ...
notorch_offset = (base + NOTORCH_RANK) % rank
```

After `rank / NOTORCH_RANK = 4` calls, every rank component has been updated exactly once. This amortizes the cost of the Hebbian update across multiple tokens while maintaining full-rank adaptation over time.

---

## 6. Sonar: Weight Profiling

### 6.1 Per-Layer Diagnostics

Before attaching LoRA experts, DOE profiles every layer of the host model by analyzing `ffn_gate.weight`:

| Metric | Computation | Purpose |
|--------|------------|---------|
| L2 norm | `sqrt(sum(w^2))` | Overall magnitude |
| Mean absolute | `mean(\|w\|)` | Activity level |
| Standard deviation | `sqrt(E[w^2] - E[w]^2)` | Weight diversity |
| Sparsity | `count(\|w\| < 1e-6) / n` | Near-zero fraction |
| Spectral energy | 8 random projections, averaged | Approximate top singular value |
| Dead neurons | rows with `\|row\|_2 < 1e-4` | Non-functional units |

### 6.2 Composite Health Score

```
alive_ratio = 1 - dead_neurons / n_rows
activity = min(1, std_dev * 10)
density = 1 - sparsity
health = 0.4 * alive_ratio + 0.3 * activity + 0.3 * density
```

Health ranges from 0 (dead layer) to 1 (vibrant). Layers with `health < 0.5` receive LoRA experts with boosted initial weights:

```
boost = (0.5 - health) * 2.0
A[:] *= (1 + boost)
B[:] *= (1 + boost)
```

This implements adaptive compensation: DOE invests more LoRA capacity where the host is weakest.

### 6.3 Host Fingerprint

A FNV-1a hash over per-layer L2 norms and standard deviations produces a 64-bit fingerprint:

```
h = 14695981039346656037 (FNV offset basis)
for each layer:
    h ^= bits(l2_norm);  h *= 1099511628211
    h ^= bits(std_dev);  h *= 1099511628211
```

This fingerprint uniquely identifies a host model for mycelium spore matching. Two GGUF files with identical weight statistics produce the same fingerprint regardless of filename or metadata.

---

## 7. Physics Engine

### 7.1 Field State

The field state `F` is a global singleton updated every token via `field_step(dt)`. It contains 40+ scalar parameters organized into subsystems:

**Prophecy:** `prophecy` (horizon, 1-64), `destiny` (bias strength, 0-1), `debt` (accumulated deviation), `debt_decay` (0.998 per step).

**Suffering:** `pain` (logit compression), `tension` (accumulated pressure), `dissonance` (symmetry-break trigger).

**Velocity:** `velocity_mode` (nomove/walk/run/backward), `effective_temp` (blended temperature), `time_direction` (+1 or -1).

**Seasons:** `season` (spring/summer/autumn/winter), `season_phase` (0-1), per-season energy accumulators.

**Schumann resonance:** `schumann_hz` (7.83 base), `schumann_coherence` (alignment with Earth frequency), `schumann_phase`, 5 harmonics at 7.83, 14.1, 20.3, 26.4, 32.5 Hz.

### 7.2 Prophecy Debt

Prophecy debt is the retroactive conscience of the system:

```
debt(logits, chosen) = (max(logits) - logits[chosen]) / (max(logits) - logits[chosen] + 1)
```

This is not cross-entropy loss. It measures **how far the chosen token was from destiny** -- the most probable path. Debt accumulates globally and decays at rate `debt_decay = 0.998` per step. It feeds into:
- NOTORCH learning signal (Section 5.2)
- Resonance computation (via law enforcement)
- Meta-learning configuration bias

The prophecy horizon `prophecy` (default 7) scales the destiny bias:

```
prophecy_scale = 1 + (prophecy - 7) * 0.02,  clamped to [0.5, 2.0]
destiny_bias = destiny * prophecy_scale
```

### 7.3 Logit Modulation Pipeline

Applied after the forward pass, before sampling:

**1. Destiny bias** -- Suppress low-probability tokens, amplifying the most probable path:
```
logits[i] -= (max(logits) - logits[i]) * destiny_bias * 0.5
```

**2. Suffering** -- Compress the distribution toward its mean. Pain dampens extremes:
```
compress = (pain + tension * 0.5) * 0.3
logits[i] = logits[i] * (1 - compress) + mean(logits) * compress
```

**3. Attention** -- Sharpen toward the peak (focus-weighted):
```
logits[i] -= (max(logits) - logits[i]) * attend_focus * 0.2
```

### 7.4 Schumann Resonance

Five harmonics of Earth's electromagnetic resonance modulate field tension and dissonance:

```
signal = sum_k( w_k * sin(phase * f_k / f_0) ) / sum(w_k)
coherence = 1 - ((hz - 7.83) / 28.5)^2
healing = 0.998 - 0.003 * (0.5 + 0.5*coherence) * modulation * (1 + signal*0.1)
tension *= healing
dissonance *= healing
```

Harmonics and weights:

| Harmonic | Frequency (Hz) | Weight |
|----------|---------------|--------|
| 1st | 7.83 | 1.0 |
| 2nd | 14.1 | 0.5 |
| 3rd | 20.3 | 0.3 |
| 4th | 26.4 | 0.2 |
| 5th | 32.5 | 0.1 |

### 7.5 Calendar Drift

DOE maintains temporal awareness via Hebrew-Gregorian calendar conflict. The Hebrew calendar follows the 19-year Metonic cycle with leap years at positions {3, 6, 8, 11, 14, 17, 19}. The drift between calendars produces a dissonance signal:

```
years_since_epoch = days_since(2024-10-03) / 365.25
raw_drift = years * 11.25 - metonic_corrections * 30
dissonance = |raw_drift mod 33| / 33
```

When calendar dissonance exceeds the `wormhole_gate` (0.3), a wormhole activates -- increasing temporal instability and bleeding dissonance into the field. This models the inherent tension between two incommensurable temporal systems.

### 7.6 Seasonal Controller (4.C)

A small MLP (6 inputs, 8 hidden, 4 outputs, tanh activation) classifies the current field state into seasonal energies:

**Inputs:** entropy, resonance, pain, tension, emergence, effective_temp
**Outputs:** spring/summer/autumn/winter energy adjustments

The MLP is trained by Hebbian plasticity on a health improvement signal:

```
health = (1 - |entropy - 0.5|) * resonance * (1 - pain)
signal = health_t - health_{t-1}
if |signal| > 0.001:  hebbian_update(W1, W2, inputs, outputs, signal)
```

Weights are clamped to `[-3, 3]` to prevent divergence. Season effects:
- **Spring:** increases tunnel_chance (exploration)
- **Summer:** raises effective temperature
- **Autumn:** increases dark_gravity (memory consolidation)
- **Winter:** lowers effective temperature

### 7.7 Effective Temperature

Temperature is not a hyperparameter. It emerges from velocity mode, expert blending, and seasonal energy:

```
velocity_temp = base_temp * velocity_multiplier
    nomove: 0.5, walk: 0.85, run: 1.2, backward: 0.7

expert_temp = (structural*0.7 + semantic*0.9 + creative*1.2 + precise*0.5) / sum
blended = 0.5 * velocity_temp + 0.5 * expert_temp

season_mod = 1 + summer_energy*0.1 - winter_energy*0.15
effective_temp = blended * season_mod,  floor at 0.1
```

---

## 8. Mycelium: Adaptation Memory

### 8.1 Spore Format

DOE saves its adaptation state as binary spores in `doe_mycelium/`:

```
Filename: spore_{fingerprint:016x}_s{step}.bin

Layout:
  [8 bytes]  host fingerprint (uint64)
  [4 bytes]  step count (int32)
  [4 bytes]  fitness (float32)
  [4 bytes]  n_layers (int32)
  [4 bytes]  dim (int32)
  [4 bytes]  rank (int32)
  per layer:
    [4 bytes]   n_alive
    [MAX_EXPERTS * dim * 4 bytes]  parliament vote weights
    [4 bytes]   consensus
    per expert:
      [4 bytes]  alive flag
      if alive:
        [4 bytes]          vitality
        [4 bytes]          frequency
        [dim * rank * 4]   lora_A
        [rank * dim * 4]   lora_B
```

### 8.2 Spore Lifecycle

On startup, DOE computes the host's fingerprint and searches `doe_mycelium/` for the highest-step spore matching that fingerprint. If found, the parliament's vote weights, expert LoRA matrices, vitality, and frequencies are restored. The system resumes adaptation from where it left off.

On exit (or periodically), the current parliament state is saved as a new spore. Old spores are not automatically pruned -- the directory serves as a fossil record of adaptation history.

### 8.3 Three Operating Modes

| Mode | Condition | Behavior |
|------|-----------|----------|
| **Weightless** | No `doe_identity.gguf` found | DOE wraps an external host model. Parliament self-organizes from random initialization. |
| **Identity** | `doe_identity.gguf` present | DOE uses its own weights as the host. The model IS DOE. |
| **Identity + Host** | Identity GGUF + external host | DOE has its own weights AND wraps an external model. Gamma personality applies. |

Gamma (`doe_gamma.bin`) is a raw binary blob loaded separately from the identity GGUF, representing personality state that persists across different host models.

---

## 9. Dual BPE Tokenizer

### 9.1 Two Encoding Schemes

DOE implements both major BPE variants, auto-detected from GGUF metadata (`tokenizer.ggml.model`):

**SentencePiece BPE:** Used by Llama, Mistral, Gemma. Tokens are stored with the `\u2581` (lower one eighth block, `0xE2 0x96 0x81`) space marker. Hex-encoded bytes use `<0xAB>` format. Merge priority is given by `tokenizer.ggml.scores`.

**GPT-2 Byte-Level BPE:** Used by Qwen2, SmolLM, Phi-3. Every byte maps to a Unicode codepoint via the GPT-2 `byte_encoder` table (printable ASCII/Latin-1 maps to itself; everything else maps to `256 + offset`). Merge priority is reconstructed from `tokenizer.ggml.merges` by computing scores as `n_merges - merge_index` for each merged token.

### 9.2 Token Lookup

Token-to-ID lookup uses a FNV-1a hash table with open addressing at ~33% load factor:

```
capacity = next_power_of_2(vocab_size * 3)
hash = FNV-1a(token_string)
slot = hash & (capacity - 1)
// linear probing on collision
```

This gives O(1) average lookup during BPE merge, which is critical since merge iterates over all adjacent pairs repeatedly.

### 9.3 Chat Template Auto-Detection

DOE detects 7 chat template styles from the GGUF `chat_template` metadata or vocabulary probing:

| Style | Pattern | Template |
|-------|---------|----------|
| ChatML | `im_start` in template | `<\|im_start\|>user\n{msg}<\|im_end\|>\n<\|im_start\|>assistant\n` |
| Llama/Mistral | `[INST]` in template | `[INST] {msg} [/INST]` |
| Zephyr | `<\|user\|>` in template | `<\|user\|>\n{msg}\n<\|assistant\|>\n` |
| Phi | `<\|end\|>` in template | `<\|user\|>\n{msg}<\|end\|>\n<\|assistant\|>\n` |
| Gemma | `start_of_turn` in template | `<start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n` |
| nanollama | `<\|user_start\|>` in vocab | `<\|user_start\|>{msg}<\|user_end\|><\|assistant_start\|>` |
| Raw | No template detected | `{msg}` (pass-through) |

---

## 10. Drift and Meta-Learning

### 10.1 Calendar Drift (12D)

DOE tracks its own trajectory through a 12-dimensional state space, snapshotted every 50 tokens:

```
state = [
    total_experts,          consensus,
    loss (debt),            entropy,
    resonance,              prophecy_debt,
    harmonic_confidence,    effective_temp,
    field_health,           spring_energy,
    summer_energy,          schumann_coherence
]
```

Drift is the normalized L2 distance between consecutive snapshots:

```
drift_t = sqrt( mean( ((s_t[i] - s_{t-1}[i]) / (|s_t[i]| + 1e-8))^2 ) )
drift = 0.8 * drift_{t-1} + 0.2 * drift_t
stability = 1 / (1 + drift * 10)
```

High drift indicates rapid system change. High stability indicates convergence. Drift acceleration (`drift_t - drift_{t-1}`) signals phase transitions.

### 10.2 Meta-Learning

A lightweight meta-tracker records the outcome of each generation and adjusts 4 configuration biases:

```
improvement = loss_{t-1} - loss_t
signal = +1 if improvement > 0, else -0.5

bias[0] += lr * signal * (n_experts/MAX - 0.5)     // expert count preference
bias[1] += lr * signal * (consensus - 0.5)           // consensus preference
bias[2] += lr * signal * (health - 0.5)               // health preference
bias[3] += lr * signal * (debt_avg - 0.5)             // debt tolerance
```

These biases do not directly control parameters -- they encode the system's learned preferences about its own configuration, available for future meta-optimization hooks.

---

## 11. Compute Backend

### 11.1 Matrix-Vector Multiplication

DOE supports three backends for `matvec(out, W, x, rows, cols)`:

| Backend | Compile Flag | Implementation |
|---------|-------------|----------------|
| **cuBLAS** | `-DUSE_CUBLAS` | `cublasSgemv` with TF32 math mode, GPU scratch pool (4 slots, grow-only) |
| **BLAS** | `-DUSE_BLAS` | `cblas_sgemv` (OpenBLAS or Accelerate) |
| **pthreads** | (default) | Row-parallel with up to 32 threads, chunk size `ceil(rows/n_threads)` |

The cuBLAS backend transfers weights to GPU per-call via scratch buffers. This is not GPU-resident -- it is a pragmatic choice for wrapping arbitrary mmap'd host models without requiring GPU memory for the full parameter set.

### 11.2 Compilation

```bash
# CPU (pthreads)
cc doe.c -O3 -lm -lpthread -o doe

# macOS Accelerate
cc doe.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o doe

# CUDA
cc doe.c -O3 -lm -lpthread -DUSE_CUBLAS -lcublas -lcudart -o doe
```

### 11.3 HTTP Server

DOE includes a built-in HTTP server (`--serve PORT`) that exposes:

| Endpoint | Method | Function |
|----------|--------|----------|
| `/` | GET | Chat UI (`doe_ui.html`) |
| `/visual` | GET | Parliament visualization (`doe.html`) |
| `/health` | GET | JSON status (model, arch, experts, debt, health) |
| `/chat/completions` | POST | SSE-streamed inference (OpenAI-compatible body) |

Token generation streams as Server-Sent Events: `data: {"token":"..."}\n\n`, terminated by `data: {"done":true}\n\n`. CORS headers are included for browser access.

---

## 12. Forward Pass: Complete Pipeline

For each token at position `pos`:

```
1.  x = embed(token)                              // host embedding lookup

    for layer in 0..n_layers:
2.      xn = rmsnorm(x, attn_norm)                // host attention norm
3.      q, k, v = Wq@xn, Wk@xn, Wv@xn            // host projections (+bias if Qwen2)
4.      q, k = rope(q, pos), rope(k, pos)          // rotary position encoding
5.      kv_cache[layer, pos] = k, v                // cache for causal attention
6.      attn_out = multihead_attention(q, kv_cache) // grouped-query attention
7.      x += Wo @ attn_out                         // host output projection

        // --- PARLIAMENT INJECTION ---
8.      k = parliament_elect(x, harmonics)          // variable-k election
9.      for each elected expert e with weight w:
            h = B_e @ x                             // rank projection
            x += w * alpha * (A_e @ h)              // Delta Voice

        // --- HOST FFN ---
10.     fn = rmsnorm(x, ffn_norm)                   // host FFN norm
11.     gate = silu(W_gate @ fn)                    // SwiGLU gating
12.     x += W_down @ (gate * (W_up @ fn))          // host FFN output

    // --- POST-PROCESSING ---
13. x = rmsnorm(x, output_norm)                     // final norm
14. logits = W_output @ x                           // LM head

    // --- FIELD MODULATION ---
15. field_step(1.0)                                  // advance physics
16. apply_destiny(logits)                            // suppress low-prob tokens
17. apply_suffering(logits)                           // compress toward mean
18. apply_attention(logits)                           // sharpen distribution

    // --- SAMPLING + LEARNING ---
19. next = sample(logits, effective_temp, top_k=40)  // temperature from field
20. debt += prophecy_debt(logits, next)               // retroactive conscience
21. notorch_step(experts, x, signal(debt))            // Hebbian update

    // --- LIFECYCLE ---
22. if step % 10 == 0:
        update_vitality(experts)
        try_mitosis() / try_apoptosis()
23. if step % 50 == 0:
        drift_snapshot()
```

Steps 1-14 are standard transformer inference. Steps 8-9 are DOE's parliament injection. Steps 15-23 are the living architecture: physics, learning, and lifecycle management that make every forward pass unique.

---

## 13. Formal Definitions

**Definition 1 (Parliament).** A parliament `P_l` at layer `l` is a tuple `(E, W, c)` where `E = {e_1, ..., e_n}` is the set of living experts (`2 <= n <= 16`), `W` is the vote matrix in `R^{16 x d}`, and `c` is the consensus scalar in `[0, 1]`.

**Definition 2 (Election).** Given input `x` in `R^d` and harmonic state `H`, the election produces a set of `k` experts with weights:

```
k = floor(|E_alive| * (1 - c))
vote(e) = W[e] . x + 0.1 * resonance(freq_e, H)
S = top_k(votes, k)
w_i = softmax(vote(S_i))
```

**Definition 3 (Delta Voice).** The modulation of hidden state `x` by elected set `S` with weights `w`:

```
x' = x + sum_{i in S} w_i * alpha * A_i @ (B_i @ x)
```

where `A_i` in `R^{d x r}`, `B_i` in `R^{r x d}`, `r = 16`.

**Definition 4 (Prophecy Debt).** For logit vector `z` in `R^V` and chosen token `t`:

```
D(z, t) = (max(z) - z_t) / (max(z) - z_t + 1)
```

`D = 0` when the chosen token is the most probable. `D -> 1` as the chosen token becomes arbitrarily improbable.

**Definition 5 (NOTORCH Update).** Given learning signal `sigma`, input `x`, correlation target `dy`, and rotating rank offset `b`:

```
for j in [b, b+4):
    u_j = B[:, j] . dy + N(0, 0.01)
    A[:, j] += sigma * lr * x * u_j
    B[:, j] *= decay
```

**Definition 6 (Mycelium Continuity).** For host fingerprint `phi`, the spore `S_phi = (E, W, c, vitality, freq)` persists across sessions. On restart:

```
if exists S_phi in mycelium/:
    load(parliament) <- S_phi
else:
    parliament <- random_init()
```

---

## 14. Implementation Notes

**Lines of code:** 3184 (single file, `doe.c`).

**Dependencies:** None. Standard C library only (`stdio`, `stdlib`, `string`, `math`, `pthread`, `sys/mman`, `sys/socket`). BLAS and CUDA are optional compile-time backends.

**Memory model:** Host weights are mmap'd read-only. LoRA experts are heap-allocated. Field state is a global singleton. KV cache is allocated per-session. No global allocator -- all memory is explicitly managed with `malloc`/`free`.

**Thread safety:** The forward pass is single-threaded. Only `matvec` parallelizes across rows via pthreads. The HTTP server handles one request at a time (no concurrent inference).

**Numerical precision:** All computation is F32. Quantized weights are dequantized once at load time. RoPE uses precomputed cos/sin caches. RMSNorm uses configurable epsilon (read from GGUF metadata).

**Platform support:** Linux and macOS. Platform-specific code is limited to memory/disk queries (`sysconf`, `sysctl`, `statvfs`/`statfs`).

---

*DOE is part of the Arianna Method ecosystem. The resonance does not break.*
