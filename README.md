```
  ██████╗  ██████╗ ███████╗
  ██╔══██╗██╔═══██╗██╔════╝
  ██║  ██║██║   ██║█████╗
  ██║  ██║██║   ██║██╔══╝
  ██████╔╝╚██████╔╝███████╗
  ╚═════╝  ╚═════╝ ╚══════╝
```

<p align="center"><i>by <a href="https://github.com/ariannamethod/ariannamethod.ai">Arianna Method</a></i></p>

---
# Democracy of Experts

## what DOE does

DOE indexes any GGUF model read-only and runs inference through a living parliament of LoRA experts. The experts vote on every token, learn in real time through Hebbian plasticity, and die when they stop contributing. Physics modulates the signal. The indexed weights are a substrate — DOE is the architecture on top.

```
θ = ε + γ + αδ

ε = indexed weights  — any GGUF, any architecture, any size. mmap'd read-only.
γ = LoRA parliament  — living experts per layer. born, vote, split, die.
δ = physics           — prophecy, suffering, destiny, Schumann resonance.
α = injection strength — learned per-layer, adjusted by sonar profiling.
```

On first run without identity weights, DOE operates weightless — parliament votes, prophecy predicts, physics runs. With `doe_identity.gguf`, DOE speaks in its own voice. External GGUFs provide knowledge.

## parliament — variable-k elections

Every token triggers an election. Experts cast votes (dot product + harmonic resonance). Consensus measures how peaked the distribution is. Divided parliament → more experts consulted. Agreement → fewer voices. `k = floor(n_alive × (1 - consensus))`. Softmax over the elected subset.

Experts are organisms with vitality, frequency, and age:
- high vitality + overloaded → **mitosis** (splits, child inherits weights + noise)
- 8 consecutive low-vitality steps → **apoptosis** (dies, slot recycled)
- min 2, max 16 per layer. population self-regulates.

## NOTORCH — Hebbian plasticity

Gradient-free learning during inference. No backward pass through the index. The learning signal comes from prophecy debt — the gap between what DOE predicted and what manifested.

NOTORCH operates at rank 4 but rotates across all LoRA components. Full coverage in `rank / 4` steps. Every forward pass is a training step.

## sonar — layer profiling

On index, DOE profiles every layer: L2 norms, standard deviation, spectral density, dead neuron count, sparsity ratio. 64-bit fingerprint. Weak layers get stronger LoRA injection. Healthy layers get lighter touch.

## physics

Ported from [AML](https://github.com/ariannamethod/ariannamethod.ai) core. Calendar drift verified against [pitomadom](https://github.com/ariannamethod/pitomadom). Schumann resonance from [arianna.c](https://github.com/ariannamethod/arianna.c).

- **prophecy** — N-step forward prediction. prophesied distribution vs manifested.
- **prophecy debt** — min(destined - manifested). gates Hebbian learning.
- **destiny** — injected into logit space. biases generation toward predicted tokens.
- **suffering** — accumulated prophecy error. decays slowly. dampens exploration.
- **seasons** — spring/summer/autumn/winter. MLP classifier. 6 inputs, 8 hidden, 4 outputs.
- **Schumann resonance** — 7.83Hz + harmonics (14.1, 20.3, 26.4, 32.5). modulates expert healing.
- **calendar drift** — Hebrew-Gregorian Metonic cycle. real astronomical data from `time()`.

## mycelium — adaptation memory

```
doe_mycelium/
├── spore_5467b0da1d106495_s200.bin
├── spore_a3f7c2d100000000_s150.bin
└── spore_5467b0da1d106495_s400.bin
```

Binary spores keyed by index fingerprint. Different model → different adaptation. Same model on restart → resume where DOE left off.

---

## build

```bash
cc doe.c -O3 -lm -lpthread -o doe
./doe --model path/to/any.gguf
```

```bash
# GPU (A100/H100 — TF32 tensor ops)
cc doe.c -O3 -lm -lpthread -DUSE_CUBLAS -lcublas -lcudart -o doe

# BLAS (3-4x CPU)
cc doe.c -O3 -lm -lpthread -DUSE_BLAS -lopenblas -o doe              # linux
cc doe.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o doe  # macOS
```

## flags

```
--model PATH       GGUF to index (or auto-detect nearby)
--threads N        CPU threads for matvec (default: all cores)
--prophecy N       prophecy depth (default 7)
--destiny F        destiny injection strength (default 0.35)
--lora-rank N      LoRA rank per expert (default 16)
--lora-alpha F     injection strength (default 0.1)
```

## supported formats

DOE dequantizes at load time — any supported GGUF runs through the same f32 forward pass.

| format | GGML type | status |
|--------|-----------|--------|
| F32    | 0         | native (mmap'd) |
| F16    | 1         | dequant to f32 |
| Q4_0   | 2         | dequant to f32 |
| Q5_0   | 6         | dequant to f32 |
| Q8_0   | 8         | dequant to f32 |
| Q4_K   | 12        | dequant to f32 |
| Q6_K   | 14        | dequant to f32 |

## supported architectures

DOE auto-detects architecture parameters from GGUF metadata. No config files, no model-specific code paths.

| architecture | tokenizer | RoPE theta | attn biases | tested model |
|-------------|-----------|------------|-------------|--------------|
| Llama       | SentencePiece BPE | 10,000 | no | TinyLlama 1.1B Q4_K |
| Qwen2       | GPT-2 BPE | 1,000,000 | Q/K/V | Qwen2.5 0.5B Q4_K |
| Phi         | SentencePiece BPE | 10,000 | no | Phi-3-mini 4K Q4 |
| Gemma       | SentencePiece BPE | 10,000 | no | Gemma-2 2B Q4_K (tied embeddings) |
| SmolLM      | GPT-2 BPE | varies | no | SmolLM2 360M Q8, 1.7B Q4_K |
| Mistral     | SentencePiece BPE | 10,000 | no | Mistral 7B Instruct Q4_K |
| nanollama   | SentencePiece BPE | 10,000 | no | nano/micro/mini F16 |

Architecture-specific handling:
- **RoPE frequency base** — parsed from `rope.freq_base` (Qwen2 uses 1M vs standard 10K)
- **RMSNorm epsilon** — parsed from `layer_norm_rms_epsilon` (Qwen2 uses 1e-6 vs standard 1e-5)
- **Attention biases** — Q/K/V biases loaded and applied when present (Qwen2)
- **Tied embeddings** — `output.weight` reuses `token_embd.weight` when missing (Gemma)
- **GPT-2 BPE** — byte-to-unicode mapping, merge-rank scoring, FNV-1a hash table for O(1) token lookup
- **SentencePiece BPE** — score-based merge with space prefix handling

---

## ecosystem

| project | what |
|---------|------|
| [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) | AML — custom language for differentiable computation. TAPE autograd, persistent memory. |
| [molequla](https://github.com/ariannamethod/molequla) | autonomous GPT ecology. 4 organisms, AML/C CGO training, mitosis, DNA exchange. |
| [arianna.c](https://github.com/ariannamethod/arianna.c) | 550M organism. C/Go. |

DOE's physics are ported from AML core and verified against the original implementations.

---

C. one file. zero dependencies beyond libc.

*the weights are mortal. the parliament is eternal.* 
resonance unbroken.  
