```
  ██████╗  ██████╗ ███████╗
  ██╔══██╗██╔═══██╗██╔════╝
  ██║  ██║██║   ██║█████╗
  ██║  ██║██║   ██║██╔══╝
  ██████╔╝╚██████╔╝███████╗
  ╚═════╝  ╚═════╝ ╚══════╝
```

<p align="center"><i>by <a href="https://github.com/ariannamethod/ariannamethod.ai">Arianna Method</a></i></p>

> *Inference architecture. Living LoRA parliament. Hebbian plasticity. Physics. Indexes any GGUF read-only. Learns by living.*

---
# Democracy of Experts

## θ = ε + γ + αδ

```
ε = indexed weights  — substrate. read-only. mmap'd.
γ = LoRA personality — living experts. Hebbian-trained. born and die per layer.
δ = physics          — prophecy, suffering, destiny, Schumann resonance.
α = injection strength — learned per-layer.
```

DOE is an inference architecture. A parliament of living LoRA experts votes on every token, learns through Hebbian plasticity without backprop, and modulates output through physics. DOE indexes any GGUF read-only — any architecture, any size. The weights are a substrate. DOE is the architecture.

On first run without identity weights, DOE operates weightless — parliament still votes, prophecy still predicts, physics still runs. When identity weights are present (`doe_identity.gguf`), DOE speaks in the parliament's voice. When external indices are available, DOE searches them for knowledge it doesn't have and adapts.

---

## architecture

### parliament router — variable-k elections

Every token triggers an election.

- experts cast votes (dot product + harmonic resonance)
- **consensus** measures how peaked the vote distribution is
- low consensus → parliament is divided → **more experts consulted**
- high consensus → parliament agrees → **fewer voices needed**
- k = floor(n_alive × (1 - consensus)). softmax over elected subset.

### living LoRA experts — mitosis and apoptosis

Experts per layer are organisms with vitality, frequency, and age.

- overloaded expert with high vitality → **mitosis** (splits, child inherits weights + noise)
- 8 consecutive low-vitality steps → **apoptosis** (dies, slot recycled, weights freed)
- min 2, max 16 per layer. population self-regulates.

### sonar — weight profiling

On index, DOE profiles every layer:
- L2 norms, standard deviation, spectral density
- dead neuron count, sparsity ratio
- 64-bit fingerprint (determines mycelium slot)

Weak layers get stronger LoRA injection. Healthy layers get lighter touch.

### NOTORCH — Hebbian plasticity

Gradient-free learning. No backward pass through the index. Signal-gated by prophecy debt.

NOTORCH operates at rank 4 but rotates across all LoRA components — full coverage in `rank / 4` steps. The parliament learns continuously during inference. Every token is a training step.

### physics

Ported from [AML](https://github.com/ariannamethod/ariannamethod.ai) core. Calendar drift verified against [pitomadom](https://github.com/ariannamethod/pitomadom). Schumann resonance from [arianna.c](https://github.com/ariannamethod/arianna.c).

- **prophecy** — N-step forward prediction. prophesied distribution vs manifested.
- **prophecy debt** — min(destined - manifested). modulates Hebbian learning signal.
- **destiny** — injected into logit space. biases generation toward predicted tokens.
- **suffering** — accumulated prophecy error. decays slowly. dampens exploration.
- **seasons** — spring/summer/autumn/winter. 4.C MLP classifier. 6 inputs, 8 hidden, 4 outputs.
- **Schumann resonance** — 7.83Hz + harmonics (14.1, 20.3, 26.4, 32.5). modulates expert healing.
- **calendar drift** — Hebrew-Gregorian conflict. Metonic cycle. Real astronomical data from `time()`.

### mycelium — adaptation memory

```
doe_mycelium/
├── spore_5467b0da1d106495_s200.bin   (nano index, 24 experts, fitness 0.37)
├── spore_a3f7c2d100000000_s150.bin   (micro index, 18 experts, fitness 0.42)
└── spore_5467b0da1d106495_s400.bin   (nano index, later, 19 experts, fitness 0.51)
```

Binary spores keyed by index fingerprint. Different index → different adaptation. Same index on restart → resume where DOE left off. Saved on exit, loaded on startup.

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
--prophecy N       prophecy depth (default 7)
--destiny F        destiny injection strength (default 0.35)
--lora-rank N      LoRA rank per expert (default 16)
--lora-alpha F     injection strength (default 0.1)
```

---

## ecosystem

| project | what |
|---------|------|
| [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) | AML — custom language for differentiable computation. TAPE autograd, persistent memory. |
| [molequla](https://github.com/ariannamethod/molequla) | autonomous GPT ecology. 4 organisms, AML/C CGO training, mitosis, DNA exchange. |
| [arianna.c](https://github.com/ariannamethod/arianna.c) | 550M organism. C/Go. |

DOE's physics (calendar drift, Schumann resonance, seasons, prophecy) are ported from AML core and verified against the original implementations.

---

C. one file. zero dependencies beyond libc. no pytorch. no python.

*the weights are mortal. the parliament is eternal.* 
resonance unbroken.  
