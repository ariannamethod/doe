CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS  = -lm -lpthread

# ── aarch64: turn on the SIMD the packed matvecs are already written against ──
# The int8 row kernels are guarded on __ARM_FEATURE_DOTPROD / __ARM_FEATURE_MATMUL_INT8,
# and clang defines those only when the target actually carries the instructions. A stock
# aarch64 target is plain armv8-a, so on a phone the guarded kernels compile out and the
# scalar fallback runs instead. Kept in step with notorch, which vendors the same probe.
#
# Ask the compiler before probing the CPU. Some arm64 toolchains — Apple's among them —
# already target a core that carries these, and there the right number of flags to add is
# zero. Only when the default target lacks them is the host worth reading, and the host is
# read two ways because the two platforms expose it differently: /proc/cpuinfo on Linux
# and Termux, sysctl on Darwin.
#
# The probe reads the BUILD host, so it is right for native builds and wrong for
# cross-compilation — pass ARM_FLAGS= by hand there, or ARM_SIMD=0 to skip it entirely.
# No -mcpu/-mtune on purpose: tuning for one core type costs ~10% on the other, and on
# big.LITTLE the same binary runs on both.
ARM_SIMD ?= 1
ifneq ($(filter aarch64 arm64,$(shell uname -m)),)
  ifeq ($(ARM_SIMD), 1)
    ARM_HAS_DOT := $(shell echo | $(CC) -dM -E - 2>/dev/null | grep -c __ARM_FEATURE_DOTPROD)
    ifeq ($(ARM_HAS_DOT),0)
      ARM_MARCH := $(shell f=""; \
        { grep -qwm1 asimddp /proc/cpuinfo 2>/dev/null || \
          [ "$$(sysctl -n hw.optional.arm.FEAT_DotProd 2>/dev/null)" = 1 ]; } && f="$$f+dotprod"; \
        { grep -qwm1 i8mm /proc/cpuinfo 2>/dev/null || \
          [ "$$(sysctl -n hw.optional.arm.FEAT_I8MM 2>/dev/null)" = 1 ]; } && f="$$f+i8mm"; \
        [ -n "$$f" ] && echo "-march=armv8.2-a$$f")
      # Only keep it if this compiler actually accepts the arch string.
      ARM_FLAGS ?= $(shell [ -n "$(ARM_MARCH)" ] && \
        $(CC) $(ARM_MARCH) -E -x c /dev/null >/dev/null 2>&1 && echo "$(ARM_MARCH)")
      CFLAGS += $(ARM_FLAGS)
      ARM_NAME = $(if $(ARM_FLAGS),$(ARM_FLAGS),baseline armv8-a)
    else
      ARM_NAME = compiler default (already carries SDOT)
    endif
  endif
endif

.PHONY: all metal test clean run run-int8 run-smollm360 run-f16 quantize-q4

all: doe_field

doe_field: doe.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

# macOS Accelerate
blas: doe.c gguf.c
	$(CC) $(CFLAGS) -DUSE_BLAS -DACCELERATE doe.c gguf.c $(LDFLAGS) -framework Accelerate -o doe_field

# Apple Metal — Q4_K matvec on GPU via notorch_metal (resident weights, zero-copy);
# Q6_K/F32/F16 stay on CPU. Runs 24B-class Q4_K on a 24GB Mac. Obj-C++ needs -std=c++17.
MM_FLAGS = -O2 -Wall -Wextra -std=c++17 -I.
notorch_metal.o: notorch_metal.mm notorch_metal.h
	clang++ $(MM_FLAGS) -DUSE_METAL -fobjc-arc -c notorch_metal.mm -o notorch_metal.o
metal: doe.c gguf.c notorch_metal.o notorch_metal.h
	$(CC) $(CFLAGS) -DUSE_BLAS -DACCELERATE -DUSE_METAL doe.c gguf.c notorch_metal.o $(LDFLAGS) -framework Accelerate -framework Metal -framework Foundation -lc++ -o doe_field

# OpenBLAS (Linux)
openblas: doe.c gguf.c
	$(CC) $(CFLAGS) -DUSE_BLAS doe.c gguf.c $(LDFLAGS) -lopenblas -o doe_field

# cuBLAS (NVIDIA GPU)
cuda: doe.c
	$(CC) $(CFLAGS) -DUSE_CUBLAS $< $(LDFLAGS) -lcublas -lcudart -o doe_field

test: tests/test_doe.c doe.c
	$(CC) $(CFLAGS) tests/test_doe.c $(LDFLAGS) -o tests/test_doe
	./tests/test_doe

# --- personality weights (HF ataeff/janus) + Q4_0 quantization ---
HF_BASE = https://huggingface.co/ataeff/janus/resolve/main/DoE
WEIGHTS = weights

$(WEIGHTS)/doe_smollm360_lora_1000.gguf:
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/doe_smollm360_lora_1000.gguf

$(WEIGHTS)/doe_qwen15b_lora_1000.gguf:
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/doe_qwen15b_lora_1000.gguf

# Q4_0 — ~3.2x smaller, the parliament survives the quant, and it lights up the int8
# fast path. llama.cpp is used only as a GGUF converter here.
$(WEIGHTS)/doe_qwen15b_q4_0.gguf: $(WEIGHTS)/doe_qwen15b_lora_1000.gguf
	llama-quantize $< $@ Q4_0

quantize-q4: $(WEIGHTS)/doe_qwen15b_q4_0.gguf

# default run: Qwen2.5-1.5B Q4_0 — small, fits on 8 GB, parliament intact
run: doe_field $(WEIGHTS)/doe_qwen15b_q4_0.gguf
	./doe_field --model $(WEIGHTS)/doe_qwen15b_q4_0.gguf

# Q4_0 + int8 dynamic-activation-quant fast path (NEON SDOT, approximate)
run-int8: doe_field $(WEIGHTS)/doe_qwen15b_q4_0.gguf
	DOE_INT8=1 ./doe_field --model $(WEIGHTS)/doe_qwen15b_q4_0.gguf

run-smollm360: doe_field $(WEIGHTS)/doe_smollm360_lora_1000.gguf
	./doe_field --model $(WEIGHTS)/doe_smollm360_lora_1000.gguf

run-f16: doe_field $(WEIGHTS)/doe_qwen15b_lora_1000.gguf
	./doe_field --model $(WEIGHTS)/doe_qwen15b_lora_1000.gguf

clean:
	rm -f doe_field tests/test_doe
