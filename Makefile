CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS  = -lm -lpthread

.PHONY: all test clean

all: doe_field

doe_field: doe.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

# macOS Accelerate
blas: doe.c
	$(CC) $(CFLAGS) -DUSE_BLAS -DACCELERATE $< $(LDFLAGS) -framework Accelerate -o doe_field

# OpenBLAS (Linux)
openblas: doe.c
	$(CC) $(CFLAGS) -DUSE_BLAS $< $(LDFLAGS) -lopenblas -o doe_field

# cuBLAS (NVIDIA GPU)
cuda: doe.c
	$(CC) $(CFLAGS) -DUSE_CUBLAS $< $(LDFLAGS) -lcublas -lcudart -o doe_field

test: tests/test_doe.c doe.c
	$(CC) $(CFLAGS) tests/test_doe.c $(LDFLAGS) -o tests/test_doe
	./tests/test_doe

# --- Personality weights from HuggingFace ---
HF_REPO  = ataeff/janus
HF_BASE  = https://huggingface.co/$(HF_REPO)/resolve/main/DoE
WEIGHTS  = weights

.PHONY: weights smollm360 qwen15b run-smollm360 run-qwen15b serve-smollm360 serve-qwen15b

weights: $(WEIGHTS)/doe_smollm360_lora_1000.gguf $(WEIGHTS)/doe_qwen15b_lora_1000.gguf

$(WEIGHTS)/doe_smollm360_lora_1000.gguf:
	@mkdir -p $(WEIGHTS)
	@echo "Downloading SmolLM2-360M DOE personality (692MB)..."
	curl -L -o $@ $(HF_BASE)/doe_smollm360_lora_1000.gguf

$(WEIGHTS)/doe_qwen15b_lora_1000.gguf:
	@mkdir -p $(WEIGHTS)
	@echo "Downloading Qwen2.5-1.5B DOE personality (3.4GB)..."
	curl -L -o $@ $(HF_BASE)/doe_qwen15b_lora_1000.gguf

smollm360: $(WEIGHTS)/doe_smollm360_lora_1000.gguf
qwen15b: $(WEIGHTS)/doe_qwen15b_lora_1000.gguf

run-smollm360: doe_field $(WEIGHTS)/doe_smollm360_lora_1000.gguf
	./doe_field --model $(WEIGHTS)/doe_smollm360_lora_1000.gguf

run-qwen15b: doe_field $(WEIGHTS)/doe_qwen15b_lora_1000.gguf
	./doe_field --model $(WEIGHTS)/doe_qwen15b_lora_1000.gguf

serve-smollm360: doe_field $(WEIGHTS)/doe_smollm360_lora_1000.gguf
	./doe_field --model $(WEIGHTS)/doe_smollm360_lora_1000.gguf --serve 8080

serve-qwen15b: doe_field $(WEIGHTS)/doe_qwen15b_lora_1000.gguf
	./doe_field --model $(WEIGHTS)/doe_qwen15b_lora_1000.gguf --serve 8080

clean:
	rm -f doe_field tests/test_doe
