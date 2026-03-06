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

clean:
	rm -f doe_field tests/test_doe
