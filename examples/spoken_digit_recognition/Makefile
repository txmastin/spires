SPIRES_ROOT ?= ../../

UNAME := $(shell uname -s)
ifeq ($(OS), Windows_NT)
    CC        := gcc
    OMP_FLAGS := -fopenmp
    OMP_LIBS  := -lgomp
    LDLIBS    := -L$(SPIRES_ROOT)/lib -lspires -llapacke -lopenblas -lm $(OMP_LIBS)
else ifeq ($(UNAME), Darwin)
    LLVM_PREFIX := $(shell brew --prefix llvm)
    CC          := $(LLVM_PREFIX)/bin/clang
    OMP_FLAGS   := -fopenmp
    OMP_LIBS    := -L$(shell brew --prefix libomp)/lib -lomp
    LDLIBS := -L$(SPIRES_ROOT)/lib -lspires -llapacke -lopenblas -lm $(OMP_LIBS) \
              -L/opt/homebrew/opt/lapack/lib \
              -L/opt/homebrew/opt/openblas/lib
else
    # Linux
    CC        := gcc
    OMP_FLAGS := -fopenmp
    OMP_LIBS  := -lgomp
    LDLIBS := -L$(SPIRES_ROOT)/lib -lspires -llapacke -lopenblas -lm $(OMP_LIBS)
endif

CFLAGS  := -O2 -Wall -Wextra -I$(SPIRES_ROOT)/include
LDFLAGS := $(OMP_FLAGS)

# -------- CUDA toggle --------
USE_CUDA ?= 0
ifeq ($(USE_CUDA), 1)
    CFLAGS  += -DUSE_CUDA
    LDLIBS  += -lcudart -lcublas
endif

BIN_DIR := bin
SRC     := spoken_digit_recognition.c
BIN     := $(BIN_DIR)/spoken_digit_recognition

.PHONY: all clean lib
all: lib $(BIN)

lib:
	$(MAKE) -C $(SPIRES_ROOT) USE_CUDA=$(USE_CUDA)   # propagate flag to root Makefile

$(BIN): $(SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR)