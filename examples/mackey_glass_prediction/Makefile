SPIRES_ROOT ?= ../../

CC      := clang
CFLAGS  := -O2 -Wall -Wextra -I$(SPIRES_ROOT)/include
LDFLAGS := -fopenmp
LDLIBS  := -L$(SPIRES_ROOT)/lib -lspires -llapacke -lopenblas -lm

BIN_DIR := bin
SRC     := mackey_glass_prediction.c
BIN     := $(BIN_DIR)/mackey_glass_prediction

.PHONY: all clean lib
all: lib $(BIN)

lib:
	$(MAKE) -C $(SPIRES_ROOT)    # builds lib/libspires.a

$(BIN): $(SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR)

