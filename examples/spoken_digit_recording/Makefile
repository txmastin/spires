# --- paths (override on command line if needed) ---
SPIRES_DIR ?= ../../

# --- toolchain ---
CC      ?= clang
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic -I$(SPIRES_DIR)/include -I$(SPIRES_DIR)/src -fopenmp
LDFLAGS ?= -L$(SPIRES_DIR)/lib
LDLIBS  ?= -lspires -lopenblas -llapacke -lm -lomp

# --- targets ---
SRC    := spoken_digit_recording.c
BIN    := bin
TARGET := $(BIN)/spoken_digit_recording

.PHONY: all clean libspires

all: $(TARGET)

$(TARGET): $(SRC) | $(BIN) libspires
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

libspires:
	$(MAKE) -C $(SPIRES_DIR)

$(BIN):
	mkdir -p $@

clean:
	rm -rf $(BIN)
