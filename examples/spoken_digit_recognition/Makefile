# --- paths (override on command line if needed) ---
SPIRES_DIR ?= ../../

# --- toolchain ---
CC      ?= clang
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic -I$(SPIRES_DIR)/include -fopenmp
LDFLAGS ?= -L$(SPIRES_DIR)/lib
LDLIBS  ?= -lspires -lopenblas -llapacke -lm -lomp

# --- targets ---
SRC    := spoken_digit_recognition.c
BIN    := bin
TARGET := $(BIN)/spoken_digit_recognition

.PHONY: all clean libspires

all: $(TARGET)

# build the example (ensure lib is built first)
$(TARGET): $(SRC) | $(BIN) libspires
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

# build the library
libspires:
	$(MAKE) -C $(SPIRES_DIR)

# output dir
$(BIN):
	mkdir -p $@

clean:
	rm -rf $(BIN)

