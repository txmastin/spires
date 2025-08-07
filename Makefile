# Paths
SRC_DIR := src
NEURON_DIR := $(SRC_DIR)/neurons
TEST_DIR := tests
BUILD_DIR := build
BIN_DIR := bin

# Tools
CC := clang
CFLAGS := -O2 -Wall -Wextra -Wpedantic -Wshadow -g
LDFLAGS := -lm
LDLIBS := -lopenblas

INCLUDES := -I$(SRC_DIR) -I$(NEURON_DIR) -I/usr/include/openblas

# Executable target
TARGET := $(BIN_DIR)/run_simulation

# Source files
SRC := $(wildcard $(SRC_DIR)/*.c) $(wildcard $(NEURON_DIR)/*.c)
TEST_SRC := $(wildcard $(TEST_DIR)/*.c)

# Object files
SRC_OBJ := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o, $(filter $(SRC_DIR)/%.c,$(SRC))) $(patsubst $(NEURON_DIR)/%.c,$(BUILD_DIR)/neurons/%.o, $(filter $(NEURON_DIR)/%.c,$(SRC)))

# All source objects except main.o (to use for tests linking)
SRC_OBJ_NO_MAIN := $(filter-out $(BUILD_DIR)/main.o, $(SRC_OBJ))

TEST_OBJ := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/tests/%.o,$(TEST_SRC))

# Final test executables (each test becomes its own binary)
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BIN_DIR)/%,$(TEST_SRC))

# Default target
all: $(TARGET) tests

# Build just src
src: $(TARGET)

# Link main binary
$(TARGET): $(SRC_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Link each test binary
$(BIN_DIR)/%: $(BUILD_DIR)/tests/%.o $(SRC_OBJ_NO_MAIN)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Compile rules
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/neurons/%.o: $(NEURON_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Build all test binaries
tests: $(TEST_BINS)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Debug helper
print-%: ; @echo $* = $($*)
