# Paths
SRC_DIR := src
NEURON_DIR := $(SRC_DIR)/neurons
TEST_DIR := tests
EXAMPLE_DIR := examples
BUILD_DIR := build
BIN_DIR := bin
LIB_DIR := lib

# Tools
CC := clang
CFLAGS := -O2 -Wall -Wextra -Wpedantic -Wshadow -g -fopenmp
LDFLAGS := -lm -fopenmp
LDLIBS := -lopenblas
INCLUDES := -I$(SRC_DIR) -I$(NEURON_DIR) -I/usr/include/openblas

# --- Library ---
# Find all .c files for the library, EXCLUDING main.c
LIB_SRC := $(filter-out $(SRC_DIR)/main.c, $(wildcard $(SRC_DIR)/*.c) $(wildcard $(NEURON_DIR)/*.c))
# Create a list of corresponding object files for the library
LIB_OBJ := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o, $(filter $(SRC_DIR)/%.c,$(LIB_SRC))) \
           $(patsubst $(NEURON_DIR)/%.c,$(BUILD_DIR)/neurons/%.o, $(filter $(NEURON_DIR)/%.c,$(LIB_SRC)))
# The final static library file
STATIC_LIB := $(LIB_DIR)/libspires.a

# --- Main Executable (if main.c exists) ---
MAIN_SRC := $(wildcard $(SRC_DIR)/main.c)
TARGET := $(patsubst $(SRC_DIR)/main.c,$(BIN_DIR)/run_simulation,$(MAIN_SRC))

# --- Tests & Examples ---
TEST_SRC := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BIN_DIR)/%,$(TEST_SRC))
EXAMPLE_SRC := $(wildcard $(EXAMPLE_DIR)/*.c)
EXAMPLE_BINS := $(patsubst $(EXAMPLE_DIR)/%.c,$(BIN_DIR)/%,$(EXAMPLE_SRC))

# Default target: build the library, the main executable, and the tests.
# Examples are NOT built by default.
all: $(STATIC_LIB) $(TARGET) tests

# --- Build Rules ---

# 1. Rule to build the static library
$(STATIC_LIB): $(LIB_OBJ)
	@mkdir -p $(LIB_DIR)
	@echo "Creating static library $@"
	ar rcs $@ $^

# 2. Rule to build the main run_simulation executable
$(TARGET): $(SRC_DIR)/main.c $(STATIC_LIB)
	@mkdir -p $(BIN_DIR)
	@echo "Building main executable $@"
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ -L$(LIB_DIR) -lspires $(LDFLAGS) $(LDLIBS)

# 3. Rule to build any test executable
$(TEST_BINS): $(BIN_DIR)/%: $(TEST_DIR)/%.c $(STATIC_LIB)
	@mkdir -p $(BIN_DIR)
	@echo "Building test $@"
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ -L$(LIB_DIR) -lspires $(LDFLAGS) $(LDLIBS)

# 4. Rule to build any example executable
$(EXAMPLE_BINS): $(BIN_DIR)/%: $(EXAMPLE_DIR)/%.c $(STATIC_LIB)
	@mkdir -p $(BIN_DIR)
	@echo "Building example $@"
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ -L$(LIB_DIR) -lspires $(LDFLAGS) $(LDLIBS)

# --- Compilation Rules for Object Files ---
# These rules compile the .c files into the .o files needed for the library
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/neurons/%.o: $(NEURON_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# --- Targets for building groups ---
lib: $(STATIC_LIB)

tests: $(TEST_BINS)

examples: $(EXAMPLE_BINS)

# --- Cleanup ---
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(LIB_DIR)

# Debug helper
print-%: ; @echo $* = $($*)

