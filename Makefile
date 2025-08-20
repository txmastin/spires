# -------- paths --------
SRC_DIR      := src
NEURON_DIR   := $(SRC_DIR)/neurons
BUILD_DIR    := build
LIB_DIR      := lib

# -------- tools/flags --------
CC      := clang
CFLAGS  := -O2 -Wall -Wextra -Wpedantic -Wshadow -g -fopenmp
INCLUDES := -Iinclude -I$(SRC_DIR) -I$(NEURON_DIR)
INCLUDES += -I/usr/include/openblas

# (Portable option: pkg-config; uncomment if available)
# LAPACKE_CFLAGS := $(shell pkg-config --cflags lapacke 2>/dev/null)
# OPENBLAS_CFLAGS:= $(shell pkg-config --cflags openblas 2>/dev/null)
# INCLUDES += $(LAPACKE_CFLAGS) $(OPENBLAS_CFLAGS)

# -------- sources/objects --------
SRCS := \
  $(SRC_DIR)/math_utils.c \
  $(SRC_DIR)/neuron.c     \
  $(SRC_DIR)/reservoir.c  \
  $(SRC_DIR)/spires_api.c \
  $(SRC_DIR)/agile.c \
  $(SRC_DIR)/spires_opt_agile.c \
  $(wildcard $(NEURON_DIR)/*.c)

OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

STATIC_LIB := $(LIB_DIR)/libspires.a

# -------- default targets --------
.PHONY: all library clean
all: library
library: $(STATIC_LIB)

# Archive rule — ensure lib/ exists first (order-only prerequisite)
$(STATIC_LIB): $(OBJS) | $(LIB_DIR)
	@echo "  AR    $@"
	ar rcs $@ $(OBJS)

# Compile C -> object (handles nested dirs; creates build subdirs)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Directory creators (real file targets, not phony)
$(BUILD_DIR):
	mkdir -p $@
$(LIB_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

