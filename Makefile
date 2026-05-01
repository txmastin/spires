# -------- paths --------
SRC_DIR      := src
NEURON_DIR   := $(SRC_DIR)/neurons
BUILD_DIR    := build
LIB_DIR      := lib

# -------- tools/flags based on operating system --------
UNAME := $(shell uname -s)
ifeq ($(OS), Windows_NT)
    CC        := gcc
    OMP_FLAGS := -fopenmp
else ifeq ($(UNAME), Darwin) 
    # Mac
    LLVM_PREFIX := $(shell brew --prefix llvm)
    CC          := $(LLVM_PREFIX)/bin/clang
    OMP_FLAGS   := -fopenmp
else
    # Linux
    CC        := gcc
    OMP_FLAGS := -fopenmp
endif

CFLAGS   := -O2 -Wall -Wextra -Wpedantic -Wshadow -g $(OMP_FLAGS)
INCLUDES := -Iinclude -I$(SRC_DIR) -I$(NEURON_DIR)

ifeq ($(OS), Windows_NT)
    INCLUDES += -I/usr/include/openblas
else ifeq ($(UNAME), Darwin)
    INCLUDES += -I$(shell brew --prefix openblas)/include
else
    INCLUDES += -I/usr/include/openblas
endif

# -------- OLD tools/flags --------
# CC      := clang
# CFLAGS  := -O2 -Wall -Wextra -Wpedantic -Wshadow -g -fopenmp
# INCLUDES := -Iinclude -I$(SRC_DIR) -I$(NEURON_DIR)
# INCLUDES += -I/usr/include/openblas

# (Portable option: pkg-config; uncomment if available)
# LAPACKE_CFLAGS := $(shell pkg-config --cflags lapacke 2>/dev/null)
# OPENBLAS_CFLAGS:= $(shell pkg-config --cflags openblas 2>/dev/null)
# INCLUDES += $(LAPACKE_CFLAGS) $(OPENBLAS_CFLAGS)


#--------- CUDA toggle ------------
USE_CUDA ?= 0

ifeq ($(USE_CUDA),1)
    NVCC      := nvcc
    NVCCFLAGS := -O2 -g -Xcompiler "-Wall -Wextra"
    CFLAGS    += -DUSE_CUDA

    CU_SRCS := $(SRC_DIR)/math_utils.cu
    CU_OBJS := $(BUILD_DIR)/math_utils_cu.o

    C_SRCS := \
      $(SRC_DIR)/neuron.c           \
      $(SRC_DIR)/reservoir.c        \
      $(SRC_DIR)/spires_api.c       \
      $(SRC_DIR)/agile.c            \
      $(SRC_DIR)/spires_opt_agile.c \
      $(wildcard $(NEURON_DIR)/*.c)

else
    CU_OBJS :=

    C_SRCS := \
      $(SRC_DIR)/math_utils.c       \
      $(SRC_DIR)/neuron.c           \
      $(SRC_DIR)/reservoir.c        \
      $(SRC_DIR)/spires_api.c       \
      $(SRC_DIR)/agile.c            \
      $(SRC_DIR)/spires_opt_agile.c \
      $(wildcard $(NEURON_DIR)/*.c)

endif

OBJS := $(C_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o) $(CU_OBJS)

STATIC_LIB := $(LIB_DIR)/libspires.a

.PHONY: all library clean
all: library
library: $(STATIC_LIB)

$(STATIC_LIB): $(OBJS) | $(LIB_DIR)
	@echo "  AR    $@"
	ar rcs $@ $(OBJS)

# -------- compile rules --------
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/math_utils_cu.o: $(SRC_DIR)/math_utils.cu | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@
$(LIB_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

