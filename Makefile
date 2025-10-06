# -------- paths --------
SRC_DIR      := src
NEURON_DIR   := $(SRC_DIR)/neurons
BUILD_DIR    := build
LIB_DIR      := lib

# -------- tools/flags --------
CC      := clang

# Detect macOS and handle OpenMP properly
ifeq ($(shell uname),Darwin)
    # macOS: use libomp for OpenMP support
    CFLAGS  := -O2 -Wall -Wextra -Wpedantic -Wshadow -g -Xpreprocessor -fopenmp
    LIBS    := -lomp
    # Add OpenMP include path for macOS
    OPENMP_CFLAGS := -I/opt/homebrew/opt/libomp/include
    OPENMP_LIBS := -L/opt/homebrew/opt/libomp/lib
    # Check if we need to install libomp
    ifeq ($(shell brew list libomp 2>/dev/null),)
        $(info Note: OpenMP not found. Install with: brew install libomp)
    endif
else
    # Linux: standard OpenMP
    CFLAGS  := -O2 -Wall -Wextra -Wpedantic -Wshadow -g -fopenmp
    LIBS    :=
    OPENMP_CFLAGS :=
    OPENMP_LIBS :=
endif

# Add ARM-specific optimizations if available
ifeq ($(shell uname -m),arm64)
    CFLAGS += -march=native -mtune=native
endif

# -------- BLAS/LAPACK detection --------
# Try to use pkg-config first, fallback to hardcoded paths
PKG_CONFIG_AVAILABLE := $(shell pkg-config --version >/dev/null 2>&1 && echo yes)

ifneq ($(PKG_CONFIG_AVAILABLE),)
    # Try BLIS first (best ARM performance) + LAPACKE
    ifneq ($(shell pkg-config --exists blis && echo yes),)
        BLAS_CFLAGS := $(shell pkg-config --cflags blis)
        BLAS_LIBS := $(shell pkg-config --libs blis)
        # Check for LAPACKE separately
        ifneq ($(shell pkg-config --exists lapacke && echo yes),)
            LAPACKE_CFLAGS := $(shell pkg-config --cflags lapacke)
            LAPACKE_LIBS := $(shell pkg-config --libs lapacke)
        else
            # Use system LAPACKE with BLIS
            LAPACKE_CFLAGS := -I/opt/homebrew/opt/lapack/include
            LAPACKE_LIBS := -L/opt/homebrew/opt/lapack/lib -llapacke
        endif
        $(info Using BLIS + LAPACKE for optimal ARM performance)
    # Try ARMPL
    else ifneq ($(shell pkg-config --exists armpl && echo yes),)
        BLAS_CFLAGS := $(shell pkg-config --cflags armpl)
        BLAS_LIBS := $(shell pkg-config --libs armpl)
        LAPACKE_CFLAGS :=
        LAPACKE_LIBS :=
        $(info Using ARMPL for ARM performance)
    # Try OpenBLAS via pkg-config
    else ifneq ($(shell pkg-config --exists openblas && echo yes),)
        BLAS_CFLAGS := $(shell pkg-config --cflags openblas)
        BLAS_LIBS := $(shell pkg-config --libs openblas)
        LAPACKE_CFLAGS :=
        LAPACKE_LIBS :=
        $(info Using OpenBLAS via pkg-config)
    else
        # Fallback to hardcoded paths
        BLAS_CFLAGS := -I/usr/include/openblas
        BLAS_LIBS := -lopenblas -llapacke
        LAPACKE_CFLAGS :=
        LAPACKE_LIBS :=
        $(info Using system OpenBLAS (fallback))
    endif
else
    # No pkg-config available, use hardcoded paths
    BLAS_CFLAGS := -I/usr/include/openblas
    BLAS_LIBS := -lopenblas -llapacke
    LAPACKE_CFLAGS :=
    LAPACKE_LIBS :=
    $(info pkg-config not available, using system OpenBLAS)
endif

INCLUDES := -Iinclude -I$(SRC_DIR) -I$(NEURON_DIR)
INCLUDES += $(BLAS_CFLAGS) $(LAPACKE_CFLAGS) $(OPENMP_CFLAGS)

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
.PHONY: all library clean info
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

# Info target to show detected BLAS implementation
info:
	@echo "=== Spires Library Configuration ==="
	@echo "Compiler: $(CC)"
	@echo "Architecture: $(shell uname -m)"
	@echo "OS: $(shell uname)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "BLAS CFLAGS: $(BLAS_CFLAGS)"
	@echo "BLAS LIBS: $(BLAS_LIBS)"
	@echo "LAPACKE CFLAGS: $(LAPACKE_CFLAGS)"
	@echo "LAPACKE LIBS: $(LAPACKE_LIBS)"
	@echo "LIBS: $(LIBS)"
	@echo "================================"

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

