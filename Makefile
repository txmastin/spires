# SPIRES build
#
#   make                                   host build (default profile)
#   make PROFILE=embedded                  cross build, see mk/embedded.mk
#   make CROSS_COMPILE=arm-none-eabi-      toolchain prefix, kernel convention
#   make V=1                               echo the commands
#   make install PREFIX=/usr/local         staged with DESTDIR=
#   make help                              every knob, with its current value
#
# Everything below uses ?= so the environment and the command line win.
# GNU make has no `--embedded`-style user flags: `--` is reserved for make's
# own options, so build variants are selected by variable, not by option.

PROFILE       ?= host
CROSS_COMPILE ?=

# -------- paths --------
SRC_DIR      := src
NEURON_DIR   := $(SRC_DIR)/neurons
SYNAPSE_DIR  := $(SRC_DIR)/synapses
# Per-profile object tree, so a host build and a cross build never share
# objects. Without this, switching PROFILE silently links mismatched code.
BUILD_DIR    := build/$(PROFILE)
# The host archive stays at lib/libspires.a: that path is the library's
# existing contract with everything that links it. Other profiles nest.
LIB_DIR      := $(if $(filter host,$(PROFILE)),lib,lib/$(PROFILE))

# -------- feature toggles --------
# Each is consumed below to pick flags; flipping one is the supported way to
# retarget the library. A profile fragment may override any of them.
OPENMP  ?= 1
BLAS    ?= openblas
DEBUG   ?= 0
LTO     ?= 0

PROFILES := host $(patsubst mk/%.mk,%,$(wildcard mk/*.mk))

# -------- profile fragment --------
# Loaded before anything is resolved from these toggles, so a profile can set
# CC, CROSS_COMPILE, BLAS, OPENMP and have it take effect. Adding a profile
# means adding mk/<name>.mk; nothing in this file needs to change.
ifneq ($(PROFILE),host)
  ifeq ($(wildcard mk/$(PROFILE).mk),)
    $(error unknown PROFILE '$(PROFILE)'; available: $(PROFILES))
  endif
  include mk/$(PROFILE).mk
endif

# -------- tools --------
# CC, AR and ARFLAGS carry built-in defaults in GNU make, and ?= is a no-op
# against those (it only fires when a variable is genuinely undefined). Test
# the origin instead, which replaces make's default while still letting the
# command line and the environment win.
ifeq ($(origin CC),default)
  CC := $(CROSS_COMPILE)clang
endif
ifeq ($(origin AR),default)
  AR := $(CROSS_COMPILE)ar
endif
ifeq ($(origin ARFLAGS),default)
  ARFLAGS := rcs
endif
RANLIB ?= $(CROSS_COMPILE)ranlib

# -------- flags --------
# CPPFLAGS: preprocessor.  CFLAGS: code generation.  LDFLAGS/LDLIBS: link.
# Kept separate so a consumer can override one without clobbering the rest.
WARNINGS ?= -Wall -Wextra -Wpedantic -Wshadow
OPTIMIZE ?= -O2

ifeq ($(DEBUG),1)
  OPTIMIZE     := -O0
  EXTRA_CFLAGS += -g3 -fno-omit-frame-pointer
else
  EXTRA_CFLAGS += -g
endif
ifeq ($(LTO),1)
  EXTRA_CFLAGS += -flto
endif

CPPFLAGS ?=
CFLAGS   ?=
LDFLAGS  ?=
LDLIBS   ?=

INCLUDES := -Iinclude -I$(SRC_DIR) -I$(NEURON_DIR) -I$(SYNAPSE_DIR)

# BLAS/LAPACK discovery: ask pkg-config, fall back to the historical path.
# BLAS=none compiles without them, which needs the linear-algebra paths
# disabled too -- see mk/embedded.mk for what that still requires.
ifeq ($(BLAS),openblas)
  BLAS_CFLAGS ?= $(shell pkg-config --cflags openblas lapacke 2>/dev/null)
  BLAS_LIBS   ?= $(shell pkg-config --libs   openblas lapacke 2>/dev/null)
  ifeq ($(strip $(BLAS_CFLAGS)),)
    BLAS_CFLAGS := -I/usr/include/openblas
  endif
  ifeq ($(strip $(BLAS_LIBS)),)
    BLAS_LIBS := -lopenblas -llapacke
  endif
else ifeq ($(BLAS),none)
  BLAS_CFLAGS :=
  BLAS_LIBS   :=
else
  $(error BLAS must be 'openblas' or 'none', got '$(BLAS)')
endif

ifeq ($(OPENMP),1)
  OPENMP_FLAGS ?= -fopenmp
else
  OPENMP_FLAGS :=
  # An OpenBLAS built with OpenMP threading reports -fopenmp through
  # pkg-config. Left alone it lands on the compile line and re-enables the
  # pragmas that OPENMP=0 was meant to switch off.
  BLAS_CFLAGS := $(filter-out -fopenmp,$(BLAS_CFLAGS))
  BLAS_LIBS   := $(filter-out -fopenmp,$(BLAS_LIBS))
endif

# -MMD -MP emit a .d per object listing its headers, so editing a header
# rebuilds what depends on it. Without this the tree needs a manual clean.
DEPFLAGS := -MMD -MP

# Feature macros. These are the contract between a toggle here and the
# #ifdefs that gate the corresponding code. NOTE: no source consults
# SPIRES_NO_BLAS yet, so BLAS=none still fails at the first #include
# <cblas.h>. The define exists so the source-side guards have something
# stable to test; until they are written, BLAS=none does not build.
FEATURE_DEFS :=
ifeq ($(BLAS),none)
  FEATURE_DEFS += -DSPIRES_NO_BLAS
endif

ALL_CPPFLAGS = $(INCLUDES) $(BLAS_CFLAGS) $(FEATURE_DEFS) $(CPPFLAGS)
ALL_CFLAGS   = $(OPTIMIZE) $(WARNINGS) $(OPENMP_FLAGS) $(EXTRA_CFLAGS) \
               $(DEPFLAGS) $(CFLAGS)
# Published for consumers via `make print-config`; this target builds an
# archive, so the library itself never links.
ALL_LDLIBS   = $(BLAS_LIBS) $(OPENMP_FLAGS) -lm $(LDLIBS)

# -------- sources --------
SRCS := \
  $(SRC_DIR)/rng.c        \
  $(SRC_DIR)/plasticity.c \
  $(SRC_DIR)/math_utils.c \
  $(SRC_DIR)/sparse.c     \
  $(SRC_DIR)/synapse.c    \
  $(SRC_DIR)/neuron.c     \
  $(SRC_DIR)/reservoir.c  \
  $(SRC_DIR)/spires_api.c \
  $(SRC_DIR)/agile.c      \
  $(SRC_DIR)/spires_opt_agile.c \
  $(wildcard $(NEURON_DIR)/*.c) \
  $(wildcard $(SYNAPSE_DIR)/*.c) \
  $(PROFILE_SRCS)

OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

STATIC_LIB := $(LIB_DIR)/libspires.a

# -------- install locations (GNU conventions) --------
PREFIX     ?= /usr/local
DESTDIR    ?=
libdir     ?= $(PREFIX)/lib
includedir ?= $(PREFIX)/include

# -------- verbosity --------
ifeq ($(V),1)
  Q :=
else
  Q := @
endif

# -------- targets --------
.PHONY: all library clean distclean install uninstall help print-config
.DEFAULT_GOAL := library

all: library
library: $(STATIC_LIB)

$(STATIC_LIB): $(OBJS) | $(LIB_DIR)
	@echo "  AR      $@"
	$(Q)$(AR) $(ARFLAGS) $@ $(OBJS)
	$(Q)$(RANLIB) $@ 2>/dev/null || true

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@echo "  CC      $<"
	$(Q)$(CC) $(ALL_CPPFLAGS) $(ALL_CFLAGS) -c $< -o $@

$(LIB_DIR):
	@mkdir -p $@

install: $(STATIC_LIB)
	@echo "  INSTALL $(DESTDIR)$(libdir)"
	$(Q)mkdir -p $(DESTDIR)$(libdir) $(DESTDIR)$(includedir)
	$(Q)cp $(STATIC_LIB) $(DESTDIR)$(libdir)/
	$(Q)cp include/spires.h $(DESTDIR)$(includedir)/

uninstall:
	$(Q)rm -f $(DESTDIR)$(libdir)/libspires.a $(DESTDIR)$(includedir)/spires.h

clean:
	$(Q)rm -rf $(BUILD_DIR) $(LIB_DIR)

distclean:
	$(Q)rm -rf build lib

print-config:
	@echo "PROFILE   = $(PROFILE)"
	@echo "CC        = $(CC)"
	@echo "BLAS      = $(BLAS)   OPENMP = $(OPENMP)   DEBUG = $(DEBUG)   LTO = $(LTO)"
	@echo "OUTPUT    = $(STATIC_LIB)"
	@echo "CPPFLAGS  = $(ALL_CPPFLAGS)"
	@echo "CFLAGS    = $(ALL_CFLAGS)"
	@echo "LDLIBS    = $(ALL_LDLIBS)"

help:
	@echo "SPIRES build"
	@echo
	@echo "  make                       build $(STATIC_LIB)"
	@echo "  make PROFILE=<name>        use mk/<name>.mk"
	@echo "  make install PREFIX=...    install, honours DESTDIR"
	@echo "  make clean | distclean     this profile | every profile"
	@echo "  make print-config          resolved flags"
	@echo
	@echo "profiles: $(PROFILES)"
	@echo
	@echo "variables (showing current value):"
	@echo "  PROFILE=$(PROFILE)"
	@echo "  CROSS_COMPILE=$(CROSS_COMPILE)      toolchain prefix, e.g. arm-none-eabi-"
	@echo "  CC=$(CC)"
	@echo "  BLAS=$(BLAS)                openblas | none"
	@echo "  OPENMP=$(OPENMP)                    1 | 0"
	@echo "  DEBUG=$(DEBUG)                     1 | 0"
	@echo "  LTO=$(LTO)                       1 | 0"
	@echo "  V=$(V)                         1 to echo commands"
	@echo
	@echo "consumers link with: $(ALL_LDLIBS)"

# Header dependencies, generated by -MMD. Must come last.
-include $(DEPS)
