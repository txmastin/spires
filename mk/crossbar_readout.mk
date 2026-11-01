# ngspice crossbar-readout profile.
#
#   make PROFILE=crossbar_readout
#
# This extends the host library with the SPICE-simulated crossbar readout.
# It requires the ngspice shared library and headers discoverable through
# pkg-config. The default host profile remains independent of ngspice.

# Add the optional crossbar implementation to the core source list.
PROFILE_SRCS := $(wildcard $(SRC_DIR)/crossbar_readout/*.c)

# Expose the crossbar configuration and run functions in the public API.
CPPFLAGS += -DSPIRES_ENABLE_CROSSBAR_READOUT

# Use pkg-config for portable ngspice discovery. The online simulator also
# uses POSIX threads to synchronize SPIRES with ngspice's background worker.
CPPFLAGS += $(shell pkg-config --cflags ngspice)
EXTRA_CFLAGS += -pthread
LDLIBS += $(shell pkg-config --libs ngspice) -pthread
