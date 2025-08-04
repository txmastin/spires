# Compiler and flags
CC = clang
CFLAGS = -Isrc -Wall -g  # -Isrc is the include path, -Wall shows warnings, -g adds debug info
LDFLAGS = -lm            # Linker flags for the math library

# Source files for the library
LIB_SRCS = src/reservoir.c src/neuron.c src/math_utils.c src/neurons/FLIF_GL.c src/neurons/LIF_Discrete.c src/neurons/LIF_Bio.c src/neurons/FLIF.c src/neurons/FLIF_Caputo.c 
# Rule to build the integration test
test_system: tests/test_neuron.c $(LIB_SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# '$^' means "all the prerequisites" (the .c files)
# '$@' means "the target" (the file name 'test_system')

# Rule to clean up compiled files
clean:
	rm -f test_system
