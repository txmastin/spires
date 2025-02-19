# Compiler
CC = clang

# Compiler flags (-fsanitize=address currently removed for compatibility with gdb)
CFLAGS = -Wall -Wextra -Wpedantic -Wshadow -g

# Linker flags
LDFLAGS = -lm

# Output executable name
TARGET = reservoir_sim

# Source files
SRC = main.c neuron.c neurons/LIF.c neurons/FLIF.c reservoir.c math_utils.c
OBJ = $(SRC:.c=.o)

# Header files (for dependencies)
DEPS = neuron.h LIF.h FLIF.h reservoir.h math_utils.h math.h

# Default target (compile everything)
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile .c files into .o object files
%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up compiled files
clean:
	rm -f $(OBJ) $(TARGET)

# Debugging: Print variables
print-%: ; @echo $* = $($*)

