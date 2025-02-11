# Compiler
CC = clang

# Compiler flags (-fsanitize=address currently removed for compatibility with gdb)
CFLAGS = -Wall -Wextra -Wpedantic -Wshadow -g

# Output executable name
TARGET = reservoir_sim

# Source files
SRC = main.c neuron.c neurons/LIF.c neurons/FLIF.c reservoir.c
OBJ = $(SRC:.c=.o)

# Header files (for dependencies)
DEPS = neuron.h LIF.h FLIF.h reservoir.h

# Default target (compile everything)
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

# Compile .c files into .o object files
%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up compiled files
clean:
	rm -f $(OBJ) $(TARGET)

# Debugging: Print variables
print-%: ; @echo $* = $($*)

