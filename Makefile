# Compiler
CC = clang

# Compiler flags
CFLAGS = -Wall -Wextra -Wpedantic -Wshadow -fsanitize=undefined -g

# Output executable name
TARGET = reservoir_sim

# Source files
SRC = main.c neuron.c LIF.c FLIF.c reservoir.c
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

