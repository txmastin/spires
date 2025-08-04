# Compiler and flags
CC = clang
CFLAGS = -Wall -Wextra -Wpedantic -Wshadow -g
LDFLAGS = -lm

# Executable
TARGET = run_simulation

# Source files (automatically find all .c files)
SRC := $(wildcard *.c) $(wildcard neurons/*.c)
OBJ := $(SRC:.c=.o)

# Header paths for dependency generation
INCLUDES := -I. -Ineurons

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Pattern rule to compile .c to .o
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJ) $(TARGET)

# Debug helper
print-%: ; @echo $* = $($*)

