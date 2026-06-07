
srcs := $(wildcard src/*.c)
target := a.out
cflags := -lm -lopenblas
CFLAGS += -Wall -Wextra -Wpedantic
# cflags += -O2
cflags += -fsanitize=address -g

.PHONY: clean run

$(target): $(srcs)
	clang $(cflags) $(srcs) -o $(target)

run: $(target)
	./$(target)

clean:
	rm -f $(target)
