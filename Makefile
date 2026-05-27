
srcs := $(wildcard src/*.c)
target := a.out
cflags := -lm -lopenblas
cflags += -O2
CFLAGS += -Wall -Wextra -Wpedantic
# cflags += -fsanitize=address -g

.PHONY: clean run

$(target): $(srcs)
	clang $(cflags) $(srcs) -o $(target)

run: $(target)
	./$(target)

clean:
	rm -f $(target)
