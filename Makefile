
srcs := $(wildcard src/*.c)
target := a.out
cflags := -lm -lopenblas
CFLAGS += -Wall -Wextra -Wpedantic
# cflags += -O2
cflags += -fsanitize=address -g

test_srcs := $(filter-out src/main.c, $(srcs)) test/overfit.c
test_target := overfit_test.out

.PHONY: all clean run test

all: $(target) $(test_target)

$(target): $(srcs)
	clang $(cflags) $^ -o $(target)

run: $(target)
	./$(target) train

$(test_target): $(test_srcs)
	clang $(cflags) $^ -o $(test_target)

test: $(test_target)
	./$(test_target)

clean:
	rm -f $(target) $(test_target)
