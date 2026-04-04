
srcs := $(wildcard src/*.c)

a.out: $(srcs)
	clang -fsanitize=address -g $(srcs) -lm -lopenblas

run: a.out
	./a.out
