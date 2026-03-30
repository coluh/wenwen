
srcs := $(wildcard src/*.c)

a.out: $(srcs)
	gcc $(srcs) -g

run: a.out
	./a.out
