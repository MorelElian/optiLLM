

CC = gcc
CFLAGS =  -O3 -Wall  -ffast-math -fno-omit-frame-pointer
CFLAGS_DUMP =  -O0 -Wall   -fno-omit-frame-pointer
LDLIBS = -lm


TARGETS = main main_dump



main_OBJS = build/main.o build/forward.o build/dumper.o


main_dump_OBJS = build/main_dump.o build/forward_dump.o build/dumper_dump.o


all: $(TARGETS)



main: $(main_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

main_dump: $(main_dump_OBJS)
	$(CC) $(CFLAGS_DUMP) -o $@ $^ $(LDLIBS)


build/%.o: %.c
	mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@


build/%_dump.o: %.c
	mkdir -p build
	$(CC) $(CFLAGS_DUMP) -DDUMPER -c $< -o $@


clean:
	rm -rf build $(TARGETS)
