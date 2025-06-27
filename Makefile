# ==== Compilation config ====

CC = gcc
CFLAGS = -O3 -Wall -ffast-math -fno-omit-frame-pointer -flto
LDLIBS = -lm

# ==== Cibles ====

TARGETS = main main_dump

# ==== Fichiers objets ====

# Objets pour version normale
main_OBJS = build/main.o build/forward.o build/dumper.o

# Objets pour version avec DUMPER
main_dump_OBJS = build/main_dump.o build/forward_dump.o build/dumper_dump.o

# ==== Règle par défaut ====

all: $(TARGETS)

# ==== Exécutables ====

main: $(main_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

main_dump: $(main_dump_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# ==== Compilation normale ====

build/%.o: %.c
	mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

# ==== Compilation avec DUMPER ====

build/%_dump.o: %.c
	mkdir -p build
	$(CC) $(CFLAGS) -DDUMPER -c $< -o $@

# ==== Nettoyage ====

clean:
	rm -rf build $(TARGETS)
