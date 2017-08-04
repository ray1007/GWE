ifeq ($(shell uname -s), Darwin)
	CC = clang
else
	CC = gcc
	CFLAGS += -pthread -lm
endif

CFLAGS += -Ofast -std=c99

all: gwe.c
	${CC} gwe.c ${CFLAGS} -o gwe

