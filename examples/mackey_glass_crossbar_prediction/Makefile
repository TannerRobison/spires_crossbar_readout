SPIRES_ROOT ?= ../../

CC := clang
CFLAGS := -O2 -Wall -Wextra -DSPIRES_ENABLE_CROSSBAR_READOUT \
	-I$(SPIRES_ROOT)/include $(shell pkg-config --cflags ngspice)
LDFLAGS := -fopenmp -pthread
LDLIBS := -L$(SPIRES_ROOT)/lib/crossbar_readout -lspires \
	-llapacke -lopenblas -lngspice -lm

BIN_DIR := bin
DATA_DIR := data
SRC := mackey_glass_crossbar_prediction.c
BIN := $(BIN_DIR)/mackey_glass_crossbar_prediction

.PHONY: all clean
all: $(BIN) | $(DATA_DIR)

$(BIN): $(SRC) $(SPIRES_ROOT)/lib/crossbar_readout/libspires.a | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(BIN_DIR) $(DATA_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR)
