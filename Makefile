
GXX     ?= g++
DOXYGEN ?= doxygen
CXXFLAGS ?= -O3 -Wall -g -fmessage-length=80

CXX11 ?= 1

CUDA_DIR ?= /usr/local/cuda

CXXFLAGS += -pthread

ifeq ($(CXX11),1)
	CXXFLAGS += -std=c++11
endif

EMBED_BEGIN = -rdynamic -Wl,-b,binary,
EMBED_END   = ,-b,default

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	# Embed a header into the executable (only supported by gcc on Linux)
	LIB += $(EMBED_BEGIN)example_headers/my_header2.cuh$(EMBED_END)
	CXXFLAGS += -D LINUX
	CUDA_LIB_DIR = $(CUDA_DIR)/lib64
else ifeq ($(UNAME_S),Darwin)
	CUDA_LIB_DIR = $(CUDA_DIR)/lib
endif

INC += -I$(CUDA_DIR)/include
LIB += -ldl -L$(CUDA_LIB_DIR) -lcuda -lcudart -lnvrtc

HEADERS = jitify.hpp \
          example_headers/my_header1.cuh.jit \
          example_headers/my_header2.cuh

jitify_example: jitify_example.cpp $(HEADERS)
	$(GXX) -o $@ $< $(CXXFLAGS) $(INC) $(LIB)

%.jit: % stringify
	./stringify $< > $@

stringify: stringify.cpp
	$(GXX) -o $@ $< -O3 -Wall

doc: jitify.hpp Doxyfile
	$(DOXYGEN) Doxyfile
.PHONY: doc

clean:
	rm -f stringify
	rm -f example_headers/*.jit
	rm -f jitify_example
.PHONY: clean
