
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

get-deps:
	sudo apt-get update
	# CMake is needed to build gtest.
	sudo apt-get install -y cmake
.PHONY: get-deps

GTEST_DIR = googletest
GTEST_LIB = $(GTEST_DIR)/build/googlemock/gtest/libgtest.a
$(GTEST_LIB):
	rm -rf $(GTEST_DIR)
	git clone https://github.com/google/googletest.git $(GTEST_DIR)
	cd $(GTEST_DIR) && git checkout release-1.8.1 && rm -rf build && mkdir build && cd build && cmake .. && make -j8

INC += -I$(GTEST_DIR)/googletest/include
LIB += -L$(GTEST_DIR)/build/googlemock/gtest -lgtest -lgtest_main -pthread

# Note that this cub dir is hard-coded in the tests too, to be looked up at runtime.
CUB_DIR = /tmp/cub
CUB_HEADER = $(CUB_DIR)/cub/cub.cuh
$(CUB_HEADER):
	rm -rf $(CUB_DIR)
	git clone https://github.com/NVlabs/cub.git $(CUB_DIR)
	cd $(CUB_DIR) && git checkout v1.8.0

INC += -I$(CUB_DIR)

jitify_test: jitify_test.cpp $(HEADERS) $(GTEST_LIB) $(CUB_HEADER)
	$(CXX) -o $@ $< -std=c++11 -O3 -Wall $(INC) $(LIB)

test: jitify_test
	./jitify_test
.PHONY: test

doc: jitify.hpp Doxyfile
	$(DOXYGEN) Doxyfile
.PHONY: doc

clean:
	rm -f stringify
	rm -f example_headers/*.jit
	rm -f jitify_example
	rm -f jitify_test
	rm -rf $(GTEST_DIR)
	rm -rf $(CUB_DIR)
.PHONY: clean
