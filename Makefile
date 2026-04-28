CC = gcc
MPICC = mpicc
NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -O2 -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L
OMPFLAGS = -fopenmp
LDFLAGS = -lm

# macOS OpenMP support via Homebrew
ifeq ($(shell uname), Darwin)
    ifeq ($(shell [ -d /opt/homebrew/opt/libomp ] && echo exists), exists)
        OMPFLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
        LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
    endif
endif
PTHREADFLAGS = -pthread
CUDA_ARCH ?= sm_89
NVFLAGS = -O2 -arch=$(CUDA_ARCH)

COMMON_SRCS = src/common/off_parser.c src/common/voxel_grid.c \
	src/common/timer.c src/common/sat.c src/common/voxelize.c
COMMON_OBJS = $(COMMON_SRCS:.c=.o)
CUDA_COMMON_OBJS = src/common/off_parser.o src/common/voxel_grid.o src/common/voxelize.o

.PHONY: all sequential openmp pthreads mpi cuda verify test clean

all: sequential openmp pthreads mpi verify test
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(MAKE) cuda; \
	else \
		echo "Skipping CUDA build: nvcc not found on PATH"; \
	fi

sequential: bin/voxelize_seq

openmp: bin/voxelize_omp

pthreads: bin/voxelize_pth

mpi: bin/voxelize_mpi

cuda:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
		echo "CUDA toolkit not configured: nvcc not found on PATH" >&2; \
		exit 1; \
	fi
	$(MAKE) bin/voxelize_cuda

verify: bin/verify

test: bin/test_sat
	./bin/test_sat

bin/voxelize_seq: src/sequential/main.c $(COMMON_OBJS) | bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bin/voxelize_omp: src/openmp/main.c $(COMMON_OBJS) | bin
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)

bin/voxelize_pth: src/pthreads/main.c src/pthreads/worker.c $(COMMON_OBJS) | bin
	$(CC) $(CFLAGS) $(PTHREADFLAGS) -o $@ $^ $(LDFLAGS)

bin/voxelize_mpi: src/mpi/main.c $(COMMON_OBJS) | bin
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

CUDA_HOST_CFLAGS = -O2 -Wall -Wextra -D_POSIX_C_SOURCE=200809L

bin/voxelize_cuda: src/cuda/main.cu src/cuda/voxelize_kernel.cu $(CUDA_COMMON_OBJS) | bin
	$(NVCC) $(NVFLAGS) -Xcompiler "$(CUDA_HOST_CFLAGS)" -o $@ $^

bin/verify: src/common/verify.c | bin
	$(CC) $(CFLAGS) -o $@ $<

bin/test_sat: src/common/test_sat.c src/common/sat.c | bin
	$(CC) -O0 -g -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L -o $@ $^ -lm

src/common/%.o: src/common/%.c
	$(CC) $(CFLAGS) -c $< -o $@

bin:
	mkdir -p bin

clean:
	rm -f src/common/*.o bin/* results/*.csv results/*.voxel
	rm -rf results/run_* results/summary
