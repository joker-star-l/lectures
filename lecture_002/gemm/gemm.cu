#include <cuda.h>
#include <stdio.h>

__global__ void gemmKernel(float *M, float *N, float *P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        float value = 0;
        for (int k = 0; k < width; ++k) {
            value += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = value;
    }
}

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

void gemm(float *M, float * N, float* P, int width) {
    float *M_d, *N_d, *P_d;
    size_t size = width * width * sizeof(float);

    cudaMalloc((void **)&M_d, size);
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&P_d, size);

    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(cdiv(width, threads.x), cdiv(width, threads.y));

    gemmKernel<<<blocks, threads>>>(M_d, N_d, P_d, width);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    const int width = 17;
    float M[width * width];
    float N[width * width];
    float P[width * width];

    // Initialize matrices M and N
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (i == j) {
                M[i * width + j] = i + 1;
            } else {
                M[i * width + j] = 0;
            }
            N[i * width + j] = 1;
        }
    }

    gemm(M, N, P, width);

    // Print the result matrix P
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.1f ", P[i * width + j]);
        }
        printf("\n");
    }

    return 0;
}
